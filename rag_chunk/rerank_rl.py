"""Stage 11 - a policy-gradient objective for the cross-encoder reranker.

Stage 8 fine-tuned ``bge-reranker-base`` with listwise softmax cross-entropy over
groups of one answer-bearing chunk and seven hard negatives, and gained +0.107
R@1 at fixed 15/0. The pool still holds the answer for 96% of questions while
R@1 sits at 0.736, which makes this the largest measured gap left in the
project. Cross-entropy optimises the log-probability of the positive; the
metric rewards where the positive lands. Stage 11 asks whether optimising the
ranking reward itself does better.

Formulation
-----------
The cross-encoder's scores over a group define a Plackett-Luce policy over
rankings (temperature ``tau``). For each group, ``G`` rankings are sampled with
the Gumbel-top-k trick and each gets the reciprocal rank of the positive as its
reward, a dense stand-in for R@1 that also rewards moving the positive from rank 5 to 2.

The gradient uses a leave-one-out baseline: each sample's advantage is its
reward minus the mean reward of the group's other samples. That is the
group-relative estimator known as GRPO, without the per-group standard
deviation rescaling, which would give near-certain groups the same weight as
uncertain ones. The log-probability is taken over the ranking's prefix up to
and including the positive. The reward depends only on that prefix, so leaving
the suffix out keeps the estimator unbiased and removes its noise.

Control
-------
Both arms run through :func:`train_arm` from the same weights with the same data
order, dropout stream, step count, schedule, clipping and precision. Ranking
samples draw from a dedicated generator so the RL arm does not perturb the
dropout masks the CE arm sees. The only difference is the loss.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import shutil
import tempfile
import time

import config as C


# --------------------------------------------------------------------------- #
# Plackett-Luce pieces (pure torch, locally testable)
# --------------------------------------------------------------------------- #
def plackett_luce_sample(scores, n_samples: int, temperature: float,
                         generator=None):
    """``(B, S, W)`` rankings sampled from PL(scores / temperature).

    Gumbel-top-k: adding independent Gumbel noise to the logits and sorting
    yields an exact Plackett-Luce sample.
    """
    import torch

    batch, width = scores.shape
    u = torch.rand((batch, n_samples, width), generator=generator,
                   device=scores.device, dtype=torch.float32)
    gumbel = -torch.log(-torch.log(u.clamp_(1e-12, 1.0 - 1e-12)))
    keys = scores.float().unsqueeze(1) / temperature + gumbel
    return keys.argsort(dim=-1, descending=True)


def positive_rank(rankings, positive: int = 0):
    """1-based rank of the positive candidate in each sampled ranking."""
    # float, not int: argmax over integer tensors is not supported on every
    # CUDA build, and this runs on the GPU during training
    return (rankings == positive).float().argmax(dim=-1) + 1


def prefix_log_prob(logits, rankings, ranks, temperature: float):
    """``(B, S)`` log-probability of each ranking's prefix through the positive.

    Position ``j`` contributes ``z[sigma_j] - logsumexp(z over the items not yet
    placed)``; positions after the positive are masked out.
    """
    import torch

    z = logits / temperature
    batch, n_samples, width = rankings.shape
    placed = torch.gather(z.unsqueeze(1).expand(batch, n_samples, width), 2,
                          rankings)
    remaining = torch.flip(torch.logcumsumexp(torch.flip(placed, [2]), dim=2),
                           [2])
    step = placed - remaining
    positions = torch.arange(width, device=logits.device).view(1, 1, width)
    mask = (positions < ranks.unsqueeze(-1)).to(step.dtype)
    return (step * mask).sum(dim=2)


def rl_loss(logits, n_samples: int, temperature: float, generator=None):
    """Group-relative policy-gradient loss for one batch of groups.

    ``logits`` is ``(B, W)`` with the positive in column 0. Returns
    ``(loss, stats)``; ``stats['live_fraction']`` is the share of groups whose
    samples disagreed on the reward, which is the share that produced any
    gradient at all.
    """
    import torch

    with torch.no_grad():
        rankings = plackett_luce_sample(logits.detach(), n_samples, temperature,
                                        generator)
        ranks = positive_rank(rankings)
        reward = 1.0 / ranks.float()
        others = (reward.sum(dim=1, keepdim=True) - reward) / (n_samples - 1)
        advantage = reward - others
        live = (reward.max(dim=1).values > reward.min(dim=1).values).float()
    log_prob = prefix_log_prob(logits.float(), rankings, ranks, temperature)
    loss = -(advantage * log_prob).mean()
    stats = {"reward_sample": float(reward.mean()),
             "live_fraction": float(live.mean()),
             "mean_abs_advantage": float(advantage.abs().mean())}
    return loss, stats


def greedy_rr(logits) -> float:
    """Mean reciprocal rank of the positive under the deterministic ranking."""
    import torch

    with torch.no_grad():
        rank = 1 + (logits[:, 1:] > logits[:, :1]).sum(dim=1)
        return float((1.0 / rank.float()).mean())


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_model(model, tokenizer, out_dir) -> None:
    """Serialise to local disk, copy onto Drive with retries, verify sizes.

    Writing the 1.1 GB safetensors file straight onto the Colab Drive mount
    occasionally fails (Stage 8 hit this), so staging locally is the default.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stage11_save_"))
    try:
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        for attempt in range(1, 4):
            try:
                for item in sorted(tmp.iterdir()):
                    shutil.copy2(item, out_dir / item.name)
                break
            except OSError as exc:
                if attempt == 3:
                    raise
                print(f"[stage11] WARN: copy attempt {attempt} failed ({exc!r}); "
                      "retrying in 15 s", flush=True)
                time.sleep(15)
        for item in tmp.iterdir():
            copied = out_dir / item.name
            if not copied.exists() or copied.stat().st_size != item.stat().st_size:
                raise OSError(f"saved file did not verify: {copied}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[stage11] saved and verified -> {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
# One arm
# --------------------------------------------------------------------------- #
def train_arm(groups: list[dict], *, arm: str, init_model: str, out_dir,
              epochs: int, lr: float, warmup_frac: float, weight_decay: float,
              groups_per_step: int, seed: int, samples: int,
              temperature: float, log_every: int, log_path=None,
              max_groups: int | None = None) -> dict:
    """Continue training ``init_model`` under ``arm`` in {"ce", "rl"}."""
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)

    if arm not in ("ce", "rl"):
        raise ValueError(f"arm must be 'ce' or 'rl', got {arm!r}")
    if max_groups:
        groups = groups[:max_groups]
    width = 1 + len(groups[0]["negs"])
    if any(1 + len(g["negs"]) != width for g in groups):
        raise SystemExit("[stage11] groups have mixed widths - rebuild them "
                         "with scripts/16_build_rerank_train_data.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(init_model)
    model = AutoModelForSequenceClassification.from_pretrained(init_model)
    model.to(device)
    model.train()

    steps_per_epoch = math.ceil(len(groups) / groups_per_step)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * warmup_frac),
        num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce_fn = torch.nn.CrossEntropyLoss()
    max_len = int(C.RERANK_MAX_LENGTH)

    rng = random.Random(seed)
    torch.manual_seed(seed)
    sampler = torch.Generator(device=device)
    sampler.manual_seed(seed)

    print(f"[stage11] arm={arm}: {len(groups)} groups x {epochs} epoch(s) = "
          f"{total_steps} steps on {device}"
          + (f", {samples} rankings/group at tau={temperature}" if arm == "rl"
             else ""), flush=True)

    log_rows: list[dict] = []
    window = {"loss": 0.0, "greedy_rr": 0.0, "reward_sample": 0.0,
              "live_fraction": 0.0, "mean_abs_advantage": 0.0, "n": 0}
    greedy_trace: list[float] = []
    live_sum, adv_sum = 0.0, 0.0
    step = 0
    started = time.perf_counter()
    for _ in range(epochs):
        order = list(range(len(groups)))
        rng.shuffle(order)
        for i in range(0, len(order), groups_per_step):
            batch = [groups[j] for j in order[i:i + groups_per_step]]
            texts_a, texts_b = [], []
            for g in batch:
                for chunk in [g["pos"]] + g["negs"]:
                    texts_a.append(g["question"])
                    texts_b.append(chunk)
            enc = tokenizer(texts_a, texts_b, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)

            optimizer.zero_grad(set_to_none=True)
            stats: dict = {}
            with torch.autocast(device_type=device, enabled=use_amp):
                logits = model(**enc).logits.view(len(batch), width)
                if arm == "ce":
                    target = torch.zeros(len(batch), dtype=torch.long,
                                         device=device)
                    loss = ce_fn(logits, target)      # positive at column 0
            if arm == "rl":
                loss, stats = rl_loss(logits.float(), samples, temperature,
                                      sampler)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            g_rr = greedy_rr(logits.detach().float())
            greedy_trace.append(g_rr)
            window["loss"] += float(loss.detach())
            window["greedy_rr"] += g_rr
            window["n"] += 1
            if arm == "rl":
                for key in ("reward_sample", "live_fraction",
                            "mean_abs_advantage"):
                    window[key] += stats[key]
                live_sum += stats["live_fraction"]
                adv_sum += stats["mean_abs_advantage"]

            if step % log_every == 0 or step == total_steps:
                n = window["n"]
                row = {"arm": arm, "step": step, "total_steps": total_steps,
                       "loss": round(window["loss"] / n, 5),
                       "greedy_rr": round(window["greedy_rr"] / n, 5),
                       "lr": scheduler.get_last_lr()[0],
                       "minutes": round((time.perf_counter() - started) / 60, 2)}
                if arm == "rl":
                    for key in ("reward_sample", "live_fraction",
                                "mean_abs_advantage"):
                        row[key] = round(window[key] / n, 5)
                log_rows.append(row)
                print(f"[stage11] {arm} step {step}/{total_steps} "
                      f"loss {row['loss']:.4f} greedy RR {row['greedy_rr']:.4f}"
                      + (f" live {row['live_fraction']:.2f}" if arm == "rl" else "")
                      + f" ({row['minutes']:.1f} min)", flush=True)
                if log_path is not None:
                    _write_log(log_path, log_rows)
                window = {k: 0.0 for k in window}
                window["n"] = 0

    save_model(model, tokenizer, out_dir)
    tenth = max(1, len(greedy_trace) // 10)
    return {
        "arm": arm, "steps": step, "device": device,
        "minutes": round((time.perf_counter() - started) / 60, 1),
        "greedy_rr_first_tenth": sum(greedy_trace[:tenth]) / tenth,
        "greedy_rr_last_tenth": sum(greedy_trace[-tenth:]) / tenth,
        "live_fraction": (live_sum / step) if arm == "rl" else None,
        "mean_abs_advantage": (adv_sum / step) if arm == "rl" else None,
    }


def _write_log(path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def arm_meta(out_dir) -> dict | None:
    """The completion marker of a trained arm, or ``None``."""
    path = pathlib.Path(out_dir) / "training_meta.json"
    if not path.exists() or not (pathlib.Path(out_dir) / "model.safetensors").exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
