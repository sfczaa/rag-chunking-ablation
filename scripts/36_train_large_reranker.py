"""Stage 15 (step 1) - train bge-reranker-large with the Stage 8 recipe, resumable.

One arm: the Stage 8 groups (2034), the Stage 8 recipe (2 epochs, lr 2e-5, 10%
warmup, weight decay 0.01, 4 groups per step, fp16, clipping at 1.0, seed 42, max
length 512), and a different base model pinned to one revision.

The loop is the resumable loop of scripts/33_train_data_scale.py with three changes
that do not alter the recipe:

- each optimizer step of 4 groups runs as forward/backward passes of
  STAGE15_MICRO_GROUPS groups, each loss weighted by its share of the step, so the
  step's gradient is the same mean over the 4 groups (dropout masks and the order
  of float operations differ, so the run is not bit-identical to one pass);
- gradient checkpointing, which recomputes activations instead of storing them;
- a guard that stops the run after more than MAX_NONFINITE non-finite losses.

The checkpoint, run lock and run identity follow Stage 14 (rag_chunk/run_guard.py).

Usage:
    python scripts/36_train_large_reranker.py --smoke --account A
    python scripts/36_train_large_reranker.py --mode fresh --account A
    python scripts/36_train_large_reranker.py --mode resume --account B
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import random
import shutil
import sys
import tempfile
import time

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402

ARM = "large"
N_STAGE8_GROUPS = 2034
MAX_NONFINITE = 10


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S14 = _load_script("33_train_data_scale.py", "stage14_train")


def arm_dir() -> pathlib.Path:
    return C.MODELS_DIR / C.STAGE15_MODEL_DIRNAME / ARM


def _progress(event: dict) -> None:
    path = C.RESULTS_LATEST_DIR / "stage15_train_progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def train_resumable(groups: list[dict], *, init_model: str, revision: str | None,
                    out: pathlib.Path, epochs: int, lr: float, groups_per_step: int,
                    micro_groups: int, grad_ckpt: bool, seed: int, ckpt_every: int,
                    account: str, save_final, stop_after: int | None = None,
                    log=_progress) -> dict:
    """scripts/33's loop with micro-batches, gradient checkpointing and a loss guard.

    ``stop_after`` ends the run after that many steps without saving final weights;
    only the smoke test uses it, to exercise a resume.
    """
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    group_width = 1 + len(groups[0]["negs"])
    if any(1 + len(g["negs"]) != group_width for g in groups):
        raise SystemExit("[stage15-train] groups have mixed widths")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    resume = S14.latest_checkpoint(out)
    if resume:
        tokenizer = AutoTokenizer.from_pretrained(str(resume[0]))
        model = AutoModelForSequenceClassification.from_pretrained(str(resume[0]))
    else:
        tokenizer = AutoTokenizer.from_pretrained(init_model, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(init_model,
                                                                   revision=revision)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    steps_per_epoch = (len(groups) + groups_per_step - 1) // groups_per_step
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=float(C.STAGE8_WEIGHT_DECAY))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * float(C.STAGE8_WARMUP_FRAC)),
        num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss()
    max_len = int(C.RERANK_MAX_LENGTH)

    rng = random.Random(seed)
    torch.manual_seed(seed)
    step, start_epoch, skip_batches = 0, 1, 0
    epoch_loss, epoch_batches, order = 0.0, 0, None
    elapsed_before, nonfinite = 0.0, 0
    if resume:
        state = torch.load(resume[0] / "trainer_state.pt", map_location="cpu",
                           weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        rng.setstate(state["py_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, start_epoch = state["step"], state["epoch"]
        skip_batches, order = state["batch_in_epoch"], state["order"]
        epoch_loss, epoch_batches = state["epoch_loss"], state["epoch_batches"]
        elapsed_before = state.get("elapsed_seconds", 0.0)
        nonfinite = state.get("nonfinite", 0)
        print(f"[stage15-train] resuming at step {step}/{total_steps} (epoch "
              f"{start_epoch}, batch {skip_batches}) from {resume[0].name}", flush=True)
        log({"account": account, "arm": ARM, "event": "resumed", "step": step,
             "total": total_steps})
    else:
        print(f"[stage15-train] {len(groups)} groups (width {group_width}) x {epochs} "
              f"epochs = {total_steps} steps on {device}; {micro_groups} groups per "
              f"pass; checkpoint every {ckpt_every} steps", flush=True)
        log({"account": account, "arm": ARM, "event": "started", "step": 0,
             "total": total_steps})

    t_start = time.perf_counter()
    last_epoch_loss = None
    for epoch in range(start_epoch, epochs + 1):
        if order is None:
            order = list(range(len(groups)))
            rng.shuffle(order)
        for b, i in enumerate(range(0, len(order), groups_per_step)):
            if b < skip_batches:
                continue
            batch = [groups[j] for j in order[i:i + groups_per_step]]
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for m in range(0, len(batch), micro_groups):
                part = batch[m:m + micro_groups]
                texts_a, texts_b = [], []
                for g in part:
                    for chunk in [g["pos"]] + g["negs"]:
                        texts_a.append(g["question"])
                        texts_b.append(chunk)
                enc = tokenizer(texts_a, texts_b, padding=True, truncation=True,
                                max_length=max_len, return_tensors="pt").to(device)
                target = torch.zeros(len(part), dtype=torch.long, device=device)
                with torch.autocast(device_type=device, enabled=use_amp):
                    logits = model(**enc).logits.view(len(part), group_width)
                    loss = loss_fn(logits, target) * (len(part) / len(batch))
                scaler.scale(loss).backward()
                step_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            if math.isfinite(step_loss):
                epoch_loss += step_loss
                epoch_batches += 1
            else:
                nonfinite += 1
                print(f"[stage15-train] WARN: non-finite loss at step {step} "
                      f"({nonfinite} so far)", flush=True)
                if nonfinite > MAX_NONFINITE:
                    log({"account": account, "arm": ARM, "event": "failed-training",
                         "step": step, "total": total_steps})
                    raise SystemExit(f"[stage15-train] FAILED-TRAINING: more than "
                                     f"{MAX_NONFINITE} non-finite losses")
            elapsed = elapsed_before + time.perf_counter() - t_start
            if step % 50 == 0 or step == total_steps:
                mem = (f", peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
                       if device == "cuda" else "")
                print(f"[stage15-train] step {step}/{total_steps} epoch {epoch} "
                      f"loss {epoch_loss / max(epoch_batches, 1):.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} ({elapsed / 60:.1f} min{mem})",
                      flush=True)
            if ckpt_every and step % ckpt_every == 0 and step < total_steps:
                state = {"step": step, "epoch": epoch, "batch_in_epoch": b + 1,
                         "order": order, "epoch_loss": epoch_loss,
                         "epoch_batches": epoch_batches,
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                         "scaler": scaler.state_dict(), "py_rng": rng.getstate(),
                         "torch_rng": torch.get_rng_state(),
                         "cuda_rng": (torch.cuda.get_rng_state_all()
                                      if device == "cuda" else None),
                         "elapsed_seconds": elapsed, "nonfinite": nonfinite}
                S14._save_checkpoint(model, tokenizer, state, out)
                log({"account": account, "arm": ARM, "event": "checkpoint",
                     "step": step, "total": total_steps})
                print(f"[stage15-train] CHECKPOINT SAVED AND VERIFIED at step "
                      f"{step}/{total_steps}. Safe to stop this runtime now.", flush=True)
            if stop_after is not None and step >= stop_after:
                return {"stopped_at": step, "total_steps": total_steps}
        last_epoch_loss = epoch_loss / max(epoch_batches, 1)
        print(f"[stage15-train] epoch {epoch}/{epochs} done - mean loss "
              f"{last_epoch_loss:.4f}", flush=True)
        order, skip_batches, epoch_loss, epoch_batches = None, 0, 0.0, 0

    save_final(model, tokenizer, out / "final")
    peak = (round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda"
            else None)
    return {"final_mean_epoch_loss": last_epoch_loss, "total_steps": total_steps,
            "group_width": group_width, "device": device, "nonfinite": nonfinite,
            "peak_gpu_gib": peak,
            "minutes": round((elapsed_before + time.perf_counter() - t_start) / 60, 1)}


def stage8_groups() -> list[dict]:
    from rag_chunk import rerank_finetune as rf

    groups = rf.load_groups()
    if len(groups) != N_STAGE8_GROUPS:
        raise SystemExit(f"[stage15] expected the {N_STAGE8_GROUPS} Stage 8 groups, "
                         f"found {len(groups)}")
    return groups


def recipe(epochs: int) -> dict:
    return {"base_model": C.STAGE15_BASE_MODEL, "base_revision": C.STAGE15_BASE_REVISION,
            "epochs": epochs, "lr": float(C.STAGE8_LR),
            "groups_per_step": int(C.STAGE8_GROUPS_PER_STEP),
            "micro_groups": int(C.STAGE15_MICRO_GROUPS),
            "gradient_checkpointing": bool(C.STAGE15_GRADIENT_CHECKPOINTING),
            "seed": int(C.STAGE8_SEED), "warmup_frac": float(C.STAGE8_WARMUP_FRAC),
            "weight_decay": float(C.STAGE8_WEIGHT_DECAY),
            "max_length": int(C.RERANK_MAX_LENGTH)}


def run_smoke(account: str, s17) -> None:
    """8 groups, 1 epoch: stop after step 1, resume from its checkpoint, finish."""
    from rag_chunk import rerank_finetune as rf

    groups = rf.load_groups()[:8]
    if len(groups) < 8:
        raise SystemExit("[stage15] smoke needs the Stage 8 groups")
    r = recipe(1)
    out = pathlib.Path(tempfile.mkdtemp(prefix="stage15_smoke_"))
    kwargs = dict(init_model=r["base_model"], revision=r["base_revision"], out=out,
                  epochs=1, lr=r["lr"], groups_per_step=r["groups_per_step"],
                  micro_groups=r["micro_groups"], grad_ckpt=r["gradient_checkpointing"],
                  seed=r["seed"], ckpt_every=1, account=account, save_final=s17._save,
                  log=lambda event: None)
    first = train_resumable(groups, stop_after=1, **kwargs)
    resumed_from = S14.latest_checkpoint(out)
    stats = train_resumable(groups, **kwargs)
    ok = ((out / "final" / "config.json").exists() and first.get("stopped_at") == 1
          and resumed_from is not None and resumed_from[1] == 1)
    shutil.rmtree(out, ignore_errors=True)
    print(f"[stage15] smoke: stopped at step 1, resumed from a verified checkpoint and "
          f"finished {stats['total_steps']} steps; peak GPU {stats['peak_gpu_gib']} GiB; "
          f"{'OK' if ok else 'FAILED'}. Nothing was kept.", flush=True)
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 15 step 1: bge-reranker-large with the Stage 8 recipe.")
    ap.add_argument("--mode", choices=("fresh", "resume"), default="resume",
                    help="fresh refuses to start if any Stage 15 training state exists")
    ap.add_argument("--account", default="unlabelled",
                    help="operational label recorded in the lock and progress log")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    ap.add_argument("--smoke", action="store_true",
                    help="8 groups, stop and resume once, temp dir, nothing kept")
    args = ap.parse_args()

    from rag_chunk import run_guard

    s17 = _load_script("17_train_reranker.py", "stage8_train")
    if args.smoke:
        run_smoke(args.account, s17)
        return

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE15_ROOT_ID)
    C.ensure_dirs()
    groups = stage8_groups()
    r = recipe(int(C.STAGE8_EPOCHS))
    identity = {"run_version": C.STAGE15_RUN_VERSION, "recipe": r,
                "trainer_sha256": run_guard.file_sha256(pathlib.Path(__file__)),
                "stage14_trainer_sha256": run_guard.file_sha256(
                    SCRIPTS / "33_train_data_scale.py"),
                "n_groups": len(groups), "groups_sha1": S14.groups_sha1(groups)}
    out = arm_dir()
    identity_path = out.parent / "run_identity.json"
    existing = identity_path.exists() or any(
        (out / d).exists() for d in S14.CKPT_DIRS + ("final",))
    if args.mode == "fresh" and existing:
        raise SystemExit("[stage15] Stage 15 training state already exists; use --mode "
                         "resume. Nothing was changed.")

    with run_guard.held_lock(C.DATA_ROOT / C.STAGE15_LOCK_FILE,
                             run_version=C.STAGE15_RUN_VERSION, account=args.account,
                             task="train", clear_stale=args.clear_stale_lock):
        for path in (out.parent, C.RESULTS_LATEST_DIR):
            run_guard.probe_directory(path)
        print(f"[stage15] run identity {run_guard.check_run_identity(identity_path, identity)}",
              flush=True)
        S14.assert_not_stage8(out)
        fingerprint = {"arm": ARM, "n_groups": len(groups),
                       "groups_sha1": identity["groups_sha1"], **r}
        marker = out / "final" / "training_meta.json"
        if marker.exists():
            done = json.loads(marker.read_text(encoding="utf-8"))
            if done.get("fingerprint") != fingerprint:
                raise SystemExit(f"[stage15] {marker} was trained with other settings")
            print(f"[stage15] already trained ({out}) - nothing to do", flush=True)
            meta = done
        else:
            stats = train_resumable(groups, init_model=r["base_model"],
                                    revision=r["base_revision"], out=out,
                                    epochs=r["epochs"], lr=r["lr"],
                                    groups_per_step=r["groups_per_step"],
                                    micro_groups=r["micro_groups"],
                                    grad_ckpt=r["gradient_checkpointing"], seed=r["seed"],
                                    ckpt_every=int(C.STAGE15_CKPT_EVERY),
                                    account=args.account, save_final=s17._save)
            meta = {"fingerprint": fingerprint, **stats}
            marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            for name in S14.CKPT_DIRS:
                shutil.rmtree(out / name, ignore_errors=True)
            _progress({"account": args.account, "arm": ARM, "event": "completed",
                       "step": stats["total_steps"], "total": stats["total_steps"]})
            print("[stage15] TRAINING COMPLETE: final weights and marker written.",
                  flush=True)

    print(f"[stage15] {meta['fingerprint']['n_groups']} groups, {meta['total_steps']} "
          f"steps, {meta['minutes']} min, final mean epoch loss "
          f"{meta['final_mean_epoch_loss']:.4f}, non-finite losses {meta['nonfinite']}")
    print("[stage15] next: python scripts/37_eval_large_reranker.py")


if __name__ == "__main__":
    main()
