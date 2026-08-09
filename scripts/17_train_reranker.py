"""Stage 8 (step 2) - fine-tune BAAI/bge-reranker-base on the mined groups.

Plain ``transformers`` + ``torch`` training loop (the sentence-transformers
*training* API has churned across major versions; inference still goes
through ``CrossEncoder``, which loads any HF sequence-classification
checkpoint directory, so the fine-tuned model plugs into the existing eval
path unchanged).

Objective: listwise softmax cross-entropy over each (1 positive +
STAGE8_NUM_NEGATIVES hard negatives) group — the standard reranker loss.
Pairs are tokenized with the same ``RERANK_MAX_LENGTH`` truncation the eval
uses, so training and inference see identical inputs.

Saves every epoch to ``models/bge_reranker_ft/epoch<i>/`` and the last epoch
to ``models/bge_reranker_ft/final/`` plus ``training_meta.json`` (base model,
data fingerprint, hyperparameters, final loss) for provenance. If a Colab
session dies mid-training, restart from the last saved epoch with
``--init-model``.

Usage:
    python scripts/17_train_reranker.py
    python scripts/17_train_reranker.py --epochs 2 --lr 2e-5
    python scripts/17_train_reranker.py --init-model <path>/epoch1 --epochs 1
    python scripts/17_train_reranker.py --max-groups 64   # tiny smoke run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402


def ft_model_dir() -> pathlib.Path:
    return C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME


def _save(model, tokenizer, out_dir: pathlib.Path) -> None:
    """Save with a local-staging fallback and retries. Writing the ~1.1 GB
    safetensors file straight onto the Colab Drive FUSE mount occasionally
    dies with a transient I/O error (os error 2); serializing to fast local
    disk first and then copying is reliable, and the copy is retried."""
    import shutil
    import tempfile

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    except Exception as exc:
        print(f"[stage8-train] WARN: direct save to {out_dir} failed "
              f"({exc!r}) — retrying via local staging")
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="stage8_save_"))
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        for attempt in range(1, 4):
            try:
                for item in sorted(tmp.iterdir()):
                    shutil.copy2(item, out_dir / item.name)
                break
            except OSError as exc2:
                if attempt == 3:
                    raise
                print(f"[stage8-train] WARN: copy attempt {attempt} failed "
                      f"({exc2!r}); retrying in 15 s")
                time.sleep(15)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[stage8-train] saved -> {out_dir}")


def _prune_previous_epoch(epoch: int) -> None:
    """Keep only the newest epoch checkpoint (~1.1 GB each — Drive quota).
    The freshly saved epoch fully supersedes the previous one for resuming."""
    import shutil

    prev = ft_model_dir() / f"epoch{epoch - 1}"
    if epoch > 1 and prev.exists():
        shutil.rmtree(prev, ignore_errors=True)
        print(f"[stage8-train] pruned older checkpoint {prev}")


def train(groups: list[dict], *, init_model: str, epochs: int, lr: float,
          groups_per_step: int, seed: int, max_groups: int | None) -> dict:
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)

    if max_groups:
        groups = groups[:max_groups]
    group_width = 1 + len(groups[0]["negs"])
    bad = [g for g in groups if 1 + len(g["negs"]) != group_width]
    if bad:
        raise SystemExit(f"[stage8-train] {len(bad)} group(s) have a "
                         f"different width than {group_width} — rebuild the "
                         "training data (script 16)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[stage8-train] WARN: no GPU — fine for a --max-groups smoke "
              "run, far too slow for real training")
    use_amp = device == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(init_model)
    model = AutoModelForSequenceClassification.from_pretrained(init_model)
    model.to(device)
    model.train()

    steps_per_epoch = (len(groups) + groups_per_step - 1) // groups_per_step
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=float(C.STAGE8_WEIGHT_DECAY))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(C.STAGE8_WARMUP_FRAC)),
        num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss()
    max_len = int(C.RERANK_MAX_LENGTH)

    print(f"[stage8-train] {len(groups)} groups (width {group_width}) x "
          f"{epochs} epochs = {total_steps} steps "
          f"({groups_per_step * group_width} pairs/step) on {device}")

    rng = random.Random(seed)
    torch.manual_seed(seed)
    step = 0
    last_epoch_loss = None
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(range(len(groups)))
        rng.shuffle(order)
        epoch_loss, epoch_batches = 0.0, 0
        for i in range(0, len(order), groups_per_step):
            batch = [groups[j] for j in order[i:i + groups_per_step]]
            texts_a, texts_b = [], []
            for g in batch:
                for chunk in [g["pos"]] + g["negs"]:
                    texts_a.append(g["question"])
                    texts_b.append(chunk)
            enc = tokenizer(texts_a, texts_b, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
            target = torch.zeros(len(batch), dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                logits = model(**enc).logits.view(len(batch), group_width)
                loss = loss_fn(logits, target)   # positive sits at index 0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            epoch_loss += float(loss.detach())
            epoch_batches += 1
            if step % 50 == 0 or step == total_steps:
                el = time.perf_counter() - t_start
                print(f"[stage8-train] step {step}/{total_steps} "
                      f"loss {epoch_loss / epoch_batches:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} "
                      f"({el / 60:.1f} min)")
        last_epoch_loss = epoch_loss / max(epoch_batches, 1)
        print(f"[stage8-train] epoch {epoch}/{epochs} done — "
              f"mean loss {last_epoch_loss:.4f}")
        _save(model, tokenizer, ft_model_dir() / f"epoch{epoch}")
        _prune_previous_epoch(epoch)

    _save(model, tokenizer, ft_model_dir() / "final")
    return {"final_mean_epoch_loss": last_epoch_loss,
            "total_steps": total_steps, "group_width": group_width,
            "device": device,
            "minutes": round((time.perf_counter() - t_start) / 60, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 8 step 2: fine-tune the cross-encoder reranker on "
                    "the mined NQ-train groups.")
    ap.add_argument("--init-model", default=None,
                    help="base checkpoint (default: config RERANKER_MODEL="
                         f"{C.RERANKER_MODEL}); pass an epoch dir to resume")
    ap.add_argument("--epochs", type=int, default=None,
                    help=f"default: config STAGE8_EPOCHS={C.STAGE8_EPOCHS}")
    ap.add_argument("--lr", type=float, default=None,
                    help=f"default: config STAGE8_LR={C.STAGE8_LR}")
    ap.add_argument("--groups-per-step", type=int, default=None,
                    help="groups per optimizer step (default: config "
                         f"STAGE8_GROUPS_PER_STEP={C.STAGE8_GROUPS_PER_STEP})")
    ap.add_argument("--seed", type=int, default=None,
                    help=f"default: config STAGE8_SEED={C.STAGE8_SEED}")
    ap.add_argument("--max-groups", type=int, default=None,
                    help="cap the group count (tiny smoke runs)")
    args = ap.parse_args()
    C.ensure_dirs()

    from rag_chunk import rerank_finetune as rf

    groups = rf.load_groups()
    if not groups:
        raise SystemExit("[stage8-train] no training groups found — run "
                         "`python scripts/16_build_rerank_train_data.py` first")

    init_model = args.init_model or C.RERANKER_MODEL
    epochs = args.epochs or int(C.STAGE8_EPOCHS)
    lr = args.lr or float(C.STAGE8_LR)
    gps = args.groups_per_step or int(C.STAGE8_GROUPS_PER_STEP)
    seed = args.seed if args.seed is not None else int(C.STAGE8_SEED)

    stats = train(groups, init_model=init_model, epochs=epochs, lr=lr,
                  groups_per_step=gps, seed=seed, max_groups=args.max_groups)

    meta = {
        "base_model": init_model,
        "n_groups": len(groups) if not args.max_groups
                    else min(len(groups), args.max_groups),
        "epochs": epochs, "lr": lr, "groups_per_step": gps, "seed": seed,
        "warmup_frac": float(C.STAGE8_WARMUP_FRAC),
        "weight_decay": float(C.STAGE8_WEIGHT_DECAY),
        "max_length": int(C.RERANK_MAX_LENGTH),
        "data_meta": rf.load_meta(),
        **stats,
    }
    meta_path = ft_model_dir() / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[stage8-train] wrote {meta_path}")
    print("[stage8-train] next (go/no-go): "
          "python scripts/18_eval_reranker_ft.py --dev")


if __name__ == "__main__":
    main()
