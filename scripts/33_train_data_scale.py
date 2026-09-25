"""Stage 14 (step 1) - train the Stage 8 recipe on more data, resumable across accounts.

Both arms start from the base model; only the groups differ, and the sets are
nested:

    4k   Stage 8 groups + Stage 11 groups
    10k  the 4k set + the three Stage 14 shards

The 2k point is the existing Stage 8 weights.

The training loop is the loop of scripts/17_train_reranker.py step for step (same
optimizer, schedule, gradient scaler, clipping, data order and seeds), with one
addition: every STAGE14_CKPT_EVERY steps it saves the full training state (weights,
optimizer, schedule, scaler, random states, the epoch's data order and position).
A resumed arm continues from that step with the same schedule, so an interruption
costs at most STAGE14_CKPT_EVERY steps. Final weights go to <arm>/final/ with a
completion marker; the resumable checkpoint is then removed.

Several Colab accounts may continue the run in one shared Drive folder, so a formal
run first checks the shared-root id, probes write access, takes the Stage 14 run
lock and compares the recorded run identity (code, data, recipe). See
rag_chunk/run_guard.py.

Usage:
    python scripts/33_train_data_scale.py --smoke --account A
    python scripts/33_train_data_scale.py --mode fresh --account A      # first start
    python scripts/33_train_data_scale.py --mode resume --account B     # any later start
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import random
import shutil
import sys
import tempfile
import time

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402

ARMS = ("4k", "10k")
N_SHARDS = 3
CKPT_DIRS = ("last", "last_new", "last_old")


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def groups_sha1(groups: list[dict]) -> str:
    h = hashlib.sha1()
    for g in groups:
        h.update(json.dumps(g, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def training_sets() -> dict[str, list[dict]]:
    """The nested 4k and 10k training sets, in a fixed concatenation order."""
    from rag_chunk import rerank_finetune as rf

    s8 = rf.load_groups()
    s11_path = C.DATA_DIR / C.STAGE11_DATA_DIRNAME / C.STAGE11_GROUPS_JSONL
    s14_dir = C.DATA_DIR / C.STAGE14_DATA_DIRNAME
    meta_path = s14_dir / "stage14_data_meta.json"
    if not s8 or not s11_path.exists():
        raise SystemExit("[stage14] the Stage 8 or Stage 11 groups are missing")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    shard_paths = [s14_dir / f"shard{k}_groups.jsonl" for k in range(N_SHARDS)]
    if any(not p.exists() or str(k) not in meta.get("shards", {})
           for k, p in enumerate(shard_paths)):
        raise SystemExit("[stage14] not every Stage 14 shard is mined - run "
                         "scripts/32_build_stage14_data.py first")
    four = s8 + _read_jsonl(s11_path)
    ten = four + [g for p in shard_paths for g in _read_jsonl(p)]
    return {"4k": four, "10k": ten}


def models_root() -> pathlib.Path:
    return C.MODELS_DIR / C.STAGE14_MODEL_DIRNAME


def arm_dir(arm: str) -> pathlib.Path:
    return models_root() / arm


def assert_not_stage8(path: pathlib.Path) -> None:
    stage8 = (C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME).resolve()
    target = path.resolve()
    if target == stage8 or stage8 in target.parents:
        raise SystemExit(f"[stage14] refusing to train into the Stage 8 directory: {target}")


# --------------------------------------------------------------------------- #
# resumable checkpoints
# --------------------------------------------------------------------------- #
def latest_checkpoint(out: pathlib.Path):
    """The verified checkpoint with the highest step among last, last_new, last_old."""
    best = None
    for name in CKPT_DIRS:
        marker = out / name / "verified.json"
        if marker.exists():
            step = json.loads(marker.read_text(encoding="utf-8"))["step"]
            if best is None or step > best[1]:
                best = (out / name, step)
    return best


def _save_checkpoint(model, tokenizer, state: dict, out: pathlib.Path) -> None:
    """Stage locally, copy to <out>/last_new, verify by reopening, then swap into last."""
    import torch
    from safetensors import safe_open

    local = pathlib.Path(tempfile.mkdtemp(prefix="stage14_ckpt_"))
    model.save_pretrained(local)
    tokenizer.save_pretrained(local)
    torch.save(state, local / "trainer_state.pt")
    new = out / "last_new"
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    for item in sorted(local.iterdir()):
        for attempt in range(1, 4):
            try:
                shutil.copy2(item, new / item.name)
                break
            except OSError as exc:
                if attempt == 3:
                    raise
                print(f"[stage14-train] WARN: copy of {item.name} failed ({exc!r}); "
                      "retrying in 15 s", flush=True)
                time.sleep(15)
    for item in local.iterdir():
        if (new / item.name).stat().st_size != item.stat().st_size:
            raise OSError(f"checkpoint file {item.name} did not verify")
    reopened = torch.load(new / "trainer_state.pt", map_location="cpu", weights_only=True)
    if reopened["step"] != state["step"]:
        raise OSError("trainer_state.pt did not reopen with the saved step")
    weights = new / "model.safetensors"
    if weights.exists():
        with safe_open(str(weights), framework="pt") as fh:
            if not list(fh.keys()):
                raise OSError("model.safetensors reopened empty")
    shutil.rmtree(local, ignore_errors=True)
    (new / "verified.json").write_text(json.dumps({"step": state["step"]}), encoding="utf-8")
    old, last = out / "last_old", out / "last"
    shutil.rmtree(old, ignore_errors=True)
    if last.exists():
        last.rename(old)
    new.rename(last)
    shutil.rmtree(old, ignore_errors=True)


def _progress(event: dict) -> None:
    path = C.RESULTS_LATEST_DIR / "stage14_train_progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def train_resumable(groups: list[dict], *, init_model: str, out: pathlib.Path, epochs: int,
                    lr: float, groups_per_step: int, seed: int, ckpt_every: int,
                    account: str, arm: str, save_final) -> dict:
    """The loop of scripts/17_train_reranker.py with resumable checkpoints."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    group_width = 1 + len(groups[0]["negs"])
    if any(1 + len(g["negs"]) != group_width for g in groups):
        raise SystemExit("[stage14-train] groups have mixed widths")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    resume = latest_checkpoint(out)
    source = str(resume[0]) if resume else init_model
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
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
    elapsed_before = 0.0
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
        print(f"[stage14-train] arm {arm}: resuming at step {step}/{total_steps} "
              f"(epoch {start_epoch}, batch {skip_batches}) from {resume[0].name}",
              flush=True)
        _progress({"account": account, "arm": arm, "event": "resumed", "step": step,
                   "total": total_steps})
    else:
        print(f"[stage14-train] arm {arm}: {len(groups)} groups (width {group_width}) x "
              f"{epochs} epochs = {total_steps} steps on {device}; checkpoint every "
              f"{ckpt_every} steps", flush=True)
        _progress({"account": account, "arm": arm, "event": "started", "step": 0,
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
                loss = loss_fn(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            epoch_loss += float(loss.detach())
            epoch_batches += 1
            elapsed = elapsed_before + time.perf_counter() - t_start
            if step % 50 == 0 or step == total_steps:
                print(f"[stage14-train] arm {arm} step {step}/{total_steps} epoch {epoch} "
                      f"loss {epoch_loss / epoch_batches:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} ({elapsed / 60:.1f} min)",
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
                         "elapsed_seconds": elapsed}
                _save_checkpoint(model, tokenizer, state, out)
                _progress({"account": account, "arm": arm, "event": "checkpoint",
                           "step": step, "total": total_steps})
                print(f"[stage14-train] CHECKPOINT SAVED AND VERIFIED: arm {arm}, step "
                      f"{step}/{total_steps}. Safe to stop this runtime now.", flush=True)
        last_epoch_loss = epoch_loss / max(epoch_batches, 1)
        print(f"[stage14-train] arm {arm} epoch {epoch}/{epochs} done - mean loss "
              f"{last_epoch_loss:.4f}", flush=True)
        order, skip_batches, epoch_loss, epoch_batches = None, 0, 0.0, 0

    save_final(model, tokenizer, out / "final")
    return {"final_mean_epoch_loss": last_epoch_loss, "total_steps": total_steps,
            "group_width": group_width, "device": device,
            "minutes": round((elapsed_before + time.perf_counter() - t_start) / 60, 1)}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 14 step 1: the Stage 8 recipe on nested 4k and 10k sets.")
    ap.add_argument("--arm", choices=ARMS + ("all",), default="all")
    ap.add_argument("--mode", choices=("fresh", "resume"), default="resume",
                    help="fresh refuses to start if any Stage 14 training state exists")
    ap.add_argument("--account", default="unlabelled",
                    help="operational label recorded in the lock and progress log")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    ap.add_argument("--smoke", action="store_true",
                    help="8 groups, 1 epoch, a checkpoint after step 1, temp dir, nothing kept")
    args = ap.parse_args()
    C.ensure_dirs()

    from rag_chunk import run_guard

    s17 = _load_script("17_train_reranker.py", "stage8_train")
    epochs = 1 if args.smoke else int(C.STAGE8_EPOCHS)
    recipe = {"base_model": C.RERANKER_MODEL, "epochs": epochs, "lr": float(C.STAGE8_LR),
              "groups_per_step": int(C.STAGE8_GROUPS_PER_STEP), "seed": int(C.STAGE8_SEED),
              "warmup_frac": float(C.STAGE8_WARMUP_FRAC),
              "weight_decay": float(C.STAGE8_WEIGHT_DECAY),
              "max_length": int(C.RERANK_MAX_LENGTH)}

    if args.smoke:
        from rag_chunk import rerank_finetune as rf

        smoke_groups = rf.load_groups()[:8]         # Stage 8 groups: runs before any mining
        if len(smoke_groups) < 8:
            raise SystemExit("[stage14] smoke needs the Stage 8 groups")
        out = pathlib.Path(tempfile.mkdtemp(prefix="stage14_smoke_"))
        stats = train_resumable(smoke_groups, init_model=C.RERANKER_MODEL, out=out,
                                epochs=1, lr=recipe["lr"],
                                groups_per_step=recipe["groups_per_step"],
                                seed=recipe["seed"], ckpt_every=1, account=args.account,
                                arm="smoke", save_final=s17._save)
        ok = (out / "final" / "config.json").exists() and latest_checkpoint(out) is not None
        shutil.rmtree(out, ignore_errors=True)
        print(f"[stage14] smoke: {stats['total_steps']} steps, checkpoint save/verify and "
              f"final save {'OK' if ok else 'FAILED'}. Nothing was kept.", flush=True)
        if not ok:
            raise SystemExit(1)
        return

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE14_ROOT_ID)
    sets = training_sets()
    identity = {"run_version": C.STAGE14_RUN_VERSION, "recipe": recipe,
                "trainer_sha256": run_guard.file_sha256(pathlib.Path(__file__)),
                "arms": {arm: {"n_groups": len(sets[arm]),
                               "groups_sha1": groups_sha1(sets[arm])} for arm in ARMS}}
    identity_path = models_root() / "run_identity.json"
    existing = identity_path.exists() or any(
        (arm_dir(a) / d).exists() for a in ARMS for d in CKPT_DIRS + ("final",))
    if args.mode == "fresh" and existing:
        raise SystemExit("[stage14] Stage 14 training state already exists; use --mode resume. "
                         "Nothing was changed.")

    wanted = ARMS if args.arm == "all" else (args.arm,)
    lock_path = C.DATA_ROOT / C.STAGE14_LOCK_FILE
    with run_guard.held_lock(lock_path, run_version=C.STAGE14_RUN_VERSION,
                             account=args.account, task="train",
                             clear_stale=args.clear_stale_lock):
        for path in (models_root(), C.RESULTS_LATEST_DIR):
            run_guard.probe_directory(path)
        print(f"[stage14] run identity {run_guard.check_run_identity(identity_path, identity)}",
              flush=True)
        summaries = {}
        for arm in wanted:
            out = arm_dir(arm)
            assert_not_stage8(out)
            fingerprint = {"arm": arm, **identity["arms"][arm], **recipe}
            marker = out / "final" / "training_meta.json"
            if marker.exists():
                done = json.loads(marker.read_text(encoding="utf-8"))
                if done.get("fingerprint") != fingerprint:
                    raise SystemExit(f"[stage14] {marker} was trained with other settings")
                print(f"[stage14] arm {arm} already trained ({out}) - skipping", flush=True)
                summaries[arm] = done
                continue
            stats = train_resumable(sets[arm], init_model=C.RERANKER_MODEL, out=out,
                                    epochs=epochs, lr=recipe["lr"],
                                    groups_per_step=recipe["groups_per_step"],
                                    seed=recipe["seed"], ckpt_every=int(C.STAGE14_CKPT_EVERY),
                                    account=args.account, arm=arm, save_final=s17._save)
            meta = {"fingerprint": fingerprint, **stats}
            marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            for name in CKPT_DIRS:
                shutil.rmtree(out / name, ignore_errors=True)
            _progress({"account": args.account, "arm": arm, "event": "completed",
                       "step": stats["total_steps"], "total": stats["total_steps"]})
            print(f"[stage14] ARM COMPLETE: {arm}, final weights and marker written.",
                  flush=True)
            summaries[arm] = meta

    print("\n[stage14] summary")
    for arm, meta in summaries.items():
        fp = meta["fingerprint"]
        print(f"  {arm}: {fp['n_groups']} groups, {meta['total_steps']} steps, "
              f"{meta['minutes']} min, final mean epoch loss {meta['final_mean_epoch_loss']:.4f}")
    ten = summaries.get("10k")
    if ten:
        n, need = ten["fingerprint"]["n_groups"], int(C.STAGE14_MIN_GROUPS_10K)
        print(f"[stage14] size condition ({need} groups for 10k): {n} - "
              f"{'met' if n >= need else 'NOT MET'}")
    print("[stage14] next: python scripts/34_eval_data_scale.py --dev")


if __name__ == "__main__":
    main()
