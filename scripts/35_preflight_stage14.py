"""Stage 14 preflight - run before any Stage 14 step, from every account.

Checks, in order, and stops at the first failure:

1. the mounted project folder is the shared one (RUN_ROOT_ID.json matches config);
2. this process can write, read back, replace and delete files in every directory
   Stage 14 writes to (a `!python` cell runs as a child process, so passing here
   also covers child-process access);
3. with --gpu, a CUDA device is visible.

Then it reports, without changing anything: the run lock, the recorded run
identity, and how far Stage 14 has got (stream, shards, arms and their latest
checkpoint, evaluation).

Usage:
    python scripts/35_preflight_stage14.py          # CPU runtime
    python scripts/35_preflight_stage14.py --gpu    # GPU runtime
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402

ARMS = ("4k", "10k")
CKPT_DIRS = ("last", "last_new", "last_old")


def training_state() -> list[str]:
    lines = []
    data = C.DATA_DIR / C.STAGE14_DATA_DIRNAME
    meta_path = data / "stage14_data_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if (data / "train_docs.jsonl").exists() and meta:
        lines.append(f"stream: done, {meta.get('n_docs')} docs / {meta.get('n_questions')} questions")
    else:
        lines.append("stream: not done")
    mined = sorted(meta.get("shards", {}))
    lines.append(f"shards mined: {len(mined)}/3 {mined}")
    for arm in ARMS:
        out = C.MODELS_DIR / C.STAGE14_MODEL_DIRNAME / arm
        marker = out / "final" / "training_meta.json"
        if marker.exists():
            m = json.loads(marker.read_text(encoding="utf-8"))
            lines.append(f"arm {arm}: complete, {m['fingerprint']['n_groups']} groups, "
                         f"{m['total_steps']} steps")
            continue
        steps = [json.loads((out / d / "verified.json").read_text(encoding="utf-8"))["step"]
                 for d in CKPT_DIRS if (out / d / "verified.json").exists()]
        lines.append(f"arm {arm}: resumable checkpoint at step {max(steps)}" if steps
                     else f"arm {arm}: not started")
    latest = C.RESULTS_LATEST_DIR
    for mode in ("dev", "final"):
        cp = latest / f"stage14_checkpoint_{mode}.jsonl"
        done = 0
        if cp.exists():
            done = sum(len(json.loads(line).get("rows", []))
                       for line in cp.read_text(encoding="utf-8").splitlines() if line.strip())
        lines.append(f"eval {mode}: {done} rows checkpointed")
    lines.append(f"verdict written: {(latest / 'stage14_verdict.json').exists()}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 14 preflight checks and state report.")
    ap.add_argument("--gpu", action="store_true", help="also require a CUDA device")
    args = ap.parse_args()

    from rag_chunk import run_guard

    print(f"[preflight] project data root: {C.DATA_ROOT}", flush=True)
    root_id = run_guard.check_root_identity(C.DATA_ROOT, C.STAGE14_ROOT_ID)
    print(f"[preflight] PASS shared-root id {root_id}", flush=True)
    for path in (C.DATA_DIR / C.STAGE14_DATA_DIRNAME, C.MODELS_DIR / C.STAGE14_MODEL_DIRNAME,
                 C.RESULTS_LATEST_DIR):
        run_guard.probe_directory(path)
        print(f"[preflight] PASS write/read/replace/delete in {path}", flush=True)
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("[preflight] FAIL no CUDA device on this runtime")
        print(f"[preflight] PASS GPU {torch.cuda.get_device_name(0)}", flush=True)

    held = run_guard.read_lock(C.DATA_ROOT / C.STAGE14_LOCK_FILE)
    if held is None:
        print("[preflight] run lock: free", flush=True)
    else:
        print(f"[preflight] run lock: HELD by account {held.get('account')} on "
              f"{held.get('host')}, task {held.get('task')}, since {held.get('started_at')}. "
              "Do not start another step unless that runtime is stopped.", flush=True)
    identity = C.MODELS_DIR / C.STAGE14_MODEL_DIRNAME / "run_identity.json"
    print(f"[preflight] run identity recorded: {identity.exists()}", flush=True)
    for line in training_state():
        print(f"[preflight] {line}", flush=True)
    print("[preflight] all checks passed", flush=True)


if __name__ == "__main__":
    main()
