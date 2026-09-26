"""Stage 15 preflight - run before any Stage 15 step, from every account.

Checks, in order, and stops at the first failure:

1. the mounted project folder is the shared one (RUN_ROOT_ID.json matches config);
2. this process can write, read back, replace and delete files where Stage 15 writes;
3. the Stage 8 groups, the Stage 8 weights and the Stage 8 archive are present;
4. at least STAGE15_MIN_FREE_GB are free under the data root (one training state of
   the large model is about 6.8 GB, the final weights about 2.2 GB);
5. with --gpu, a CUDA device is visible.

Then it reports, without changing anything: the run lock, the recorded run identity,
the training state and the evaluation state.

Usage:
    python scripts/38_preflight_stage15.py --gpu
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402

CKPT_DIRS = ("last", "last_new", "last_old")


def state_lines() -> list[str]:
    out = C.MODELS_DIR / C.STAGE15_MODEL_DIRNAME / "large"
    marker = out / "final" / "training_meta.json"
    if marker.exists():
        m = json.loads(marker.read_text(encoding="utf-8"))
        lines = [f"training: complete, {m['total_steps']} steps, {m['minutes']} min"]
    else:
        steps = [json.loads((out / d / "verified.json").read_text(encoding="utf-8"))["step"]
                 for d in CKPT_DIRS if (out / d / "verified.json").exists()]
        lines = [f"training: resumable checkpoint at step {max(steps)}" if steps
                 else "training: not started"]
    scores = C.RESULTS_LATEST_DIR / "stage15_scores"
    cached = sorted(p.stem for p in scores.glob("*.npy")) if scores.exists() else []
    lines.append(f"evaluation: cached reranker scores {cached or 'none'}")
    lines.append(f"verdict written: {(C.RESULTS_LATEST_DIR / 'stage15_verdict.json').exists()}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 15 preflight checks and state report.")
    ap.add_argument("--gpu", action="store_true", help="also require a CUDA device")
    args = ap.parse_args()

    from rag_chunk import rerank_finetune as rf
    from rag_chunk import run_guard

    print(f"[preflight] project data root: {C.DATA_ROOT}", flush=True)
    root_id = run_guard.check_root_identity(C.DATA_ROOT, C.STAGE15_ROOT_ID)
    print(f"[preflight] PASS shared-root id {root_id}", flush=True)
    for path in (C.MODELS_DIR / C.STAGE15_MODEL_DIRNAME, C.RESULTS_LATEST_DIR):
        run_guard.probe_directory(path)
        print(f"[preflight] PASS write/read/replace/delete in {path}", flush=True)

    n_groups = len(rf.load_groups())
    ft = C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final" / "model.safetensors"
    archive = C.RESULTS_DIR / "stage8" / "final" / C.STAGE8_RESULTS_CSV
    missing = [name for name, ok in (("Stage 8 groups (2034)", n_groups == 2034),
                                     ("Stage 8 weights", ft.exists()),
                                     ("Stage 8 archive", archive.exists())) if not ok]
    if missing:
        raise SystemExit(f"[preflight] FAIL missing {missing}")
    print("[preflight] PASS Stage 8 groups, weights and archive present", flush=True)

    free_gb = shutil.disk_usage(C.DATA_ROOT).free / 1e9
    need = float(C.STAGE15_MIN_FREE_GB)
    if free_gb < need:
        raise SystemExit(f"[preflight] FAIL {free_gb:.1f} GB free under the data root, "
                         f"{need:.0f} GB needed")
    print(f"[preflight] PASS {free_gb:.1f} GB free under the data root", flush=True)

    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("[preflight] FAIL no CUDA device on this runtime")
        props = torch.cuda.get_device_properties(0)
        print(f"[preflight] PASS GPU {props.name}, {props.total_memory / 2**30:.1f} GiB",
              flush=True)

    held = run_guard.read_lock(C.DATA_ROOT / C.STAGE15_LOCK_FILE)
    if held is None:
        print("[preflight] run lock: free", flush=True)
    else:
        print(f"[preflight] run lock: HELD by account {held.get('account')} on "
              f"{held.get('host')}, task {held.get('task')}, since {held.get('started_at')}. "
              "Do not start another step unless that runtime is stopped.", flush=True)
    identity = C.MODELS_DIR / C.STAGE15_MODEL_DIRNAME / "run_identity.json"
    print(f"[preflight] run identity recorded: {identity.exists()}", flush=True)
    for line in state_lines():
        print(f"[preflight] {line}", flush=True)
    print("[preflight] all checks passed", flush=True)


if __name__ == "__main__":
    main()
