"""Stage 17 preflight - run before any Stage 17 step, from every account.

Checks, in order, and stops at the first failure:

1. the mounted project folder is the shared one (RUN_ROOT_ID.json matches config);
2. this process can write, read back, replace and delete files where Stage 17 writes;
3. the Stage 6 bench, the three weight files and the stage8/stage15 archives exist;
4. the pinned reader revision resolves on the Hugging Face Hub;
5. with --gpu, a CUDA device is visible (the run expects a T4).

Then it reports, without changing anything: the run lock, the run identity, the
contexts and the saved generations per arm.

Usage:
    python scripts/44_preflight_stage17.py --gpu
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import shutil
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S41 = _load_script("41_stage17_contexts.py", "stage17_contexts")
MIN_FREE_GB = 1


def state_lines() -> list[str]:
    latest = C.RESULTS_LATEST_DIR
    lines = [f"run identity recorded: {(latest / C.STAGE17_IDENTITY_JSON).exists()}"]
    ctx = latest / C.STAGE17_CONTEXTS_JSONL
    if ctx.exists():
        meta, _ = S41.read_contexts(ctx)
        lines.append(f"contexts: written, valid {meta['valid']}")
    else:
        lines.append("contexts: not written")
    gen_dir = latest / C.STAGE17_GENERATIONS_DIRNAME
    for path in sorted(gen_dir.glob("*.jsonl")) if gen_dir.exists() else []:
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        lines.append(f"generations {path.stem}: {n} rows")
    lines.append(f"verdict written: {(latest / C.STAGE17_VERDICT_JSON).exists()}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 17 preflight checks and state report.")
    ap.add_argument("--gpu", action="store_true", help="also require a CUDA device")
    args = ap.parse_args()

    from rag_chunk import run_guard

    print(f"[preflight] project data root: {C.DATA_ROOT}", flush=True)
    root_id = run_guard.check_root_identity(C.DATA_ROOT, C.STAGE17_ROOT_ID)
    print(f"[preflight] PASS shared-root id {root_id}", flush=True)
    for path in (C.RESULTS_LATEST_DIR,
                 C.RESULTS_LATEST_DIR / C.STAGE17_GENERATIONS_DIRNAME):
        run_guard.probe_directory(path)
        print(f"[preflight] PASS write/read/replace/delete in {path}", flush=True)

    docs, questions, _ = S41.S16.B16.stage6_bench()
    archives = (C.RESULTS_DIR / "stage8" / "final" / C.STAGE8_RESULTS_CSV,
                C.RESULTS_DIR / "stage15" / "final" / C.STAGE15_RESULTS_CSV)
    missing = [str(p) for p in list(S41.weight_paths().values()) + list(archives)
               if not p.exists()]
    if missing:
        raise SystemExit(f"[preflight] FAIL missing {missing}")
    print(f"[preflight] PASS Stage 6 bench ({len(docs)} docs / {len(questions)} questions), "
          "weights and archives present", flush=True)

    free_gb = shutil.disk_usage(C.DATA_ROOT).free / 1e9
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"[preflight] FAIL {free_gb:.1f} GB free under the data root")
    print(f"[preflight] PASS {free_gb:.1f} GB free under the data root", flush=True)

    from huggingface_hub import model_info

    name, revision = S41.reader_spec()
    info = model_info(name, revision=revision)
    if info.sha != revision:
        raise SystemExit(f"[preflight] FAIL {name} resolved to {info.sha}, expected {revision}")
    print(f"[preflight] PASS reader {name} at {revision}", flush=True)

    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("[preflight] FAIL no CUDA device on this runtime")
        props = torch.cuda.get_device_properties(0)
        note = "" if "T4" in props.name else " (the run expects a T4)"
        print(f"[preflight] PASS GPU {props.name}, {props.total_memory / 2**30:.1f} GiB{note}",
              flush=True)

    held = run_guard.read_lock(C.DATA_ROOT / C.STAGE17_LOCK_FILE)
    if held is None:
        print("[preflight] run lock: free", flush=True)
    else:
        print(f"[preflight] run lock: HELD by account {held.get('account')} on "
              f"{held.get('host')}, task {held.get('task')}, since {held.get('started_at')}. "
              "Do not start another step unless that runtime is stopped.", flush=True)
    for line in state_lines():
        print(f"[preflight] {line}", flush=True)
    print("[preflight] all checks passed", flush=True)


if __name__ == "__main__":
    main()
