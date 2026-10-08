"""Stage 18 preflight - check the shared run and report saved state.

Checks the holdout bench, weights, Stage 16 archive, reader revision and optional
GPU. Reports the lock, identity, contexts, partial configs and generations.
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


S45 = _load_script("45_stage18_contexts.py", "stage18_contexts")
MIN_FREE_GB = 1


def state_lines() -> list[str]:
    latest = C.RESULTS_LATEST_DIR
    lines = [f"run identity recorded: {(latest / C.STAGE18_IDENTITY_JSON).exists()}"]
    ctx = latest / C.STAGE18_CONTEXTS_JSONL
    if ctx.exists():
        meta, _ = S45.S41.read_contexts(ctx)
        lines.append(f"contexts: written, valid {meta['valid']}")
    else:
        lines.append("contexts: not written")
    for label in S45.CONFIG_ARMS:
        path = S45.partial_path(latest, label)
        if path.exists():
            meta, rows = S45.S41.read_contexts(path)
            lines.append(f"partial {label}: {len(rows)} rows, "
                         f"{meta['n_questions']} questions")
        else:
            lines.append(f"partial {label}: not written")
    gen_dir = latest / C.STAGE18_GENERATIONS_DIRNAME
    for arm in S45.ARMS:
        path = gen_dir / f"{arm}.jsonl"
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()) if path.exists() else 0
        lines.append(f"generations {arm}: {n} rows")
    lines.append(f"verdict written: {(latest / C.STAGE18_VERDICT_JSON).exists()}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 18 preflight checks and state report.")
    ap.add_argument("--gpu", action="store_true", help="also require a CUDA device")
    args = ap.parse_args()
    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE14_ROOT_ID)
    print("[preflight] PASS shared-root id", flush=True)
    for path in (C.RESULTS_LATEST_DIR,
                 C.RESULTS_LATEST_DIR / C.STAGE18_GENERATIONS_DIRNAME):
        run_guard.probe_directory(path)
        print(f"[preflight] PASS write/read/replace/delete in {path.name}", flush=True)

    bench_dir = S45.S39.bench_dir()
    bench_files = [bench_dir / name for name in
                   ("docs.jsonl", "questions.jsonl", "holdout_meta.json")]
    stage6_dir = S45.S39.stage6_dir()
    stage6_files = [stage6_dir / name for name in ("docs.jsonl", "questions.jsonl")]
    archive = C.RESULTS_DIR / "stage16" / "final" / C.STAGE16_RESULTS_CSV
    missing = [str(p.relative_to(C.DATA_ROOT)) for p in bench_files + stage6_files +
               list(S45.weight_paths().values()) + [archive]
               if not p.exists()]
    if missing:
        raise SystemExit(f"[preflight] FAIL missing {missing}")
    docs, questions, _, _ = S45.bench()
    S45.S39.stage6_bench()
    print(f"[preflight] PASS holdout bench ({len(docs)} docs / {len(questions)} "
          "questions), weights and Stage 16 archive present", flush=True)

    free_gb = shutil.disk_usage(C.DATA_ROOT).free / 1e9
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"[preflight] FAIL {free_gb:.1f} GB free under the data root")
    print(f"[preflight] PASS {free_gb:.1f} GB free under the data root", flush=True)

    from huggingface_hub import model_info
    name, revision = S45.S41.reader_spec()
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

    held = run_guard.read_lock(C.DATA_ROOT / C.STAGE18_LOCK_FILE)
    if held is None:
        print("[preflight] run lock: free", flush=True)
    else:
        print(f"[preflight] run lock: HELD by account {held.get('account')}, "
              f"task {held.get('task')}, since {held.get('started_at')}. "
              "Do not start another step unless that runtime is stopped.", flush=True)
    for line in state_lines():
        print(f"[preflight] {line}", flush=True)
    print("[preflight] all checks passed", flush=True)


if __name__ == "__main__":
    main()
