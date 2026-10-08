"""Stage 18 (step 2) - reader outputs for the holdout arms.

Uses the Stage 17 reader, prompt and generation functions. Saved arm files resume
from their last complete row.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S45 = _load_script("45_stage18_contexts.py", "stage18_contexts")
S42 = _load_script("42_stage17_generate.py", "stage17_generate")


def run(gen_dir: pathlib.Path) -> None:
    from rag_chunk import run_guard

    latest = C.RESULTS_LATEST_DIR
    ident = latest / C.STAGE18_IDENTITY_JSON
    if not ident.exists():
        raise SystemExit("[stage18] no run identity; run scripts/45_stage18_contexts.py first")
    _, questions, titles_sha1, _ = S45.bench()
    run_guard.check_run_identity(ident, S45.run_identity(titles_sha1, questions))
    ctx_path = latest / C.STAGE18_CONTEXTS_JSONL
    if not ctx_path.exists():
        raise SystemExit("[stage18] no contexts; run scripts/45_stage18_contexts.py first")
    meta, ctx_rows = S45.S41.read_contexts(ctx_path)
    if meta["n_questions"] != len(questions):
        raise SystemExit("[stage18] contexts and bench question counts differ")
    if not meta["valid"]:
        print(f"[stage18] INVALID contexts ({meta['why_invalid']}); generation skipped",
              flush=True)
        return
    if set(ctx_rows) != set(S45.CONFIG_ARMS["fixed15"] +
                            S45.CONFIG_ARMS["fixed6"] +
                            S45.CONFIG_ARMS["bilstm15"] + ("gold",)):
        raise SystemExit("[stage18] contexts arms differ")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("[stage18] no CUDA device on this runtime")
    import transformers

    gpu = torch.cuda.get_device_name(0)
    versions = f"torch {torch.__version__}, transformers {transformers.__version__}"
    run_guard.probe_directory(gen_dir)
    seen = sorted({r["gpu"] for arm in S45.ARMS
                   for r in S42.read_rows(gen_dir / f"{arm}.jsonl")} - {gpu})
    if seen:
        raise SystemExit(f"[stage18] saved rows came from {seen}, this runtime has {gpu}")
    kept = S42.kept_questions(questions)
    print(f"[stage18] {len(kept)} scored questions; GPU {gpu}; {versions}", flush=True)
    t0 = time.perf_counter()
    model, tok = S42.load_reader("cuda")
    print(f"[stage18] reader loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    for arm in S45.ARMS:
        S42.generate_arm(arm, S42.work_items(arm, kept, ctx_rows), questions, model,
                         tok, gpu, gen_dir / f"{arm}.jsonl", versions=versions)
    print("[stage18] every arm is complete; run scripts/47_stage18_score.py", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 18 step 2: reader outputs per arm.")
    ap.add_argument("--smoke", action="store_true", help="Stage 17 NQ-train reader check")
    ap.add_argument("--account", default="unlabelled", help="label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()
    if args.smoke:
        S42.smoke()
        return
    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE14_ROOT_ID)
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE18_LOCK_FILE,
                             run_version=C.STAGE18_RUN_VERSION, account=args.account,
                             task="generate", clear_stale=args.clear_stale_lock):
        run(C.RESULTS_LATEST_DIR / C.STAGE18_GENERATIONS_DIRNAME)


if __name__ == "__main__":
    main()
