"""Stage 16 (step 2) - the Stage 15 comparison on the holdout NQ bench.

Arms, each reordering one identical BGE top-20 pool at fixed 15/0:

    bge                 dense order
    rerank20_ft         Stage 8 weights
    rerank20_s15_large  Stage 15 weights

Primary comparison: rerank20_s15_large minus rerank20_ft, R@1, paired over the
holdout questions. The criteria are in docs/stage16_holdout_bench.md.

The bench is read from NQ_DIR/STAGE16_BENCH_DIRNAME and never rebuilt here. Both
weight files are hashed and must match the values in config.py.

Usage:
    python scripts/40_eval_holdout.py --smoke        # 40 questions, temp dir
    python scripts/40_eval_holdout.py --account A
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S11 = _load_script("28_eval_reranker_rl.py", "stage11_eval")
S15 = _load_script("37_eval_large_reranker.py", "stage15_eval")
B16 = _load_script("39_build_holdout_bench.py", "stage16_build")
ARM_FT = S11.ARM_FT
ARM_LARGE = S15.ARM_LARGE_FT
PAIRS = ((ARM_LARGE, ARM_FT),)
CONFIG = (15, 0)
SMOKE_QUESTIONS = 40
CHECKPOINT = "stage16_checkpoint_final.jsonl"
VERDICT_JSON = "stage16_verdict.json"
SCORES_DIR = "stage16_scores"


def _mean_ci(diffs):
    return S11._mean_ci(diffs)


def _paired(by) -> list[dict]:
    out = []
    for a, b in PAIRS:
        for metric, key in (("recall@1", "hits1"), ("recall@5", "hits5")):
            diffs = [x - y for x, y in zip(by[a][key], by[b][key])]
            mean, lo, hi = _mean_ci(diffs)
            out.append({"bench": "holdout", "metric": metric, "comparison": f"{a} - {b}",
                        "mean": round(mean, 4), "ci95_low": round(lo, 4),
                        "ci95_high": round(hi, 4), "n_questions": len(diffs)})
    return out


def _verdict(diffs, valid, why_invalid, n_questions, floor):
    """Pre-registered rule on unrounded values: (verdict, why, mean, lo, hi)."""
    m, lo, hi = _mean_ci(diffs)
    ci = f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    need = int(C.STAGE16_MIN_QUESTIONS)
    if not valid:
        return "INVALID", why_invalid, m, lo, hi
    if n_questions < need:
        return ("INCONCLUSIVE-SIZE", f"the holdout bench has {n_questions} questions, "
                f"fewer than {need}", m, lo, hi)
    if lo > 0 and m >= floor:
        return ("CONFIRMED", f"large beats Stage 8 by {m:+.4f} R@1, {ci}, at or above the "
                f"{floor} floor", m, lo, hi)
    if lo > 0:
        return ("BELOW-FLOOR", f"large beats Stage 8 by {m:+.4f} R@1, {ci}, below the "
                f"{floor} floor", m, lo, hi)
    if hi < 0:
        return "LARGE-WORSE", f"large trails Stage 8 by {m:+.4f} R@1, {ci}", m, lo, hi
    return ("NOT-CONFIRMED", f"large - Stage 8 = {m:+.4f} R@1, {ci}: the interval "
            f"includes 0", m, lo, hi)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def stage15_hits() -> dict | None:
    """Per-question hits of the Stage 15 run for the pooled comparison."""
    for folder in (C.RESULTS_DIR / "stage15" / "final", C.RESULTS_LATEST_DIR):
        path = folder / "stage15_checkpoint_final.jsonl"
        if path.exists():
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows += json.loads(line).get("rows", [])
            return {r["arm"]: r for r in rows}
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 16 step 2: large vs Stage 8 on the holdout bench.")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_QUESTIONS} questions, temp dir, no verdict")
    ap.add_argument("--account", default="unlabelled", help="label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()
    if args.smoke:
        evaluate(args)
        return

    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE16_ROOT_ID)
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE16_LOCK_FILE,
                             run_version=C.STAGE16_RUN_VERSION, account=args.account,
                             task="eval", clear_stale=args.clear_stale_lock):
        run_guard.probe_directory(C.RESULTS_LATEST_DIR)
        evaluate(args)


def evaluate(args) -> None:
    bench = B16.bench_dir()
    meta_path = bench / "holdout_meta.json"
    if not meta_path.exists():
        raise SystemExit("[stage16] no holdout bench - run scripts/39_build_holdout_bench.py")
    bench_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    docs = B16._read_jsonl(bench / "docs.jsonl")
    questions = B16._read_jsonl(bench / "questions.jsonl")
    s6_docs, s6_questions, _ = B16.stage6_bench()
    overlap = len({d["title"] for d in s6_docs} & {d["title"] for d in docs})
    q_overlap = len({q["question"] for q in s6_questions} & {q["question"] for q in questions})

    ft_dir = C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"
    large_dir = C.MODELS_DIR / C.STAGE15_MODEL_DIRNAME / "large" / "final"
    hashes = {}
    for arm, d in ((ARM_FT, ft_dir), (ARM_LARGE, large_dir)):
        w = d / "model.safetensors"
        if not w.exists():
            raise SystemExit(f"[stage16] no weights at {w}")
        hashes[arm] = sha256(w)
    expected = {ARM_FT: C.STAGE16_STAGE8_SHA256, ARM_LARGE: C.STAGE16_LARGE_SHA256}
    bad_hash = [arm for arm in hashes if hashes[arm] != expected[arm]]
    print(f"[stage16] weights: {'match' if not bad_hash else f'MISMATCH {bad_hash}'}; "
          f"title overlap with Stage 6 {overlap}, question overlap {q_overlap}", flush=True)

    if args.smoke:
        questions = questions[:SMOKE_QUESTIONS]
    print(f"[stage16] holdout bench: {len(docs)} docs / {len(questions)} questions"
          f"{' (smoke)' if args.smoke else ''}", flush=True)

    C.apply(RETRIEVAL_EMBED_MODEL="BAAI/bge-base-en-v1.5", RETRIEVAL_EMBED_NORMALIZE=True)
    C.apply(NQ_DIR=bench)

    import torch
    from sentence_transformers import CrossEncoder

    from rag_chunk.large_eval import append_checkpoint, load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = int(C.RERANK_MAX_LENGTH)
    sources = {ARM_FT: str(ft_dir), ARM_LARGE: str(large_dir)}

    def loader(path):
        return lambda: CrossEncoder(path, max_length=max_len, device=device)

    latest = (pathlib.Path(tempfile.mkdtemp(prefix="stage16_eval_smoke_")) if args.smoke
              else C.RESULTS_LATEST_DIR)
    scorers = {arm: S15.cached_scorer(arm, loader(p), latest / SCORES_DIR)
               for arm, p in sources.items()}
    checkpoint = latest / CHECKPOINT
    meta = {"stage": "stage16", "n_docs": len(docs), "n_questions": len(questions),
            "retrieval_model": C.RETRIEVAL_EMBED_MODEL, "models": sources,
            "weights_sha256": hashes}
    done = load_checkpoint(checkpoint, meta)
    if not done and not checkpoint.exists():
        append_checkpoint(checkpoint, {"meta": meta})
    rows = list(done)
    if not rows:
        rows = S11._eval_config(docs, questions, CONFIG[0], CONFIG[1], scorers)
        for r in rows:
            mp = latest / SCORES_DIR / f"{r['arm']}.json"
            if mp.exists():
                r["rerank_seconds"] = json.loads(mp.read_text(encoding="utf-8"))["seconds"]
        append_checkpoint(checkpoint, {"rows": rows})
    by = {r["arm"]: r for r in rows}
    paired = _paired(by)
    for p in paired:
        print(f"[stage16] {p['metric']:9s} {p['comparison']:36s} {p['mean']:+.4f} "
              f"[{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]")

    if args.smoke:
        shutil.rmtree(latest, ignore_errors=True)
        print("[stage16] smoke: every arm scored and the paired table built. The numbers are "
              "not results and nothing was kept.", flush=True)
        return

    s15 = stage15_hits()
    if s15 is not None:
        for metric, key in (("recall@1", "hits1"), ("recall@5", "hits5")):
            diffs = ([x - y for x, y in zip(s15[ARM_LARGE][key], s15[ARM_FT][key])]
                     + [x - y for x, y in zip(by[ARM_LARGE][key], by[ARM_FT][key])])
            mean, lo, hi = _mean_ci(diffs)
            paired.append({"bench": "stage6 + holdout", "metric": metric,
                           "comparison": f"{ARM_LARGE} - {ARM_FT}", "mean": round(mean, 4),
                           "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                           "n_questions": len(diffs)})

    from rag_chunk import rerank_finetune as rf
    train_titles = {d["title"] for d in rf.load_bench("train")[0]}
    train_overlap = len(train_titles & {d["title"] for d in docs})

    S11._write_rows(latest / C.STAGE16_RESULTS_CSV, rows)
    S11._write_rows(latest / C.STAGE16_PAIRED_CSV, paired)
    valid = not bad_hash and overlap == 0 and q_overlap == 0
    why_invalid = (f"weights differ from the pre-registered hashes: {bad_hash}" if bad_hash
                   else f"the bench overlaps the Stage 6 bench ({overlap} titles, "
                        f"{q_overlap} questions)")
    diffs = [a - b for a, b in zip(by[ARM_LARGE]["hits1"], by[ARM_FT]["hits1"])]
    floor = float(C.STAGE16_PRACTICAL_FLOOR)
    verdict, why, m, lo, hi = _verdict(diffs, valid, why_invalid, len(questions), floor)

    lines = ["# Stage 16 - the Stage 15 comparison on a holdout NQ bench", "",
             f"Verdict: {verdict}. {why}.", "",
             f"- holdout bench: {len(docs)} docs / {len(questions)} questions from "
             f"{bench_meta['rows_scanned']} rows; title overlap with Stage 6 {overlap}, "
             f"question overlap {q_overlap}",
             f"- titles also in the Stage 8 training documents: {train_overlap}",
             f"- weights: {'match the pre-registered hashes' if not bad_hash else 'MISMATCH'}",
             f"- primary comparison: fixed 15/0, R@1, {ARM_LARGE} - {ARM_FT}, paired over "
             f"{len(diffs)} questions; threshold {floor}",
             "", "| arm | R@1 | R@3 | R@5 | pool@20 | s per question, load included |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        sec = r.get("rerank_seconds")
        per_q = f"{sec / len(questions):.3f}" if sec is not None else "-"
        lines.append(f"| {r['arm']} | {r['recall@1']:.4f} | {r['recall@3']:.4f} | "
                     f"{r['recall@5']:.4f} | {r[f'pool_recall@{S11.DEPTH}']:.4f} | {per_q} |")
    lines += ["", "| bench | metric | comparison | mean | 95% CI | n |",
              "| --- | --- | --- | --- | --- | --- |"]
    for p in paired:
        lines.append(f"| {p['bench']} | {p['metric']} | {p['comparison']} | {p['mean']:+.4f} | "
                     f"[{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}] | {p['n_questions']} |")
    (latest / C.STAGE16_SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (latest / VERDICT_JSON).write_text(json.dumps(
        {"verdict": verdict, "why": why, "valid": valid, "n_questions": len(diffs),
         "large_minus_stage8_r1": m, "ci95": [lo, hi], "floor": floor,
         "title_overlap": overlap, "question_overlap": q_overlap,
         "train_title_overlap": train_overlap, "weights_sha256": hashes}, indent=2),
        encoding="utf-8")
    print(f"[stage16] wrote {latest / C.STAGE16_SUMMARY_MD}")
    print(f"\n[stage16] VERDICT: {verdict} - {why}.")


if __name__ == "__main__":
    main()
