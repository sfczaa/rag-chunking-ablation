"""Stage 15 (step 2) - does the larger fine-tuned reranker rank better?

Every arm reorders one identical BGE top-20 pool at fixed 15/0:

    bge                 dense order, used for the pipeline check
    rerank20            off-the-shelf bge-reranker-base, used for the pipeline check
    rerank20_ft         Stage 8 weights (base, 2034 groups)
    rerank20_large      off-the-shelf bge-reranker-large, reported only
    rerank20_s15_large  bge-reranker-large, Stage 8 recipe and groups

Primary comparison: fixed 15/0, R@1, rerank20_s15_large minus rerank20_ft, paired
over questions. The criteria are in docs/stage15_large_reranker.md.

Each reranker's scores are saved as soon as they are computed, keyed by a hash of
the scored pairs, so a restarted evaluation reuses finished rerankers. Models are
loaded one at a time.

Scoring and statistics helpers come from scripts/28_eval_reranker_rl.py and are
loaded unchanged, so some log lines start with [stage11].

Usage:
    python scripts/37_eval_large_reranker.py --smoke        # 40 questions, temp dir
    python scripts/37_eval_large_reranker.py --account A    # the claim
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import time

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S11 = _load_script("28_eval_reranker_rl.py", "stage11_eval")
ARM_BGE, ARM_OTS, ARM_FT = S11.ARM_BGE, S11.ARM_OTS, S11.ARM_FT
ARM_LARGE_OTS, ARM_LARGE_FT = "rerank20_large", "rerank20_s15_large"
PAIRS = ((ARM_LARGE_FT, ARM_FT), (ARM_LARGE_OTS, ARM_OTS), (ARM_LARGE_FT, ARM_LARGE_OTS))
CONFIG = (15, 0)
SMOKE_QUESTIONS = 40
CHECKPOINT = "stage15_checkpoint_final.jsonl"
VERDICT_JSON = "stage15_verdict.json"
SCORES_DIR = "stage15_scores"


def _paired(rows) -> list[dict]:
    """Stage 15 pairs; the same estimator scripts/28 uses."""
    by = {r["arm"]: r for r in rows}
    out = []
    for a, b in PAIRS:
        ra, rb = by.get(a), by.get(b)
        if ra is None or rb is None:
            continue
        for metric, key in (("recall@1", "hits1"), ("recall@5", "hits5")):
            diffs = [x - y for x, y in zip(ra[key], rb[key])]
            mean, lo, hi = S11._mean_ci(diffs)
            out.append({"config": f"fixed {CONFIG[0]}/{CONFIG[1]}", "metric": metric,
                        "comparison": f"{a} - {b}", "mean": round(mean, 4),
                        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                        "n_questions": len(diffs)})
    return out


def _verdict(diffs, valid, floor):
    """Pre-registered rule on unrounded values: (verdict, why, mean, lo, hi)."""
    m, lo, hi = S11._mean_ci(diffs)
    ci = f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    if not valid:
        return ("INVALID", "the dense or off-the-shelf rows did not reproduce "
                "stage8/final", m, lo, hi)
    if lo > 0 and m >= floor:
        return ("LARGE-BETTER", f"large beats Stage 8 by {m:+.4f} R@1, {ci}, at or above "
                f"the {floor} floor", m, lo, hi)
    if hi < 0:
        return "LARGE-WORSE", f"large trails Stage 8 by {m:+.4f} R@1, {ci}", m, lo, hi
    if lo > 0:
        return ("TIE", f"large - Stage 8 = {m:+.4f} R@1, {ci}: excludes 0 but below the "
                f"{floor} floor", m, lo, hi)
    return ("TIE", f"large - Stage 8 = {m:+.4f} R@1, {ci}: not a detectable difference",
            m, lo, hi)


def cached_scorer(arm: str, load_model, cache_dir: pathlib.Path):
    """Score pairs with the model ``load_model()`` returns, loading it only when the
    scores are not cached, and releasing it afterwards."""
    import numpy as np

    def run(pairs):
        if not pairs:
            return np.zeros(0, dtype="float32")
        h = hashlib.sha1()
        for q, c in pairs:
            h.update(q.encode("utf-8") + b"\x00" + c.encode("utf-8") + b"\x01")
        key = h.hexdigest()
        path, meta = cache_dir / f"{arm}.npy", cache_dir / f"{arm}.json"
        if path.exists() and meta.exists():
            if json.loads(meta.read_text(encoding="utf-8")).get("pairs_sha1") == key:
                print(f"[stage15]   {arm}: scores reused from {path.name}", flush=True)
                return np.load(path)
        t0 = time.perf_counter()
        model = load_model()
        scores = np.asarray(model.predict(pairs, batch_size=int(C.RERANK_BATCH_SIZE),
                                          show_progress_bar=False), dtype="float32")
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, scores)
        tmp.replace(path)
        meta.write_text(json.dumps({"pairs_sha1": key, "n_pairs": len(pairs),
                                    "seconds": time.perf_counter() - t0}),
                        encoding="utf-8")
        return scores
    return run


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 15 step 2: bge-reranker-large against the Stage 8 reranker.")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_QUESTIONS} questions, off-the-shelf large in place of "
                         "the trained arm, temp dir, no verdict")
    ap.add_argument("--account", default="unlabelled",
                    help="operational label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()
    if args.smoke:
        evaluate(args)
        return

    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE15_ROOT_ID)
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE15_LOCK_FILE,
                             run_version=C.STAGE15_RUN_VERSION, account=args.account,
                             task="eval", clear_stale=args.clear_stale_lock):
        run_guard.probe_directory(C.RESULTS_LATEST_DIR)
        evaluate(args)


def evaluate(args) -> None:
    from rag_chunk import rerank_rl as rr

    ft_dir = C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"
    if not (ft_dir / "model.safetensors").exists():
        raise SystemExit(f"[stage15] no Stage 8 weights at {ft_dir}")
    large_dir = C.MODELS_DIR / C.STAGE15_MODEL_DIRNAME / "large" / "final"
    large_meta = rr.arm_meta(large_dir)
    if large_meta is None and not args.smoke:
        raise SystemExit("[stage15] no completed training - run "
                         "scripts/36_train_large_reranker.py first")

    s8 = C.RESULTS_DIR / "stage8" / "final" / C.STAGE8_RESULTS_CSV
    if not s8.exists():
        raise SystemExit(f"[stage15] missing {s8} - the evaluation checks itself "
                         "against the Stage 8 archive")
    with open(s8, newline="", encoding="utf-8") as fh:
        archived = list(csv.DictReader(fh))

    C.apply(RETRIEVAL_EMBED_MODEL="BAAI/bge-base-en-v1.5", RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()
    n = int(C.N_NQ_DOCS_LARGE)
    C.apply(N_NQ_DOCS=n)
    C.apply(NQ_DIR=C.NQ_DIR / f"large_n{n}")
    from rag_chunk import nq_data
    docs, questions = nq_data.prepare_nq()
    if args.smoke:
        questions = questions[:SMOKE_QUESTIONS]
    print(f"[stage15] {len(docs)} docs / {len(questions)} questions"
          f"{' (smoke)' if args.smoke else ''}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder

    from rag_chunk.large_eval import append_checkpoint, load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = int(C.RERANK_MAX_LENGTH)
    rev = C.STAGE15_BASE_REVISION
    large_ft_source = C.STAGE15_BASE_MODEL if args.smoke else str(large_dir)
    sources = {
        ARM_OTS: (C.RERANKER_MODEL, None),
        ARM_FT: (str(ft_dir), None),
        ARM_LARGE_OTS: (C.STAGE15_BASE_MODEL, rev),
        ARM_LARGE_FT: (large_ft_source, rev if args.smoke else None),
    }

    def loader(path, revision):
        def load():
            kwargs = {"revision": revision} if revision else {}
            return CrossEncoder(path, max_length=max_len, device=device, **kwargs)
        return load

    latest = (pathlib.Path(tempfile.mkdtemp(prefix="stage15_eval_smoke_")) if args.smoke
              else C.RESULTS_LATEST_DIR)
    scorers = {arm: cached_scorer(arm, loader(*src), latest / SCORES_DIR)
               for arm, src in sources.items()}

    checkpoint = latest / CHECKPOINT
    meta = {"stage": "stage15", "n_docs": len(docs), "n_questions": len(questions),
            "retrieval_model": C.RETRIEVAL_EMBED_MODEL,
            "models": {arm: f"{p}@{r}" if r else p for arm, (p, r) in sources.items()},
            "large_fingerprint": large_meta["fingerprint"] if large_meta else None}
    done = load_checkpoint(checkpoint, meta)
    if not done and not checkpoint.exists():
        append_checkpoint(checkpoint, {"meta": meta})
    rows = list(done)
    if rows:
        print("[stage15] fixed 15/0 already in the checkpoint", flush=True)
    else:
        print("[stage15] fixed 15/0", flush=True)
        rows = S11._eval_config(docs, questions, CONFIG[0], CONFIG[1], scorers)
        for r in rows:
            meta_path = latest / SCORES_DIR / f"{r['arm']}.json"
            if meta_path.exists():
                r["rerank_seconds"] = json.loads(
                    meta_path.read_text(encoding="utf-8"))["seconds"]
        append_checkpoint(checkpoint, {"rows": rows})

    paired = _paired(rows)
    for p in paired:
        print(f"[stage15] {p['metric']:9s} {p['comparison']:40s} {p['mean']:+.4f} "
              f"[{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]")
    by = {r["arm"]: r for r in rows}
    for arm in sources:
        sec = by[arm].get("rerank_seconds")
        if sec is not None:
            print(f"[stage15] {arm}: {sec / len(questions):.3f} s per question", flush=True)

    if args.smoke:
        shutil.rmtree(latest, ignore_errors=True)
        print("[stage15] smoke: every arm scored and the paired table built. The numbers "
              "are not results and nothing was kept.", flush=True)
        return

    S11._write_rows(latest / C.STAGE15_RESULTS_CSV, rows)
    S11._write_rows(latest / C.STAGE15_PAIRED_CSV, paired)
    check_ok, check_rows = S11._check(rows, archived)
    questions_ok = {int(float(r["n_questions"])) for r in archived} == {len(questions)}
    valid = check_ok and questions_ok
    S11._write_rows(latest / C.STAGE15_CHECK_CSV, check_rows)

    diffs = [a - b for a, b in zip(by[ARM_LARGE_FT]["hits1"], by[ARM_FT]["hits1"])]
    floor = float(C.STAGE15_PRACTICAL_FLOOR)
    verdict, why, m, lo, hi = _verdict(diffs, valid, floor)
    if not questions_ok and check_ok:
        why = f"the bench has {len(questions)} questions, not the archived count"

    lines = ["# Stage 15 - a larger cross-encoder under the Stage 8 recipe", "",
             f"Verdict: {verdict}. {why}.", "",
             f"- Stage 6 bench: {len(docs)} docs / {len(questions)} questions; all arms "
             f"rerank the same BGE top-{S11.DEPTH} pool",
             f"- pipeline check (bge, off-the-shelf vs stage8/final, within "
             f"{S11.CHECK_TOLERANCE}, same chunk and question counts): "
             f"{'PASS' if valid else 'FAIL'}",
             f"- primary comparison: fixed 15/0, R@1, {ARM_LARGE_FT} - {ARM_FT}, paired "
             f"over {len(diffs)} questions; threshold {floor}",
             f"- base model {C.STAGE15_BASE_MODEL} at {C.STAGE15_BASE_REVISION}",
             f"- training: {large_meta['total_steps']} steps, {large_meta['minutes']} min, "
             f"final mean epoch loss {large_meta['final_mean_epoch_loss']:.4f}, "
             f"non-finite losses {large_meta['nonfinite']}",
             "", "| arm | R@1 | R@3 | R@5 | pool@20 | s per question, load included |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        sec = r.get("rerank_seconds")
        per_q = f"{sec / len(questions):.3f}" if sec is not None else "-"
        lines.append(f"| {r['arm']} | {r['recall@1']:.4f} | {r['recall@3']:.4f} | "
                     f"{r['recall@5']:.4f} | {r[f'pool_recall@{S11.DEPTH}']:.4f} | {per_q} |")
    lines += ["", "| metric | comparison | mean | 95% CI |", "| --- | --- | --- | --- |"]
    for p in paired:
        lines.append(f"| {p['metric']} | {p['comparison']} | {p['mean']:+.4f} | "
                     f"[{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}] |")
    (latest / C.STAGE15_SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (latest / VERDICT_JSON).write_text(json.dumps(
        {"verdict": verdict, "why": why, "check_ok": check_ok,
         "questions_ok": questions_ok, "large_minus_stage8_r1": m, "ci95": [lo, hi],
         "floor": floor, "n_questions": len(diffs)}, indent=2), encoding="utf-8")
    print(f"[stage15] wrote {latest / C.STAGE15_SUMMARY_MD}")
    print(f"\n[stage15] VERDICT: {verdict} - {why}.")


if __name__ == "__main__":
    main()
