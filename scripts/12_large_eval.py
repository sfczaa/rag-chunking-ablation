"""Stage 6 - larger-scale robustness evaluation.

Re-runs the Stage 3/5 protocol at ~1000 NQ docs/questions (``N_NQ_DOCS_LARGE``)
to test whether the Stage 1-5 conclusions replicate at larger N. Nothing else
changes: same dataset source, same chunking grids, same boundary models and
weights, same BGE dense retriever, same off-the-shelf reranker (top-20 only —
no rerank50, no fine-tuning).

Two modes:

    --check   N=200 sanity mode. Uses the same cached corpus as Stages 3/5 and
              compares every row (bge on all 30 configs, rerank20 on the
              STAGE6_RERANK_CONFIGS subset) against the archived Stage 5 rows
              — all deltas should be 0.0000. Run this BEFORE the large run.
    (default) the large eval. The bigger corpus caches under a separate
              nq/large_n<N>/ folder, so the 200-doc cache is never touched.
              The eval set differs from Stages 3/5, so no exact-delta check is
              possible; instead stage6_direction_check.csv reports whether the
              four Stage 1-5 direction claims replicate.

Both modes are restart/resume-safe: every finished config is appended to a
JSONL checkpoint in results/latest/, and re-running the same command skips
finished configs (--fresh discards the checkpoint). The cleanup of
results/latest/ runs only after all prechecks, weights and the corpus load,
never deletes anything without a byte-identical archived copy, and leaves the
stage6_* working files alone.

Usage:
    python scripts/12_large_eval.py --check
    python scripts/12_large_eval.py
    python scripts/12_large_eval.py --n-docs 500
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import shutil
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402


REQUIRED_STAGE5_FILES = (
    C.RERANK_SWEEP_CSV,
    C.RERANK_BEST_JSON,
    "stage5_reranker_summary.md",
)
STAGE6_CHECK_CSV = "stage6_check_vs_stage5.csv"
CHECKPOINT_FILES = {
    "check": "stage6_checkpoint_check.jsonl",
    "large": "stage6_checkpoint_large.jsonl",
}
# In check mode every (arm, config) re-runs the exact Stage 5 pipeline on the
# same cached corpus, so recall deltas should be 0.0000; anything above one
# question (~0.005 at n=203) means the setup differs.
CHECK_TOLERANCE = 0.005


def _stage_final_dir(stage: str) -> pathlib.Path:
    return C.RESULTS_DIR / stage / "final"


def _ensure_stage5_archived() -> pathlib.Path:
    stage5_dir = _stage_final_dir("stage5")
    missing = [name for name in REQUIRED_STAGE5_FILES
               if not (stage5_dir / name).exists()]
    if missing:
        raise SystemExit(
            "[stage6] the Stage 5 archive is incomplete.\n"
            f"[stage6] Missing from {stage5_dir}: {', '.join(missing)}\n\n"
            "Stage 6 checks its rows against the archived Stage 5 run. "
            "Archive Stage 5 first (copy-only, does not rerun anything):\n"
            "  python scripts/save_stage_results.py --stage stage5"
        )
    return stage5_dir


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _ensure_stage5_models(stage5_rows: list[dict], retrieval_model: str,
                          reranker_model: str) -> None:
    """The archived Stage 5 baseline must use the same models Stage 6 is about
    to run, or neither the check mode nor the direction references hold."""
    retr = {row.get("retrieval_embedding_model", "") for row in stage5_rows}
    if retr != {retrieval_model}:
        raise SystemExit(
            f"[stage6] Stage 5 archive uses retrieval model(s) {sorted(retr)}, "
            f"but Stage 6 is configured for {retrieval_model!r}.")
    rer = {row.get("reranker_model", "") for row in stage5_rows
           if str(row.get("arm", "")).startswith("rerank")}
    if rer and rer != {reranker_model}:
        raise SystemExit(
            f"[stage6] Stage 5 archive uses reranker(s) {sorted(rer)}, "
            f"but Stage 6 is configured for {reranker_model!r}.")


def _sha1(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_latest() -> None:
    """Remove archived files from ``results/latest/`` so the Stage 6 archive
    stays pure. Only files with a byte-identical copy in some stage archive are
    deleted, and the ``stage6_*`` working files (checkpoints + outputs of an
    in-progress Stage 6) are always left alone — resume depends on them."""
    latest = C.RESULTS_LATEST_DIR
    if not latest.exists():
        return
    archives = [d for d in (_stage_final_dir(s)
                            for s in ("stage3", "stage4", "stage5", "stage6"))
                if d.is_dir()]
    stale = [p for p in sorted(latest.iterdir())
             if p.is_file() and not p.name.startswith("stage6_")]
    unarchived = [p.name for p in stale
                  if not any((a / p.name).exists() and _sha1(a / p.name) == _sha1(p)
                             for a in archives)]
    if unarchived:
        raise SystemExit(
            "[stage6] results/latest/ has file(s) with no identical copy in any "
            f"stage archive: {', '.join(unarchived)}\n"
            "[stage6] Archive them first so nothing is lost, e.g.:\n"
            "  python scripts/save_stage_results.py --stage stage5"
        )
    for p in stale:
        p.unlink()
    if stale:
        print(f"[stage6] cleared {len(stale)} archived file(s) from {latest}")


# --------------------------------------------------------------------------- #
# Check mode: every row must reproduce the archived Stage 5 run exactly
# --------------------------------------------------------------------------- #
def _as_float(row: dict, col: str):
    val = row.get(col)
    if val in ("", None):
        return None
    return float(val)


def _build_stage5_check(stage5_rows: list[dict], stage6_rows: list[dict]):
    """Match Stage 6 rows to the archived Stage 5 rows by (arm, chunking
    config). Returns ``(columns, rows, stats)``."""
    from rag_chunk.hybrid import config_key, config_label

    ks = sorted(C.RECALL_KS)
    s5 = {(str(r["arm"]), config_key(r)): r for r in stage5_rows}
    cols = ["arm", "method", "chunk_config",
            "stage5_n_chunks", "stage6_n_chunks", "delta_n_chunks",
            "stage5_avg_chunk_size", "stage6_avg_chunk_size"]
    for k in ks:
        cols += [f"stage5_recall@{k}", f"stage6_recall@{k}", f"delta_recall@{k}"]

    rows = []
    stats = {"unmatched": 0, "n_chunks_mismatch": 0, "max_abs_recall_delta": 0.0}
    for r6 in stage6_rows:
        r5 = s5.get((str(r6["arm"]), config_key(r6)))
        out = {"arm": r6["arm"], "method": r6["method"],
               "chunk_config": config_label(r6),
               "stage6_n_chunks": r6["n_chunks"],
               "stage6_avg_chunk_size": f"{r6['avg_chunk_size']:.3f}"}
        if r5 is None:
            stats["unmatched"] += 1
            rows.append(out)
            continue
        n5 = int(_as_float(r5, "n_chunks"))
        out["stage5_n_chunks"] = n5
        out["delta_n_chunks"] = r6["n_chunks"] - n5
        if out["delta_n_chunks"] != 0:
            stats["n_chunks_mismatch"] += 1
        out["stage5_avg_chunk_size"] = f"{_as_float(r5, 'avg_chunk_size'):.3f}"
        for k in ks:
            v5 = _as_float(r5, f"recall@{k}")
            v6 = round(r6[f"recall@{k}"], 4)
            delta = round(v6 - v5, 4)
            stats["max_abs_recall_delta"] = max(stats["max_abs_recall_delta"],
                                                abs(delta))
            out[f"stage5_recall@{k}"] = f"{v5:.4f}"
            out[f"stage6_recall@{k}"] = f"{v6:.4f}"
            out[f"delta_recall@{k}"] = f"{delta:.4f}"
        rows.append(out)
    return cols, rows, stats


def _write_stage5_check(stage5_rows, stage6_rows, path: pathlib.Path) -> dict:
    cols, rows, stats = _build_stage5_check(stage5_rows, stage6_rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stage6] wrote {path} ({len(rows)} rows)")

    if stats["unmatched"]:
        print(f"[stage6] WARN: {stats['unmatched']} row(s) had no matching "
              "Stage 5 row (was the archived Stage 5 run the full grid?)")
    if stats["n_chunks_mismatch"]:
        print(f"[stage6] WARN: {stats['n_chunks_mismatch']} config(s) produced "
              "a different chunk count than Stage 5 — chunking is NOT "
              "identical, do not trust this run.")
    d = stats["max_abs_recall_delta"]
    if stats["unmatched"] == 0 and stats["n_chunks_mismatch"] == 0 and d == 0.0:
        print("[stage6] check OK: Stage 6 reproduces the archived Stage 5 rows "
              "exactly — proceed to the large run.")
    elif d <= CHECK_TOLERANCE and stats["n_chunks_mismatch"] == 0:
        print(f"[stage6] check: max |recall delta| vs Stage 5 = {d:.4f} "
              "(within one question — acceptable, but inspect "
              f"{STAGE6_CHECK_CSV} before the large run).")
    else:
        print(f"[stage6] WARN: max |recall delta| vs Stage 5 = {d:.4f} exceeds "
              f"{CHECK_TOLERANCE} — Stage 6 does not reproduce the baseline. "
              "Fix this before the large run.")
    return stats


# --------------------------------------------------------------------------- #
# Large-mode summary
# --------------------------------------------------------------------------- #
def _best_bge(rows: list[dict]) -> dict | None:
    from rag_chunk.sweep import _rank_key

    bge = [r for r in rows if r["arm"] == "bge"]
    return max(bge, key=_rank_key) if bge else None


def _write_summary(out_dir: pathlib.Path, rows, matched, checks,
                   n_docs: int, n_questions: int, n_requested: int) -> None:
    from rag_chunk.hybrid import config_label
    from rag_chunk.large_eval import rerank_arm, rerank_depth

    ks = sorted(C.RECALL_KS)
    arm = rerank_arm()
    lines = [
        "# Stage 6 - larger-scale robustness evaluation",
        "",
        f"- Eval set: **{n_docs} docs / {n_questions} questions** "
        f"(requested ~{n_requested} docs; the stream stops at the N-th usable "
        "document, so the counts are reported, not assumed)",
        f"- Boundary/chunking embedding model: `{C.BOUNDARY_EMBED_MODEL}`",
        f"- Dense retrieval embedding model: `{C.RETRIEVAL_EMBED_MODEL}`",
        f"- Reranker: `{C.RERANKER_MODEL}` (off-the-shelf, top-{rerank_depth()} "
        "only, no fine-tuning)",
        f"- bge arm: full 30-config grid; {arm} arm: "
        f"{len(matched)} selected configs",
        "",
    ]
    best = _best_bge(rows)
    if best:
        recalls = " ".join(f"R@{k}={best[f'recall@{k}']:.4f}" for k in ks)
        lines += [f"Best bge config: `{best['method']}`, "
                  f"`{config_label(best)}` — {recalls}", ""]
    if matched:
        lines += [f"## {arm} vs bge on the selected configs", ""]
        header = "| config | pool_recall@%d |" % rerank_depth()
        header += "".join(f" ΔR@{k} |" for k in ks)
        lines += [header, "|" + "---|" * (2 + len(ks))]
        for m in matched:
            cells = f"| {m['method']} {m['chunk_config']} " \
                    f"| {m[f'pool_recall@{rerank_depth()}']:.4f} |"
            cells += "".join(f" {m[f'{arm}_minus_bge@{k}']:+.4f} |" for k in ks)
            lines.append(cells)
        lines.append("")
    lines += ["## Direction checks (do the Stage 1-5 conclusions replicate?)",
              ""]
    for c in checks:
        val = "" if c["value"] is None else f" — observed {c['value']}"
        lines.append(f"- **{c['replicates']}** — {c['claim']}: {c['metric']}"
                     f"{val} (rule: {c['rule']}; small-eval reference: "
                     f"{c['reference_small_eval']})")
    n_yes = sum(1 for c in checks if c["replicates"] == "yes")
    lines += ["", f"**{n_yes}/{len(checks)} direction checks replicate.**", ""]
    path = out_dir / C.STAGE6_SUMMARY_MD
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stage6] wrote {path}")


def _print_summary(rows, matched, checks, n_docs: int, n_questions: int) -> None:
    from rag_chunk.hybrid import config_label
    from rag_chunk.large_eval import rerank_arm

    ks = sorted(C.RECALL_KS)
    arm = rerank_arm()
    print(f"\n[stage6] {len(rows)} rows scored on {n_docs} docs / "
          f"{n_questions} questions")
    best = _best_bge(rows)
    if best:
        recalls = " ".join(f"R@{k}={best[f'recall@{k}']:.4f}" for k in ks)
        print(f"[stage6] best bge config: {best['method']} "
              f"{config_label(best)}  {recalls}")
    for m in matched:
        cells = "  ".join(f"R@{k} {m[f'{arm}_minus_bge@{k}']:+.4f}" for k in ks)
        print(f"[stage6] {arm} - bge @ {m['method']} {m['chunk_config']}: "
              f"{cells}")
    print("\n[stage6] direction checks:")
    for c in checks:
        val = "" if c["value"] is None else f" (observed {c['value']})"
        print(f"  [{c['replicates']:>3}] {c['claim']}{val}")
    n_yes = sum(1 for c in checks if c["replicates"] == "yes")
    print(f"[stage6] {n_yes}/{len(checks)} direction checks replicate.\n")


def _snapshot_latest(out_dir: pathlib.Path) -> pathlib.Path:
    snap = (C.RESULTS_RUNS_DIR /
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage6_large_eval")
    snap.mkdir(parents=True, exist_ok=True)
    for name in (C.STAGE6_RESULTS_CSV, C.STAGE6_MATCHED_CSV,
                 C.STAGE6_DIRECTION_CSV, C.STAGE6_SUMMARY_MD,
                 STAGE6_CHECK_CSV):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, snap / name)
    print(f"[stage6] snapshot saved -> {snap}")
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 6: larger-scale robustness evaluation (bge on the "
                    "full grid + rerank20 on selected configs).")
    ap.add_argument("--check", action="store_true",
                    help="N=200 sanity mode: reproduce the archived Stage 5 "
                         "rows exactly on the same cached corpus")
    ap.add_argument("--n-docs", type=int, default=None,
                    help="large-eval corpus size "
                         f"(default: config N_NQ_DOCS_LARGE={C.N_NQ_DOCS_LARGE})")
    ap.add_argument("--fresh", action="store_true",
                    help="discard this mode's checkpoint and start over")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="dense retrieval model; must match the Stage 5 archive")
    ap.add_argument("--retrieval-dim", type=int, default=None,
                    help="optional retrieval embedding dimension override")
    ap.add_argument("--reranker-model", default=None,
                    help="pretrained cross-encoder "
                         f"(default: config RERANKER_MODEL={C.RERANKER_MODEL})")
    ap.add_argument("--rerank-batch", type=int, default=None,
                    help="cross-encoder batch size "
                         f"(default: config RERANK_BATCH_SIZE={C.RERANK_BATCH_SIZE})")
    ap.add_argument("--save-run", action="store_true",
                    help="also snapshot Stage 6 artifacts to "
                         "results/runs/<timestamp>_stage6_large_eval/")
    args = ap.parse_args()
    if args.check and args.n_docs is not None:
        raise SystemExit("[stage6] --check always uses the default corpus size; "
                         "drop --n-docs")

    stage5_dir = _ensure_stage5_archived()
    stage5_rows = _read_csv(stage5_dir / C.RERANK_SWEEP_CSV)
    reranker_model = args.reranker_model or C.RERANKER_MODEL
    _ensure_stage5_models(stage5_rows, args.retrieval_model, reranker_model)

    mode = "check" if args.check else "large"
    default_n = C.N_NQ_DOCS
    n_docs = default_n if mode == "check" else (args.n_docs or C.N_NQ_DOCS_LARGE)

    overrides = {
        "RETRIEVAL_EMBED_MODEL": args.retrieval_model,
        "RETRIEVAL_EMBED_DIM": args.retrieval_dim,
        "RETRIEVAL_EMBED_NORMALIZE": True,
        "RERANK_DEPTHS": (20,),          # Stage 6 scope: top-20 only
        "N_NQ_DOCS": n_docs,
        "RERANKER_MODEL": reranker_model,
    }
    if args.rerank_batch is not None:
        overrides["RERANK_BATCH_SIZE"] = args.rerank_batch
    C.apply(**overrides)
    C.ensure_dirs()

    # The large corpus caches under a separate folder so the default 200-doc
    # cache (Stages 1-5 + check mode) is never overwritten or rebuilt.
    if n_docs != default_n:
        large_nq = C.NQ_DIR / f"large_n{n_docs}"
        C.apply(NQ_DIR=large_nq)
        large_nq.mkdir(parents=True, exist_ok=True)
        print(f"[stage6] large corpus caches separately -> {large_nq}")

    # Heavy imports + weights + reranker + corpus all load BEFORE latest/ is
    # touched, so any failure leaves the archived files and checkpoints intact.
    from rag_chunk import large_eval, nq_data, rerank, training

    print("[stage6] mode:", mode)
    print("[stage6] boundary/chunking embeddings stay on:", C.BOUNDARY_EMBED_MODEL)
    print("[stage6] dense retrieval embeddings:", C.RETRIEVAL_EMBED_MODEL)
    print(f"[stage6] reranker: {C.RERANKER_MODEL} (off-the-shelf) "
          f"depth={large_eval.rerank_depth()} batch={C.RERANK_BATCH_SIZE}")
    print("[stage6] Stage 5 archive:", stage5_dir)

    bilstm = training.load_model("bilstm")
    transformer = training.load_model("transformer")
    rerank.load_reranker()
    docs, questions = nq_data.prepare_nq()
    print(f"[stage6] eval set: {len(docs)} docs, {len(questions)} questions "
          f"(requested ~{n_docs} docs)")

    latest = C.RESULTS_LATEST_DIR
    ckpt = latest / CHECKPOINT_FILES[mode]
    if args.fresh and ckpt.exists():
        ckpt.unlink()
        print(f"[stage6] --fresh: discarded checkpoint {ckpt.name}")

    _clean_latest()
    rows = large_eval.run_stage6(
        docs, questions, bilstm=bilstm, transformer_model=transformer,
        mode=mode, checkpoint_path=ckpt)

    if mode == "check":
        _write_stage5_check(stage5_rows, rows, latest / STAGE6_CHECK_CSV)
        print("[stage6] check mode done. If the check is OK, run the large "
              "eval:\n  python scripts/12_large_eval.py")
        return

    large_eval.write_results_csv(rows, latest / C.STAGE6_RESULTS_CSV)
    matched = large_eval.matched_summary(rows)
    large_eval.write_matched_csv(matched, latest / C.STAGE6_MATCHED_CSV)
    checks = large_eval.direction_checks(rows, len(questions))
    large_eval.write_direction_csv(checks, latest / C.STAGE6_DIRECTION_CSV)
    _write_summary(latest, rows, matched, checks,
                   len(docs), len(questions), n_docs)
    _print_summary(rows, matched, checks, len(docs), len(questions))
    if args.save_run:
        _snapshot_latest(latest)

    print("[stage6] complete. Review the direction checks above, then archive:")
    print("  python scripts/save_stage_results.py --stage stage6")


if __name__ == "__main__":
    main()
