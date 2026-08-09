"""Stage 4 - hybrid retrieval ablation (BM25 + BGE + RRF).

Runs the Stage 3 fixed / BiLSTM / Transformer chunking sweep unchanged, but
scores every chunking config with three retrievers over the identical chunks:

    bge   dense BGE retrieval (exactly the Stage 3 path)
    bm25  lexical Okapi BM25 (pure numpy, no new dependency)
    rrf   Reciprocal Rank Fusion of the BGE and BM25 rankings

Dataset size, chunking logic, boundary models/weights, and the BGE model are
all unchanged. Before writing ``artifacts/results/latest/``, this script
requires Stage 3 to be archived under ``artifacts/results/stage3/final/`` (the
copy-only ``save_stage_results.py --stage stage3``) so the BGE baseline is not
lost when ``latest`` is overwritten. The archived Stage 3 rows are then used as
a consistency check: the Stage 4 ``bge`` arm must reproduce them.

Usage:
    python scripts/10_sweep_hybrid_retrieval.py
    python scripts/10_sweep_hybrid_retrieval.py --quick
    python scripts/10_sweep_hybrid_retrieval.py --rrf-k 60 --fuse-depth 50
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


REQUIRED_STAGE3_FILES = (
    C.SWEEP_RESULTS_CSV,
    C.BEST_CONFIG_JSON,
    C.FAIR_TABLE_CSV,
    C.RECALL_PLOT_PNG,
    C.MODEL_PLOT_PNG,
    "stage3_bge_retrieval_summary.md",
)
STAGE4_SUMMARY_MD = "stage4_hybrid_retrieval_summary.md"
STAGE3_CHECK_CSV = "stage3_vs_stage4_bge_check.csv"
# The bge arm re-runs the exact Stage 3 pipeline, so its recall deltas vs the
# archived baseline should be 0.0000; anything above one question (~0.005 at
# n=203) means the setup differs and the run should not be trusted.
CHECK_TOLERANCE = 0.005


def _stage3_final_dir() -> pathlib.Path:
    return C.RESULTS_DIR / "stage3" / "final"


def _ensure_stage3_archived() -> pathlib.Path:
    stage3_dir = _stage3_final_dir()
    missing = [name for name in REQUIRED_STAGE3_FILES if not (stage3_dir / name).exists()]
    if missing:
        raise SystemExit(
            "[stage4] Stage 3 latest results are not archived yet.\n"
            f"[stage4] Missing from {stage3_dir}: {', '.join(missing)}\n\n"
            "Run this first (copy-only, does not rerun Stage 3):\n"
            "  python scripts/save_stage_results.py --stage stage3"
        )
    return stage3_dir


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _ensure_stage3_is_bge(stage3_rows: list[dict], retrieval_model: str) -> None:
    """The archived baseline must have been produced with the same dense
    retrieval model Stage 4 is about to use, or the bge-arm check is meaningless."""
    models = {row.get("retrieval_embedding_model", "") for row in stage3_rows}
    if models != {retrieval_model}:
        raise SystemExit(
            f"[stage4] Stage 3 archive was built with retrieval model(s) "
            f"{sorted(models)}, but Stage 4 is configured for {retrieval_model!r}.\n"
            "[stage4] Pass --retrieval-model to match the archive, or re-archive "
            "the intended Stage 3 run."
        )


def _sha1(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_latest(stage3_dir: pathlib.Path) -> None:
    """Remove files from ``results/latest/`` so the Stage 4 archive stays pure —
    but only files whose byte-identical copy exists in the Stage 3 archive, so
    nothing unarchived is ever deleted."""
    latest = C.RESULTS_LATEST_DIR
    if not latest.exists():
        return
    stale = [p for p in sorted(latest.iterdir()) if p.is_file()]
    unarchived = [p.name for p in stale
                  if not (stage3_dir / p.name).exists()
                  or _sha1(stage3_dir / p.name) != _sha1(p)]
    if unarchived:
        raise SystemExit(
            "[stage4] results/latest/ has file(s) with no identical copy in the "
            f"Stage 3 archive: {', '.join(unarchived)}\n"
            "[stage4] Re-archive first so nothing is lost:\n"
            "  python scripts/save_stage_results.py --stage stage3"
        )
    for p in stale:
        p.unlink()
    if stale:
        print(f"[stage4] cleared {len(stale)} archived Stage 3 file(s) from {latest}")


# --------------------------------------------------------------------------- #
# Stage 3 consistency check (bge arm must reproduce the archived baseline)
# --------------------------------------------------------------------------- #
def _as_float(row: dict, col: str):
    val = row.get(col)
    if val in ("", None):
        return None
    return float(val)


def _build_stage3_check(stage3_rows: list[dict], stage4_rows: list[dict]):
    """Match Stage 4's bge rows to the archived Stage 3 rows by chunking config.

    Returns ``(columns, rows, stats)`` where stats reports unmatched configs,
    n_chunks mismatches and the max |recall delta| (on 4-dp rounded values, the
    precision the Stage 3 CSV stores).
    """
    from rag_chunk.hybrid import config_key, config_label

    ks = sorted(C.RECALL_KS)
    s3_by_key = {config_key(r): r for r in stage3_rows}
    cols = ["method", "chunk_config",
            "stage3_n_chunks", "stage4_n_chunks", "delta_n_chunks",
            "stage3_avg_chunk_size", "stage4_avg_chunk_size"]
    for k in ks:
        cols += [f"stage3_recall@{k}", f"stage4_recall@{k}", f"delta_recall@{k}"]

    rows = []
    stats = {"unmatched": 0, "n_chunks_mismatch": 0, "max_abs_recall_delta": 0.0}
    for r4 in [r for r in stage4_rows if r["retriever"] == "bge"]:
        r3 = s3_by_key.get(config_key(r4))
        out = {"method": r4["method"], "chunk_config": config_label(r4),
               "stage4_n_chunks": r4["n_chunks"],
               "stage4_avg_chunk_size": f"{r4['avg_chunk_size']:.3f}"}
        if r3 is None:
            stats["unmatched"] += 1
            rows.append(out)
            continue
        n3 = int(_as_float(r3, "n_chunks"))
        out["stage3_n_chunks"] = n3
        out["delta_n_chunks"] = r4["n_chunks"] - n3
        if out["delta_n_chunks"] != 0:
            stats["n_chunks_mismatch"] += 1
        out["stage3_avg_chunk_size"] = f"{_as_float(r3, 'avg_chunk_size'):.3f}"
        for k in ks:
            v3 = _as_float(r3, f"recall@{k}")
            v4 = round(r4[f"recall@{k}"], 4)
            delta = round(v4 - v3, 4)
            stats["max_abs_recall_delta"] = max(stats["max_abs_recall_delta"], abs(delta))
            out[f"stage3_recall@{k}"] = f"{v3:.4f}"
            out[f"stage4_recall@{k}"] = f"{v4:.4f}"
            out[f"delta_recall@{k}"] = f"{delta:.4f}"
        rows.append(out)
    return cols, rows, stats


def _write_stage3_check(stage3_rows, stage4_rows, path: pathlib.Path) -> dict:
    cols, rows, stats = _build_stage3_check(stage3_rows, stage4_rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stage4] wrote {path} ({len(rows)} rows)")

    if stats["unmatched"]:
        print(f"[stage4] WARN: {stats['unmatched']} bge row(s) had no matching "
              "Stage 3 row (was the archived Stage 3 run the full grid?)")
    if stats["n_chunks_mismatch"]:
        print(f"[stage4] WARN: {stats['n_chunks_mismatch']} config(s) produced a "
              "different chunk count than Stage 3 — chunking is NOT identical, "
              "do not trust this run.")
    d = stats["max_abs_recall_delta"]
    if stats["unmatched"] == 0 and stats["n_chunks_mismatch"] == 0 and d == 0.0:
        print("[stage4] check OK: the bge arm reproduces the archived Stage 3 "
              "baseline exactly.")
    elif d <= CHECK_TOLERANCE and stats["n_chunks_mismatch"] == 0:
        print(f"[stage4] check: max |recall delta| vs Stage 3 = {d:.4f} "
              "(within one question — acceptable, but inspect "
              f"{STAGE3_CHECK_CSV} before archiving).")
    else:
        print(f"[stage4] WARN: max |recall delta| vs Stage 3 = {d:.4f} exceeds "
              f"{CHECK_TOLERANCE} — the bge arm does not reproduce the baseline. "
              "Inspect before archiving.")
    return stats


def _write_summary(stage3_dir: pathlib.Path, out_dir: pathlib.Path,
                   stats: dict) -> pathlib.Path:
    import json

    from rag_chunk.hybrid import config_label

    best_path = out_dir / C.HYBRID_BEST_JSON
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    lines = [
        "# Stage 4 - hybrid retrieval ablation (BM25 + BGE + RRF)",
        "",
        f"- Boundary/chunking embedding model: `{C.BOUNDARY_EMBED_MODEL}`",
        f"- Dense retrieval embedding model: `{C.RETRIEVAL_EMBED_MODEL}`",
        f"- Query instruction: `{C.retrieval_query_instruction()}`",
        f"- BM25: Okapi, k1={C.BM25_K1}, b={C.BM25_B}, "
        "lower-case alphanumeric tokens, Lucene idf",
        f"- RRF: k={C.RRF_K}, fuse depth={C.HYBRID_FUSE_DEPTH} per retriever",
        f"- Stage 3 baseline compared from: `{stage3_dir}`",
        f"- Stage 3 bge-arm check: max |recall delta| = "
        f"{stats['max_abs_recall_delta']:.4f}, "
        f"n_chunks mismatches = {stats['n_chunks_mismatch']}, "
        f"unmatched = {stats['unmatched']}",
        "",
        "Outputs in this folder:",
        f"- `{C.HYBRID_SWEEP_CSV}`: one row per (retriever, chunking config)",
        f"- `{C.HYBRID_BEST_JSON}`: best config per retriever + overall",
        f"- `{C.HYBRID_MATCHED_CSV}`: bge / bm25 / rrf side by side per config",
        f"- `{STAGE3_CHECK_CSV}`: Stage 4 bge arm vs archived Stage 3 rows",
        f"- `{C.HYBRID_SCATTER_PNG}` and `{C.HYBRID_RETRIEVER_PLOT_PNG}`: plots",
    ]
    per = best.get("per_retriever", {})
    if per:
        lines += ["", "Best config per retriever:"]
        for name in ("bge", "bm25", "rrf"):
            cfg = per.get(name, {}).get("config")
            if cfg:
                recalls = " ".join(
                    f"R@{k}={cfg[f'recall@{k}']:.4f}" for k in sorted(C.RECALL_KS))
                lines.append(f"- {name}: `{cfg['method']}`, "
                             f"`{config_label(cfg)}` — {recalls}")
    path = out_dir / STAGE4_SUMMARY_MD
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stage4] wrote {path}")
    return path


def _snapshot_latest(out_dir: pathlib.Path) -> pathlib.Path:
    snap = C.RESULTS_RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage4_hybrid"
    snap.mkdir(parents=True, exist_ok=True)
    for name in (
        C.HYBRID_SWEEP_CSV,
        C.HYBRID_BEST_JSON,
        C.HYBRID_MATCHED_CSV,
        STAGE3_CHECK_CSV,
        C.HYBRID_SCATTER_PNG,
        C.HYBRID_RETRIEVER_PLOT_PNG,
        STAGE4_SUMMARY_MD,
    ):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, snap / name)
    print(f"[stage4] snapshot saved -> {snap}")
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 4: score the fixed/BiLSTM/Transformer sweep with "
                    "BGE-only, BM25-only, and BGE+BM25 RRF retrieval.")
    ap.add_argument("--quick", action="store_true",
                    help="use the smaller QUICK_* grids for a fast sanity sweep")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="dense retrieval model; must match the Stage 3 archive "
                         "(default: BAAI/bge-base-en-v1.5)")
    ap.add_argument("--retrieval-dim", type=int, default=None,
                    help="optional retrieval embedding dimension override")
    ap.add_argument("--rrf-k", type=int, default=None,
                    help=f"RRF constant (default: config RRF_K={C.RRF_K})")
    ap.add_argument("--fuse-depth", type=int, default=None,
                    help="candidates taken from each retriever before fusion "
                         f"(default: config HYBRID_FUSE_DEPTH={C.HYBRID_FUSE_DEPTH})")
    ap.add_argument("--save-run", action="store_true",
                    help="also snapshot Stage 4 artifacts to "
                         "results/runs/<timestamp>_stage4_hybrid/")
    args = ap.parse_args()

    stage3_dir = _ensure_stage3_archived()
    stage3_rows = _read_csv(stage3_dir / C.SWEEP_RESULTS_CSV)
    _ensure_stage3_is_bge(stage3_rows, args.retrieval_model)

    overrides = {
        "RETRIEVAL_EMBED_MODEL": args.retrieval_model,
        "RETRIEVAL_EMBED_DIM": args.retrieval_dim,
        "RETRIEVAL_EMBED_NORMALIZE": True,
    }
    if args.rrf_k is not None:
        overrides["RRF_K"] = args.rrf_k
    if args.fuse_depth is not None:
        overrides["HYBRID_FUSE_DEPTH"] = args.fuse_depth
    C.apply(**overrides)
    C.ensure_dirs()

    # Heavy imports + model weights load BEFORE latest/ is cleared, so an
    # environment or weights failure leaves the archived Stage 3 files in place.
    from rag_chunk import hybrid, training

    print("[stage4] boundary/chunking embeddings stay on:", C.BOUNDARY_EMBED_MODEL)
    print("[stage4] dense retrieval embeddings:", C.RETRIEVAL_EMBED_MODEL)
    print(f"[stage4] BM25 k1={C.BM25_K1} b={C.BM25_B}; "
          f"RRF k={C.RRF_K} depth={C.HYBRID_FUSE_DEPTH}")
    print("[stage4] Stage 3 archive:", stage3_dir)

    bilstm = training.load_model("bilstm")
    transformer = training.load_model("transformer")

    _clean_latest(stage3_dir)
    rows = hybrid.run_hybrid_sweep(
        bilstm,
        transformer_model=transformer,
        quick=args.quick,
        out_dir=C.RESULTS_LATEST_DIR,
    )

    stats = _write_stage3_check(
        stage3_rows, rows, C.RESULTS_LATEST_DIR / STAGE3_CHECK_CSV)
    _write_summary(stage3_dir, C.RESULTS_LATEST_DIR, stats)
    if args.save_run:
        _snapshot_latest(C.RESULTS_LATEST_DIR)

    print("[stage4] complete. Verify the check above, then archive:")
    print("  python scripts/save_stage_results.py --stage stage4")


if __name__ == "__main__":
    main()
