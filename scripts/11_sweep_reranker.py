"""Stage 5 - cross-encoder reranking on top of the BGE retrieval baseline.

Runs the Stage 3/4 fixed / BiLSTM / Transformer chunking sweep unchanged, but
scores every chunking config with the dense baseline plus rerank arms over the
identical chunks:

    bge       dense BGE retrieval (exactly the Stage 3/4 path)
    rerank20  BGE top-20 candidates reordered by a pretrained cross-encoder
    rerank50  BGE top-50 candidates reordered by the same cross-encoder

Dataset size, chunking logic, boundary models/weights, and the BGE dense
retriever are all unchanged; the reranker is off-the-shelf (no training or
fine-tuning). Before writing ``artifacts/results/latest/``, this script
requires the Stage 3 archive (``results/stage3/final/``) as the bge-arm
consistency baseline, and refuses to delete anything from ``latest/`` that has
no byte-identical copy in a stage archive — so Stage 3/4 finals are only ever
read, never modified.

Usage:
    python scripts/11_sweep_reranker.py
    python scripts/11_sweep_reranker.py --quick
    python scripts/11_sweep_reranker.py --depths 20,50 --rerank-batch 32
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
    "stage3_bge_retrieval_summary.md",
)
STAGE5_SUMMARY_MD = "stage5_reranker_summary.md"
STAGE3_CHECK_CSV = "stage3_vs_stage5_bge_check.csv"
# The bge arm re-runs the exact Stage 3 pipeline, so its recall deltas vs the
# archived baseline should be 0.0000; anything above one question (~0.005 at
# n=203) means the setup differs and the run should not be trusted.
CHECK_TOLERANCE = 0.005


def _stage_final_dir(stage: str) -> pathlib.Path:
    return C.RESULTS_DIR / stage / "final"


def _ensure_stage3_archived() -> pathlib.Path:
    stage3_dir = _stage_final_dir("stage3")
    missing = [name for name in REQUIRED_STAGE3_FILES if not (stage3_dir / name).exists()]
    if missing:
        raise SystemExit(
            "[stage5] the Stage 3 archive (the bge-arm baseline) is incomplete.\n"
            f"[stage5] Missing from {stage3_dir}: {', '.join(missing)}\n\n"
            "Stage 5 checks its bge arm against the archived Stage 3 rows. "
            "Archive Stage 3 first (copy-only, does not rerun anything):\n"
            "  python scripts/save_stage_results.py --stage stage3"
        )
    return stage3_dir


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _ensure_stage3_is_bge(stage3_rows: list[dict], retrieval_model: str) -> None:
    """The archived baseline must have been produced with the same dense
    retrieval model Stage 5 is about to use, or the bge-arm check is meaningless."""
    models = {row.get("retrieval_embedding_model", "") for row in stage3_rows}
    if models != {retrieval_model}:
        raise SystemExit(
            f"[stage5] Stage 3 archive was built with retrieval model(s) "
            f"{sorted(models)}, but Stage 5 is configured for {retrieval_model!r}.\n"
            "[stage5] Pass --retrieval-model to match the archive, or re-archive "
            "the intended Stage 3 run."
        )


def _sha1(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_latest() -> None:
    """Remove files from ``results/latest/`` so the Stage 5 archive stays pure —
    but only files whose byte-identical copy exists in *some* stage archive
    (stage3/stage4/stage5 final), so nothing unarchived is ever deleted."""
    latest = C.RESULTS_LATEST_DIR
    if not latest.exists():
        return
    archives = [d for d in (_stage_final_dir(s)
                            for s in ("stage3", "stage4", "stage5"))
                if d.is_dir()]
    stale = [p for p in sorted(latest.iterdir()) if p.is_file()]
    unarchived = [p.name for p in stale
                  if not any((a / p.name).exists() and _sha1(a / p.name) == _sha1(p)
                             for a in archives)]
    if unarchived:
        raise SystemExit(
            "[stage5] results/latest/ has file(s) with no identical copy in any "
            f"stage archive: {', '.join(unarchived)}\n"
            "[stage5] Archive them first so nothing is lost, e.g.:\n"
            "  python scripts/save_stage_results.py --stage stage4"
        )
    for p in stale:
        p.unlink()
    if stale:
        print(f"[stage5] cleared {len(stale)} archived file(s) from {latest}")


# --------------------------------------------------------------------------- #
# Stage 3 consistency check (bge arm must reproduce the archived baseline)
# --------------------------------------------------------------------------- #
def _as_float(row: dict, col: str):
    val = row.get(col)
    if val in ("", None):
        return None
    return float(val)


def _build_stage3_check(stage3_rows: list[dict], stage5_rows: list[dict]):
    """Match Stage 5's bge rows to the archived Stage 3 rows by chunking config.

    Returns ``(columns, rows, stats)`` where stats reports unmatched configs,
    n_chunks mismatches and the max |recall delta| (on 4-dp rounded values, the
    precision the Stage 3 CSV stores).
    """
    from rag_chunk.hybrid import config_key, config_label

    ks = sorted(C.RECALL_KS)
    s3_by_key = {config_key(r): r for r in stage3_rows}
    cols = ["method", "chunk_config",
            "stage3_n_chunks", "stage5_n_chunks", "delta_n_chunks",
            "stage3_avg_chunk_size", "stage5_avg_chunk_size"]
    for k in ks:
        cols += [f"stage3_recall@{k}", f"stage5_recall@{k}", f"delta_recall@{k}"]

    rows = []
    stats = {"unmatched": 0, "n_chunks_mismatch": 0, "max_abs_recall_delta": 0.0}
    for r5 in [r for r in stage5_rows if r["arm"] == "bge"]:
        r3 = s3_by_key.get(config_key(r5))
        out = {"method": r5["method"], "chunk_config": config_label(r5),
               "stage5_n_chunks": r5["n_chunks"],
               "stage5_avg_chunk_size": f"{r5['avg_chunk_size']:.3f}"}
        if r3 is None:
            stats["unmatched"] += 1
            rows.append(out)
            continue
        n3 = int(_as_float(r3, "n_chunks"))
        out["stage3_n_chunks"] = n3
        out["delta_n_chunks"] = r5["n_chunks"] - n3
        if out["delta_n_chunks"] != 0:
            stats["n_chunks_mismatch"] += 1
        out["stage3_avg_chunk_size"] = f"{_as_float(r3, 'avg_chunk_size'):.3f}"
        for k in ks:
            v3 = _as_float(r3, f"recall@{k}")
            v5 = round(r5[f"recall@{k}"], 4)
            delta = round(v5 - v3, 4)
            stats["max_abs_recall_delta"] = max(stats["max_abs_recall_delta"], abs(delta))
            out[f"stage3_recall@{k}"] = f"{v3:.4f}"
            out[f"stage5_recall@{k}"] = f"{v5:.4f}"
            out[f"delta_recall@{k}"] = f"{delta:.4f}"
        rows.append(out)
    return cols, rows, stats


def _write_stage3_check(stage3_rows, stage5_rows, path: pathlib.Path) -> dict:
    cols, rows, stats = _build_stage3_check(stage3_rows, stage5_rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stage5] wrote {path} ({len(rows)} rows)")

    if stats["unmatched"]:
        print(f"[stage5] WARN: {stats['unmatched']} bge row(s) had no matching "
              "Stage 3 row (was the archived Stage 3 run the full grid?)")
    if stats["n_chunks_mismatch"]:
        print(f"[stage5] WARN: {stats['n_chunks_mismatch']} config(s) produced a "
              "different chunk count than Stage 3 — chunking is NOT identical, "
              "do not trust this run.")
    d = stats["max_abs_recall_delta"]
    if stats["unmatched"] == 0 and stats["n_chunks_mismatch"] == 0 and d == 0.0:
        print("[stage5] check OK: the bge arm reproduces the archived Stage 3 "
              "baseline exactly.")
    elif d <= CHECK_TOLERANCE and stats["n_chunks_mismatch"] == 0:
        print(f"[stage5] check: max |recall delta| vs Stage 3 = {d:.4f} "
              "(within one question — acceptable, but inspect "
              f"{STAGE3_CHECK_CSV} before archiving).")
    else:
        print(f"[stage5] WARN: max |recall delta| vs Stage 3 = {d:.4f} exceeds "
              f"{CHECK_TOLERANCE} — the bge arm does not reproduce the baseline. "
              "Inspect before archiving.")
    return stats


def _write_summary(stage3_dir: pathlib.Path, out_dir: pathlib.Path,
                   stats: dict) -> pathlib.Path:
    import json

    from rag_chunk.hybrid import config_label
    from rag_chunk.rerank import arms, rerank_depths

    best_path = out_dir / C.RERANK_BEST_JSON
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    lines = [
        "# Stage 5 - cross-encoder reranking (BGE top-k + pretrained reranker)",
        "",
        f"- Boundary/chunking embedding model: `{C.BOUNDARY_EMBED_MODEL}`",
        f"- Dense retrieval embedding model: `{C.RETRIEVAL_EMBED_MODEL}`",
        f"- Query instruction: `{C.retrieval_query_instruction()}`",
        f"- Reranker: `{C.RERANKER_MODEL}` (off-the-shelf, no fine-tuning; "
        f"batch={C.RERANK_BATCH_SIZE}, max_length={C.RERANK_MAX_LENGTH})",
        f"- Candidate-pool depths: {rerank_depths()}",
        f"- Stage 3 baseline compared from: `{stage3_dir}`",
        f"- Stage 3 bge-arm check: max |recall delta| = "
        f"{stats['max_abs_recall_delta']:.4f}, "
        f"n_chunks mismatches = {stats['n_chunks_mismatch']}, "
        f"unmatched = {stats['unmatched']}",
        "",
        "Outputs in this folder:",
        f"- `{C.RERANK_SWEEP_CSV}`: one row per (arm, chunking config), incl. "
        "pool_recall@depth (the rerank ceiling) and rerank_seconds",
        f"- `{C.RERANK_BEST_JSON}`: best config per arm + overall",
        f"- `{C.RERANK_MATCHED_CSV}`: arms side by side per config with "
        "rerank-vs-bge deltas",
        f"- `{STAGE3_CHECK_CSV}`: Stage 5 bge arm vs archived Stage 3 rows",
        f"- `{C.RERANK_SCATTER_PNG}` and `{C.RERANK_COMPARISON_PNG}`: plots",
    ]
    per = best.get("per_arm", {})
    if per:
        lines += ["", "Best config per arm:"]
        for name in arms():
            cfg = per.get(name, {}).get("config")
            if cfg:
                recalls = " ".join(
                    f"R@{k}={cfg[f'recall@{k}']:.4f}" for k in sorted(C.RECALL_KS))
                lines.append(f"- {name}: `{cfg['method']}`, "
                             f"`{config_label(cfg)}` — {recalls}")
    path = out_dir / STAGE5_SUMMARY_MD
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stage5] wrote {path}")
    return path


def _snapshot_latest(out_dir: pathlib.Path) -> pathlib.Path:
    snap = C.RESULTS_RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage5_rerank"
    snap.mkdir(parents=True, exist_ok=True)
    for name in (
        C.RERANK_SWEEP_CSV,
        C.RERANK_BEST_JSON,
        C.RERANK_MATCHED_CSV,
        STAGE3_CHECK_CSV,
        C.RERANK_SCATTER_PNG,
        C.RERANK_COMPARISON_PNG,
        STAGE5_SUMMARY_MD,
    ):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, snap / name)
    print(f"[stage5] snapshot saved -> {snap}")
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 5: score the fixed/BiLSTM/Transformer sweep with the "
                    "BGE baseline plus cross-encoder rerank arms.")
    ap.add_argument("--quick", action="store_true",
                    help="use the smaller QUICK_* grids for a fast sanity sweep")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="dense retrieval model; must match the Stage 3 archive "
                         "(default: BAAI/bge-base-en-v1.5)")
    ap.add_argument("--retrieval-dim", type=int, default=None,
                    help="optional retrieval embedding dimension override")
    ap.add_argument("--reranker-model", default=None,
                    help="pretrained cross-encoder "
                         f"(default: config RERANKER_MODEL={C.RERANKER_MODEL})")
    ap.add_argument("--depths", default=None,
                    help="comma-separated candidate-pool depths "
                         f"(default: config RERANK_DEPTHS={C.RERANK_DEPTHS})")
    ap.add_argument("--rerank-batch", type=int, default=None,
                    help="cross-encoder batch size "
                         f"(default: config RERANK_BATCH_SIZE={C.RERANK_BATCH_SIZE})")
    ap.add_argument("--save-run", action="store_true",
                    help="also snapshot Stage 5 artifacts to "
                         "results/runs/<timestamp>_stage5_rerank/")
    args = ap.parse_args()

    stage3_dir = _ensure_stage3_archived()
    stage3_rows = _read_csv(stage3_dir / C.SWEEP_RESULTS_CSV)
    _ensure_stage3_is_bge(stage3_rows, args.retrieval_model)

    overrides = {
        "RETRIEVAL_EMBED_MODEL": args.retrieval_model,
        "RETRIEVAL_EMBED_DIM": args.retrieval_dim,
        "RETRIEVAL_EMBED_NORMALIZE": True,
    }
    if args.reranker_model is not None:
        overrides["RERANKER_MODEL"] = args.reranker_model
    if args.depths is not None:
        overrides["RERANK_DEPTHS"] = tuple(
            int(d) for d in args.depths.split(",") if d.strip())
    if args.rerank_batch is not None:
        overrides["RERANK_BATCH_SIZE"] = args.rerank_batch
    C.apply(**overrides)
    C.ensure_dirs()

    # Heavy imports + model weights load BEFORE latest/ is cleared, so an
    # environment, weights, or reranker-download failure leaves the archived
    # files in place.
    from rag_chunk import rerank, training

    print("[stage5] boundary/chunking embeddings stay on:", C.BOUNDARY_EMBED_MODEL)
    print("[stage5] dense retrieval embeddings:", C.RETRIEVAL_EMBED_MODEL)
    print(f"[stage5] reranker: {C.RERANKER_MODEL} (off-the-shelf) "
          f"depths={rerank.rerank_depths()} batch={C.RERANK_BATCH_SIZE}")
    print("[stage5] Stage 3 archive:", stage3_dir)

    bilstm = training.load_model("bilstm")
    transformer = training.load_model("transformer")
    rerank.load_reranker()

    _clean_latest()
    rows = rerank.run_rerank_sweep(
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

    print("[stage5] complete. Verify the check above, then archive:")
    print("  python scripts/save_stage_results.py --stage stage5")


if __name__ == "__main__":
    main()
