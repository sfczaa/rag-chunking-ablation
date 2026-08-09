"""Stage 3 - BGE retrieval embedding ablation.

Runs the existing fixed / BiLSTM / Transformer chunking sweep with a separate
retrieval embedding model, defaulting to ``BAAI/bge-base-en-v1.5``. Boundary and
chunking embeddings stay on the existing MiniLM path and existing Stage 2
weights are loaded as-is.

Before writing ``artifacts/results/latest/``, this script requires Stage 2 to be
archived under ``artifacts/results/stage2/final/`` so the MiniLM baseline is not
lost when ``latest`` is overwritten.

Usage:
    python scripts/9_sweep_bge_retrieval.py
    python scripts/9_sweep_bge_retrieval.py --quick
    python scripts/9_sweep_bge_retrieval.py --retrieval-model BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402


REQUIRED_STAGE2_FILES = (
    C.SWEEP_RESULTS_CSV,
    C.BEST_CONFIG_JSON,
    C.FAIR_TABLE_CSV,
    C.RECALL_PLOT_PNG,
    C.MODEL_PLOT_PNG,
)
STAGE3_SUMMARY_MD = "stage3_bge_retrieval_summary.md"
# Same matched MiniLM-vs-BGE table as fair_comparison_table.csv, written under an
# unambiguous name because fair_comparison_table.csv is a filename reused across
# stages (bilstm-vs-fixed in Stage 1/2, Stage 2-vs-Stage 3 here).
STAGE3_MATCHED_CSV = "stage2_vs_stage3_matched.csv"


def _stage2_final_dir() -> pathlib.Path:
    return C.RESULTS_DIR / "stage2" / "final"


def _ensure_stage2_archived() -> pathlib.Path:
    stage2_dir = _stage2_final_dir()
    missing = [name for name in REQUIRED_STAGE2_FILES if not (stage2_dir / name).exists()]
    if missing:
        raise SystemExit(
            "[stage3] Stage 2 latest results are not archived yet.\n"
            f"[stage3] Missing from {stage2_dir}: {', '.join(missing)}\n\n"
            "Run this first:\n"
            "  python scripts/save_stage_results.py --stage stage2"
        )
    return stage2_dir


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _cell(row: dict, col: str):
    val = row.get(col)
    if val in ("", None):
        return None
    return val


def _as_int(row: dict, col: str):
    val = _cell(row, col)
    if val is None:
        return None
    return int(float(val))


def _as_float(row: dict, col: str):
    val = _cell(row, col)
    if val is None:
        return None
    return float(val)


def _row_key(row: dict) -> tuple:
    method = str(row["method"]).strip()
    if method == "fixed":
        return ("fixed", _as_int(row, "fixed_size"), _as_int(row, "fixed_overlap"))
    return (
        method,
        _cell(row, "semantic_policy") or "target",
        _as_int(row, "semantic_target_size"),
        _as_int(row, "semantic_min_size"),
        _as_int(row, "semantic_max_size"),
        _as_int(row, "semantic_overlap"),
    )


def _config_label(row: dict) -> str:
    method = str(row["method"]).strip()
    if method == "fixed":
        return f"fixed_size={_as_int(row, 'fixed_size')},overlap={_as_int(row, 'fixed_overlap')}"
    return (
        f"target={_as_int(row, 'semantic_target_size')},"
        f"min={_as_int(row, 'semantic_min_size')},"
        f"max={_as_int(row, 'semantic_max_size')},"
        f"overlap={_as_int(row, 'semantic_overlap')}"
    )


def _model_label(row: dict, preferred: str, fallback: str) -> str:
    return str(_cell(row, preferred) or _cell(row, fallback) or "")


def _fmt(val, digits: int = 4):
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return val


def _build_stage2_vs_stage3_table(
    stage2_rows: list[dict],
    stage3_rows: list[dict],
) -> tuple[list[str], list[dict], int]:
    """Build the MiniLM-vs-BGE table at matching chunking configs.

    Returns ``(columns, rows, n_unmatched)`` so the identical table can be
    written to more than one file without recomputing it.
    """
    stage2_by_key = {_row_key(row): row for row in stage2_rows}
    ks = sorted(C.RECALL_KS)
    cols = [
        "method", "chunk_config",
        "stage2_retrieval_embedding_model", "stage3_retrieval_embedding_model",
        "boundary_embedding_model",
        "stage2_avg_chunk_size", "stage3_avg_chunk_size", "delta_avg_chunk_size",
        "stage2_n_chunks", "stage3_n_chunks", "delta_n_chunks",
    ]
    for k in ks:
        cols += [f"stage2_recall@{k}", f"stage3_recall@{k}", f"delta_recall@{k}"]

    rows = []
    unmatched = 0
    for row3 in stage3_rows:
        row2 = stage2_by_key.get(_row_key(row3), {})
        if not row2:
            unmatched += 1
        avg2 = _as_float(row2, "avg_chunk_size") if row2 else None
        avg3 = _as_float(row3, "avg_chunk_size")
        n2 = _as_int(row2, "n_chunks") if row2 else None
        n3 = _as_int(row3, "n_chunks")
        out = {
            "method": row3["method"],
            "chunk_config": _config_label(row3),
            "stage2_retrieval_embedding_model": _model_label(
                row2, "retrieval_embedding_model", "embedding_model") if row2 else "",
            "stage3_retrieval_embedding_model": _model_label(
                row3, "retrieval_embedding_model", "embedding_model"),
            "boundary_embedding_model": _model_label(
                row3, "boundary_embedding_model", "embedding_model"),
            "stage2_avg_chunk_size": _fmt(avg2, 3),
            "stage3_avg_chunk_size": _fmt(avg3, 3),
            "delta_avg_chunk_size": _fmt(avg3 - avg2, 3) if avg2 is not None else "",
            "stage2_n_chunks": n2 if n2 is not None else "",
            "stage3_n_chunks": n3 if n3 is not None else "",
            "delta_n_chunks": n3 - n2 if n2 is not None else "",
        }
        for k in ks:
            r2 = _as_float(row2, f"recall@{k}") if row2 else None
            r3 = _as_float(row3, f"recall@{k}")
            out[f"stage2_recall@{k}"] = _fmt(r2)
            out[f"stage3_recall@{k}"] = _fmt(r3)
            out[f"delta_recall@{k}"] = _fmt(r3 - r2) if r2 is not None else ""
        rows.append(out)
    return cols, rows, unmatched


def _write_stage2_vs_stage3_tables(
    stage2_rows: list[dict],
    stage3_rows: list[dict],
    paths: list[pathlib.Path],
) -> None:
    """Write the MiniLM-vs-BGE matched table to every path in ``paths``.

    The identical content is written under ``fair_comparison_table.csv`` (the
    name README/notebook already reference) and ``stage2_vs_stage3_matched.csv``
    (an unambiguous name), so a Stage 3 reader is not tripped up by the reused
    fair-table filename.
    """
    cols, rows, unmatched = _build_stage2_vs_stage3_table(stage2_rows, stage3_rows)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[stage3] wrote {path} ({len(rows)} rows)")
    if unmatched:
        print(f"[stage3] WARN: {unmatched} BGE row(s) had no matching Stage 2 row")


def _write_summary(stage2_dir: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    best_path = out_dir / C.BEST_CONFIG_JSON
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    selected = best.get("config", {})
    lines = [
        "# Stage 3 - BGE retrieval embedding ablation",
        "",
        f"- Boundary/chunking embedding model: `{C.BOUNDARY_EMBED_MODEL}`",
        f"- Retrieval embedding model: `{C.RETRIEVAL_EMBED_MODEL}`",
        f"- Retrieval embeddings normalized: `{C.RETRIEVAL_EMBED_NORMALIZE}`",
        f"- Query instruction: `{C.retrieval_query_instruction()}`",
        f"- Stage 2 archive compared from: `{stage2_dir}`",
        "",
        "Outputs in this folder:",
        f"- `{C.SWEEP_RESULTS_CSV}`: Stage 3 BGE sweep rows",
        f"- `{C.BEST_CONFIG_JSON}`: best Stage 3 BGE config",
        f"- `{C.FAIR_TABLE_CSV}`: matching-config Stage 2 MiniLM vs Stage 3 BGE table",
        f"- `{STAGE3_MATCHED_CSV}`: the same matched table under an unambiguous name",
        f"- `{C.RECALL_PLOT_PNG}` and `{C.MODEL_PLOT_PNG}`: Stage 3 BGE plots",
    ]
    if selected:
        lines += [
            "",
            "Best Stage 3 config:",
            f"- method: `{selected.get('method')}`",
            f"- chunk config: `{_config_label(selected)}`",
        ]
    path = out_dir / STAGE3_SUMMARY_MD
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stage3] wrote {path}")
    return path


def _snapshot_latest(out_dir: pathlib.Path) -> pathlib.Path:
    snap = C.RESULTS_RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_stage3_bge"
    snap.mkdir(parents=True, exist_ok=True)
    for name in (
        C.SWEEP_RESULTS_CSV,
        C.BEST_CONFIG_JSON,
        C.FAIR_TABLE_CSV,
        STAGE3_MATCHED_CSV,
        C.RECALL_PLOT_PNG,
        C.MODEL_PLOT_PNG,
        STAGE3_SUMMARY_MD,
    ):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, snap / name)
    print(f"[stage3] snapshot saved -> {snap}")
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 3: run fixed/BiLSTM/Transformer sweep with BGE retrieval embeddings.")
    ap.add_argument("--quick", action="store_true",
                    help="use the smaller QUICK_* grids for a fast sanity sweep")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="SentenceTransformer retrieval model (default: BAAI/bge-base-en-v1.5)")
    ap.add_argument("--retrieval-dim", type=int, default=None,
                    help="optional retrieval embedding dimension override")
    ap.add_argument("--query-instruction", default=None,
                    help="override retrieval query instruction prefix")
    ap.add_argument("--no-query-instruction", action="store_true",
                    help="disable retrieval query instruction prefix")
    ap.add_argument("--save-run", action="store_true",
                    help="also snapshot Stage 3 artifacts to results/runs/<timestamp>_stage3_bge/")
    ap.add_argument("--save-sweep-index", action="store_true",
                    help="persist each per-config BGE FAISS index under a BGE-specific sweep path")
    args = ap.parse_args()

    stage2_dir = _ensure_stage2_archived()

    overrides = {
        "RETRIEVAL_EMBED_MODEL": args.retrieval_model,
        "RETRIEVAL_EMBED_DIM": args.retrieval_dim,
        "RETRIEVAL_EMBED_NORMALIZE": True,
    }
    if args.no_query_instruction:
        overrides["RETRIEVAL_QUERY_INSTRUCTION"] = ""
    elif args.query_instruction is not None:
        overrides["RETRIEVAL_QUERY_INSTRUCTION"] = args.query_instruction
    C.apply(**overrides)
    C.ensure_dirs()

    # Heavy imports happen only after arg parsing and archive validation.
    from rag_chunk import sweep, training

    print("[stage3] boundary/chunking embeddings stay on:", C.BOUNDARY_EMBED_MODEL)
    print("[stage3] retrieval embeddings:", C.RETRIEVAL_EMBED_MODEL)
    print("[stage3] Stage 2 archive:", stage2_dir)

    bilstm = training.load_model("bilstm")
    transformer = training.load_model("transformer")
    rows = sweep.run_sweep(
        bilstm,
        transformer_model=transformer,
        quick=args.quick,
        save_run=False,
        save_sweep_index=args.save_sweep_index,
        out_dir=C.RESULTS_LATEST_DIR,
    )

    stage2_rows = _read_csv(stage2_dir / C.SWEEP_RESULTS_CSV)
    _write_stage2_vs_stage3_tables(
        stage2_rows,
        rows,
        [C.RESULTS_LATEST_DIR / C.FAIR_TABLE_CSV,
         C.RESULTS_LATEST_DIR / STAGE3_MATCHED_CSV],
    )
    _write_summary(stage2_dir, C.RESULTS_LATEST_DIR)
    if args.save_run:
        _snapshot_latest(C.RESULTS_LATEST_DIR)

    print("[stage3] complete. Archive when satisfied:")
    print("  python scripts/save_stage_results.py --stage stage3")


if __name__ == "__main__":
    main()
