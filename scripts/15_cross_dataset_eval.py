"""Stage 7 - cross-dataset robustness check (TriviaQA rc.wikipedia, bge-only).

Re-runs the Stage 3 protocol on a different QA dataset to test whether the
headline conclusion — chunk size dominates Recall, chunking method ties —
survives a change of dataset. Nothing else changes: same chunking grids (30
configs), same boundary models/weights, same BGE dense retriever and query
instruction, same doc-constrained Recall@k. bge arm only: no BM25/RRF, no
reranking, no fine-tuning.

Two modes:

    --check   NQ sanity mode. Runs the 30-config bge-only sweep on the same
              cached 200-doc NQ corpus as Stage 3 and compares every row
              against the archived stage3/final rows — all deltas must be
              0.0000. This proves the Stage 7 code path (including the
              multi-gold metric extension) IS the Stage 3 pipeline. Run this
              BEFORE the TriviaQA run.
    (default) the TriviaQA rc.wikipedia eval (STAGE7_N_QUESTIONS kept
              questions; loader stats and actual counts are reported in every
              output). The eval set differs from NQ, so no exact-delta check
              is possible; stage7_direction_check.csv re-tests the direction
              claims with explicit rules instead.

Both modes are restart/resume-safe: every finished config is appended to a
JSONL checkpoint in results/latest/ (--fresh discards it). The cleanup of
results/latest/ never deletes a file without a byte-identical archived copy
and leaves stage7_* working files alone.

Usage:
    python scripts/15_cross_dataset_eval.py --check
    python scripts/15_cross_dataset_eval.py
    python scripts/15_cross_dataset_eval.py --n-questions 300
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402


REQUIRED_STAGE3_FILES = (C.SWEEP_RESULTS_CSV, C.BEST_CONFIG_JSON)
STAGE7_CHECK_CSV = "stage7_check_vs_stage3.csv"
CHECKPOINT_FILES = {
    "check": "stage7_checkpoint_check.jsonl",
    "trivia": "stage7_checkpoint_trivia.jsonl",
}
# In check mode every config re-runs the exact Stage 3 pipeline on the same
# cached corpus, so recall deltas should be 0.0000; anything above one
# question (~0.005 at n=203) means the setup differs.
CHECK_TOLERANCE = 0.005
# The archived Stage 3 eval set (see PROGRESS.md); its CSV does not store the
# counts itself.
NQ_SMALL_DOCS = 200
NQ_SMALL_QUESTIONS = 203

# Same visual language as the Stage 1-6 figures.
METHOD_COLORS = {"fixed": "#888888", "bilstm": "#1f77b4", "transformer": "#d62728"}
OVERLAP_MARKERS = {0: "o", 1: "^"}


def _stage_final_dir(stage: str) -> pathlib.Path:
    return C.RESULTS_DIR / stage / "final"


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _ensure_stage3_archived() -> pathlib.Path:
    stage3_dir = _stage_final_dir("stage3")
    missing = [name for name in REQUIRED_STAGE3_FILES
               if not (stage3_dir / name).exists()]
    if missing:
        raise SystemExit(
            "[stage7] the Stage 3 archive is incomplete.\n"
            f"[stage7] Missing from {stage3_dir}: {', '.join(missing)}\n\n"
            "Stage 7 checks its rows against the archived Stage 3 run and "
            "will not start without it.")
    return stage3_dir


def _ensure_stage3_models(stage3_rows: list[dict], retrieval_model: str) -> None:
    retr = {row.get("retrieval_embedding_model", "") for row in stage3_rows}
    if retr != {retrieval_model}:
        raise SystemExit(
            f"[stage7] Stage 3 archive uses retrieval model(s) {sorted(retr)}, "
            f"but Stage 7 is configured for {retrieval_model!r}.")


def _sha1(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_latest() -> None:
    """Remove archived files from ``results/latest/`` so the Stage 7 archive
    stays pure. Only files with a byte-identical copy in some stage archive
    are deleted; ``stage7_*`` working files (checkpoints + outputs of an
    in-progress Stage 7) are always left alone."""
    latest = C.RESULTS_LATEST_DIR
    if not latest.exists():
        return
    archives = [d for d in (_stage_final_dir(s) for s in
                            ("stage2", "stage3", "stage4", "stage5", "stage6",
                             "stage7"))
                if d.is_dir()]
    stale = [p for p in sorted(latest.iterdir())
             if p.is_file() and not p.name.startswith("stage7_")]
    unarchived = [p.name for p in stale
                  if not any((a / p.name).exists() and _sha1(a / p.name) == _sha1(p)
                             for a in archives)]
    if unarchived:
        raise SystemExit(
            "[stage7] results/latest/ has file(s) with no identical copy in "
            f"any stage archive: {', '.join(unarchived)}\n"
            "[stage7] Archive them first so nothing is lost, e.g.:\n"
            "  python scripts/save_stage_results.py --stage stage6"
        )
    for p in stale:
        p.unlink()
    if stale:
        print(f"[stage7] cleared {len(stale)} archived file(s) from {latest}")


# --------------------------------------------------------------------------- #
# The 30-config bge-only grid (checkpointed)
# --------------------------------------------------------------------------- #
def _checkpoint_meta(docs, questions, mode: str, dataset: str) -> dict:
    """Fingerprint of the run a checkpoint belongs to; a resumed run must
    match exactly or it aborts instead of mixing rows."""
    return {
        "stage": "stage7",
        "mode": mode,
        "dataset": dataset,
        "n_docs": len(docs),
        "n_questions": len(questions),
        "retrieval_model": C.RETRIEVAL_EMBED_MODEL,
        "boundary_model": C.BOUNDARY_EMBED_MODEL,
    }


def _run_grid(docs, questions, *, bilstm, transformer_model, mode: str,
              dataset: str, checkpoint_path) -> list[dict]:
    """Score the full 30-config grid with the bge arm only. Resume-safe via
    ``checkpoint_path``; returns rows in the canonical sweep order."""
    from rag_chunk import sweep
    from rag_chunk.hybrid import config_key
    from rag_chunk.large_eval import (_key_for, append_checkpoint, config_plan,
                                      load_checkpoint)

    plan = config_plan()
    meta = _checkpoint_meta(docs, questions, mode, dataset)
    done_rows = load_checkpoint(checkpoint_path, meta)
    if not done_rows and not pathlib.Path(checkpoint_path).exists():
        append_checkpoint(checkpoint_path, {"meta": meta})
    done_keys = {config_key(r) for r in done_rows}

    remaining = [cfg for cfg in plan if _key_for(*cfg) not in done_keys]
    if done_rows:
        print(f"[stage7] resuming: {len(plan) - len(remaining)}/{len(plan)} "
              "configs already in the checkpoint")

    needed = {m for m, _, _ in remaining if m in ("bilstm", "transformer")}
    models = {}
    for mtype, model in (("bilstm", bilstm), ("transformer", transformer_model)):
        if mtype in needed:
            if model is None:
                raise ValueError(f"{mtype} configs remain but no {mtype} model given")
            models[mtype] = model
    probs_by_type = sweep._precompute_boundary_probs(docs, models) if models else {}

    rows = list(done_rows)
    done_ct = len(plan) - len(remaining)
    for method, size, overlap in plan:
        if _key_for(method, size, overlap) in done_keys:
            continue
        done_ct += 1
        if method == "fixed":
            print(f"[stage7] ({done_ct}/{len(plan)}) fixed        "
                  f"size={size} overlap={overlap}")
            row = sweep._eval_config("fixed", docs, questions,
                                     fixed_size=size, fixed_overlap=overlap)
        else:
            mn, mx = sweep._semantic_window(size)
            print(f"[stage7] ({done_ct}/{len(plan)}) {method:<12} target={size} "
                  f"(min={mn},max={mx}) overlap={overlap}")
            row = sweep._eval_config(
                method, docs, questions,
                boundary_probs_by_id=probs_by_type[method],
                semantic_policy="target", semantic_target_size=size,
                semantic_min_size=mn, semantic_max_size=mx,
                semantic_overlap=overlap)
        row["arm"] = "bge"
        row["n_docs"] = len(docs)
        row["n_questions"] = len(questions)
        row["dataset"] = dataset
        append_checkpoint(checkpoint_path, {"rows": [row]})
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Check mode: every row must reproduce the archived Stage 3 run exactly
# --------------------------------------------------------------------------- #
def _as_float(row: dict, col: str):
    val = row.get(col)
    if val in ("", None):
        return None
    return float(val)


def _write_stage3_check(stage3_rows, stage7_rows, path: pathlib.Path) -> None:
    from rag_chunk.hybrid import config_key, config_label

    ks = sorted(C.RECALL_KS)
    s3 = {config_key(r): r for r in stage3_rows}
    cols = ["method", "chunk_config",
            "stage3_n_chunks", "stage7_n_chunks", "delta_n_chunks",
            "stage3_avg_chunk_size", "stage7_avg_chunk_size"]
    for k in ks:
        cols += [f"stage3_recall@{k}", f"stage7_recall@{k}", f"delta_recall@{k}"]

    rows = []
    stats = {"unmatched": 0, "n_chunks_mismatch": 0, "max_abs_recall_delta": 0.0}
    for r7 in stage7_rows:
        r3 = s3.get(config_key(r7))
        out = {"method": r7["method"], "chunk_config": config_label(r7),
               "stage7_n_chunks": r7["n_chunks"],
               "stage7_avg_chunk_size": f"{r7['avg_chunk_size']:.3f}"}
        if r3 is None:
            stats["unmatched"] += 1
            rows.append(out)
            continue
        n3 = int(_as_float(r3, "n_chunks"))
        out["stage3_n_chunks"] = n3
        out["delta_n_chunks"] = r7["n_chunks"] - n3
        if out["delta_n_chunks"] != 0:
            stats["n_chunks_mismatch"] += 1
        out["stage3_avg_chunk_size"] = f"{_as_float(r3, 'avg_chunk_size'):.3f}"
        for k in ks:
            v3 = _as_float(r3, f"recall@{k}")
            v7 = round(r7[f"recall@{k}"], 4)
            delta = round(v7 - v3, 4)
            stats["max_abs_recall_delta"] = max(stats["max_abs_recall_delta"],
                                                abs(delta))
            out[f"stage3_recall@{k}"] = f"{v3:.4f}"
            out[f"stage7_recall@{k}"] = f"{v7:.4f}"
            out[f"delta_recall@{k}"] = f"{delta:.4f}"
        rows.append(out)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stage7] wrote {path} ({len(rows)} rows)")

    if stats["unmatched"]:
        print(f"[stage7] WARN: {stats['unmatched']} row(s) had no matching "
              "Stage 3 row (was the archived Stage 3 run the full grid?)")
    if stats["n_chunks_mismatch"]:
        print(f"[stage7] WARN: {stats['n_chunks_mismatch']} config(s) produced "
              "a different chunk count than Stage 3 — chunking is NOT "
              "identical, do not trust this run.")
    d = stats["max_abs_recall_delta"]
    if stats["unmatched"] == 0 and stats["n_chunks_mismatch"] == 0 and d == 0.0:
        print("[stage7] check OK: Stage 7 reproduces the archived Stage 3 rows "
              "exactly — proceed to the TriviaQA run.")
    elif d <= CHECK_TOLERANCE and stats["n_chunks_mismatch"] == 0:
        print(f"[stage7] check: max |recall delta| vs Stage 3 = {d:.4f} "
              "(within one question — acceptable, but inspect "
              f"{STAGE7_CHECK_CSV} before the TriviaQA run).")
    else:
        print(f"[stage7] WARN: max |recall delta| vs Stage 3 = {d:.4f} exceeds "
              f"{CHECK_TOLERANCE} — Stage 7 does not reproduce the baseline. "
              "Fix this before the TriviaQA run.")


# --------------------------------------------------------------------------- #
# Matched-methods summary + direction checks (TriviaQA mode)
# --------------------------------------------------------------------------- #
def _matched_summary(rows: list[dict]) -> list[dict]:
    """Per (nominal size, overlap) cell: the three methods' R@5 side by side
    with the method spread — claims 1b/2 at a glance."""
    from rag_chunk.large_eval import _nominal_size

    topk = max(C.RECALL_KS)
    cells: dict[tuple, dict] = {}
    for r in rows:
        size = _nominal_size(r)
        ov = (r["fixed_overlap"] if r["method"] == "fixed"
              else r["semantic_overlap"])
        cell = cells.setdefault((size, ov), {"nominal_size": size, "overlap": ov})
        cell[f"recall{topk}_{r['method']}"] = r[f"recall@{topk}"]
        cell[f"avg_size_{r['method']}"] = r["avg_chunk_size"]
    out = []
    for key in sorted(cells):
        cell = cells[key]
        vals = [cell.get(f"recall{topk}_{m}") for m in METHOD_COLORS]
        vals = [v for v in vals if v is not None]
        cell[f"method_spread@{topk}"] = (
            (max(vals) - min(vals)) if len(vals) >= 2 else None)
        out.append(cell)
    return out


def _direction_checks(rows: list[dict], n_questions: int) -> list[dict]:
    """Re-test the size-vs-method direction claims on the TriviaQA rows.

    Rules are explicit (see docs/stage7_cross_dataset.md); SE comes from the
    mean bge R@5 at the actual question count. A claim failing to replicate
    is a finding to interpret, not an error to fix."""
    from rag_chunk.large_eval import _nominal_size, _pearson
    from rag_chunk.sweep import _rank_key

    topk = max(C.RECALL_KS)
    pbar = sum(r[f"recall@{topk}"] for r in rows) / len(rows)
    se = (pbar * (1 - pbar) / n_questions) ** 0.5

    checks: list[dict] = []

    def add(claim, metric, value, rule, reference, replicates):
        checks.append({
            "claim": claim, "metric": metric,
            "value": None if value is None else round(value, 4),
            "rule": rule, "reference_nq": reference, "replicates": replicates,
        })

    # 1a. size > method: recall tracks chunk size.
    r = _pearson([r["avg_chunk_size"] for r in rows],
                 [r[f"recall@{topk}"] for r in rows]) if len(rows) >= 2 else None
    add("1. chunk size matters more than chunking method",
        f"pearson_r(avg_chunk_size, recall@{topk}) over the 30 bge configs",
        r, "r >= 0.5",
        "NQ: r ~ 0.77 (n=203), 0.95 (n=1032)",
        "n/a" if r is None else ("yes" if r >= 0.5 else "no"))

    # 1b. ... and the size effect exceeds the method spread.
    by_size: dict[int, list[float]] = {}
    cells: dict[tuple, dict[str, float]] = {}
    for row in rows:
        size = _nominal_size(row)
        ov = (row["fixed_overlap"] if row["method"] == "fixed"
              else row["semantic_overlap"])
        by_size.setdefault(size, []).append(row[f"recall@{topk}"])
        cells.setdefault((size, ov), {})[row["method"]] = row[f"recall@{topk}"]
    size_effect = method_spread = None
    if by_size:
        lo, hi = min(by_size), max(by_size)
        if lo != hi:
            size_effect = (sum(by_size[hi]) / len(by_size[hi])
                           - sum(by_size[lo]) / len(by_size[lo]))
        spreads = [max(v.values()) - min(v.values())
                   for v in cells.values() if len(v) >= 2]
        method_spread = sum(spreads) / len(spreads) if spreads else None
    ok = (size_effect is not None and method_spread is not None
          and size_effect > method_spread)
    add("1. chunk size matters more than chunking method",
        f"recall@{topk} size effect (largest vs smallest size) minus mean "
        "method spread at matched (size, overlap)",
        None if size_effect is None or method_spread is None
        else size_effect - method_spread,
        "size effect > method spread",
        "NQ n=203: size ~ +0.07 vs method <= 0.014; n=1032: 0.052 > spread",
        "yes" if ok else "no")

    # 2. methods tie at the (large-chunk) matched cells.
    top_size = max(by_size) if by_size else None
    spreads15 = [max(v.values()) - min(v.values())
                 for (size, _), v in cells.items()
                 if size == top_size and len(v) >= 2]
    max_spread = max(spreads15, default=None)
    ok = max_spread is not None and max_spread < 2 * se
    add("2. methods tie at matched (large) chunk size",
        f"max method spread of recall@{topk} over the size-{top_size} cells",
        max_spread,
        f"max spread < 2 SE ({2 * se:.4f})",
        "NQ stage3: spread 0.020 at size 15/ov0 (~1 SE = 0.019)",
        "n/a" if max_spread is None else ("yes" if ok else "no"))

    # 3. the best config still sits at a large chunk size.
    best = max(rows, key=_rank_key)
    best_size = _nominal_size(best)
    ok = best_size in (12, 15)
    add("3. best chunk size is large",
        "nominal size of the best-recall@5 config (project ranking)",
        float(best_size),
        "best nominal size in {12, 15}",
        "NQ: best size 15 at n=203 and at n=1032",
        "yes" if ok else "no")

    return checks


# --------------------------------------------------------------------------- #
# Writers + figure (TriviaQA mode)
# --------------------------------------------------------------------------- #
def _results_columns() -> list[str]:
    from rag_chunk.sweep import _sweep_columns

    return ["arm"] + _sweep_columns() + ["n_docs", "n_questions", "dataset"]


def _write_results_csv(rows: list[dict], path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _results_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[stage7] wrote {path}  ({len(rows)} rows)")


def _write_matched_csv(table: list[dict], path) -> None:
    from rag_chunk.sweep import _fmt

    topk = max(C.RECALL_KS)
    cols = (["nominal_size", "overlap"]
            + [f"recall{topk}_{m}" for m in METHOD_COLORS]
            + [f"method_spread@{topk}"]
            + [f"avg_size_{m}" for m in METHOD_COLORS])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[stage7] wrote {path}  ({len(table)} rows)")


def _write_direction_csv(checks: list[dict], path) -> None:
    cols = ["claim", "metric", "value", "rule", "reference_nq", "replicates"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in checks:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    print(f"[stage7] wrote {path}  ({len(checks)} rows)")


def _overlap_of(row: dict) -> int:
    field = "fixed_overlap" if row["method"] == "fixed" else "semantic_overlap"
    return int(float(row[field]))


def _scatter_panel(ax, rows: list[dict], topk: int, title: str,
                   n_questions: int) -> None:
    import numpy as np

    xs = np.array([float(r["avg_chunk_size"]) for r in rows])
    ys = np.array([float(r[f"recall@{topk}"]) for r in rows])
    for r in rows:
        ax.scatter(float(r["avg_chunk_size"]), float(r[f"recall@{topk}"]),
                   c=METHOD_COLORS.get(r["method"], "#333333"),
                   marker=OVERLAP_MARKERS.get(_overlap_of(r), "o"),
                   s=80, edgecolor="white", linewidth=0.6, zorder=3)
    slope, intercept = np.polyfit(xs, ys, 1)
    pear = float(np.corrcoef(xs, ys)[0, 1])
    xline = np.array([xs.min(), xs.max()])
    ax.plot(xline, slope * xline + intercept,
            color="black", linestyle="--", alpha=0.6, zorder=2)
    pbar = float(ys.mean())
    se = (pbar * (1 - pbar) / n_questions) ** 0.5
    ax.text(0.02, 0.98,
            f"Pearson r(size, R@{topk}) = {pear:.2f}\n1 SE = {se:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_title(title)
    ax.set_xlabel("Average chunk size (sentences)")
    ax.grid(True, alpha=0.3)


def _plot_scatter(nq_rows: list[dict], trivia_rows: list[dict],
                  n_docs: int, n_questions: int, path: pathlib.Path) -> None:
    """Claim 1 in one glance: NQ (Stage 3 archive) and TriviaQA side by side —
    does the upward size trend transfer, with the method colours intermixed?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    topk = max(C.RECALL_KS)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    _scatter_panel(ax_l, nq_rows, topk,
                   f"NQ — {NQ_SMALL_DOCS} docs / {NQ_SMALL_QUESTIONS} questions "
                   "(Stage 3 archive)", NQ_SMALL_QUESTIONS)
    _scatter_panel(ax_r, trivia_rows, topk,
                   f"TriviaQA rc.wikipedia — {n_docs} docs / "
                   f"{n_questions} questions", n_questions)
    ax_l.set_ylabel(f"Doc-constrained Recall@{topk}")

    handles = (
        [Line2D([0], [0], marker="s", linestyle="", color=METHOD_COLORS[m],
                label=m) for m in METHOD_COLORS]
        + [Line2D([0], [0], marker=mk, linestyle="", color="#555555",
                  label=f"overlap={ov}") for ov, mk in OVERLAP_MARKERS.items()]
        + [Line2D([0], [0], color="black", linestyle="--", alpha=0.6,
                  label="size trend")]
    )
    ax_l.legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle("BGE-only recall vs chunk size: does the size effect "
                 "transfer across datasets?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[stage7] wrote {path}")


def _write_summary(out_dir: pathlib.Path, rows, matched, checks,
                   loader_meta: dict | None, n_docs: int, n_questions: int,
                   n_requested: int) -> None:
    from rag_chunk.hybrid import config_label
    from rag_chunk.sweep import _rank_key

    topk = max(C.RECALL_KS)
    ks = sorted(C.RECALL_KS)
    lines = [
        "# Stage 7 - cross-dataset robustness check (TriviaQA rc.wikipedia)",
        "",
        f"- Dataset: `{C.STAGE7_DATASET}` / `{C.STAGE7_DATASET_CONFIG}` / "
        f"`{C.STAGE7_SPLIT}` — full Wikipedia entity pages bundled in the "
        "dataset (no fetching).",
        f"- Eval set: **{n_docs} docs / {n_questions} questions** "
        f"(requested {n_requested} kept questions).",
        "- Gold documents: the question's entity pages whose sentence-joined "
        "text contains the kept answer string (`answer.value` first, then "
        "aliases). **Distant supervision** — known to contain the answer, "
        "not human-verified to support it (weaker than NQ's annotated gold).",
        f"- Boundary/chunking embedding model: `{C.BOUNDARY_EMBED_MODEL}` "
        "(Stage 2 weights, unchanged)",
        f"- Dense retrieval embedding model: `{C.RETRIEVAL_EMBED_MODEL}`",
        "- Arm: **bge only** (no BM25/RRF, no reranking, no fine-tuning)",
        "",
    ]
    if loader_meta:
        stats = loader_meta.get("stats", {})
        lines += [
            "## Loader statistics",
            "",
            f"- Scanned {stats.get('scanned')} validation rows to keep "
            f"{stats.get('kept')} questions.",
            f"- Dropped: {stats.get('dropped_no_answer_in_pages')} with no "
            "answer candidate in any entity page, "
            f"{stats.get('dropped_no_usable_page')} with no usable page; "
            f"{stats.get('pages_too_short')} pages under 2 sentences.",
            f"- Multi-gold questions (answer in 2 pages): "
            f"{stats.get('multi_gold_questions')}.",
            f"- Document length (sentences): median "
            f"{loader_meta.get('median_doc_sentences')}, mean "
            f"{loader_meta.get('mean_doc_sentences')}, range "
            f"{loader_meta.get('min_doc_sentences')}-"
            f"{loader_meta.get('max_doc_sentences')} (comparability guard: "
            f"median >= {2 * max(C.FIXED_SIZE_GRID)} passed).",
            "",
        ]
    best = max(rows, key=_rank_key)
    recalls = " ".join(f"R@{k}={best[f'recall@{k}']:.4f}" for k in ks)
    lines += [f"Best bge config: `{best['method']}`, "
              f"`{config_label(best)}` — {recalls}", ""]
    lines += [f"## Methods side by side (recall@{topk} per size x overlap)", ""]
    header = "| size | overlap |" + "".join(f" {m} |" for m in METHOD_COLORS)
    header += " spread |"
    lines += [header, "|" + "---|" * (3 + len(METHOD_COLORS))]
    for m in matched:
        cells = f"| {m['nominal_size']} | {m['overlap']} |"
        for meth in METHOD_COLORS:
            v = m.get(f"recall{topk}_{meth}")
            cells += f" {v:.4f} |" if v is not None else " |"
        sp = m.get(f"method_spread@{topk}")
        cells += f" {sp:.4f} |" if sp is not None else " |"
        lines.append(cells)
    lines += ["", "## Direction checks (does the NQ headline transfer?)", ""]
    for c in checks:
        val = "" if c["value"] is None else f" — observed {c['value']}"
        lines.append(f"- **{c['replicates']}** — {c['claim']}: {c['metric']}"
                     f"{val} (rule: {c['rule']}; NQ reference: "
                     f"{c['reference_nq']})")
    n_yes = sum(1 for c in checks if c["replicates"] == "yes")
    lines += ["", f"**{n_yes}/{len(checks)} direction checks replicate.**", ""]
    path = out_dir / C.STAGE7_SUMMARY_MD
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stage7] wrote {path}")


def _print_summary(rows, checks, n_docs: int, n_questions: int) -> None:
    from rag_chunk.hybrid import config_label
    from rag_chunk.sweep import _rank_key

    ks = sorted(C.RECALL_KS)
    print(f"\n[stage7] {len(rows)} rows scored on {n_docs} docs / "
          f"{n_questions} questions")
    best = max(rows, key=_rank_key)
    recalls = " ".join(f"R@{k}={best[f'recall@{k}']:.4f}" for k in ks)
    print(f"[stage7] best bge config: {best['method']} "
          f"{config_label(best)}  {recalls}")
    print("\n[stage7] direction checks:")
    for c in checks:
        val = "" if c["value"] is None else f" (observed {c['value']})"
        print(f"  [{c['replicates']:>3}] {c['claim']}{val}")
    n_yes = sum(1 for c in checks if c["replicates"] == "yes")
    print(f"[stage7] {n_yes}/{len(checks)} direction checks replicate.\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 7: cross-dataset robustness check — the 30-config "
                    "bge-only sweep on TriviaQA rc.wikipedia.")
    ap.add_argument("--check", action="store_true",
                    help="NQ sanity mode: reproduce the archived Stage 3 rows "
                         "exactly on the same cached 200-doc corpus")
    ap.add_argument("--n-questions", type=int, default=None,
                    help="kept TriviaQA questions "
                         f"(default: config STAGE7_N_QUESTIONS="
                         f"{C.STAGE7_N_QUESTIONS})")
    ap.add_argument("--fresh", action="store_true",
                    help="discard this mode's checkpoint and start over")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="dense retrieval model; must match the Stage 3 archive")
    ap.add_argument("--retrieval-dim", type=int, default=None,
                    help="optional retrieval embedding dimension override")
    args = ap.parse_args()
    if args.check and args.n_questions is not None:
        raise SystemExit("[stage7] --check always uses the cached NQ corpus; "
                         "drop --n-questions")

    stage3_dir = _ensure_stage3_archived()
    stage3_rows = _read_csv(stage3_dir / C.SWEEP_RESULTS_CSV)
    _ensure_stage3_models(stage3_rows, args.retrieval_model)

    mode = "check" if args.check else "trivia"
    n_questions_req = args.n_questions or C.STAGE7_N_QUESTIONS

    C.apply(
        RETRIEVAL_EMBED_MODEL=args.retrieval_model,
        RETRIEVAL_EMBED_DIM=args.retrieval_dim,
        RETRIEVAL_EMBED_NORMALIZE=True,
    )
    C.ensure_dirs()

    # Heavy imports + weights + corpus all load BEFORE latest/ is touched, so
    # any failure leaves the archived files and checkpoints intact.
    from rag_chunk import cross_dataset, nq_data, training

    print("[stage7] mode:", mode)
    print("[stage7] boundary/chunking embeddings stay on:",
          C.BOUNDARY_EMBED_MODEL)
    print("[stage7] dense retrieval embeddings:", C.RETRIEVAL_EMBED_MODEL)
    print("[stage7] Stage 3 archive:", stage3_dir)

    bilstm = training.load_model("bilstm")
    transformer = training.load_model("transformer")
    if mode == "check":
        dataset = "nq"
        docs, questions = nq_data.prepare_nq()      # the 200-doc cached corpus
        loader_meta = None
    else:
        dataset = cross_dataset.DATASET_LABEL
        docs, questions = cross_dataset.prepare_trivia(n_questions_req)
        loader_meta = cross_dataset.load_meta(n_questions_req)
    print(f"[stage7] eval set ({dataset}): {len(docs)} docs, "
          f"{len(questions)} questions")

    latest = C.RESULTS_LATEST_DIR
    ckpt = latest / CHECKPOINT_FILES[mode]
    if args.fresh and ckpt.exists():
        ckpt.unlink()
        print(f"[stage7] --fresh: discarded checkpoint {ckpt.name}")

    _clean_latest()
    rows = _run_grid(docs, questions, bilstm=bilstm,
                     transformer_model=transformer, mode=mode,
                     dataset=dataset, checkpoint_path=ckpt)

    if mode == "check":
        _write_stage3_check(stage3_rows, rows, latest / STAGE7_CHECK_CSV)
        print("[stage7] check mode done. If the check is OK, run the TriviaQA "
              "eval:\n  python scripts/15_cross_dataset_eval.py")
        return

    _write_results_csv(rows, latest / C.STAGE7_RESULTS_CSV)
    matched = _matched_summary(rows)
    _write_matched_csv(matched, latest / C.STAGE7_MATCHED_CSV)
    checks = _direction_checks(rows, len(questions))
    _write_direction_csv(checks, latest / C.STAGE7_DIRECTION_CSV)
    _plot_scatter(stage3_rows, rows, len(docs), len(questions),
                  latest / C.STAGE7_SCATTER_PNG)
    _write_summary(latest, rows, matched, checks, loader_meta,
                   len(docs), len(questions), n_questions_req)
    _print_summary(rows, checks, len(docs), len(questions))

    print("[stage7] complete. Review the direction checks above, then archive:")
    print("  python scripts/save_stage_results.py --stage stage7")


if __name__ == "__main__":
    main()
