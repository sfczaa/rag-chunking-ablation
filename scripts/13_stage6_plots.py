"""Stage 6 figures: plot the large-eval results against the archived Stage 5 run.

Reads (no experiment is rerun):
    results/stage5/final/rerank_sweep_results.csv   -- small eval (200 docs / 203 q)
    results/stage5/final/rerank_matched.csv
    results/latest/stage6_large_eval_results.csv    -- large eval (~1000 docs)
    results/latest/stage6_matched_summary.csv

Writes into ``results/latest/`` (run BEFORE ``save_stage_results.py --stage stage6``
so the PNGs are archived together with the CSVs):
    stage6_size_vs_recall.png  -- bge-only R@5 vs avg chunk size, small vs large
                                  eval side by side (claim 1: size dominates)
    stage6_rerank_delta.png    -- rerank20-bge dRecall@1 on the 5 matched configs,
                                  small vs large eval (claims 3-4)

Usage:
    python scripts/13_stage6_plots.py
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

# The archived Stage 5 eval set (see README.md); the Stage 5 CSVs do not
# store the counts themselves.
SMALL_EVAL_DOCS = 200
SMALL_EVAL_QUESTIONS = 203

# Same visual language as the Stage 1-5 figures (sweep.plot_size_vs_recall_scatter).
METHOD_COLORS = {"fixed": "#888888", "bilstm": "#1f77b4", "transformer": "#d62728"}
OVERLAP_MARKERS = {0: "o", 1: "^"}


def _load_rows(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _overlap_of(row: dict) -> int:
    field = "fixed_overlap" if row["method"] == "fixed" else "semantic_overlap"
    return int(float(row[field]))


def _short_label(method: str, chunk_config: str) -> str:
    parts = dict(p.split("=", 1) for p in chunk_config.split(","))
    ov = parts["overlap"]
    if method == "fixed":
        return f"fixed {parts['fixed_size']}/{ov}"
    return f"{method} t{parts['target']}/{ov}"


def _two_se(rows: list[dict], recall_col: str, n_questions: int) -> float:
    pbar = sum(float(r[recall_col]) for r in rows) / len(rows)
    return 2.0 * (pbar * (1.0 - pbar) / n_questions) ** 0.5


def _scatter_panel(ax, rows: list[dict], topk: int, n_docs: int, n_questions: int) -> None:
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
    ax.set_title(f"{n_docs} docs / {n_questions} questions")
    ax.set_xlabel("Average chunk size (sentences)")
    ax.grid(True, alpha=0.3)


def plot_size_vs_recall(small_rows: list[dict], large_rows: list[dict],
                        n_docs: int, n_questions: int, path: pathlib.Path) -> None:
    """Claim 1 in one glance: the upward size trend has the same shape at both
    scales while the method colours stay intermixed along it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    topk = max(C.RECALL_KS)
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    _scatter_panel(ax_s, small_rows, topk, SMALL_EVAL_DOCS, SMALL_EVAL_QUESTIONS)
    _scatter_panel(ax_l, large_rows, topk, n_docs, n_questions)
    ax_s.set_ylabel(f"Doc-constrained Recall@{topk}")

    handles = (
        [Line2D([0], [0], marker="s", linestyle="", color=METHOD_COLORS[m], label=m)
         for m in METHOD_COLORS]
        + [Line2D([0], [0], marker=mk, linestyle="", color="#555555",
                  label=f"overlap={ov}") for ov, mk in OVERLAP_MARKERS.items()]
        + [Line2D([0], [0], color="black", linestyle="--", alpha=0.6,
                  label="size trend")]
    )
    ax_s.legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle("BGE-only recall vs chunk size: the size effect replicates at ~5x scale")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[stage6-plots] wrote {path}")


def plot_rerank_delta(small_matched: list[dict], large_matched: list[dict],
                      n_questions: int, path: pathlib.Path) -> None:
    """Claims 3-4: the rerank20 gain lives at small chunks and shrinks with scale;
    every size-15 delta sits inside the +/-2 SE noise band of the large eval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    small_by_key = {(r["method"], r["chunk_config"]): r for r in small_matched}
    pairs = [(r, small_by_key[(r["method"], r["chunk_config"])]) for r in large_matched]

    labels = [_short_label(r["method"], r["chunk_config"]) for r, _ in pairs]
    d_small = [float(s["rerank20_minus_bge@1"]) for _, s in pairs]
    d_large = [float(r["rerank20_minus_bge@1"]) for r, _ in pairs]
    se2_small = _two_se([s for _, s in pairs], "bge_recall@1", SMALL_EVAL_QUESTIONS)
    se2_large = _two_se([r for r, _ in pairs], "bge_recall@1", n_questions)

    x = np.arange(len(pairs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, deltas, label in (
        (-w / 2, d_small, f"n={SMALL_EVAL_QUESTIONS} (Stage 5)"),
        (+w / 2, d_large, f"n={n_questions} (Stage 6)"),
    ):
        bars = ax.bar(x + offset, deltas, w, label=label)
        ax.bar_label(bars, fmt="%+.3f", padding=2, fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    for y in (se2_large, -se2_large):
        ax.axhline(y, color="#888888", linestyle="--", linewidth=1,
                   label=f"+/-2 SE (n={n_questions})" if y > 0 else None)
    ax.text(0.98, 0.98, f"2 SE at n={SMALL_EVAL_QUESTIONS}: {se2_small:.3f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xticks(x, labels)
    ax.set_ylabel("Delta Recall@1 (rerank20 - bge)")
    ax.set_title("Reranker gain: concentrated at small chunks, none at size 15")
    ax.legend(loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 0.90))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[stage6-plots] wrote {path}")


def main() -> None:
    stage5_final = C.RESULTS_DIR / "stage5" / "final"
    inputs = {
        "stage5 sweep": stage5_final / C.RERANK_SWEEP_CSV,
        "stage5 matched": stage5_final / C.RERANK_MATCHED_CSV,
        "stage6 results": C.RESULTS_LATEST_DIR / C.STAGE6_RESULTS_CSV,
        "stage6 matched": C.RESULTS_LATEST_DIR / C.STAGE6_MATCHED_CSV,
    }
    missing = [f"  {name}: {p}" for name, p in inputs.items() if not p.exists()]
    if missing:
        raise SystemExit("[stage6-plots] missing input file(s):\n" + "\n".join(missing)
                         + "\nrun `python scripts/12_large_eval.py` first.")

    stage5_bge = [r for r in _load_rows(inputs["stage5 sweep"]) if r["arm"] == "bge"]
    stage6_rows = _load_rows(inputs["stage6 results"])
    stage6_bge = [r for r in stage6_rows if r["arm"] == "bge"]
    n_docs = int(stage6_rows[0]["n_docs"])
    n_questions = int(stage6_rows[0]["n_questions"])

    plot_size_vs_recall(stage5_bge, stage6_bge, n_docs, n_questions,
                        C.RESULTS_LATEST_DIR / C.STAGE6_SIZE_PLOT_PNG)
    plot_rerank_delta(_load_rows(inputs["stage5 matched"]),
                      _load_rows(inputs["stage6 matched"]),
                      n_questions, C.RESULTS_LATEST_DIR / C.STAGE6_DELTA_PLOT_PNG)


if __name__ == "__main__":
    main()
