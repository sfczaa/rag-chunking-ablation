"""Portfolio figure: best doc-constrained Recall@5 per stage (Stages 1-6, 8).

Reads ONLY the archived stage finals (no experiment is rerun):
    results/stage2/final/sweep_results.csv              -- Stages 1-2 (MiniLM)
    results/stage3/final/sweep_results.csv              -- Stage 3 (BGE)
    results/stage4/final/hybrid_sweep_results.csv       -- Stage 4 (BGE/BM25/RRF)
    results/stage5/final/rerank_sweep_results.csv       -- Stage 5 (+ CE rerank)
    results/stage6/final/stage6_large_eval_results.csv  -- Stage 6 (5x scale)
    results/stage8/final/stage8_ft_eval_results.csv     -- Stage 8 (+ FT rerank)

Stage 1 has no archive of its own (results/latest/ was overwritten before it
was snapshotted -- see results/stage1/final/README.md); its numbers are the
fixed + BiLSTM rows of the Stage 2 sweep, which re-ran the identical
deterministic grid unchanged while adding the Transformer arm.

Stages 1-5 share one eval set (200 docs / 203 questions), so their best-R@5
points sit on one line. Stage 6 re-evaluates the same pipeline on a fresh
1000-doc / 1032-question corpus: its absolute number is NOT comparable to
Stages 1-5 (5x more distractor docs in the index), so it is drawn detached
behind a dashed connector as a scale check, not a regression. Stage 8 shares
the Stage 6 eval set exactly (its bge/rerank20 rows reproduce stage6/final
byte-for-byte), so Stage 6 -> 8 is a solid, directly comparable segment.
Stage 7 (TriviaQA) is a different dataset and is not drawn on this axis.

Writes:
    results/portfolio/best_r5_evolution.png

Usage:
    python scripts/14_evolution_plot.py
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

# The archived Stage 1-5 eval set (see PROGRESS.md); those CSVs do not store
# the counts themselves.
SMALL_EVAL_QUESTIONS = 203

# Same visual language as the Stage 1-6 figures.
METHOD_COLORS = {"fixed": "#888888", "bilstm": "#1f77b4", "transformer": "#d62728"}
METHOD_X_OFFSET = {"fixed": -0.16, "bilstm": 0.0, "transformer": 0.16}


def _load_rows(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rank_key(r: dict):
    """The project ranking rule: R@5, then R@3, R@1, then smaller chunks."""
    return (float(r["recall@5"]), float(r["recall@3"]), float(r["recall@1"]),
            -float(r["avg_chunk_size"]))


def _describe(r: dict) -> str:
    if r["method"] == "fixed":
        cfg = f"fixed {r['fixed_size']}/{int(float(r['fixed_overlap']))}"
    else:
        cfg = (f"{r['method']} t{r['semantic_target_size']}"
               f"/{int(float(r['semantic_overlap']))}")
    arm = r.get("arm") or r.get("retriever")
    return f"{arm} {cfg}" if arm else cfg


def _one_se(p: float, n: int) -> float:
    return (p * (1.0 - p) / n) ** 0.5


def load_stages() -> list[dict]:
    """One entry per stage: label, change description, rows, question count."""
    finals = {n: C.RESULTS_DIR / n / "final" for n in
              ("stage2", "stage3", "stage4", "stage5", "stage6", "stage8")}
    inputs = {
        "stage2": finals["stage2"] / C.SWEEP_RESULTS_CSV,
        "stage3": finals["stage3"] / C.SWEEP_RESULTS_CSV,
        "stage4": finals["stage4"] / C.HYBRID_SWEEP_CSV,
        "stage5": finals["stage5"] / C.RERANK_SWEEP_CSV,
        "stage6": finals["stage6"] / C.STAGE6_RESULTS_CSV,
        "stage8": finals["stage8"] / C.STAGE8_RESULTS_CSV,
    }
    missing = [f"  {name}: {p}" for name, p in inputs.items() if not p.exists()]
    if missing:
        raise SystemExit("[evolution] missing archived input(s):\n"
                         + "\n".join(missing))

    rows = {name: _load_rows(p) for name, p in inputs.items()}
    n_large = int(rows["stage6"][0]["n_questions"])
    return [
        dict(label="Stage 1", change="chunking\noptimizer (MiniLM)",
             rows=[r for r in rows["stage2"] if r["method"] != "transformer"],
             n=SMALL_EVAL_QUESTIONS),
        dict(label="Stage 2", change="+ Transformer\nboundary",
             rows=rows["stage2"], n=SMALL_EVAL_QUESTIONS),
        dict(label="Stage 3", change="embedder\nMiniLM → BGE",
             rows=rows["stage3"], n=SMALL_EVAL_QUESTIONS),
        dict(label="Stage 4", change="+ BM25 / RRF\n(no gain)",
             rows=rows["stage4"], n=SMALL_EVAL_QUESTIONS),
        dict(label="Stage 5", change="+ CE rerank\ntop-20",
             rows=rows["stage5"], n=SMALL_EVAL_QUESTIONS),
        dict(label="Stage 6", change="5× eval scale\n(new eval set)",
             rows=rows["stage6"], n=n_large),
        dict(label="Stage 8", change="+ fine-tuned\nCE reranker",
             rows=rows["stage8"], n=n_large),
    ]


def plot(stages: list[dict], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 5.5))

    best_vals = []
    for x, st in enumerate(stages):
        best = max(st["rows"], key=_rank_key)
        best_vals.append(float(best["recall@5"]))
        print(f"[evolution] {st['label']} best: {_describe(best)} "
              f"R@5={best['recall@5']}", end="")
        for method in METHOD_COLORS:
            method_rows = [r for r in st["rows"] if r["method"] == method]
            if not method_rows:
                continue
            v = float(max(method_rows, key=_rank_key)["recall@5"])
            print(f" | {method}={v:.4f}", end="")
            ax.scatter(x + METHOD_X_OFFSET[method], v, c=METHOD_COLORS[method],
                       s=70, edgecolor="white", linewidth=0.6, zorder=3)
        print()

    xs = list(range(len(stages)))
    ses = [_one_se(v, st["n"]) for v, st in zip(best_vals, stages)]
    # Stages 1-5: one eval set, one line. Stage 6: new eval set -> detached
    # behind a dotted connector. Stage 8 shares the Stage 6 eval set, so
    # Stage 6 -> 8 is a solid, directly comparable segment.
    ax.errorbar(xs[:5], best_vals[:5], yerr=ses[:5], color="black", marker="o",
                markersize=5, capsize=3, linewidth=1.5, zorder=4,
                label=f"best overall (n={SMALL_EVAL_QUESTIONS})")
    ax.plot(xs[4:6], best_vals[4:6], color="#555555", linestyle=":",
            linewidth=1.2, zorder=2)
    ax.errorbar(xs[5:], best_vals[5:], yerr=ses[5:], color="black", marker="o",
                markersize=6, markerfacecolor="white", capsize=3, linewidth=1.5,
                zorder=4, label=f"best overall (n={stages[5]['n']}, new eval set)")
    for x, v in zip(xs, best_vals):
        # the Stage 6 label sits below its marker, clear of both connectors
        dxy, ha = (((0, -18), "center") if x == xs[5] else ((0, 10), "center"))
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                    xytext=dxy, ha=ha, fontsize=9, zorder=5)

    handles = [Line2D([0], [0], marker="o", linestyle="",
                      color=METHOD_COLORS[m], label=f"best {m}")
               for m in METHOD_COLORS]
    handles += ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="lower left", fontsize=8)

    ax.set_xticks(xs, [f"{st['label']}\n{st['change']}" for st in stages],
                  fontsize=9)
    ax.set_ylabel("Best doc-constrained Recall@5")
    ax.set_title("Best Recall@5 per stage — one controlled change at a time:\n"
                 "the embedder and in-domain reranker fine-tuning move the "
                 "ceiling, the chunking method never does",
                 fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    lo = min(m - s for m, s in zip(best_vals, ses)) - 0.025
    hi = max(m + s for m, s in zip(best_vals, ses)) + 0.025
    ax.set_ylim(min(lo, 0.84), hi)
    ax.text(0.99, 0.02,
            "Stages 6 and 8 share one 1000-doc / 1032-question eval set "
            "(Stages 1–5: 200 / 203); Stage 6's absolute drop reflects\n"
            "5× more distractor documents, not a code change (35/35 rows "
            "exact). Stage 8 changes only the reranker weights\n(10/10 "
            "baseline rows exact). Stage 7 (TriviaQA) is a different dataset "
            "and is not drawn on this axis.",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
            color="#444444")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evolution] wrote {path}")


def main() -> None:
    plot(load_stages(), C.RESULTS_DIR / C.PORTFOLIO_DIRNAME / C.EVOLUTION_PLOT_PNG)


if __name__ == "__main__":
    main()
