"""Phase 6 — chunking sweep optimizer.

Sweeps fixed-size and learned **target-size** chunking across chunk sizes and
overlap settings, scores each on Natural Questions doc-constrained Recall@k, and
exports the artifacts that turn this project from a single comparison into a real
optimizer:

    <results>/sweep_results.csv          one row per (method, config)
    <results>/best_config.json           the validation-best configuration
    <results>/fair_comparison_table.csv  BiLSTM vs the closest-size fixed row
    <results>/recall_vs_chunk_size.png   recall@k vs avg chunk size, 4 series
    <results>/model_comparison.png       best fixed vs best BiLSTM (optional)

Design notes
------------
* The question it answers is *not* "does learned chunking win?" but: **at the
  same approximate chunk size and overlap, does learned semantic cutting pick
  better boundaries than fixed-size cutting?** Hence every learned config is
  size-matched to a fixed config in the fair table.
* Per-config FAISS indices are built **in memory and discarded** — nothing is
  persisted under ``nq/`` unless you opt in (``save_sweep_index``), so the
  sweep never litters the data folder.
* The boundary model runs **once per document**; all learned configs reuse those
  probabilities (cheap pure-Python re-chunking), which is what keeps the full
  grid Colab-friendly.

Heavy deps (numpy / matplotlib / faiss / torch) are imported lazily inside
functions, matching the rest of the package, so ``import rag_chunk.sweep`` and
the pure-Python helpers (:func:`select_best_config`, :func:`fair_comparison_table`)
work without them installed.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import config as C


# --------------------------------------------------------------------------- #
# Grids
# --------------------------------------------------------------------------- #
def _grids(quick: bool) -> tuple[list[int], list[int], list[int]]:
    """Return ``(fixed_sizes, target_sizes, overlaps)`` for the chosen grid."""
    if quick:
        return (list(C.QUICK_FIXED_SIZE_GRID),
                list(C.QUICK_TARGET_SIZE_GRID),
                list(C.QUICK_OVERLAP_GRID))
    return (list(C.FIXED_SIZE_GRID),
            list(C.TARGET_SIZE_GRID),
            list(C.OVERLAP_GRID))


def _semantic_window(target: int) -> tuple[int, int]:
    """Learned [min, max] window derived from a target size (per the spec)."""
    return max(2, target - 4), target + 4


# --------------------------------------------------------------------------- #
# One configuration -> one scored row
# --------------------------------------------------------------------------- #
def _precompute_boundary_probs(docs: list[dict], models_by_type: dict) -> dict:
    """Run each learned model once per document, sharing sentence embeddings.

    ``models_by_type`` maps a model type (``"bilstm"`` / ``"transformer"``) to a
    model. Sentence embeddings are encoded once per doc and reused across models,
    and every learned config later re-chunks from these cached probabilities.
    Returns ``{model_type: {doc_key: per-boundary probs}}``.
    """
    from rag_chunk import chunking, embedding

    out: dict = {mt: {} for mt in models_by_type}
    for d in docs:
        sents = d["sentences"]
        key = d.get("title", d.get("id"))
        sent_emb = embedding.encode(sents, normalize=False)   # once per doc
        for mt, model in models_by_type.items():
            out[mt][key] = chunking.predict_boundary_probs(sents, sent_emb, model)
    return out


def _eval_config(
    method: str,
    docs: list[dict],
    questions: list[dict],
    *,
    model=None,
    boundary_probs_by_id: dict | None = None,
    fixed_size: int | None = None,
    fixed_overlap: int | None = None,
    semantic_policy: str | None = None,
    semantic_target_size: int | None = None,
    semantic_min_size: int | None = None,
    semantic_max_size: int | None = None,
    semantic_overlap: int | None = None,
    threshold: float | None = None,
    save_prefix: Path | None = None,
) -> dict:
    """Build an in-memory index for one config and score Recall@k on NQ."""
    from rag_chunk import metrics, retrieval

    index = retrieval.build_index_for_config(
        method, docs, model=model,
        fixed_size=fixed_size, fixed_overlap=fixed_overlap,
        semantic_policy=semantic_policy, semantic_target_size=semantic_target_size,
        semantic_min_size=semantic_min_size, semantic_max_size=semantic_max_size,
        semantic_overlap=semantic_overlap, threshold=threshold,
        boundary_probs_by_id=boundary_probs_by_id, save_prefix=save_prefix,
    )
    rec = metrics.recall_at_k(index, questions, C.RECALL_KS)

    is_learned = method in ("bilstm", "transformer")
    row: dict = {
        "method": method,
        "model_type": method if is_learned else "none",
        "embedding_model": C.RETRIEVAL_EMBED_MODEL,
        "boundary_embedding_model": C.BOUNDARY_EMBED_MODEL,
        "retrieval_embedding_model": C.RETRIEVAL_EMBED_MODEL,
    }
    for k in C.RECALL_KS:
        row[f"recall@{k}"] = rec["doc_constrained"][k]
    for k in C.RECALL_KS:
        row[f"recall_unconstrained@{k}"] = rec["unconstrained"][k]
    row.update({
        "avg_chunk_size": index.avg_chunk_size(),
        "n_chunks": len(index.chunk_texts),
        "fixed_size": fixed_size if method == "fixed" else None,
        "fixed_overlap": fixed_overlap if method == "fixed" else None,
        "semantic_policy": semantic_policy if is_learned else None,
        "semantic_target_size": semantic_target_size if is_learned else None,
        "semantic_min_size": semantic_min_size if is_learned else None,
        "semantic_max_size": semantic_max_size if is_learned else None,
        "semantic_overlap": semantic_overlap if is_learned else None,
        # only meaningful for the threshold policy; None for target-size cutting
        "boundary_threshold": (
            threshold if (is_learned and semantic_policy == "threshold") else None),
    })
    # `index` (and its FAISS handle) goes out of scope on return -> reclaimed.
    return row


# --------------------------------------------------------------------------- #
# Sweep driver
# --------------------------------------------------------------------------- #
def run_sweep(
    model=None,
    quick: bool = False,
    save_run: bool = False,
    save_sweep_index: bool = False,
    out_dir=None,
    docs: list[dict] | None = None,
    questions: list[dict] | None = None,
    transformer_model=None,
) -> list[dict]:
    """Run the sweep and write all artifacts. Returns the rows.

    Sweeps **fixed-size + BiLSTM** by default (Stage 1). Pass
    ``transformer_model`` to also sweep the **Transformer** boundary model
    (Stage 2) — fixed + bilstm + transformer — under the same MiniLM embeddings,
    the same NQ corpus, the same Recall@k metric and the same target-size policy,
    so the only thing that differs is the boundary model.

    Parameters
    ----------
    model : trained BiLSTM, or ``None`` to ``training.load_model("bilstm")``.
    transformer_model : trained Transformer to also include, or ``None`` to skip
        it (Stage 1 behaviour — fixed + bilstm only).
    quick : use the smaller ``QUICK_*`` grids for a fast sanity sweep.
    save_run : also snapshot the artifacts to a timestamped
        ``results/runs/<ts>_sweep/`` folder (off by default).
    save_sweep_index : persist each per-config FAISS index under
        ``nq/indices/sweep/`` (off by default — normally in-memory only).
    out_dir : where the "latest" artifacts go (default ``RESULTS_LATEST_DIR``).
    docs, questions : reuse an already-prepared NQ corpus (else prepared here).
    """
    from rag_chunk import nq_data, training

    C.ensure_dirs()
    if model is None:
        model = training.load_model("bilstm")
    if docs is None or questions is None:
        docs, questions = nq_data.prepare_nq()

    # Learned models to sweep (insertion order = sweep + report order).
    learned = {"bilstm": model}
    if transformer_model is not None:
        learned["transformer"] = transformer_model

    fixed_sizes, target_sizes, overlaps = _grids(quick)
    probs_by_type = _precompute_boundary_probs(docs, learned)

    sweep_idx_dir = C.retrieval_index_sweep_dir() if save_sweep_index else None
    if sweep_idx_dir is not None:
        sweep_idx_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total = (len(fixed_sizes) + len(learned) * len(target_sizes)) * len(overlaps)
    done = 0
    for overlap in overlaps:
        for size in fixed_sizes:
            done += 1
            print(f"[sweep] ({done}/{total}) fixed        size={size} overlap={overlap}")
            prefix = (sweep_idx_dir / f"fixed_s{size}_o{overlap}") if sweep_idx_dir else None
            rows.append(_eval_config(
                "fixed", docs, questions,
                fixed_size=size, fixed_overlap=overlap, save_prefix=prefix))
        for mtype in learned:
            for target in target_sizes:
                done += 1
                mn, mx = _semantic_window(target)
                print(f"[sweep] ({done}/{total}) {mtype:<12} target={target} "
                      f"(min={mn},max={mx}) overlap={overlap}")
                prefix = ((sweep_idx_dir / f"{mtype}_t{target}_o{overlap}")
                          if sweep_idx_dir else None)
                rows.append(_eval_config(
                    mtype, docs, questions,
                    boundary_probs_by_id=probs_by_type[mtype],
                    semantic_policy="target", semantic_target_size=target,
                    semantic_min_size=mn, semantic_max_size=mx,
                    semantic_overlap=overlap, save_prefix=prefix))

    out_dir = Path(out_dir) if out_dir is not None else C.RESULTS_LATEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best = select_best_config(rows)
    fair = fair_comparison_table(rows)
    _write_sweep_csv(rows, out_dir / C.SWEEP_RESULTS_CSV)
    _write_best_json(best, out_dir / C.BEST_CONFIG_JSON)
    _write_fair_csv(fair, out_dir / C.FAIR_TABLE_CSV)
    plot_recall_vs_chunk_size(rows, out_dir / C.RECALL_PLOT_PNG)
    plot_model_comparison(rows, out_dir / C.MODEL_PLOT_PNG)
    plot_size_vs_recall_scatter(rows, out_dir / C.SIZE_SCATTER_PNG,
                                n_questions=len(questions))
    _print_summary(rows, best, out_dir)

    if save_run:
        snap = C.RESULTS_RUNS_DIR / f"{_timestamp()}_sweep"
        snap.mkdir(parents=True, exist_ok=True)
        for name in (C.SWEEP_RESULTS_CSV, C.BEST_CONFIG_JSON, C.FAIR_TABLE_CSV,
                     C.RECALL_PLOT_PNG, C.MODEL_PLOT_PNG, C.SIZE_SCATTER_PNG):
            src = out_dir / name
            if src.exists():
                shutil.copy2(src, snap / name)
        print(f"[sweep] snapshot saved -> {snap}")

    return rows


# --------------------------------------------------------------------------- #
# Best config + fair table (pure Python — no heavy deps)
# --------------------------------------------------------------------------- #
def _rank_key(row: dict) -> tuple:
    """Ranking key: maximise doc-constrained recall@k (largest k first), then
    prefer a smaller average chunk size."""
    ks = sorted(C.RECALL_KS, reverse=True)
    return tuple(row[f"recall@{k}"] for k in ks) + (-row["avg_chunk_size"],)


def select_best_config(rows: list[dict]) -> dict:
    """Pick the validation-best row.

    Ranking: maximise recall@5, tie-break recall@3, then recall@1, then the
    smaller average chunk size (a tie that retrieves equally well but with
    tighter chunks is the better default).
    """
    if not rows:
        return {}
    best = max(rows, key=_rank_key)
    metrics = {f"recall@{k}": best[f"recall@{k}"] for k in sorted(C.RECALL_KS)}
    metrics["avg_chunk_size"] = best["avg_chunk_size"]
    return {
        "ranking": ("max doc-constrained recall@5, then recall@3, then recall@1, "
                    "then min avg_chunk_size"),
        "selected_metrics": metrics,
        "config": best,
    }


def _overlap_of(row: dict) -> int | None:
    return row["fixed_overlap"] if row["method"] == "fixed" else row["semantic_overlap"]


def fair_comparison_table(rows: list[dict]) -> list[dict]:
    """For each learned (BiLSTM) row, find the fixed row with the same overlap and
    closest average chunk size, and report the Recall@k deltas (bilstm - fixed).
    """
    fixed_rows = [r for r in rows if r["method"] == "fixed"]
    bilstm_rows = [r for r in rows if r["method"] == "bilstm"]
    ks = sorted(C.RECALL_KS)
    table: list[dict] = []
    for b in bilstm_rows:
        ov = _overlap_of(b)
        cands = [f for f in fixed_rows if _overlap_of(f) == ov]
        if not cands:
            continue
        f = min(cands, key=lambda r: abs(r["avg_chunk_size"] - b["avg_chunk_size"]))
        row = {
            "overlap": ov,
            "bilstm_target_size": b["semantic_target_size"],
            "bilstm_avg_chunk_size": b["avg_chunk_size"],
            "fixed_size": f["fixed_size"],
            "fixed_avg_chunk_size": f["avg_chunk_size"],
            "delta_avg_chunk_size": b["avg_chunk_size"] - f["avg_chunk_size"],
        }
        for k in ks:
            row[f"bilstm_recall@{k}"] = b[f"recall@{k}"]
            row[f"fixed_recall@{k}"] = f[f"recall@{k}"]
            row[f"delta_recall@{k}"] = b[f"recall@{k}"] - f[f"recall@{k}"]
        table.append(row)
    return table


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_recall_vs_chunk_size(rows: list[dict], path) -> None:
    """x = avg chunk size, y = doc-constrained Recall@max(k); one line per
    (method, overlap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topk = max(C.RECALL_KS)
    # Transformer series only render if transformer rows exist (Stage 2); in
    # Stage 1 they are empty and skipped, so the plot is unchanged.
    series = [
        ("fixed", 0, "Fixed overlap=0", "o", "-"),
        ("fixed", 1, "Fixed overlap=1", "o", "--"),
        ("bilstm", 0, "BiLSTM overlap=0", "s", "-"),
        ("bilstm", 1, "BiLSTM overlap=1", "s", "--"),
        ("transformer", 0, "Transformer overlap=0", "^", "-"),
        ("transformer", 1, "Transformer overlap=1", "^", "--"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotted = False
    for method, ov, label, marker, ls in series:
        pts = sorted(
            (r["avg_chunk_size"], r[f"recall@{topk}"])
            for r in rows if r["method"] == method and _overlap_of(r) == ov
        )
        if not pts:
            continue
        plotted = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=marker, linestyle=ls, label=label)
    ax.set_xlabel("Average chunk size (sentences)")
    ax.set_ylabel(f"Doc-constrained Recall@{topk}")
    ax.set_title("RAG retrieval recall vs chunk size (NQ): learned vs fixed")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[sweep] wrote {path}")


def plot_model_comparison(rows: list[dict], path) -> None:
    """Optional: grouped Recall@k bars for the best config of each method present
    — fixed vs bilstm (Stage 1), plus transformer when its rows exist (Stage 2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def best_of(method: str):
        cand = [r for r in rows if r["method"] == method]
        return max(cand, key=_rank_key) if cand else None

    def label_of(method: str, row: dict) -> str:
        if method == "fixed":
            return f"Fixed (size={row['fixed_size']}, ov={row['fixed_overlap']})"
        return (f"{method.capitalize()} (target={row['semantic_target_size']}, "
                f"ov={row['semantic_overlap']})")

    present = [(m, best_of(m)) for m in ("fixed", "bilstm", "transformer")]
    present = [(m, r) for m, r in present if r is not None]
    if len(present) < 2:
        return
    ks = sorted(C.RECALL_KS)
    x = np.arange(len(ks))
    n = len(present)
    w = min(0.38, 0.8 / n)            # n==2 -> 0.38 (Stage 1 chart unchanged)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (method, row) in enumerate(present):
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, [row[f"recall@{k}"] for k in ks], w,
                      label=label_of(method, row))
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, [f"Recall@{k}" for k in ks])
    ax.set_ylabel("Recall (doc-constrained)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Best config per method (ranked by Recall@%d)" % max(ks))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[sweep] wrote {path}")


def plot_size_vs_recall_scatter(rows: list[dict], path, n_questions: int | None = None) -> None:
    """Scatter Recall@max(k) vs average chunk size, coloured by method.

    Built to show the Stage 2 finding in one glance: recall tracks chunk **size**
    (a clear upward trend, with the Pearson ``r`` annotated) while the chunking
    **method** colours stay intermixed along that trend — i.e. method differences
    sit inside the size effect and the per-question sampling noise. ``n_questions``
    (if given) annotates the 1-SE noise floor so the eye can judge significance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    topk = max(C.RECALL_KS)
    colors = {"fixed": "#888888", "bilstm": "#1f77b4", "transformer": "#d62728"}
    markers = {0: "o", 1: "^"}

    xs = np.array([r["avg_chunk_size"] for r in rows], dtype=float)
    ys = np.array([r[f"recall@{topk}"] for r in rows], dtype=float)
    if xs.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in rows:
        ax.scatter(r["avg_chunk_size"], r[f"recall@{topk}"],
                   c=colors.get(r["method"], "#333333"),
                   marker=markers.get(_overlap_of(r), "o"),
                   s=80, edgecolor="white", linewidth=0.6, zorder=3)

    # Global trend line + Pearson r — the "size dominates" evidence.
    annot = ""
    if xs.size >= 2 and xs.std() > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        pear = float(np.corrcoef(xs, ys)[0, 1])
        xline = np.array([xs.min(), xs.max()])
        ax.plot(xline, slope * xline + intercept,
                color="black", linestyle="--", alpha=0.6, zorder=2)
        annot = f"Pearson r(size, R@{topk}) = {pear:.2f}"
    if n_questions:
        pbar = float(ys.mean())
        se = (pbar * (1 - pbar) / n_questions) ** 0.5
        annot += f"\n1 SE ≈ {se:.3f}  (n={n_questions} questions)"
    if annot:
        ax.text(0.02, 0.98, annot, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    present = [m for m in colors if any(r["method"] == m for r in rows)]
    method_handles = [Line2D([0], [0], marker="s", linestyle="", color=colors[m],
                             label=m) for m in present]
    overlap_handles = [Line2D([0], [0], marker=mk, linestyle="", color="#555555",
                              label=f"overlap={ov}") for ov, mk in markers.items()]
    trend_handle = [Line2D([0], [0], color="black", linestyle="--", alpha=0.6,
                           label="size trend")]
    ax.legend(handles=method_handles + overlap_handles + trend_handle,
              loc="lower right", fontsize=9)

    ax.set_xlabel("Average chunk size (sentences)")
    ax.set_ylabel(f"Doc-constrained Recall@{topk}")
    ax.set_title("RAG recall vs chunk size (NQ): size dominates, methods overlap")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[sweep] wrote {path}")


# --------------------------------------------------------------------------- #
# Writers / formatting
# --------------------------------------------------------------------------- #
def _sweep_columns() -> list[str]:
    cols = [
        "method", "model_type", "embedding_model",
        "boundary_embedding_model", "retrieval_embedding_model",
    ]
    cols += [f"recall@{k}" for k in C.RECALL_KS]
    cols += [f"recall_unconstrained@{k}" for k in C.RECALL_KS]
    cols += ["avg_chunk_size", "n_chunks", "fixed_size", "fixed_overlap",
             "semantic_policy", "semantic_target_size", "semantic_min_size",
             "semantic_max_size", "semantic_overlap", "boundary_threshold"]
    return cols


def _fmt(col: str, val):
    if val is None:
        return ""
    if isinstance(val, float):
        if col in ("avg_chunk_size", "delta_avg_chunk_size",
                   "bilstm_avg_chunk_size", "fixed_avg_chunk_size"):
            return f"{val:.3f}"
        if col == "boundary_threshold":
            return f"{val:.3f}"
        return f"{val:.4f}"          # recall / delta_recall columns
    return val


def _write_sweep_csv(rows: list[dict], path) -> None:
    cols = _sweep_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[sweep] wrote {path}  ({len(rows)} rows)")


def _fair_columns() -> list[str]:
    cols = ["overlap", "bilstm_target_size", "bilstm_avg_chunk_size",
            "fixed_size", "fixed_avg_chunk_size", "delta_avg_chunk_size"]
    for k in sorted(C.RECALL_KS):
        cols += [f"bilstm_recall@{k}", f"fixed_recall@{k}", f"delta_recall@{k}"]
    return cols


def _write_fair_csv(fair: list[dict], path) -> None:
    cols = _fair_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in fair:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[sweep] wrote {path}  ({len(fair)} rows)")


def _json_default(o):
    # be robust if a numpy scalar ever sneaks in
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _write_best_json(best: dict, path) -> None:
    Path(path).write_text(
        json.dumps(best, indent=2, default=_json_default), encoding="utf-8")
    print(f"[sweep] wrote {path}")


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _print_summary(rows: list[dict], best: dict, out_dir) -> None:
    ks = sorted(C.RECALL_KS)
    topk = max(ks)
    ordered = sorted(rows, key=_rank_key, reverse=True)
    print(f"\n[sweep] {len(rows)} configs scored — ranked by doc-constrained "
          f"Recall@{topk}:\n")
    head = f"{'method':11} {'size':>5} {'ov':>3} {'avg':>6} " + \
           " ".join(f"R@{k:<4}" for k in ks)
    print(head)
    print("-" * len(head))
    for r in ordered:
        size = r["fixed_size"] if r["method"] == "fixed" else r["semantic_target_size"]
        ov = _overlap_of(r)
        cells = " ".join(f"{r[f'recall@{k}']:.3f}" for k in ks)
        print(f"{r['method']:11} {str(size):>5} {str(ov):>3} "
              f"{r['avg_chunk_size']:6.2f} {cells}")
    if best:
        c = best["config"]
        size = c["fixed_size"] if c["method"] == "fixed" else c["semantic_target_size"]
        print(f"\n[sweep] best: {c['method']} size/target={size} "
              f"overlap={_overlap_of(c)} "
              f"avg={c['avg_chunk_size']:.2f} "
              + " ".join(f"R@{k}={c[f'recall@{k}']:.3f}" for k in ks))
    print(f"[sweep] artifacts -> {out_dir}\n")
