"""Stage 5 — cross-encoder reranking on top of the BGE dense retrieval baseline.

Reuses the Stage 3/4 protocol unchanged — the same NQ corpus and questions, the
same fixed / BiLSTM / Transformer chunking grids, the same boundary models and
weights, the same BGE dense retriever — and adds one new step over the
*identical* chunks: BGE retrieves a top-``d`` candidate pool per question, and
an off-the-shelf pretrained cross-encoder (``C.RERANKER_MODEL``, no training or
fine-tuning) rescores the (question, chunk) pairs and reorders the pool.

Arms compared per chunking config (one row each):

* ``bge``       — dense-only baseline, exactly the Stage 3/4 bge path;
* ``rerank<d>`` — BGE top-``d`` candidates reordered by the cross-encoder,
                  one arm per depth in ``C.RERANK_DEPTHS`` (default 20 and 50).

For every chunking config the chunks are built ONE time and the dense index
embeds them once; the deeper pools extend the shallower ones (same dense
ranking), so the cross-encoder scores each (question, chunk) pair exactly once
— the pairs for ranks 1..20 are timed separately from ranks 21..50, keeping
the per-depth latency honest. All arms are scored by the same
``metrics.recall_from_retrieved`` code path.

Each row also records the candidate-pool ceiling ``pool_recall@d`` (was the
answer chunk present in the BGE top-``d`` pool at all?) — reranking can never
recall a chunk the pool missed.

Artifacts (written under ``RESULTS_LATEST_DIR`` by :func:`run_rerank_sweep`):

    rerank_sweep_results.csv        one row per (arm, method, config)
    rerank_best_config.json         best row per arm + overall best
    rerank_matched.csv              bge / rerank20 / rerank50 side-by-side
    rerank_recall_vs_chunk_size.png Recall@1 vs avg chunk size by arm
    rerank_comparison.png           best config per arm, grouped Recall@k bars

Heavy deps (numpy / matplotlib / faiss / torch / sentence-transformers) are
imported lazily inside functions, matching the rest of the package.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import config as C
from rag_chunk.hybrid import config_key, config_label  # shared config identity

BASELINE_ARM = "bge"


def rerank_depths() -> list[int]:
    """Active candidate-pool depths, ascending and de-duplicated. Each must be
    at least max(RECALL_KS) or the rerank arm could not even fill the top-k."""
    depths = sorted({int(d) for d in C.RERANK_DEPTHS})
    bad = [d for d in depths if d < max(C.RECALL_KS)]
    if bad:
        raise ValueError(f"RERANK_DEPTHS {bad} below max(RECALL_KS)={max(C.RECALL_KS)}")
    return depths


def arms() -> list[str]:
    """Arm names: the dense baseline plus one rerank arm per pool depth."""
    return [BASELINE_ARM] + [f"rerank{d}" for d in rerank_depths()]


# --------------------------------------------------------------------------- #
# Cross-encoder loading / scoring (off-the-shelf weights only)
# --------------------------------------------------------------------------- #
_RERANKER = None
_RERANKER_NAME = None


def load_reranker(model_name: str | None = None):
    """Load the pretrained cross-encoder once (cached singleton). Stage 5 does
    no reranker training or fine-tuning — the weights are used as published."""
    global _RERANKER, _RERANKER_NAME

    from sentence_transformers import CrossEncoder

    name = model_name or C.RERANKER_MODEL
    if _RERANKER is None or _RERANKER_NAME != name:
        device = None
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        _RERANKER = CrossEncoder(name, max_length=int(C.RERANK_MAX_LENGTH),
                                 device=device)
        _RERANKER_NAME = name
        print(f"[rerank] loaded cross-encoder {name} on {device or 'auto'}")
    return _RERANKER


def score_pairs(pairs: list[tuple[str, str]]):
    """Cross-encoder relevance score per (question, chunk_text) pair, as a 1-D
    float32 numpy array. Only the ordering matters, so the model's default
    activation is used as-is (any monotonic transform reranks identically)."""
    import numpy as np

    if not pairs:
        return np.zeros(0, dtype="float32")
    scores = load_reranker().predict(
        pairs, batch_size=int(C.RERANK_BATCH_SIZE), show_progress_bar=False)
    return np.asarray(scores, dtype="float32")


def rerank_order(scores) -> list[int]:
    """Candidate order after reranking: score desc, ties broken by the original
    dense rank (ascending) — fully deterministic. ``scores[i]`` is the score of
    the candidate at dense rank ``i``."""
    import numpy as np

    s = np.asarray(scores, dtype="float32")
    return [int(i) for i in np.lexsort((np.arange(s.size), -s))]


# --------------------------------------------------------------------------- #
# One chunking config -> one row per arm (bge / rerank20 / rerank50)
# --------------------------------------------------------------------------- #
def _eval_config_rerank(
    method: str,
    docs: list[dict],
    questions: list[dict],
    *,
    scorer=None,
    boundary_probs_by_id: dict | None = None,
    fixed_size: int | None = None,
    fixed_overlap: int | None = None,
    semantic_policy: str | None = None,
    semantic_target_size: int | None = None,
    semantic_min_size: int | None = None,
    semantic_max_size: int | None = None,
    semantic_overlap: int | None = None,
) -> list[dict]:
    """Build one config's chunks, then score the dense baseline and every
    rerank arm on them.

    The dense index embeds the chunks once and retrieves one max-depth pool per
    question; each rerank arm reorders a prefix of that pool. ``scorer``
    defaults to the real cross-encoder (:func:`score_pairs`); tests inject a
    fake. Returns one row per arm, sharing every chunking field.
    """
    import numpy as np

    from rag_chunk import metrics, retrieval

    if scorer is None:
        scorer = score_pairs

    dense = retrieval.build_index_for_config(
        method, docs,
        fixed_size=fixed_size, fixed_overlap=fixed_overlap,
        semantic_policy=semantic_policy, semantic_target_size=semantic_target_size,
        semantic_min_size=semantic_min_size, semantic_max_size=semantic_max_size,
        semantic_overlap=semantic_overlap,
        boundary_probs_by_id=boundary_probs_by_id,
    )

    queries = [q["question"] for q in questions]
    depths = rerank_depths()
    maxk = max(C.RECALL_KS)
    pool = dense.search_chunks(queries, max(depths))     # dense order

    # Candidate-pool ceiling: reranking cannot recover a chunk the pool missed.
    pool_rec = metrics.recall_from_retrieved(pool, questions, tuple(depths))

    # Score each pool tier once — ranks (prev, d] across all questions in one
    # batched call. The tier for ranks 1..20 is exactly what a depth-20-only
    # run would score, so the cumulative per-depth timings stay honest.
    scores_by_q: list[list[float]] = [[] for _ in questions]
    seconds_at: dict[int, float] = {}
    pairs_at: dict[int, int] = {}
    prev, cum_s, cum_p = 0, 0.0, 0
    for d in depths:
        tier_pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        for qi, (qtext, cands) in enumerate(zip(queries, pool)):
            for c in cands[prev:d]:
                tier_pairs.append((qtext, c["text"]))
                owners.append(qi)
        t0 = time.perf_counter()
        tier_scores = np.asarray(scorer(tier_pairs), dtype="float32")
        cum_s += time.perf_counter() - t0
        cum_p += len(tier_pairs)
        seconds_at[d], pairs_at[d] = cum_s, cum_p
        for qi, s in zip(owners, tier_scores):
            scores_by_q[qi].append(float(s))
        prev = d

    retrieved_by_arm = {BASELINE_ARM: [cands[:maxk] for cands in pool]}
    for d in depths:
        rer = []
        for cands, qscores in zip(pool, scores_by_q):
            n = min(d, len(cands))
            order = rerank_order(qscores[:n])
            rer.append([cands[j] for j in order[:maxk]])
        retrieved_by_arm[f"rerank{d}"] = rer

    is_learned = method in ("bilstm", "transformer")
    base = {
        "method": method,
        "model_type": method if is_learned else "none",
        "boundary_embedding_model": C.BOUNDARY_EMBED_MODEL,
        "embedding_model": C.RETRIEVAL_EMBED_MODEL,
        "retrieval_embedding_model": C.RETRIEVAL_EMBED_MODEL,
        "avg_chunk_size": dense.avg_chunk_size(),
        "n_chunks": len(dense.chunk_texts),
        "fixed_size": fixed_size if method == "fixed" else None,
        "fixed_overlap": fixed_overlap if method == "fixed" else None,
        "semantic_policy": semantic_policy if is_learned else None,
        "semantic_target_size": semantic_target_size if is_learned else None,
        "semantic_min_size": semantic_min_size if is_learned else None,
        "semantic_max_size": semantic_max_size if is_learned else None,
        "semantic_overlap": semantic_overlap if is_learned else None,
        "boundary_threshold": None,          # target-size policy only in Stage 5
    }
    for d in depths:
        base[f"pool_recall@{d}"] = pool_rec["doc_constrained"][d]

    rows: list[dict] = []
    for name in arms():
        rec = metrics.recall_from_retrieved(
            retrieved_by_arm[name], questions, C.RECALL_KS)
        row = dict(base)
        row["arm"] = name
        if name == BASELINE_ARM:
            row["rerank_depth"] = None
            row["reranker_model"] = "none"
            row["rerank_seconds"] = None
            row["n_pairs"] = None
        else:
            d = int(name[len("rerank"):])
            row["rerank_depth"] = d
            row["reranker_model"] = C.RERANKER_MODEL
            row["rerank_seconds"] = seconds_at[d]
            row["n_pairs"] = pairs_at[d]
        for k in C.RECALL_KS:
            row[f"recall@{k}"] = rec["doc_constrained"][k]
        for k in C.RECALL_KS:
            row[f"recall_unconstrained@{k}"] = rec["unconstrained"][k]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Sweep driver
# --------------------------------------------------------------------------- #
def run_rerank_sweep(
    model=None,
    transformer_model=None,
    quick: bool = False,
    out_dir=None,
    docs: list[dict] | None = None,
    questions: list[dict] | None = None,
    scorer=None,
) -> list[dict]:
    """Run the Stage 5 rerank sweep and write all artifacts. Returns the rows.

    Same grids, boundary models, corpus and metric as the Stage 3/4 sweeps —
    the added dimension is the arm (bge / rerank20 / rerank50), so each
    chunking config yields one row per arm.
    """
    from rag_chunk import nq_data, sweep, training

    C.ensure_dirs()
    if model is None:
        model = training.load_model("bilstm")
    if docs is None or questions is None:
        docs, questions = nq_data.prepare_nq()

    learned = {"bilstm": model}
    if transformer_model is not None:
        learned["transformer"] = transformer_model

    fixed_sizes, target_sizes, overlaps = sweep._grids(quick)
    probs_by_type = sweep._precompute_boundary_probs(docs, learned)

    rows: list[dict] = []
    total = (len(fixed_sizes) + len(learned) * len(target_sizes)) * len(overlaps)
    done = 0
    for overlap in overlaps:
        for size in fixed_sizes:
            done += 1
            print(f"[rerank] ({done}/{total}) fixed        size={size} overlap={overlap}")
            rows += _eval_config_rerank(
                "fixed", docs, questions, scorer=scorer,
                fixed_size=size, fixed_overlap=overlap)
        for mtype in learned:
            for target in target_sizes:
                done += 1
                mn, mx = sweep._semantic_window(target)
                print(f"[rerank] ({done}/{total}) {mtype:<12} target={target} "
                      f"(min={mn},max={mx}) overlap={overlap}")
                rows += _eval_config_rerank(
                    mtype, docs, questions, scorer=scorer,
                    boundary_probs_by_id=probs_by_type[mtype],
                    semantic_policy="target", semantic_target_size=target,
                    semantic_min_size=mn, semantic_max_size=mx,
                    semantic_overlap=overlap)

    out_dir = Path(out_dir) if out_dir is not None else C.RESULTS_LATEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best = best_configs(rows)
    matched = arm_matched_table(rows)
    _write_rerank_csv(rows, out_dir / C.RERANK_SWEEP_CSV)
    _write_best_json(best, out_dir / C.RERANK_BEST_JSON)
    _write_matched_csv(matched, out_dir / C.RERANK_MATCHED_CSV)
    plot_recall_vs_size_by_arm(rows, out_dir / C.RERANK_SCATTER_PNG,
                               n_questions=len(questions))
    plot_arm_comparison(rows, out_dir / C.RERANK_COMPARISON_PNG)
    _print_summary(rows, best, out_dir)
    return rows


# --------------------------------------------------------------------------- #
# Best configs + matched table (pure Python — no heavy deps)
# --------------------------------------------------------------------------- #
def best_configs(rows: list[dict]) -> dict:
    """Best row per arm plus the overall best, using the same ranking as the
    Stage 1-4 sweeps (max recall@5, then @3, @1, then min avg chunk size)."""
    from rag_chunk import sweep

    per = {name: sweep.select_best_config([r for r in rows if r["arm"] == name])
           for name in arms()}
    return {
        "ranking": ("per arm: max doc-constrained recall@5, then recall@3, "
                    "then recall@1, then min avg_chunk_size"),
        "per_arm": per,
        "overall": sweep.select_best_config(rows),
    }


def arm_matched_table(rows: list[dict]) -> list[dict]:
    """One row per chunking config with all arms side by side, the rerank-vs-bge
    deltas at every k, and the candidate-pool ceiling per depth."""
    ks = sorted(C.RECALL_KS)
    depths = rerank_depths()
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault(config_key(r), {})[r["arm"]] = r
    table: list[dict] = []
    for group in by_key.values():           # insertion order = sweep order
        any_row = next(iter(group.values()))
        out = {
            "method": any_row["method"],
            "chunk_config": config_label(any_row),
            "avg_chunk_size": any_row["avg_chunk_size"],
            "n_chunks": any_row["n_chunks"],
        }
        for d in depths:
            out[f"pool_recall@{d}"] = any_row.get(f"pool_recall@{d}")
        for k in ks:
            for name in arms():
                out[f"{name}_recall@{k}"] = (
                    group[name][f"recall@{k}"] if name in group else None)
            for d in depths:
                nm = f"rerank{d}"
                if nm in group and BASELINE_ARM in group:
                    out[f"rerank{d}_minus_bge@{k}"] = (
                        group[nm][f"recall@{k}"] - group[BASELINE_ARM][f"recall@{k}"])
        table.append(out)
    return table


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
_ARM_PALETTE = ["#1f77b4", "#9467bd", "#d62728", "#8c564b", "#e377c2"]


def _arm_colors() -> dict[str, str]:
    return {name: _ARM_PALETTE[i % len(_ARM_PALETTE)]
            for i, name in enumerate(arms())}


def plot_recall_vs_size_by_arm(rows: list[dict], path,
                               n_questions: int | None = None) -> None:
    """Scatter Recall@1 vs average chunk size, coloured by arm, with a per-arm
    trend line. Recall@1 is the Stage 5 goal metric — reranking targets the top
    of the ranking, where the dense baseline has the most headroom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    k1 = min(C.RECALL_KS)
    colors = _arm_colors()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = False
    for name in arms():
        sub = [r for r in rows if r["arm"] == name]
        if not sub:
            continue
        plotted = True
        xs = np.array([r["avg_chunk_size"] for r in sub], dtype=float)
        ys = np.array([r[f"recall@{k1}"] for r in sub], dtype=float)
        ax.scatter(xs, ys, c=colors[name], s=60,
                   edgecolor="white", linewidth=0.6, zorder=3)
        if xs.size >= 2 and xs.std() > 0:
            slope, intercept = np.polyfit(xs, ys, 1)
            xline = np.array([xs.min(), xs.max()])
            ax.plot(xline, slope * xline + intercept,
                    color=colors[name], linestyle="--", alpha=0.7, zorder=2)
    if not plotted:
        plt.close(fig)
        return
    if n_questions:
        ys_all = np.array([r[f"recall@{k1}"] for r in rows], dtype=float)
        pbar = float(ys_all.mean())
        se = (pbar * (1 - pbar) / n_questions) ** 0.5
        ax.text(0.02, 0.98, f"1 SE ≈ {se:.3f}  (n={n_questions} questions)",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    handles = [Line2D([0], [0], marker="o", linestyle="--",
                      color=colors[name], label=name)
               for name in arms() if any(r["arm"] == name for r in rows)]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_xlabel("Average chunk size (sentences)")
    ax.set_ylabel(f"Doc-constrained Recall@{k1}")
    ax.set_title("Cross-encoder reranking (NQ): BGE vs reranked top-k")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[rerank] wrote {path}")


def plot_arm_comparison(rows: list[dict], path) -> None:
    """Grouped Recall@k bars for the best config of each arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from rag_chunk.sweep import _rank_key

    colors = _arm_colors()
    present = []
    for name in arms():
        sub = [r for r in rows if r["arm"] == name]
        if sub:
            present.append((name, max(sub, key=_rank_key)))
    if len(present) < 2:
        return
    ks = sorted(C.RECALL_KS)
    x = np.arange(len(ks))
    n = len(present)
    w = min(0.38, 0.8 / n)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (name, row) in enumerate(present):
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, [row[f"recall@{k}"] for k in ks], w,
                      color=colors[name],
                      label=f"{name} ({row['method']}, {config_label(row)})")
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, [f"Recall@{k}" for k in ks])
    ax.set_ylabel("Recall (doc-constrained)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Best config per arm (ranked by Recall@%d)" % max(ks))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[rerank] wrote {path}")


# --------------------------------------------------------------------------- #
# Writers / formatting
# --------------------------------------------------------------------------- #
def _rerank_columns() -> list[str]:
    from rag_chunk.sweep import _sweep_columns

    cols = ["arm"] + _sweep_columns()
    cols += ["rerank_depth", "reranker_model", "rerank_seconds", "n_pairs"]
    cols += [f"pool_recall@{d}" for d in rerank_depths()]
    return cols


def _write_rerank_csv(rows: list[dict], path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _rerank_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[rerank] wrote {path}  ({len(rows)} rows)")


def _matched_columns() -> list[str]:
    cols = ["method", "chunk_config", "avg_chunk_size", "n_chunks"]
    cols += [f"pool_recall@{d}" for d in rerank_depths()]
    for k in sorted(C.RECALL_KS):
        cols += [f"{name}_recall@{k}" for name in arms()]
        cols += [f"rerank{d}_minus_bge@{k}" for d in rerank_depths()]
    return cols


def _write_matched_csv(table: list[dict], path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _matched_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[rerank] wrote {path}  ({len(table)} rows)")


def _write_best_json(best: dict, path) -> None:
    from rag_chunk.sweep import _json_default

    Path(path).write_text(
        json.dumps(best, indent=2, default=_json_default), encoding="utf-8")
    print(f"[rerank] wrote {path}")


def _print_summary(rows: list[dict], best: dict, out_dir) -> None:
    ks = sorted(C.RECALL_KS)
    depths = rerank_depths()
    n_configs = len(rows) // len(arms()) if rows else 0
    print(f"\n[rerank] {len(rows)} rows scored "
          f"({n_configs} configs x {len(arms())} arms)")

    # Candidate-pool ceiling: reranking cannot beat these.
    bge_rows = [r for r in rows if r["arm"] == BASELINE_ARM]
    if bge_rows:
        pools = "  ".join(
            f"@{d}={sum(r[f'pool_recall@{d}'] for r in bge_rows) / len(bge_rows):.4f}"
            for d in depths)
        print(f"[rerank] mean BGE candidate-pool recall (rerank ceiling): {pools}")

    # Mean rerank-vs-bge deltas across matched configs, per k.
    matched = arm_matched_table(rows)
    for d in depths:
        cells = []
        for k in ks:
            vals = [m[f"rerank{d}_minus_bge@{k}"] for m in matched
                    if m.get(f"rerank{d}_minus_bge@{k}") is not None]
            if vals:
                cells.append(f"R@{k} {sum(vals) / len(vals):+.4f}")
        if cells:
            print(f"[rerank] mean delta rerank{d} - bge over "
                  f"{len(matched)} configs: " + "  ".join(cells))

    # Rerank cost (per config; questions share one batched scoring pass).
    for d in depths:
        sub = [r for r in rows if r["arm"] == f"rerank{d}" and r.get("rerank_seconds")]
        if not sub:
            continue
        mean_s = sum(r["rerank_seconds"] for r in sub) / len(sub)
        mean_p = sum(r["n_pairs"] for r in sub) / len(sub)
        per_pair = f"; {mean_s / mean_p * 1000:.1f} ms/pair" if mean_p else ""
        print(f"[rerank] rerank{d} cost: mean {mean_s:.1f}s per config "
              f"({mean_p:.0f} pairs{per_pair})")

    head = (f"{'arm':9} {'method':11} {'config':>28} "
            + " ".join(f"R@{k:<4}" for k in ks))
    print(f"\n[rerank] best config per arm (ranked by Recall@{max(ks)}):")
    print(head)
    print("-" * len(head))
    for name in arms():
        b = best["per_arm"].get(name, {}).get("config")
        if b:
            cells = " ".join(f"{b[f'recall@{k}']:.3f}" for k in ks)
            print(f"{name:9} {b['method']:11} {config_label(b):>28} {cells}")
    print(f"[rerank] artifacts -> {out_dir}\n")
