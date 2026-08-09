"""Stage 4 — hybrid retrieval ablation: BM25 + dense (BGE) + RRF fusion.

Reuses the Stage 3 protocol unchanged — the same NQ corpus and questions, the
same fixed / BiLSTM / Transformer chunking grids, the same boundary models and
weights, the same BGE dense retrieval — and adds two retrievers over the
*identical* chunks:

* ``bm25`` — classic lexical Okapi BM25 (pure numpy, no extra dependency);
* ``rrf``  — Reciprocal Rank Fusion of the BGE and BM25 rankings.

For every chunking config the sweep builds ONE set of chunks, embeds it once
for the dense FAISS index, builds one BM25 index over the same texts, and
scores all three retrievers with the same doc-constrained Recall@k — so within
a config the only thing that differs between the three rows is the retriever.

Artifacts (written under ``RESULTS_LATEST_DIR`` by :func:`run_hybrid_sweep`):

    hybrid_sweep_results.csv        one row per (retriever, method, config)
    hybrid_best_config.json         best row per retriever + overall best
    hybrid_retriever_matched.csv    bge / bm25 / rrf side-by-side per config
    hybrid_recall_vs_chunk_size.png Recall@5 vs avg chunk size by retriever
    hybrid_retriever_comparison.png best config per retriever, grouped bars

Heavy deps (numpy / matplotlib / faiss / torch) are imported lazily inside
functions, matching the rest of the package.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import config as C

RETRIEVERS = ("bge", "bm25", "rrf")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric word tokens — applied identically to chunks and
    queries. Deliberately simple (no stemming / stopwords): a standard, fully
    deterministic BM25 baseline with no extra dependency."""
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# BM25 (Okapi) over the same chunk id space as the dense index
# --------------------------------------------------------------------------- #
class BM25Index:
    """Okapi BM25 with the Lucene idf ``log(1 + (N - df + 0.5) / (df + 0.5))``
    (always positive) and ``C.BM25_K1`` / ``C.BM25_B`` read at build time.

    Chunk ids are the row indices of ``chunk_texts`` — the same id space as the
    dense ``ChunkIndex`` built from the same texts, so RRF can fuse the two.
    """

    def __init__(self, chunk_texts: list[str]) -> None:
        import numpy as np

        self.k1 = float(C.BM25_K1)
        self.b = float(C.BM25_B)
        self.n_docs = len(chunk_texts)
        docs = [tokenize(t) for t in chunk_texts]
        self.doc_len = np.array([len(d) for d in docs], dtype="float32")
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, toks in enumerate(docs):
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                ids, tfs = postings.setdefault(t, ([], []))
                ids.append(i)
                tfs.append(c)
        self.postings = {
            t: (np.asarray(ids, dtype="int64"), np.asarray(tfs, dtype="float32"))
            for t, (ids, tfs) in postings.items()
        }
        self.idf = {
            t: float(np.log(1.0 + (self.n_docs - len(ids) + 0.5) / (len(ids) + 0.5)))
            for t, (ids, _) in self.postings.items()
        }

    def search_ids(self, queries: list[str], k: int) -> list[list[int]]:
        """Top-k chunk ids per query, deterministic (score desc, then id asc).
        A repeated query term contributes once per occurrence (query tf)."""
        import numpy as np

        k = min(k, self.n_docs)
        # per-doc length normalisation, shared across queries
        denom_norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        out: list[list[int]] = []
        for q in queries:
            scores = np.zeros(self.n_docs, dtype="float32")
            for t in tokenize(q):
                post = self.postings.get(t)
                if post is None:
                    continue
                ids, tf = post
                scores[ids] += self.idf[t] * tf * (self.k1 + 1.0) / (tf + denom_norm[ids])
            order = np.lexsort((np.arange(self.n_docs), -scores))
            out.append([int(i) for i in order[:k]])
        return out


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def rrf_fuse(ranked_lists: list[list[int]], k: int, rrf_k: int | None = None) -> list[int]:
    """Fuse ranked id lists: ``score(d) = sum over lists of 1/(rrf_k + rank)``
    with 1-based ranks. Ties break on the best rank any list gave the id, then
    on the id itself, so fusion is fully deterministic. Returns the top-k ids.
    """
    if rrf_k is None:
        rrf_k = C.RRF_K
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if rank < best_rank.get(cid, 1 << 30):
                best_rank[cid] = rank
    fused = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    return fused[:k]


# --------------------------------------------------------------------------- #
# Config keys / labels (shared with the Stage 4 script's Stage 3 check)
# --------------------------------------------------------------------------- #
def _num(val):
    """Int from a native or CSV-string cell; None for blank."""
    if val in ("", None):
        return None
    return int(float(val))


def config_key(row: dict) -> tuple:
    """Chunking-config identity of a row — works on both native sweep rows and
    rows read back from a CSV (string cells)."""
    method = str(row["method"]).strip()
    if method == "fixed":
        return ("fixed", _num(row.get("fixed_size")), _num(row.get("fixed_overlap")))
    return (
        method,
        str(row.get("semantic_policy") or "target"),
        _num(row.get("semantic_target_size")),
        _num(row.get("semantic_min_size")),
        _num(row.get("semantic_max_size")),
        _num(row.get("semantic_overlap")),
    )


def config_label(row: dict) -> str:
    method = str(row["method"]).strip()
    if method == "fixed":
        return (f"fixed_size={_num(row.get('fixed_size'))},"
                f"overlap={_num(row.get('fixed_overlap'))}")
    return (
        f"target={_num(row.get('semantic_target_size'))},"
        f"min={_num(row.get('semantic_min_size'))},"
        f"max={_num(row.get('semantic_max_size'))},"
        f"overlap={_num(row.get('semantic_overlap'))}"
    )


# --------------------------------------------------------------------------- #
# One chunking config -> three scored rows (bge / bm25 / rrf)
# --------------------------------------------------------------------------- #
def _retrieved_chunks(index, ids_lists: list[list[int]]) -> list[list[dict]]:
    """Map per-query id lists to the ``{'text','doc_id'}`` dicts the metric eats."""
    return [
        [{"text": index.chunk_texts[i], "doc_id": index.chunk_doc_ids[i]} for i in ids]
        for ids in ids_lists
    ]


def _eval_config_hybrid(
    method: str,
    docs: list[dict],
    questions: list[dict],
    *,
    boundary_probs_by_id: dict | None = None,
    fixed_size: int | None = None,
    fixed_overlap: int | None = None,
    semantic_policy: str | None = None,
    semantic_target_size: int | None = None,
    semantic_min_size: int | None = None,
    semantic_max_size: int | None = None,
    semantic_overlap: int | None = None,
) -> list[dict]:
    """Build one config's chunks, then score all three retrievers on them.

    The dense index embeds the chunks once (the expensive part); BM25 indexes
    the identical texts; RRF fuses the two depth-``C.HYBRID_FUSE_DEPTH``
    rankings. Returns one row per retriever, sharing every chunking field.
    """
    from rag_chunk import metrics, retrieval

    dense = retrieval.build_index_for_config(
        method, docs,
        fixed_size=fixed_size, fixed_overlap=fixed_overlap,
        semantic_policy=semantic_policy, semantic_target_size=semantic_target_size,
        semantic_min_size=semantic_min_size, semantic_max_size=semantic_max_size,
        semantic_overlap=semantic_overlap,
        boundary_probs_by_id=boundary_probs_by_id,
    )
    bm25 = BM25Index(dense.chunk_texts)

    queries = [q["question"] for q in questions]
    maxk = max(C.RECALL_KS)
    depth = max(int(C.HYBRID_FUSE_DEPTH), maxk)
    bge_ids = dense.search_ids(queries, depth)
    bm25_ids = bm25.search_ids(queries, depth)
    ids_by_retriever = {
        "bge": [ids[:maxk] for ids in bge_ids],
        "bm25": [ids[:maxk] for ids in bm25_ids],
        "rrf": [rrf_fuse([g, b], maxk) for g, b in zip(bge_ids, bm25_ids)],
    }

    is_learned = method in ("bilstm", "transformer")
    base = {
        "method": method,
        "model_type": method if is_learned else "none",
        "boundary_embedding_model": C.BOUNDARY_EMBED_MODEL,
        "avg_chunk_size": dense.avg_chunk_size(),
        "n_chunks": len(dense.chunk_texts),
        "fixed_size": fixed_size if method == "fixed" else None,
        "fixed_overlap": fixed_overlap if method == "fixed" else None,
        "semantic_policy": semantic_policy if is_learned else None,
        "semantic_target_size": semantic_target_size if is_learned else None,
        "semantic_min_size": semantic_min_size if is_learned else None,
        "semantic_max_size": semantic_max_size if is_learned else None,
        "semantic_overlap": semantic_overlap if is_learned else None,
        "boundary_threshold": None,          # target-size policy only in Stage 4
    }
    rows: list[dict] = []
    for name in RETRIEVERS:
        rec = metrics.recall_from_retrieved(
            _retrieved_chunks(dense, ids_by_retriever[name]), questions, C.RECALL_KS)
        row = dict(base)
        row["retriever"] = name
        dense_model = "none" if name == "bm25" else C.RETRIEVAL_EMBED_MODEL
        row["embedding_model"] = dense_model
        row["retrieval_embedding_model"] = dense_model
        row["rrf_k"] = int(C.RRF_K) if name == "rrf" else None
        row["fuse_depth"] = depth if name == "rrf" else None
        for k in C.RECALL_KS:
            row[f"recall@{k}"] = rec["doc_constrained"][k]
        for k in C.RECALL_KS:
            row[f"recall_unconstrained@{k}"] = rec["unconstrained"][k]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Sweep driver
# --------------------------------------------------------------------------- #
def run_hybrid_sweep(
    model=None,
    transformer_model=None,
    quick: bool = False,
    out_dir=None,
    docs: list[dict] | None = None,
    questions: list[dict] | None = None,
) -> list[dict]:
    """Run the Stage 4 hybrid sweep and write all artifacts. Returns the rows.

    Same grids, boundary models, corpus and metric as the Stage 3 sweep — the
    added dimension is the retriever (bge / bm25 / rrf), so each chunking
    config yields three rows instead of one.
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
            print(f"[hybrid] ({done}/{total}) fixed        size={size} overlap={overlap}")
            rows += _eval_config_hybrid(
                "fixed", docs, questions,
                fixed_size=size, fixed_overlap=overlap)
        for mtype in learned:
            for target in target_sizes:
                done += 1
                mn, mx = sweep._semantic_window(target)
                print(f"[hybrid] ({done}/{total}) {mtype:<12} target={target} "
                      f"(min={mn},max={mx}) overlap={overlap}")
                rows += _eval_config_hybrid(
                    mtype, docs, questions,
                    boundary_probs_by_id=probs_by_type[mtype],
                    semantic_policy="target", semantic_target_size=target,
                    semantic_min_size=mn, semantic_max_size=mx,
                    semantic_overlap=overlap)

    out_dir = Path(out_dir) if out_dir is not None else C.RESULTS_LATEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best = best_configs(rows)
    matched = retriever_matched_table(rows)
    _write_hybrid_csv(rows, out_dir / C.HYBRID_SWEEP_CSV)
    _write_best_json(best, out_dir / C.HYBRID_BEST_JSON)
    _write_matched_csv(matched, out_dir / C.HYBRID_MATCHED_CSV)
    plot_recall_vs_size_by_retriever(rows, out_dir / C.HYBRID_SCATTER_PNG,
                                     n_questions=len(questions))
    plot_retriever_comparison(rows, out_dir / C.HYBRID_RETRIEVER_PLOT_PNG)
    _print_summary(rows, best, out_dir)
    return rows


# --------------------------------------------------------------------------- #
# Best configs + matched table (pure Python — no heavy deps)
# --------------------------------------------------------------------------- #
def best_configs(rows: list[dict]) -> dict:
    """Best row per retriever plus the overall best, using the same ranking as
    the Stage 1-3 sweeps (max recall@5, then @3, @1, then min avg chunk size)."""
    from rag_chunk import sweep

    per = {name: sweep.select_best_config([r for r in rows if r["retriever"] == name])
           for name in RETRIEVERS}
    return {
        "ranking": ("per retriever: max doc-constrained recall@5, then recall@3, "
                    "then recall@1, then min avg_chunk_size"),
        "per_retriever": per,
        "overall": sweep.select_best_config(rows),
    }


def retriever_matched_table(rows: list[dict]) -> list[dict]:
    """One row per chunking config with the three retrievers side by side and
    the fusion deltas (rrf minus each single retriever)."""
    ks = sorted(C.RECALL_KS)
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault(config_key(r), {})[r["retriever"]] = r
    table: list[dict] = []
    for group in by_key.values():           # insertion order = sweep order
        any_row = next(iter(group.values()))
        out = {
            "method": any_row["method"],
            "chunk_config": config_label(any_row),
            "avg_chunk_size": any_row["avg_chunk_size"],
            "n_chunks": any_row["n_chunks"],
        }
        for k in ks:
            for name in RETRIEVERS:
                out[f"{name}_recall@{k}"] = (
                    group[name][f"recall@{k}"] if name in group else None)
            if "rrf" in group and "bge" in group:
                out[f"rrf_minus_bge@{k}"] = (
                    group["rrf"][f"recall@{k}"] - group["bge"][f"recall@{k}"])
            if "rrf" in group and "bm25" in group:
                out[f"rrf_minus_bm25@{k}"] = (
                    group["rrf"][f"recall@{k}"] - group["bm25"][f"recall@{k}"])
        table.append(out)
    return table


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
_RETRIEVER_COLORS = {"bge": "#1f77b4", "bm25": "#ff7f0e", "rrf": "#2ca02c"}


def plot_recall_vs_size_by_retriever(rows: list[dict], path,
                                     n_questions: int | None = None) -> None:
    """Scatter Recall@max(k) vs average chunk size, coloured by retriever, with
    a per-retriever trend line — the retriever separation across all configs in
    one glance (the Stage 4 analogue of the Stage 2 size scatter)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    topk = max(C.RECALL_KS)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = False
    for name in RETRIEVERS:
        sub = [r for r in rows if r["retriever"] == name]
        if not sub:
            continue
        plotted = True
        xs = np.array([r["avg_chunk_size"] for r in sub], dtype=float)
        ys = np.array([r[f"recall@{topk}"] for r in sub], dtype=float)
        ax.scatter(xs, ys, c=_RETRIEVER_COLORS[name], s=60,
                   edgecolor="white", linewidth=0.6, zorder=3)
        if xs.size >= 2 and xs.std() > 0:
            slope, intercept = np.polyfit(xs, ys, 1)
            xline = np.array([xs.min(), xs.max()])
            ax.plot(xline, slope * xline + intercept,
                    color=_RETRIEVER_COLORS[name], linestyle="--",
                    alpha=0.7, zorder=2)
    if not plotted:
        plt.close(fig)
        return
    if n_questions:
        ys_all = np.array([r[f"recall@{topk}"] for r in rows], dtype=float)
        pbar = float(ys_all.mean())
        se = (pbar * (1 - pbar) / n_questions) ** 0.5
        ax.text(0.02, 0.98, f"1 SE ≈ {se:.3f}  (n={n_questions} questions)",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    handles = [Line2D([0], [0], marker="o", linestyle="--",
                      color=_RETRIEVER_COLORS[name], label=name)
               for name in RETRIEVERS if any(r["retriever"] == name for r in rows)]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_xlabel("Average chunk size (sentences)")
    ax.set_ylabel(f"Doc-constrained Recall@{topk}")
    ax.set_title("Hybrid retrieval (NQ): BGE vs BM25 vs RRF across chunk configs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[hybrid] wrote {path}")


def plot_retriever_comparison(rows: list[dict], path) -> None:
    """Grouped Recall@k bars for the best config of each retriever."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from rag_chunk.sweep import _rank_key

    present = []
    for name in RETRIEVERS:
        sub = [r for r in rows if r["retriever"] == name]
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
                      color=_RETRIEVER_COLORS[name],
                      label=f"{name} ({row['method']}, {config_label(row)})")
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, [f"Recall@{k}" for k in ks])
    ax.set_ylabel("Recall (doc-constrained)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Best config per retriever (ranked by Recall@%d)" % max(ks))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[hybrid] wrote {path}")


# --------------------------------------------------------------------------- #
# Writers / formatting
# --------------------------------------------------------------------------- #
def _hybrid_columns() -> list[str]:
    from rag_chunk.sweep import _sweep_columns

    return ["retriever"] + _sweep_columns() + ["rrf_k", "fuse_depth"]


def _write_hybrid_csv(rows: list[dict], path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _hybrid_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[hybrid] wrote {path}  ({len(rows)} rows)")


def _matched_columns() -> list[str]:
    cols = ["method", "chunk_config", "avg_chunk_size", "n_chunks"]
    for k in sorted(C.RECALL_KS):
        cols += [f"{name}_recall@{k}" for name in RETRIEVERS]
        cols += [f"rrf_minus_bge@{k}", f"rrf_minus_bm25@{k}"]
    return cols


def _write_matched_csv(table: list[dict], path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _matched_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[hybrid] wrote {path}  ({len(table)} rows)")


def _write_best_json(best: dict, path) -> None:
    from rag_chunk.sweep import _json_default

    Path(path).write_text(
        json.dumps(best, indent=2, default=_json_default), encoding="utf-8")
    print(f"[hybrid] wrote {path}")


def _print_summary(rows: list[dict], best: dict, out_dir) -> None:
    from rag_chunk.sweep import _rank_key

    ks = sorted(C.RECALL_KS)
    topk = max(ks)
    print(f"\n[hybrid] {len(rows)} rows scored "
          f"({len(rows) // len(RETRIEVERS)} configs x {len(RETRIEVERS)} retrievers) "
          f"— top 10 by doc-constrained Recall@{topk}:\n")
    head = (f"{'retriever':9} {'method':11} {'config':>28} {'avg':>6} "
            + " ".join(f"R@{k:<4}" for k in ks))
    print(head)
    print("-" * len(head))
    for r in sorted(rows, key=_rank_key, reverse=True)[:10]:
        cells = " ".join(f"{r[f'recall@{k}']:.3f}" for k in ks)
        print(f"{r['retriever']:9} {r['method']:11} {config_label(r):>28} "
              f"{r['avg_chunk_size']:6.2f} {cells}")
    for name in RETRIEVERS:
        b = best["per_retriever"].get(name, {}).get("config")
        if b:
            print(f"[hybrid] best {name:5}: {b['method']} {config_label(b)} "
                  + " ".join(f"R@{k}={b[f'recall@{k}']:.3f}" for k in ks))
    print(f"[hybrid] artifacts -> {out_dir}\n")
