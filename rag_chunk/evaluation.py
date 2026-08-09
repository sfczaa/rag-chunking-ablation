"""Phase 5 — compare BiLSTM chunking vs fixed-size baseline on NQ.

Produces:
  * a printed comparison table,
  * ``results/recall_comparison.csv``,
  * ``results/recall_comparison.png`` (grouped Recall@k bar chart).
"""

from __future__ import annotations

import csv

import config as C


def evaluate_all(model, rebuild: bool = False) -> dict:
    from rag_chunk import metrics, nq_data, retrieval

    C.ensure_dirs()
    bi_prefix = retrieval.index_prefix("bilstm")
    fx_prefix = retrieval.index_prefix("fixed")

    # NQ corpus first, so we can decide whether the cached indices are still valid
    # for the *current* config (threshold, chunk size, model weights, corpus size).
    docs, questions = nq_data.prepare_nq()
    want = retrieval.current_signature(docs, questions, C.BOUNDARY_THRESHOLD)
    have = retrieval.load_manifest()
    cache_ok = (
        retrieval.ChunkIndex.exists(bi_prefix)
        and retrieval.ChunkIndex.exists(fx_prefix)
        and have == want
    )

    if not rebuild and cache_ok:
        bilstm = retrieval.ChunkIndex.load(bi_prefix)
        fixed = retrieval.ChunkIndex.load(fx_prefix)
        print("[eval] loaded cached FAISS indices (manifest matches config)")
    else:
        if not rebuild and have is not None and have != want:
            print("[eval] config changed since indices were built — rebuilding")
        # pass threshold explicitly so the rebuilt indices match `want`'s signature
        built = retrieval.build_indexes(model, threshold=C.BOUNDARY_THRESHOLD)
        bilstm, fixed, questions = built["bilstm"], built["fixed"], built["questions"]

    recall_bilstm = metrics.recall_at_k(bilstm, questions, C.RECALL_KS)
    recall_fixed = metrics.recall_at_k(fixed, questions, C.RECALL_KS)
    boundary = metrics.boundary_f1(model, split="test")

    results = {
        "n_questions": len(questions),
        "fixed": {
            "recall": recall_fixed["doc_constrained"],
            "recall_unconstrained": recall_fixed["unconstrained"],
            "avg_chunk_size": fixed.avg_chunk_size(),
            "n_chunks": len(fixed.chunk_texts),
        },
        "bilstm": {
            "recall": recall_bilstm["doc_constrained"],
            "recall_unconstrained": recall_bilstm["unconstrained"],
            "avg_chunk_size": bilstm.avg_chunk_size(),
            "n_chunks": len(bilstm.chunk_texts),
        },
        "boundary_f1": boundary,
    }

    _print_table(results)
    _write_csv(results)
    _plot(results)
    return results


def _print_table(r: dict) -> None:
    ks = C.RECALL_KS
    header = "| Method | " + " | ".join(f"Recall@{k}" for k in ks) + " | Avg Chunk Size |"
    sep = "|" + "---|" * (len(ks) + 2)
    print(f"\nEvaluated on {r['n_questions']} NQ questions")
    print("Recall@k = doc-constrained (hit must come from the question's gold "
          "document)\n")
    print(header)
    print(sep)
    for name, key in (("Fixed-size (baseline)", "fixed"), ("BiLSTM chunking (ours)", "bilstm")):
        row = r[key]
        cells = " | ".join(f"{row['recall'][k]:.3f}" for k in ks)
        print(f"| {name} | {cells} | {row['avg_chunk_size']:.2f} |")
    print("\n(reference) unconstrained Recall@k — answer may match any document, "
          "so these over-count:")
    for name, key in (("Fixed-size", "fixed"), ("BiLSTM", "bilstm")):
        cells = " | ".join(f"{r[key]['recall_unconstrained'][k]:.3f}" for k in ks)
        print(f"  {name:10s}: {cells}")
    b = r["boundary_f1"]
    print(f"\nWikipedia test-set Boundary F1: "
          f"P={b['precision']:.3f}  R={b['recall']:.3f}  F1={b['f1']:.3f}\n")


def _write_csv(r: dict) -> None:
    ks = C.RECALL_KS
    with open(C.RESULTS_TABLE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric"] + [f"recall@{k}" for k in ks]
                   + ["avg_chunk_size", "n_chunks"])
        for name, key in (("fixed", "fixed"), ("bilstm", "bilstm")):
            row = r[key]
            w.writerow([name, "doc_constrained"]
                       + [f"{row['recall'][k]:.4f}" for k in ks]
                       + [f"{row['avg_chunk_size']:.3f}", row["n_chunks"]])
            w.writerow([name, "unconstrained"]
                       + [f"{row['recall_unconstrained'][k]:.4f}" for k in ks]
                       + [f"{row['avg_chunk_size']:.3f}", row["n_chunks"]])
    print(f"[eval] wrote {C.RESULTS_TABLE}")


def _plot(r: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ks = C.RECALL_KS
    x = np.arange(len(ks))
    width = 0.38
    fixed_vals = [r["fixed"]["recall"][k] for k in ks]
    bilstm_vals = [r["bilstm"]["recall"][k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width / 2, fixed_vals, width, label="Fixed-size (baseline)")
    b2 = ax.bar(x + width / 2, bilstm_vals, width, label="BiLSTM chunking (ours)")
    ax.set_xticks(x, [f"Recall@{k}" for k in ks])
    ax.set_ylabel("Recall (doc-constrained)")
    ax.set_ylim(0, 1.0)
    ax.set_title("RAG retrieval: learned vs fixed-size chunking (NQ)")
    ax.legend()
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.RESULTS_FIGURE, dpi=150)
    plt.close(fig)
    print(f"[eval] wrote {C.RESULTS_FIGURE}")
