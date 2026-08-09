"""Stage 8 addendum - does the fine-tuned reranker's gain TRANSFER cross-dataset?

Stage 8 fine-tuned ``BAAI/bge-reranker-base`` on NQ-train hard negatives and
measured ft-ots ΔR@1 = +0.087..+0.107 on the NQ (Stage 6) bench. That gain is
in-domain (NQ train -> NQ val). This script answers the one caveat we always
state: **does it survive a change of evaluation dataset?**

It re-runs the EXACT Stage 8 final protocol -- the same 5 STAGE6_RERANK_CONFIGS,
the same three arms sharing one BGE top-20 pool (``bge`` / off-the-shelf
``rerank20`` / fine-tuned ``rerank20_ft``), the same fine-tuned checkpoint -- but
on the **Stage 7 TriviaQA rc.wikipedia bench** instead of NQ. Only the eval
dataset changes, so a difference in the ft-ots delta is a transfer effect, not a
protocol difference. The in-domain Stage 8 deltas are read from the archive and
printed next to the cross-dataset ones.

GPU + Drive assets required (run on Colab):
    - fine-tuned reranker at ``models/bge_reranker_ft/final`` (Stage 8),
    - the TriviaQA bench cache (Stage 7 built it for the same n-questions),
    - the archived ``stage8/final`` for the in-domain comparison.

Honest reading, stated in the output: TriviaQA gold is distant-supervised
(weaker than NQ's annotated gold) and n is only ~300, so the 2 SE band is wide
(~0.057 at R@1) and absolute recall is NOT comparable to NQ. The question here
is directional: does ft still beat the off-the-shelf reranker on a dataset it
was never tuned on?

Usage:
    python scripts/21_reranker_transfer.py                 # n = STAGE7_N_QUESTIONS
    python scripts/21_reranker_transfer.py --n-questions 300
    python scripts/21_reranker_transfer.py --fresh         # discard checkpoint
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

DEPTH = 20                          # matches Stage 5/6/8 rerank20
ARM_BGE, ARM_OTS, ARM_FT = "bge", "rerank20", "rerank20_ft"
CHECKPOINT_FILE = "stage8_transfer_checkpoint.jsonl"
RESULTS_CSV = "stage8_transfer_results.csv"
MATCHED_CSV = "stage8_transfer_matched.csv"
SUMMARY_MD = "stage8_transfer_summary.md"
DELTA_PNG = "stage8_transfer_delta.png"


def _default_ft_dir() -> pathlib.Path:
    return C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"


def _stage_final_dir(stage: str) -> pathlib.Path:
    return C.RESULTS_DIR / stage / "final"


# --------------------------------------------------------------------------- #
# One config -> three rows (bge / rerank20 / rerank20_ft), one shared pool.
# Faithful to scripts/18_eval_reranker_ft.py::_eval_config_arms.
# --------------------------------------------------------------------------- #
def _eval_config_arms(method, docs, questions, *, scorers,
                      boundary_probs_by_id=None, fixed_size=None,
                      fixed_overlap=None, semantic_policy=None,
                      semantic_target_size=None, semantic_min_size=None,
                      semantic_max_size=None, semantic_overlap=None) -> list[dict]:
    from rag_chunk import metrics, retrieval
    from rag_chunk.rerank import rerank_order

    dense = retrieval.build_index_for_config(
        method, docs,
        fixed_size=fixed_size, fixed_overlap=fixed_overlap,
        semantic_policy=semantic_policy,
        semantic_target_size=semantic_target_size,
        semantic_min_size=semantic_min_size,
        semantic_max_size=semantic_max_size,
        semantic_overlap=semantic_overlap,
        boundary_probs_by_id=boundary_probs_by_id)

    queries = [q["question"] for q in questions]
    maxk = max(C.RECALL_KS)
    pool = dense.search_chunks(queries, DEPTH)
    pool_rec = metrics.recall_from_retrieved(pool, questions, (DEPTH,))

    pairs, counts = [], []
    for qtext, cands in zip(queries, pool):
        pairs.extend((qtext, c["text"]) for c in cands)
        counts.append(len(cands))

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
        "boundary_threshold": None,
        f"pool_recall@{DEPTH}": pool_rec["doc_constrained"][DEPTH],
        "n_docs": len(docs),
        "n_questions": len(questions),
    }

    def finish(arm, retrieved, model_label, seconds, n_pairs):
        rec = metrics.recall_from_retrieved(retrieved, questions, C.RECALL_KS)
        row = dict(base)
        row.update({"arm": arm,
                    "rerank_depth": None if arm == ARM_BGE else DEPTH,
                    "reranker_model": model_label,
                    "rerank_seconds": seconds, "n_pairs": n_pairs})
        for k in C.RECALL_KS:
            row[f"recall@{k}"] = rec["doc_constrained"][k]
        for k in C.RECALL_KS:
            row[f"recall_unconstrained@{k}"] = rec["unconstrained"][k]
        return row

    rows = [finish(ARM_BGE, [cands[:maxk] for cands in pool], "none",
                   None, None)]
    for arm, (scorer, label) in scorers.items():
        t0 = time.perf_counter()
        scores = scorer(pairs)
        secs = time.perf_counter() - t0
        retrieved, off = [], 0
        for cands, n in zip(pool, counts):
            order = rerank_order(scores[off:off + n])
            retrieved.append([cands[j] for j in order[:maxk]])
            off += n
        rows.append(finish(arm, retrieved, label, secs, len(pairs)))
    return rows


def _run_configs(configs, docs, questions, *, scorers, dataset,
                 checkpoint_path, models) -> list[dict]:
    from rag_chunk import sweep
    from rag_chunk.hybrid import config_key
    from rag_chunk.large_eval import (_key_for, append_checkpoint,
                                      load_checkpoint)

    meta = {"stage": "stage8_transfer", "dataset": dataset,
            "n_docs": len(docs), "n_questions": len(questions),
            "retrieval_model": C.RETRIEVAL_EMBED_MODEL,
            "boundary_model": C.BOUNDARY_EMBED_MODEL,
            "reranker_base": C.RERANKER_MODEL,
            "ft_model": scorers[ARM_FT][1]}
    done_rows = load_checkpoint(checkpoint_path, meta)
    if not done_rows and not pathlib.Path(checkpoint_path).exists():
        append_checkpoint(checkpoint_path, {"meta": meta})
    done_keys = {config_key(r) for r in done_rows}

    remaining = [cfg for cfg in configs if _key_for(*cfg) not in done_keys]
    if done_rows:
        print(f"[transfer] resuming: {len(configs) - len(remaining)}/"
              f"{len(configs)} configs already in the checkpoint")

    needed = {m for m, _, _ in remaining if m in ("bilstm", "transformer")}
    probs_by_type = (sweep._precompute_boundary_probs(
        docs, {m: models[m] for m in needed}) if needed else {})

    rows = list(done_rows)
    for i, (method, size, overlap) in enumerate(configs, 1):
        if _key_for(method, size, overlap) in done_keys:
            continue
        print(f"[transfer] ({i}/{len(configs)}) {method} size={size} "
              f"overlap={overlap}")
        if method == "fixed":
            new_rows = _eval_config_arms(
                method, docs, questions, scorers=scorers,
                fixed_size=size, fixed_overlap=overlap)
        else:
            mn, mx = sweep._semantic_window(size)
            new_rows = _eval_config_arms(
                method, docs, questions, scorers=scorers,
                boundary_probs_by_id=probs_by_type[method],
                semantic_policy="target", semantic_target_size=size,
                semantic_min_size=mn, semantic_max_size=mx,
                semantic_overlap=overlap)
        append_checkpoint(checkpoint_path, {"rows": new_rows})
        rows += new_rows
    return rows


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _matched_summary(rows) -> list[dict]:
    from rag_chunk.hybrid import config_key, config_label

    ks = sorted(C.RECALL_KS)
    by_key: dict[tuple, dict[str, dict]] = {}
    order: list[tuple] = []
    for r in rows:
        key = config_key(r)
        if key not in by_key:
            order.append(key)
        by_key.setdefault(key, {})[r["arm"]] = r
    out = []
    for key in order:
        group = by_key[key]
        if not all(a in group for a in (ARM_BGE, ARM_OTS, ARM_FT)):
            continue
        any_row = group[ARM_BGE]
        row = {"method": any_row["method"],
               "chunk_config": config_label(any_row),
               "avg_chunk_size": any_row["avg_chunk_size"],
               f"pool_recall@{DEPTH}": any_row[f"pool_recall@{DEPTH}"]}
        for k in ks:
            row[f"bge_recall@{k}"] = group[ARM_BGE][f"recall@{k}"]
            row[f"ots_recall@{k}"] = group[ARM_OTS][f"recall@{k}"]
            row[f"ft_recall@{k}"] = group[ARM_FT][f"recall@{k}"]
            row[f"ft_minus_ots@{k}"] = (group[ARM_FT][f"recall@{k}"]
                                        - group[ARM_OTS][f"recall@{k}"])
            row[f"ft_minus_bge@{k}"] = (group[ARM_FT][f"recall@{k}"]
                                        - group[ARM_BGE][f"recall@{k}"])
        out.append(row)
    return out


def _read_indomain_deltas() -> dict[str, dict[str, float]]:
    """From stage8/final matched summary: {config_label: {ft_minus_ots@k: v}}."""
    path = _stage_final_dir("stage8") / C.STAGE8_MATCHED_CSV
    if not path.exists():
        print(f"[transfer] WARN: {path} not found — in-domain column omitted")
        return {}
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            label = f"{r['method']} {r['chunk_config']}"
            out[label] = {c: float(r[c]) for c in r
                          if c.startswith("ft_minus_ots@") and r[c] not in ("", None)}
    return out


def _two_se(rows, n_questions: int) -> float:
    k1 = min(C.RECALL_KS)
    vals = [r[f"recall@{k1}"] for r in rows if r["arm"] == ARM_BGE]
    pbar = sum(vals) / len(vals) if vals else 0.5
    return 2.0 * (pbar * (1.0 - pbar) / n_questions) ** 0.5


def _verdict(matched, se2: float) -> tuple[str, float | None]:
    k1 = min(C.RECALL_KS)
    primary = next((m for m in matched if m["method"] == "fixed"
                    and "size=15" in m["chunk_config"]
                    and "overlap=0" in m["chunk_config"]), None)
    if primary is None:
        return "n/a", None
    d = primary[f"ft_minus_ots@{k1}"]
    if d >= se2:
        return "TRANSFERS", d          # gain clears the (wide) noise band
    if d > 0:
        return "DIRECTIONAL", d         # positive but within noise at this n
    return "NO-TRANSFER", d             # no gain / regression


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _columns() -> list[str]:
    from rag_chunk.sweep import _sweep_columns

    return (["arm"] + _sweep_columns()
            + ["rerank_depth", "reranker_model", "rerank_seconds", "n_pairs",
               f"pool_recall@{DEPTH}", "n_docs", "n_questions", "dataset"])


def _write_rows_csv(rows, dataset, path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rec = dict(r, dataset=dataset)
            w.writerow({c: _fmt(c, rec.get(c)) for c in cols})
    print(f"[transfer] wrote {path}  ({len(rows)} rows)")


def _write_matched_csv(table, indomain, path) -> None:
    from rag_chunk.sweep import _fmt

    ks = sorted(C.RECALL_KS)
    cols = ["method", "chunk_config", "avg_chunk_size", f"pool_recall@{DEPTH}"]
    for k in ks:
        cols += [f"bge_recall@{k}", f"ots_recall@{k}", f"ft_recall@{k}",
                 f"ft_minus_ots@{k}", f"ft_minus_bge@{k}",
                 f"indomain_ft_minus_ots@{k}"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in table:
            label = f"{r['method']} {r['chunk_config']}"
            ind = indomain.get(label, {})
            rec = dict(r)
            for k in ks:
                rec[f"indomain_ft_minus_ots@{k}"] = ind.get(f"ft_minus_ots@{k}")
            w.writerow({c: _fmt(c, rec.get(c)) for c in cols})
    print(f"[transfer] wrote {path}  ({len(table)} rows)")


def _plot_delta(matched, n_questions: int, se2: float, path) -> None:
    """Cross-dataset ft-ots ΔR@1 per config vs the ±2 SE band, with the
    in-domain (NQ) delta drawn as a hollow marker for contrast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    k1 = min(C.RECALL_KS)
    labels = [f"{m['method']}\n{m['chunk_config']}" for m in matched]
    d_cross = [m[f"ft_minus_ots@{k1}"] for m in matched]
    d_in = [m.get(f"indomain_ft_minus_ots@{k1}") for m in matched]

    x = np.arange(len(matched))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, d_cross, 0.55, color="#2ca02c",
                  label=f"TriviaQA ft−ots ΔR@{k1} (n={n_questions})")
    ax.bar_label(bars, fmt="%+.3f", padding=2, fontsize=8)
    have_in = [(xi, v) for xi, v in zip(x, d_in) if v is not None]
    if have_in:
        ax.scatter([xi for xi, _ in have_in], [v for _, v in have_in],
                   marker="D", s=70, facecolors="none", edgecolors="#9467bd",
                   linewidths=1.6, zorder=4,
                   label="in-domain NQ ft−ots (Stage 8)")
    ax.axhline(0, color="#333", lw=0.8)
    for y in (se2, -se2):
        ax.axhline(y, color="#888", ls="--", lw=1,
                   label=f"±2 SE (n={n_questions})" if y > 0 else None)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel(f"Δ Recall@{k1} (ft − off-the-shelf)")
    ax.set_title("Does the NQ-tuned reranker's gain transfer to TriviaQA?")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[transfer] wrote {path}")


def _write_summary(out_dir, matched, indomain, verdict, se2,
                   n_docs, n_questions) -> None:
    ks = sorted(C.RECALL_KS)
    k1 = min(ks)
    lines = [
        "# Stage 8 addendum — cross-dataset transfer of the fine-tuned reranker",
        "",
        f"- Eval set: **TriviaQA rc.wikipedia — {n_docs} docs / {n_questions} "
        "questions** (Stage 7 bench; distant-supervised gold).",
        f"- Reranker fine-tuned on **NQ-train** (Stage 8) — a different dataset; "
        "this measures transfer, not in-domain fit.",
        f"- Same 5 configs / 3 arms / one shared BGE top-{DEPTH} pool as the "
        "Stage 8 final eval; only the eval dataset changed.",
        f"- 2 SE at n={n_questions} ≈ **{se2:.4f}** (wide — small eval set).",
        "",
        f"**Verdict (fixed 15/0): {verdict[0]}** "
        + (f"(cross-dataset ft−ots ΔR@{k1} = {verdict[1]:+.4f})"
           if verdict[1] is not None else ""),
        "",
        "## ft − off-the-shelf, cross-dataset vs in-domain",
        "",
        f"| config | pool@{DEPTH} | bge R@{k1} | ots R@{k1} | ft R@{k1} "
        f"| TriviaQA ft−ots ΔR@{k1} | NQ (in-domain) ΔR@{k1} |",
        "|" + "---|" * 7,
    ]
    for m in matched:
        label = f"{m['method']} {m['chunk_config']}"
        ind = indomain.get(label, {}).get(f"ft_minus_ots@{k1}")
        ind_s = f"{ind:+.4f}" if ind is not None else "—"
        lines.append(
            f"| {label} | {m[f'pool_recall@{DEPTH}']:.4f} "
            f"| {m[f'bge_recall@{k1}']:.4f} | {m[f'ots_recall@{k1}']:.4f} "
            f"| {m[f'ft_recall@{k1}']:.4f} | {m[f'ft_minus_ots@{k1}']:+.4f} "
            f"| {ind_s} |")
    lines += [
        "",
        "## How to read this",
        "",
        "- **TRANSFERS** — ft still beats the off-the-shelf reranker on a "
        "dataset it was never tuned on, above the noise band. Strengthens the "
        "Stage 8 claim.",
        "- **DIRECTIONAL** — ft is still ahead but the gap is within 2 SE at "
        f"n={n_questions}; consistent with a real but smaller transfer, not "
        "provable at this eval size. Honest, still useful.",
        "- **NO-TRANSFER** — the in-domain gain does not carry over; the "
        "reranker learned NQ-specific ranking. Also an honest, publishable "
        "boundary.",
        "",
        "> **Caveats (state these).** TriviaQA gold is distant-supervised "
        "(answer-string match on entity pages), weaker than NQ's annotated "
        f"gold; n={n_questions} makes the 2 SE band ~{se2:.3f}; absolute "
        "recall is not comparable to the NQ bench. The headline "
        "\"size dominates, method ties\" is unaffected either way — this only "
        "tests the reranking lever.",
        "",
        "_Generated by `scripts/21_reranker_transfer.py`._",
        "",
    ]
    (out_dir / SUMMARY_MD).write_text("\n".join(lines), encoding="utf-8")
    print(f"[transfer] wrote {out_dir / SUMMARY_MD}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 8 addendum: cross-dataset transfer of the "
                    "fine-tuned reranker on the TriviaQA bench.")
    ap.add_argument("--n-questions", type=int, default=None,
                    help=f"kept TriviaQA questions (default STAGE7_N_QUESTIONS="
                         f"{C.STAGE7_N_QUESTIONS}; reuse the same value Stage 7 "
                         "cached to avoid re-downloading)")
    ap.add_argument("--ft-model", default=None,
                    help="fine-tuned checkpoint dir (default "
                         "models/bge_reranker_ft/final)")
    ap.add_argument("--fresh", action="store_true",
                    help="discard the checkpoint and start over")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="dense retrieval model; must match the archives")
    args = ap.parse_args()

    ft_dir = pathlib.Path(args.ft_model) if args.ft_model else _default_ft_dir()
    if not (ft_dir / "config.json").exists():
        raise SystemExit(f"[transfer] no fine-tuned model at {ft_dir} — train "
                         "it first (scripts/17_train_reranker.py) or pass "
                         "--ft-model")

    n_req = args.n_questions or C.STAGE7_N_QUESTIONS
    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()

    from rag_chunk import cross_dataset, training

    models = {"bilstm": training.load_model("bilstm"),
              "transformer": training.load_model("transformer")}
    docs, questions = cross_dataset.prepare_trivia(n_req)
    dataset = cross_dataset.DATASET_LABEL
    print(f"[transfer] eval set ({dataset}): {len(docs)} docs, "
          f"{len(questions)} questions")
    print(f"[transfer] off-the-shelf: {C.RERANKER_MODEL}")
    print(f"[transfer] fine-tuned:    {ft_dir}")

    from sentence_transformers import CrossEncoder
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = None
    if device != "cuda":
        print("[transfer] WARN: no GPU — reranking on CPU is slow but correct")
    max_len = int(C.RERANK_MAX_LENGTH)
    ots_model = CrossEncoder(C.RERANKER_MODEL, max_length=max_len, device=device)
    ft_model = CrossEncoder(str(ft_dir), max_length=max_len, device=device)

    def _scorer(model):
        def run(pairs):
            import numpy as np
            if not pairs:
                return np.zeros(0, dtype="float32")
            return np.asarray(
                model.predict(pairs, batch_size=int(C.RERANK_BATCH_SIZE),
                              show_progress_bar=False), dtype="float32")
        return run

    scorers = {ARM_OTS: (_scorer(ots_model), C.RERANKER_MODEL),
               ARM_FT: (_scorer(ft_model), str(ft_dir))}

    latest = C.RESULTS_LATEST_DIR
    ckpt = latest / CHECKPOINT_FILE
    if args.fresh and ckpt.exists():
        ckpt.unlink()
        print(f"[transfer] --fresh: discarded {ckpt.name}")

    rows = _run_configs(list(C.STAGE6_RERANK_CONFIGS), docs, questions,
                        scorers=scorers, dataset=dataset,
                        checkpoint_path=ckpt, models=models)

    matched = _matched_summary(rows)
    indomain = _read_indomain_deltas()
    se2 = _two_se(rows, len(questions))
    verdict = _verdict(matched, se2)
    k1 = min(C.RECALL_KS)

    _write_rows_csv(rows, dataset, latest / RESULTS_CSV)
    _write_matched_csv(matched, indomain, latest / MATCHED_CSV)
    _plot_delta([dict(m, **{f"indomain_ft_minus_ots@{k1}":
                            indomain.get(f"{m['method']} {m['chunk_config']}",
                                         {}).get(f"ft_minus_ots@{k1}")})
                 for m in matched], len(questions), se2, latest / DELTA_PNG)
    _write_summary(latest, matched, indomain, verdict, se2,
                   len(docs), len(questions))

    print(f"\n[transfer] 2 SE (n={len(questions)}) = {se2:.4f}")
    for m in matched:
        label = f"{m['method']} {m['chunk_config']}"
        ind = indomain.get(label, {}).get(f"ft_minus_ots@{k1}")
        ind_s = f"{ind:+.4f}" if ind is not None else "  —  "
        print(f"[transfer]   {label:<22} TriviaQA ft−ots@{k1}="
              f"{m[f'ft_minus_ots@{k1}']:+.4f}   NQ in-domain={ind_s}")
    print(f"\n[transfer] VERDICT (fixed 15/0): {verdict[0]}"
          + (f"  (ft−ots ΔR@{k1} = {verdict[1]:+.4f} vs 2 SE {se2:.4f})"
             if verdict[1] is not None else ""))
    print("[transfer] review stage8_transfer_summary.md, then archive if kept.")


if __name__ == "__main__":
    main()
