"""Stage 9 - does a *fine-tuned* reranker change the rerank-depth verdict?

Stage 5 concluded that reranking the BGE top-50 is always worse than the top-20:
the extra candidates added more noise than signal. That was measured with the
**off-the-shelf** cross-encoder. Stage 8 then showed the off-the-shelf reranker
is weak here (fine-tuning it bought +0.107 R@1), and the archived pool recalls
say the deeper pool genuinely holds more answers (Stage 5, n=203:
pool@20 0.9754 -> pool@50 0.9951 at transformer 15/1). So the trade-off may
invert once the ranker is strong enough to sort 50 candidates.

The experiment is a single controlled question: **at a fixed chunking config,
does depth 50 beat depth 20 for the fine-tuned reranker, and does it still lose
for the off-the-shelf one?**

Cost trick (why this is ~1 GPU-hour, not 2.5): cross-encoder scores are
independent per (question, chunk) pair, so the depth-20 arm is exactly the
depth-50 scores restricted to the first 20 dense candidates. Every depth is
scored **once** at ``max(depths)`` and the shallower depths are derived by
slicing — no pair is ever scored twice.

That also makes the reproduction check free: the derived depth-20 rows must
match the archived ``stage8/final`` rows exactly, and only then do the depth-50
rows mean anything.

Runtime on a T4 (from the archived Stage 6 timings, ~2.5x the depth-20 cost):
``fixed 15/0`` + ``fixed 6/0`` with both rerankers ~= 1.5 h; ``--configs all``
~= 4.1 h, which will not fit one free Colab session — use the checkpoint.

Usage:
    python scripts/22_rerank_depth.py                     # 2 configs, depths 20/50
    python scripts/22_rerank_depth.py --configs all
    python scripts/22_rerank_depth.py --depths 20,50,100
    python scripts/22_rerank_depth.py --fresh
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

ARM_BGE = "bge"
CHECKPOINT_FILE = "stage9_depth_checkpoint.jsonl"
RESULTS_CSV = "stage9_depth_results.csv"
MATCHED_CSV = "stage9_depth_matched.csv"
SUMMARY_MD = "stage9_depth_summary.md"
DELTA_PNG = "stage9_depth_delta.png"
CHECK_CSV = "stage9_check_vs_stage8.csv"
CHECK_TOLERANCE = 0.005

# Default: the deployment config plus the small-chunk config where off-the-shelf
# reranking helped most in Stage 6 — the two cases the claim is about.
DEFAULT_CONFIGS = (("fixed", 15, 0), ("fixed", 6, 0))


def _stage_final_dir(stage: str) -> pathlib.Path:
    return C.RESULTS_DIR / stage / "final"


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _default_ft_dir() -> pathlib.Path:
    return C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"


def _arm_name(depth: int, kind: str) -> str:
    return f"rerank{depth}" if kind == "ots" else f"rerank{depth}_ft"


# --------------------------------------------------------------------------- #
# One config -> rows for every (arm, depth), from ONE scoring pass
# --------------------------------------------------------------------------- #
def _eval_config_depths(method, docs, questions, *, scorers, depths,
                        boundary_probs_by_id=None, fixed_size=None,
                        fixed_overlap=None, semantic_policy=None,
                        semantic_target_size=None, semantic_min_size=None,
                        semantic_max_size=None, semantic_overlap=None) -> list[dict]:
    from rag_chunk import metrics, retrieval
    from rag_chunk.rerank import rerank_order

    max_depth = max(depths)
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
    pool = dense.search_chunks(queries, max_depth)          # deepest pool once

    # Ceiling per depth: is the answer anywhere in the top-d candidates?
    pool_rec = {d: metrics.recall_from_retrieved(
        [cands[:d] for cands in pool], questions, (d,))["doc_constrained"][d]
        for d in depths}

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
        "n_docs": len(docs),
        "n_questions": len(questions),
    }
    for d in depths:
        base[f"pool_recall@{d}"] = pool_rec[d]

    def finish(arm, retrieved, depth, model_label, seconds, n_pairs):
        rec = metrics.recall_from_retrieved(retrieved, questions, C.RECALL_KS)
        row = dict(base)
        row.update({"arm": arm, "rerank_depth": depth,
                    "reranker_model": model_label,
                    "rerank_seconds": seconds, "n_pairs": n_pairs})
        for k in C.RECALL_KS:
            row[f"recall@{k}"] = rec["doc_constrained"][k]
        for k in C.RECALL_KS:
            row[f"recall_unconstrained@{k}"] = rec["unconstrained"][k]
        return row

    rows = [finish(ARM_BGE, [cands[:maxk] for cands in pool], None, "none",
                   None, None)]

    for kind, (scorer, label) in scorers.items():
        t0 = time.perf_counter()
        scores = scorer(pairs)                       # scored ONCE at max_depth
        secs = time.perf_counter() - t0
        print(f"[stage9]   {kind}: {len(pairs)} pairs in {secs / 60:.1f} min",
              flush=True)
        for d in depths:
            retrieved, off = [], 0
            for cands, n in zip(pool, counts):
                take = min(d, n)
                # depth-d rerank == the top-d dense candidates, sorted by the
                # scores they already received in the depth-max pass
                order = rerank_order(scores[off:off + take])
                retrieved.append([cands[j] for j in order[:maxk]])
                off += n
            share = secs * d / max_depth              # attributed cost at depth d
            rows.append(finish(_arm_name(d, kind), retrieved, d, label,
                               share, sum(min(d, n) for n in counts)))
    return rows


def _run_configs(configs, docs, questions, *, scorers, depths, checkpoint_path,
                 models) -> list[dict]:
    from rag_chunk import sweep
    from rag_chunk.hybrid import config_key
    from rag_chunk.large_eval import (_key_for, append_checkpoint,
                                      load_checkpoint)

    meta = {"stage": "stage9", "dataset": "nq-large",
            "n_docs": len(docs), "n_questions": len(questions),
            "depths": list(depths),
            "retrieval_model": C.RETRIEVAL_EMBED_MODEL,
            "boundary_model": C.BOUNDARY_EMBED_MODEL,
            "reranker_base": C.RERANKER_MODEL,
            "ft_model": scorers["ft"][1]}
    done_rows = load_checkpoint(checkpoint_path, meta)
    if not done_rows and not pathlib.Path(checkpoint_path).exists():
        append_checkpoint(checkpoint_path, {"meta": meta})
    done_keys = {config_key(r) for r in done_rows}

    remaining = [cfg for cfg in configs if _key_for(*cfg) not in done_keys]
    if done_rows:
        print(f"[stage9] resuming: {len(configs) - len(remaining)}/"
              f"{len(configs)} configs already done")

    needed = {m for m, _, _ in remaining if m in ("bilstm", "transformer")}
    probs_by_type = (sweep._precompute_boundary_probs(
        docs, {m: models[m] for m in needed}) if needed else {})

    rows = list(done_rows)
    for i, (method, size, overlap) in enumerate(configs, 1):
        if _key_for(method, size, overlap) in done_keys:
            continue
        print(f"[stage9] ({i}/{len(configs)}) {method} size={size} "
              f"overlap={overlap}", flush=True)
        if method == "fixed":
            new_rows = _eval_config_depths(
                method, docs, questions, scorers=scorers, depths=depths,
                fixed_size=size, fixed_overlap=overlap)
        else:
            mn, mx = sweep._semantic_window(size)
            new_rows = _eval_config_depths(
                method, docs, questions, scorers=scorers, depths=depths,
                boundary_probs_by_id=probs_by_type[method],
                semantic_policy="target", semantic_target_size=size,
                semantic_min_size=mn, semantic_max_size=mx,
                semantic_overlap=overlap)
        append_checkpoint(checkpoint_path, {"rows": new_rows})
        rows += new_rows
    return rows


# --------------------------------------------------------------------------- #
# Check: the derived depth-20 rows must reproduce stage8/final
# --------------------------------------------------------------------------- #
def _as_float(row: dict, col: str):
    v = row.get(col)
    return None if v in ("", None) else float(v)


def _write_stage8_check(stage8_rows, rows, depths, path) -> bool | None:
    """Depth-20 arms re-run the exact Stage 8 pipeline, so they must reproduce
    it. Returns True/False, or None when depth 20 was not evaluated."""
    from rag_chunk.hybrid import config_key, config_label

    if 20 not in depths:
        return None
    ks = sorted(C.RECALL_KS)
    s8 = {(str(r["arm"]), config_key(r)): r for r in stage8_rows}
    cols = ["arm", "method", "chunk_config"]
    for k in ks:
        cols += [f"stage8_recall@{k}", f"stage9_recall@{k}", f"delta_recall@{k}"]

    out, worst, unmatched = [], 0.0, 0
    for r in rows:
        arm = r["arm"]
        if arm not in (ARM_BGE, "rerank20", "rerank20_ft"):
            continue
        r8 = s8.get((arm, config_key(r)))
        rec = {"arm": arm, "method": r["method"],
               "chunk_config": config_label(r)}
        if r8 is None:
            unmatched += 1
            out.append(rec)
            continue
        for k in ks:
            v8 = _as_float(r8, f"recall@{k}")
            v9 = round(r[f"recall@{k}"], 4)
            delta = round(v9 - v8, 4)
            worst = max(worst, abs(delta))
            rec[f"stage8_recall@{k}"] = f"{v8:.4f}"
            rec[f"stage9_recall@{k}"] = f"{v9:.4f}"
            rec[f"delta_recall@{k}"] = f"{delta:.4f}"
        out.append(rec)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"[stage9] wrote {path} ({len(out)} rows)")

    ok = unmatched == 0 and worst <= CHECK_TOLERANCE
    if unmatched:
        print(f"[stage9] WARN: {unmatched} row(s) had no Stage 8 counterpart")
    if ok and worst == 0.0:
        print("[stage9] check OK: depth-20 arms reproduce stage8/final exactly "
              "— the depth-50 rows are trustworthy.")
    elif ok:
        print(f"[stage9] check: max |delta| vs Stage 8 = {worst:.4f} (within "
              "one question).")
    else:
        print(f"[stage9] WARN: check FAILED (max |delta| = {worst:.4f}) — do "
              "NOT trust the depth-50 rows until this is explained.")
    return ok


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _two_se(rows, n_questions: int) -> float:
    k1 = min(C.RECALL_KS)
    vals = [r[f"recall@{k1}"] for r in rows if r["arm"] == ARM_BGE]
    p = sum(vals) / len(vals) if vals else 0.5
    return 2.0 * (p * (1 - p) / n_questions) ** 0.5


def _matched(rows, depths) -> list[dict]:
    """Per config: each reranker's R@k at every depth, plus deep-minus-shallow."""
    from rag_chunk.hybrid import config_key, config_label

    ks = sorted(C.RECALL_KS)
    shallow, deep = min(depths), max(depths)
    by_key, order = {}, []
    for r in rows:
        key = config_key(r)
        if key not in by_key:
            order.append(key)
        by_key.setdefault(key, {})[r["arm"]] = r

    out = []
    for key in order:
        g = by_key[key]
        if ARM_BGE not in g:
            continue
        any_row = g[ARM_BGE]
        rec = {"method": any_row["method"],
               "chunk_config": config_label(any_row),
               "avg_chunk_size": any_row["avg_chunk_size"]}
        for d in depths:
            rec[f"pool_recall@{d}"] = any_row.get(f"pool_recall@{d}")
        for k in ks:
            rec[f"bge_recall@{k}"] = g[ARM_BGE][f"recall@{k}"]
            for kind in ("ots", "ft"):
                for d in depths:
                    a = _arm_name(d, kind)
                    if a in g:
                        rec[f"{kind}{d}_recall@{k}"] = g[a][f"recall@{k}"]
                a_deep, a_shal = _arm_name(deep, kind), _arm_name(shallow, kind)
                if a_deep in g and a_shal in g:
                    rec[f"{kind}_deep_minus_shallow@{k}"] = (
                        g[a_deep][f"recall@{k}"] - g[a_shal][f"recall@{k}"])
        out.append(rec)
    return out


def _columns(depths) -> list[str]:
    from rag_chunk.sweep import _sweep_columns

    return (["arm"] + _sweep_columns()
            + ["rerank_depth", "reranker_model", "rerank_seconds", "n_pairs"]
            + [f"pool_recall@{d}" for d in depths]
            + ["n_docs", "n_questions"])


def _write_rows_csv(rows, depths, path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _columns(depths)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[stage9] wrote {path}  ({len(rows)} rows)")


def _write_matched_csv(table, depths, path) -> None:
    from rag_chunk.sweep import _fmt

    ks = sorted(C.RECALL_KS)
    cols = ["method", "chunk_config", "avg_chunk_size"]
    cols += [f"pool_recall@{d}" for d in depths]
    for k in ks:
        cols.append(f"bge_recall@{k}")
        for kind in ("ots", "ft"):
            cols += [f"{kind}{d}_recall@{k}" for d in depths]
            cols.append(f"{kind}_deep_minus_shallow@{k}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[stage9] wrote {path}  ({len(table)} rows)")


def _plot(matched, depths, se2, path) -> None:
    """Deep-minus-shallow ΔR@1 per config for each reranker, against ±2 SE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    k1 = min(C.RECALL_KS)
    shallow, deep = min(depths), max(depths)
    labels = [f"{m['method']}\n{m['chunk_config']}" for m in matched]
    x = np.arange(len(matched))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for off, kind, colour, name in ((-w / 2, "ots", "#9467bd", "off-the-shelf"),
                                    (+w / 2, "ft", "#2ca02c", "fine-tuned")):
        vals = [m.get(f"{kind}_deep_minus_shallow@{k1}") or 0.0 for m in matched]
        bars = ax.bar(x + off, vals, w, color=colour,
                      label=f"{name}: depth {deep} − depth {shallow}")
        ax.bar_label(bars, fmt="%+.3f", padding=2, fontsize=8)
    ax.axhline(0, color="#333", lw=0.8)
    for y in (se2, -se2):
        ax.axhline(y, color="#888", ls="--", lw=1,
                   label=f"±2 SE" if y > 0 else None)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel(f"Δ Recall@{k1} (deeper pool − shallower pool)")
    ax.set_title("Does a stronger reranker make the deeper pool pay off?")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[stage9] wrote {path}")


def _write_summary(out_dir, matched, depths, se2, check_ok, n_docs,
                   n_questions, *, smoke: bool = False) -> None:
    ks = sorted(C.RECALL_KS)
    k1 = min(ks)
    shallow, deep = min(depths), max(depths)
    lines = [
        f"# Stage 9 — rerank depth revisited with a fine-tuned reranker",
        "",]
    if smoke:
        lines += ["> **SMOKE RUN — not a result.** Only the first "
                  f"{n_questions} questions were scored and the Stage 8 "
                  "reproduction check was skipped. Do not archive or cite.", ""]
    lines += [
        f"- Eval set: NQ bench, **{n_docs} docs / {n_questions} questions** "
        "(the Stage 6/8 bench).",
        f"- Depths compared: {', '.join(str(d) for d in depths)}. Every depth is "
        f"scored once at depth {deep}; shallower depths are the same scores "
        "restricted to the top-d dense candidates, so no pair is scored twice.",
        f"- 2 SE at n={n_questions} ≈ **{se2:.4f}**.",
    ]
    if check_ok is not None:
        lines.append("- Check vs `stage8/final` (depth-20 arms): "
                     + ("**OK (exact)**" if check_ok else
                        "**FAILED — do not trust the deep rows**"))
    lines += [
        "",
        "## Question",
        "",
        "Stage 5 found reranking the top-50 always lost to the top-20 — but "
        "that used the **off-the-shelf** cross-encoder. The deeper pool "
        "demonstrably holds more answers, so the verdict should depend on how "
        "well the ranker sorts them. Does fine-tuning flip it?",
        "",
        "## Results",
        "",
        f"| config | pool@{shallow} | pool@{deep} | ots@{shallow} | ots@{deep} "
        f"| **ots Δ** | ft@{shallow} | ft@{deep} | **ft Δ** |",
        "|" + "---|" * 9,
    ]
    for m in matched:
        def g(key, fmt="{:.4f}"):
            v = m.get(key)
            return "—" if v is None else fmt.format(v)
        lines.append(
            f"| {m['method']} {m['chunk_config']} "
            f"| {g(f'pool_recall@{shallow}')} | {g(f'pool_recall@{deep}')} "
            f"| {g(f'ots{shallow}_recall@{k1}')} | {g(f'ots{deep}_recall@{k1}')} "
            f"| {g(f'ots_deep_minus_shallow@{k1}', '{:+.4f}')} "
            f"| {g(f'ft{shallow}_recall@{k1}')} | {g(f'ft{deep}_recall@{k1}')} "
            f"| {g(f'ft_deep_minus_shallow@{k1}', '{:+.4f}')} |")
    lines += [
        "",
        f"All figures are doc-constrained Recall@{k1}. A Δ above "
        f"+{se2:.4f} (2 SE) is a real gain; below −{se2:.4f} a real loss; in "
        "between, the depth does not matter at this sample size.",
        "",
        "> **Honest reading.** Deeper pools cost ~2.5x the reranking time for "
        "at most the pool-ceiling difference. Report the Δ against that cost, "
        "and note that the chunking headline (size dominates, method ties) is "
        "untouched — this only tunes the reranking stage.",
        "",
        "_Generated by `scripts/22_rerank_depth.py`._",
        "",
    ]
    path = out_dir / (f"smoke_{SUMMARY_MD}" if smoke else SUMMARY_MD)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[stage9] wrote {path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 9: re-test rerank depth with the fine-tuned "
                    "cross-encoder on the Stage 6 bench.")
    ap.add_argument("--configs", default="default",
                    help="'default' (fixed 15/0 + 6/0, ~1.5 h), 'all' (the 5 "
                         "Stage 6 rerank configs, ~4.1 h), or e.g. 'fixed:15:0'")
    ap.add_argument("--depths", default="20,50",
                    help="comma-separated rerank depths (default 20,50)")
    ap.add_argument("--ft-model", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="discard the checkpoint and start over")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="SMOKE ONLY: score just the first N questions. The "
                         "numbers are not comparable to the archives and the "
                         "Stage 8 check is skipped; use to verify the run "
                         "end to end before committing ~1.5 h of GPU.")
    args = ap.parse_args()

    depths = sorted({int(d) for d in args.depths.split(",") if d.strip()})
    if len(depths) < 2:
        raise SystemExit("[stage9] give at least two depths, e.g. --depths 20,50")

    if args.configs == "default":
        configs = list(DEFAULT_CONFIGS)
    elif args.configs == "all":
        configs = list(C.STAGE6_RERANK_CONFIGS)
    else:
        configs = []
        for spec in args.configs.split(","):
            m, s, o = spec.split(":")
            configs.append((m, int(s), int(o)))

    ft_dir = pathlib.Path(args.ft_model) if args.ft_model else _default_ft_dir()
    if not (ft_dir / "config.json").exists():
        raise SystemExit(f"[stage9] no fine-tuned model at {ft_dir}")

    stage8_csv = _stage_final_dir("stage8") / C.STAGE8_RESULTS_CSV
    if not stage8_csv.exists():
        raise SystemExit(f"[stage9] missing {stage8_csv} — the depth-20 arms "
                         "are checked against the Stage 8 archive")
    stage8_rows = _read_csv(stage8_csv)

    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()
    n = int(C.N_NQ_DOCS_LARGE)
    C.apply(N_NQ_DOCS=n)
    C.apply(NQ_DIR=C.NQ_DIR / f"large_n{n}")

    from rag_chunk import nq_data, training

    models = {"bilstm": training.load_model("bilstm"),
              "transformer": training.load_model("transformer")}
    docs, questions = nq_data.prepare_nq()
    smoke = args.max_questions is not None
    if smoke:
        questions = questions[:args.max_questions]
        print(f"[stage9] *** SMOKE MODE: {len(questions)} questions — results "
              "are NOT comparable to the archives ***")
    print(f"[stage9] bench: {len(docs)} docs / {len(questions)} questions")
    print(f"[stage9] configs: {configs}")
    print(f"[stage9] depths: {depths}")

    from sentence_transformers import CrossEncoder
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = None
    if device != "cuda":
        print("[stage9] WARN: no GPU — this will take many hours on CPU")
    max_len = int(C.RERANK_MAX_LENGTH)

    def _scorer(model):
        def run(pairs):
            import numpy as np
            if not pairs:
                return np.zeros(0, dtype="float32")
            return np.asarray(
                model.predict(pairs, batch_size=int(C.RERANK_BATCH_SIZE),
                              show_progress_bar=False), dtype="float32")
        return run

    scorers = {
        "ots": (_scorer(CrossEncoder(C.RERANKER_MODEL, max_length=max_len,
                                     device=device)), C.RERANKER_MODEL),
        "ft": (_scorer(CrossEncoder(str(ft_dir), max_length=max_len,
                                    device=device)), str(ft_dir)),
    }

    latest = C.RESULTS_LATEST_DIR
    # A smoke run must never share a checkpoint (or an output name) with the
    # real one, or a partial 20-question run would masquerade as the result.
    ckpt = latest / (CHECKPOINT_FILE.replace(".jsonl", "_smoke.jsonl")
                     if smoke else CHECKPOINT_FILE)
    if args.fresh and ckpt.exists():
        ckpt.unlink()
        print(f"[stage9] --fresh: discarded {ckpt.name}")

    rows = _run_configs(configs, docs, questions, scorers=scorers,
                        depths=depths, checkpoint_path=ckpt, models=models)

    matched = _matched(rows, depths)
    se2 = _two_se(rows, len(questions))
    k1 = min(C.RECALL_KS)
    shallow, deep = min(depths), max(depths)

    # Smoke outputs are prefixed so they can never be mistaken for — or
    # archived as — the real run.
    def out(name: str) -> pathlib.Path:
        return latest / (f"smoke_{name}" if smoke else name)

    _write_rows_csv(rows, depths, out(RESULTS_CSV))
    if smoke:
        check_ok = None
        print("[stage9] smoke mode: Stage 8 reproduction check skipped")
    else:
        check_ok = _write_stage8_check(stage8_rows, rows, depths,
                                       latest / CHECK_CSV)
    _write_matched_csv(matched, depths, out(MATCHED_CSV))
    _plot(matched, depths, se2, out(DELTA_PNG))
    _write_summary(latest, matched, depths, se2, check_ok, len(docs),
                   len(questions), smoke=smoke)

    print(f"\n[stage9] 2 SE = {se2:.4f}   (depth {deep} vs {shallow}, R@{k1})")
    for m in matched:
        d_ots = m.get(f"ots_deep_minus_shallow@{k1}")
        d_ft = m.get(f"ft_deep_minus_shallow@{k1}")
        verdict = ("n/a" if d_ft is None else
                   "DEEPER WINS" if d_ft > se2 else
                   "DEEPER LOSES" if d_ft < -se2 else "NO DIFFERENCE")
        print(f"[stage9]   {m['method']} {m['chunk_config']:<26} "
              f"ots {d_ots:+.4f}  ft {d_ft:+.4f}   -> ft: {verdict}")
    if smoke:
        print("\n[stage9] smoke run complete — pipeline works end to end. "
              "Re-run without --max-questions for the real numbers.")
    else:
        print("\n[stage9] complete. Review, then archive:")
        print("  python scripts/save_stage_results.py --stage stage9")


if __name__ == "__main__":
    main()
