"""Stage 8 (steps 3-4) - evaluate the fine-tuned reranker.

Three arms per chunking config, all sharing ONE identical BGE top-20 pool so
the comparison is purely about ranking quality:

    bge          the dense-only baseline order;
    rerank20     the pool reordered by the OFF-THE-SHELF cross-encoder;
    rerank20_ft  the pool reordered by the FINE-TUNED cross-encoder.

Two modes:

    --dev     go/no-go gate. Scores the held-out NQ-train dev bench (built by
              script 16) at fixed 15/0 (primary) and fixed 6/0 (context) and
              prints the verdict: dev ΔR@1 (ft - off-the-shelf) at fixed 15/0
              >= STAGE8_GO_THRESHOLD -> GO; <= 0 -> NO-GO (honest negative
              result); in between -> judgment call, at most one retry.
    (default) the final eval on the Stage 6 bench (nq/large_n1000 cache, 1032
              questions, the 5 STAGE6_RERANK_CONFIGS). Built-in check
              discipline: the bge and rerank20 rows re-run the exact Stage 6
              pipeline, so they must reproduce the archived stage6/final rows
              exactly (stage8_check_vs_stage6.csv) — only then does the
              rerank20_ft row mean anything.

The final mode checkpoints per config to results/latest/ and is resume-safe;
it refuses to run if stage6/final is missing or was built with different
models.

Usage:
    python scripts/18_eval_reranker_ft.py --dev
    python scripts/18_eval_reranker_ft.py
    python scripts/18_eval_reranker_ft.py --ft-model <dir>   # non-default ckpt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

DEPTH = 20                         # matches Stage 5/6 rerank20
ARM_BGE, ARM_OTS, ARM_FT = "bge", "rerank20", "rerank20_ft"
CHECKPOINT_FILES = {"dev": "stage8_checkpoint_dev.jsonl",
                    "final": "stage8_checkpoint_final.jsonl"}
CHECK_TOLERANCE = 0.005
DEV_CONFIGS = (("fixed", 15, 0), ("fixed", 6, 0))


def _stage_final_dir(stage: str) -> pathlib.Path:
    return C.RESULTS_DIR / stage / "final"


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _default_ft_dir() -> pathlib.Path:
    return C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"


def _sha1(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_latest() -> None:
    """Same contract as Stages 6/7: only delete files with a byte-identical
    archived copy; never touch stage8_* working files."""
    latest = C.RESULTS_LATEST_DIR
    if not latest.exists():
        return
    archives = [d for d in (_stage_final_dir(s) for s in
                            ("stage2", "stage3", "stage4", "stage5", "stage6",
                             "stage7", "stage8"))
                if d.is_dir()]
    stale = [p for p in sorted(latest.iterdir())
             if p.is_file() and not p.name.startswith("stage8_")]
    unarchived = [p.name for p in stale
                  if not any((a / p.name).exists() and _sha1(a / p.name) == _sha1(p)
                             for a in archives)]
    if unarchived:
        raise SystemExit(
            "[stage8] results/latest/ has file(s) with no identical copy in "
            f"any stage archive: {', '.join(unarchived)}\n"
            "[stage8] Archive them first so nothing is lost.")
    for p in stale:
        p.unlink()
    if stale:
        print(f"[stage8] cleared {len(stale)} archived file(s) from {latest}")


# --------------------------------------------------------------------------- #
# One config -> three rows (bge / rerank20 / rerank20_ft), one shared pool
# --------------------------------------------------------------------------- #
def _eval_config_arms(method, docs, questions, *, scorers,
                      boundary_probs_by_id=None, fixed_size=None,
                      fixed_overlap=None, semantic_policy=None,
                      semantic_target_size=None, semantic_min_size=None,
                      semantic_max_size=None, semantic_overlap=None) -> list[dict]:
    """``scorers`` maps rerank arm name -> (scorer_fn, model_label). The dense
    pool is retrieved once; every rerank arm reorders the same candidates."""
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


def _run_configs(configs, docs, questions, *, scorers, mode, dataset,
                 checkpoint_path, models) -> list[dict]:
    """Checkpointed loop over (method, size, overlap) configs."""
    from rag_chunk import sweep
    from rag_chunk.hybrid import config_key
    from rag_chunk.large_eval import (_key_for, append_checkpoint,
                                      load_checkpoint)

    meta = {"stage": "stage8", "mode": mode, "dataset": dataset,
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
        print(f"[stage8] resuming: {len(configs) - len(remaining)}/"
              f"{len(configs)} configs already in the checkpoint")

    needed = {m for m, _, _ in remaining if m in ("bilstm", "transformer")}
    probs_by_type = (sweep._precompute_boundary_probs(
        docs, {m: models[m] for m in needed}) if needed else {})

    rows = list(done_rows)
    for i, (method, size, overlap) in enumerate(configs, 1):
        if _key_for(method, size, overlap) in done_keys:
            continue
        print(f"[stage8] ({i}/{len(configs)}) {method} size={size} "
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
# Writers / check / summary
# --------------------------------------------------------------------------- #
def _columns() -> list[str]:
    from rag_chunk.sweep import _sweep_columns

    return (["arm"] + _sweep_columns()
            + ["rerank_depth", "reranker_model", "rerank_seconds", "n_pairs",
               f"pool_recall@{DEPTH}", "n_docs", "n_questions"])


def _write_rows_csv(rows, path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[stage8] wrote {path}  ({len(rows)} rows)")


def _as_float(row: dict, col: str):
    val = row.get(col)
    if val in ("", None):
        return None
    return float(val)


def _write_stage6_check(stage6_rows, rows, path) -> bool:
    """The bge and rerank20 rows must reproduce the archived Stage 6 rows
    exactly. Returns True when the check passes."""
    from rag_chunk.hybrid import config_key, config_label

    ks = sorted(C.RECALL_KS)
    s6 = {(str(r["arm"]), config_key(r)): r for r in stage6_rows}
    cols = ["arm", "method", "chunk_config",
            "stage6_n_chunks", "stage8_n_chunks", "delta_n_chunks"]
    for k in ks:
        cols += [f"stage6_recall@{k}", f"stage8_recall@{k}", f"delta_recall@{k}"]

    out_rows = []
    stats = {"unmatched": 0, "n_chunks_mismatch": 0, "max_abs_recall_delta": 0.0}
    for r8 in rows:
        if r8["arm"] not in (ARM_BGE, ARM_OTS):
            continue
        r6 = s6.get((str(r8["arm"]), config_key(r8)))
        out = {"arm": r8["arm"], "method": r8["method"],
               "chunk_config": config_label(r8),
               "stage8_n_chunks": r8["n_chunks"]}
        if r6 is None:
            stats["unmatched"] += 1
            out_rows.append(out)
            continue
        n6 = int(_as_float(r6, "n_chunks"))
        out["stage6_n_chunks"] = n6
        out["delta_n_chunks"] = r8["n_chunks"] - n6
        if out["delta_n_chunks"] != 0:
            stats["n_chunks_mismatch"] += 1
        for k in ks:
            v6 = _as_float(r6, f"recall@{k}")
            v8 = round(r8[f"recall@{k}"], 4)
            delta = round(v8 - v6, 4)
            stats["max_abs_recall_delta"] = max(stats["max_abs_recall_delta"],
                                                abs(delta))
            out[f"stage6_recall@{k}"] = f"{v6:.4f}"
            out[f"stage8_recall@{k}"] = f"{v8:.4f}"
            out[f"delta_recall@{k}"] = f"{delta:.4f}"
        out_rows.append(out)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[stage8] wrote {path} ({len(out_rows)} rows)")

    d = stats["max_abs_recall_delta"]
    ok = (stats["unmatched"] == 0 and stats["n_chunks_mismatch"] == 0
          and d <= CHECK_TOLERANCE)
    if stats["unmatched"]:
        print(f"[stage8] WARN: {stats['unmatched']} row(s) had no matching "
              "Stage 6 row")
    if stats["n_chunks_mismatch"]:
        print(f"[stage8] WARN: {stats['n_chunks_mismatch']} config(s) with a "
              "different chunk count — chunking is NOT identical")
    if ok and d == 0.0:
        print("[stage8] check OK: bge + rerank20 rows reproduce the archived "
              "Stage 6 rows exactly — the rerank20_ft rows are trustworthy.")
    elif ok:
        print(f"[stage8] check: max |recall delta| vs Stage 6 = {d:.4f} "
              "(within one question — inspect before trusting).")
    else:
        print(f"[stage8] WARN: check FAILED (max |recall delta| = {d:.4f}) — "
              "do NOT trust the rerank20_ft rows until this is explained.")
    return ok


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


def _write_matched_csv(table, path) -> None:
    from rag_chunk.sweep import _fmt

    ks = sorted(C.RECALL_KS)
    cols = ["method", "chunk_config", "avg_chunk_size", f"pool_recall@{DEPTH}"]
    for k in ks:
        cols += [f"bge_recall@{k}", f"ots_recall@{k}", f"ft_recall@{k}",
                 f"ft_minus_ots@{k}", f"ft_minus_bge@{k}"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[stage8] wrote {path}  ({len(table)} rows)")


def _two_se(rows, n_questions: int) -> float:
    k1 = min(C.RECALL_KS)
    vals = [r[f"recall@{k1}"] for r in rows if r["arm"] == ARM_BGE]
    pbar = sum(vals) / len(vals) if vals else 0.5
    return 2.0 * (pbar * (1.0 - pbar) / n_questions) ** 0.5


def _plot_delta(matched, n_questions: int, path) -> None:
    """ΔR@1 vs bge for the off-the-shelf and fine-tuned rerankers per config,
    with the ±2 SE band — the Stage 8 question in one picture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    k1 = min(C.RECALL_KS)
    labels = [f"{m['method']}\n{m['chunk_config']}" for m in matched]
    d_ots = [m[f"ots_recall@{k1}"] - m[f"bge_recall@{k1}"] for m in matched]
    d_ft = [m[f"ft_recall@{k1}"] - m[f"bge_recall@{k1}"] for m in matched]
    pbar = float(np.mean([m[f"bge_recall@{k1}"] for m in matched]))
    se2 = 2.0 * (pbar * (1 - pbar) / n_questions) ** 0.5

    x = np.arange(len(matched))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, deltas, label, color in (
            (-w / 2, d_ots, "off-the-shelf rerank20", "#9467bd"),
            (+w / 2, d_ft, "fine-tuned rerank20", "#2ca02c")):
        bars = ax.bar(x + offset, deltas, w, label=label, color=color)
        ax.bar_label(bars, fmt="%+.3f", padding=2, fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    for y in (se2, -se2):
        ax.axhline(y, color="#888888", linestyle="--", linewidth=1,
                   label=f"±2 SE (n={n_questions})" if y > 0 else None)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel(f"Δ Recall@{k1} vs bge")
    ax.set_title("Does fine-tuning the cross-encoder move the needle?")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[stage8] wrote {path}")


def _gate_verdict(matched) -> tuple[str, float | None]:
    k1 = min(C.RECALL_KS)
    primary = next((m for m in matched if m["method"] == "fixed"
                    and "15" in m["chunk_config"]), None)
    if primary is None:
        return "n/a", None
    delta = primary[f"ft_minus_ots@{k1}"]
    thr = float(C.STAGE8_GO_THRESHOLD)
    if delta >= thr:
        return "GO", delta
    if delta <= 0:
        return "NO-GO", delta
    return "GRAY-ZONE", delta


def _write_summary(out_dir, matched, check_ok: bool | None, gate,
                   n_docs, n_questions, mode) -> None:
    ks = sorted(C.RECALL_KS)
    k1 = min(ks)
    lines = ["# Stage 8 - fine-tuned reranker evaluation", "",
             f"- Mode: **{mode}** — {n_docs} docs / {n_questions} questions",
             f"- Base reranker: `{C.RERANKER_MODEL}` (off-the-shelf arm)",
             "- Arms share one identical BGE top-20 pool per config.", ""]
    if check_ok is not None:
        lines.append("- Built-in check vs stage6/final: "
                     + ("**OK (exact)**" if check_ok else "**FAILED — do not "
                        "trust the ft rows until explained**"))
    if gate[1] is not None:
        lines.append(f"- Go/no-go gate (fixed 15/0, ΔR@{k1} ft−ots = "
                     f"{gate[1]:+.4f}, threshold {C.STAGE8_GO_THRESHOLD}): "
                     f"**{gate[0]}**")
    lines += ["", "## Per-config results", ""]
    header = (f"| config | pool@{DEPTH} | bge R@{k1} | ots R@{k1} "
              f"| ft R@{k1} | ft−ots ΔR@{k1} | ft−bge ΔR@{k1} | ft R@5 |")
    lines += [header, "|" + "---|" * 8]
    for m in matched:
        lines.append(
            f"| {m['method']} {m['chunk_config']} "
            f"| {m[f'pool_recall@{DEPTH}']:.4f} "
            f"| {m[f'bge_recall@{k1}']:.4f} | {m[f'ots_recall@{k1}']:.4f} "
            f"| {m[f'ft_recall@{k1}']:.4f} | {m[f'ft_minus_ots@{k1}']:+.4f} "
            f"| {m[f'ft_minus_bge@{k1}']:+.4f} | {m['ft_recall@5']:.4f} |")
    path = out_dir / C.STAGE8_SUMMARY_MD
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stage8] wrote {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 8: evaluate the fine-tuned reranker (dev go/no-go "
                    "or the final Stage 6 bench).")
    ap.add_argument("--dev", action="store_true",
                    help="go/no-go mode on the held-out NQ-train dev bench")
    ap.add_argument("--ft-model", default=None,
                    help="fine-tuned checkpoint dir (default: "
                         "models/bge_reranker_ft/final)")
    ap.add_argument("--fresh", action="store_true",
                    help="discard this mode's checkpoint and start over")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="dense retrieval model; must match the archives")
    args = ap.parse_args()

    mode = "dev" if args.dev else "final"
    ft_dir = pathlib.Path(args.ft_model) if args.ft_model else _default_ft_dir()
    if not (ft_dir / "config.json").exists():
        raise SystemExit(f"[stage8] no fine-tuned model at {ft_dir} — run "
                         "`python scripts/17_train_reranker.py` first")

    stage6_rows = None
    if mode == "final":
        stage6_dir = _stage_final_dir("stage6")
        s6_csv = stage6_dir / C.STAGE6_RESULTS_CSV
        if not s6_csv.exists():
            raise SystemExit(f"[stage8] missing {s6_csv} — the final mode "
                             "checks itself against the Stage 6 archive")
        stage6_rows = _read_csv(s6_csv)
        retr = {r.get("retrieval_embedding_model", "") for r in stage6_rows}
        if retr != {args.retrieval_model}:
            raise SystemExit(f"[stage8] Stage 6 archive retrieval model(s) "
                             f"{sorted(retr)} != {args.retrieval_model!r}")
        rer = {r.get("reranker_model", "") for r in stage6_rows
               if str(r.get("arm", "")).startswith("rerank")}
        if rer and rer != {C.RERANKER_MODEL}:
            raise SystemExit(f"[stage8] Stage 6 archive reranker {sorted(rer)}"
                             f" != {C.RERANKER_MODEL!r} (the off-the-shelf arm "
                             "must match)")

    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()

    from rag_chunk import rerank_finetune as rf

    models = {}
    if mode == "final":
        # The Stage 6 corpus caches under its own folder; never rebuild it.
        n = int(C.N_NQ_DOCS_LARGE)
        C.apply(N_NQ_DOCS=n)
        large_nq = C.NQ_DIR / f"large_n{n}"
        C.apply(NQ_DIR=large_nq)
        from rag_chunk import nq_data, training
        models["bilstm"] = training.load_model("bilstm")
        models["transformer"] = training.load_model("transformer")
        docs, questions = nq_data.prepare_nq()
        dataset = "nq-large"
        configs = list(C.STAGE6_RERANK_CONFIGS)
    else:
        docs, questions = rf.load_bench("dev")
        if not docs or not questions:
            raise SystemExit("[stage8] dev bench missing — run "
                             "`python scripts/16_build_rerank_train_data.py` "
                             "first")
        dataset = "nq-train-dev"
        configs = list(DEV_CONFIGS)
    print(f"[stage8] mode={mode} eval set ({dataset}): {len(docs)} docs, "
          f"{len(questions)} questions")
    print(f"[stage8] off-the-shelf: {C.RERANKER_MODEL}")
    print(f"[stage8] fine-tuned:    {ft_dir}")

    # Two cross-encoders, loaded once, bypassing the rerank.py singleton.
    from sentence_transformers import CrossEncoder
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = None
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
    ckpt = latest / CHECKPOINT_FILES[mode]
    if args.fresh and ckpt.exists():
        ckpt.unlink()
        print(f"[stage8] --fresh: discarded checkpoint {ckpt.name}")
    _clean_latest()

    rows = _run_configs(configs, docs, questions, scorers=scorers, mode=mode,
                        dataset=dataset, checkpoint_path=ckpt, models=models)

    matched = _matched_summary(rows)
    gate = _gate_verdict(matched)
    k1 = min(C.RECALL_KS)

    if mode == "dev":
        _write_rows_csv(rows, latest / C.STAGE8_DEV_RESULTS_CSV)
        se2 = _two_se(rows, len(questions))
        print(f"\n[stage8] dev bench ({len(questions)} questions, "
              f"2 SE = {se2:.4f}):")
        for m in matched:
            print(f"[stage8]   {m['method']} {m['chunk_config']}: "
                  f"bge R@{k1}={m[f'bge_recall@{k1}']:.4f}  "
                  f"ots={m[f'ots_recall@{k1}']:.4f}  "
                  f"ft={m[f'ft_recall@{k1}']:.4f}  "
                  f"ft−ots={m[f'ft_minus_ots@{k1}']:+.4f}")
        print(f"\n[stage8] GO/NO-GO verdict (fixed 15/0, threshold "
              f"+{C.STAGE8_GO_THRESHOLD}): {gate[0]} "
              f"(ΔR@{k1} ft−ots = {gate[1]:+.4f})")
        if gate[0] == "GO":
            print("[stage8] -> run the final eval: "
                  "python scripts/18_eval_reranker_ft.py")
        elif gate[0] == "NO-GO":
            print("[stage8] -> stop-loss: archive this as the honest negative "
                  "result (no final run).")
        else:
            print("[stage8] -> gray zone: at most ONE retry (more data / "
                  "epochs), then decide. No threshold-shopping.")
        gate_md = latest / "stage8_dev_gate.md"
        gate_md.write_text(
            f"# Stage 8 go/no-go (dev bench)\n\n- dev bench: {len(docs)} docs"
            f" / {len(questions)} questions (NQ train split, disjoint)\n"
            f"- ΔR@{k1} (ft − off-the-shelf) at fixed 15/0: {gate[1]:+.4f}\n"
            f"- threshold: +{C.STAGE8_GO_THRESHOLD}; 2 SE = {se2:.4f}\n"
            f"- verdict: **{gate[0]}**\n", encoding="utf-8")
        print(f"[stage8] wrote {gate_md}")
        return

    # final mode
    _write_rows_csv(rows, latest / C.STAGE8_RESULTS_CSV)
    check_ok = _write_stage6_check(stage6_rows, rows,
                                   latest / C.STAGE8_CHECK_CSV)
    _write_matched_csv(matched, latest / C.STAGE8_MATCHED_CSV)
    _plot_delta(matched, len(questions), latest / C.STAGE8_DELTA_PNG)
    _write_summary(latest, matched, check_ok, gate, len(docs),
                   len(questions), mode)

    se2 = _two_se(rows, len(questions))
    size15 = [m[f"ft_minus_ots@{k1}"] for m in matched
              if "15" in m["chunk_config"]]
    print(f"\n[stage8] headline: max size-15 ΔR@{k1} (ft−ots) = "
          f"{max(size15):+.4f} vs 2 SE = {se2:.4f}"
          if size15 else "\n[stage8] no size-15 configs found")
    print("[stage8] complete. Review, then archive:")
    print("  python scripts/save_stage_results.py --stage stage8")


if __name__ == "__main__":
    main()
