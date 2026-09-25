"""Stage 11 (steps 2-3) - evaluate the RL objective against its listwise control.

Every arm reorders one identical BGE top-20 pool per config, so the comparison
is purely about ranking:

    bge              dense order (pipeline check)
    rerank20         off-the-shelf cross-encoder (pipeline check, final only)
    rerank20_ft      the Stage 8 weights both Stage 11 arms started from
    rerank20_s11_ce  + one epoch of listwise cross-entropy on fresh groups (control)
    rerank20_s11_rl  + one epoch of the policy-gradient objective on the same groups

Per-question hits are kept for every arm, so differences are paired: two arms
answering the same questions share most of their noise, and the paired interval
is far tighter than comparing two recalls independently.

Modes:

    --dev     cost gate on the Stage 8 dev bench at fixed 15/0.
    (default) the Stage 6 bench at fixed 15/0, the primary claim. Add
              ``--configs 15:0,6:0`` for the secondary fixed 6/0 rows; the
              checkpoint is shared, so that resumes instead of re-scoring 15/0.

Checks. ``bge`` and ``rerank20`` are deterministic re-runs of archived rows and
must reproduce ``stage8/final`` exactly, or the run is INVALID. ``rerank20_ft``
is compared with its archived row too, but only reported: the Stage 8 weights
on Drive were retrained after the originals were lost, so a small difference
there measures training nondeterminism, not a broken pipeline. The Stage 11
comparisons use the current weights throughout, so they are unaffected.

Usage:
    python scripts/28_eval_reranker_rl.py --dev
    python scripts/28_eval_reranker_rl.py                       # primary, fixed 15/0
    python scripts/28_eval_reranker_rl.py --configs 15:0,6:0    # + secondary
    python scripts/28_eval_reranker_rl.py --max-questions 40    # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

DEPTH = 20
CHECK_TOLERANCE = 0.005
SMOKE_PREFIX = "smoke_"
ARM_BGE, ARM_OTS, ARM_FT = "bge", "rerank20", "rerank20_ft"
ARM_CE, ARM_RL = "rerank20_s11_ce", "rerank20_s11_rl"
DEV_CONFIGS = ((15, 0),)
FINAL_CONFIGS = ((15, 0), (6, 0))
PAIRS = ((ARM_RL, ARM_CE), (ARM_RL, ARM_FT), (ARM_CE, ARM_FT))


def _hits(retrieved, questions, k: int) -> list[int]:
    """Per-question doc-constrained hit@k, the rule metrics.py uses."""
    from rag_chunk.metrics import normalize_text

    out = []
    for q, chunks in zip(questions, retrieved):
        ans = normalize_text(q["answer"])
        gold = tuple(q.get("doc_titles") or ()) or (q.get("doc_title"),)
        out.append(int(any(ans in normalize_text(c["text"]) and c["doc_id"] in gold
                           for c in chunks[:k])))
    return out


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _parse_configs(spec: str | None) -> tuple:
    """Final-mode configs; the primary, fixed 15/0, is always included."""
    if not spec:
        return ((15, 0),)
    chosen: list[tuple[int, int]] = []
    for part in spec.split(","):
        size, overlap = (int(x) for x in part.strip().split(":"))
        if (size, overlap) not in FINAL_CONFIGS:
            raise SystemExit(f"[stage11] unknown config {part!r}; choose from "
                             + ", ".join(f"{a}:{b}" for a, b in FINAL_CONFIGS))
        if (size, overlap) not in chosen:
            chosen.append((size, overlap))
    if (15, 0) not in chosen:
        raise SystemExit("[stage11] the primary config 15:0 must be included")
    return tuple(sorted(chosen, reverse=True))


def _eval_config(docs, questions, size, overlap, scorers) -> list[dict]:
    from rag_chunk import metrics, retrieval
    from rag_chunk.rerank import rerank_order

    dense = retrieval.build_index_for_config(
        "fixed", docs, fixed_size=size, fixed_overlap=overlap)
    queries = [q["question"] for q in questions]
    maxk = max(C.RECALL_KS)
    pool = dense.search_chunks(queries, DEPTH)
    pool_rec = metrics.recall_from_retrieved(pool, questions, (DEPTH,))
    pairs, counts = [], []
    for qtext, cands in zip(queries, pool):
        pairs.extend((qtext, c["text"]) for c in cands)
        counts.append(len(cands))

    retrieved_by_arm = {ARM_BGE: ([cands[:maxk] for cands in pool], None)}
    for arm, scorer in scorers.items():
        t0 = time.perf_counter()
        scores = scorer(pairs)
        seconds = time.perf_counter() - t0
        retrieved, off = [], 0
        for cands, n in zip(pool, counts):
            order = rerank_order(scores[off:off + n])
            retrieved.append([cands[j] for j in order[:maxk]])
            off += n
        retrieved_by_arm[arm] = (retrieved, seconds)
        print(f"[stage11]   {arm}: {len(pairs)} pairs in {seconds / 60:.1f} min",
              flush=True)

    rows = []
    for arm, (retrieved, seconds) in retrieved_by_arm.items():
        rec = metrics.recall_from_retrieved(retrieved, questions, C.RECALL_KS)
        h1, h5 = _hits(retrieved, questions, 1), _hits(retrieved, questions, 5)
        # the paired analysis must count exactly what the archived metric counts
        for k, h in ((1, h1), (5, h5)):
            if abs(sum(h) / len(h) - rec["doc_constrained"][k]) > 1e-12:
                raise SystemExit(f"[stage11] per-question hits disagree with "
                                 f"metrics.py for {arm} at k={k}")
        row = {"arm": arm, "method": "fixed", "fixed_size": size,
               "fixed_overlap": overlap, "n_chunks": len(dense.chunk_texts),
               "avg_chunk_size": dense.avg_chunk_size(),
               f"pool_recall@{DEPTH}": pool_rec["doc_constrained"][DEPTH],
               "n_docs": len(docs), "n_questions": len(questions),
               "rerank_seconds": seconds}
        for k in C.RECALL_KS:
            row[f"recall@{k}"] = rec["doc_constrained"][k]
        row["hits1"], row["hits5"] = h1, h5
        rows.append(row)
    return rows


def _write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    rows = [{k: v for k, v in r.items() if k not in ("hits1", "hits5")}
            for r in rows]
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stage11] wrote {path} ({len(rows)} rows)")


def _paired(rows) -> list[dict]:
    by = {(r["fixed_size"], r["fixed_overlap"], r["arm"]): r for r in rows}
    configs = sorted({(r["fixed_size"], r["fixed_overlap"]) for r in rows},
                     reverse=True)
    out = []
    for size, overlap in configs:
        for a, b in PAIRS:
            ra, rb = by.get((size, overlap, a)), by.get((size, overlap, b))
            if ra is None or rb is None:
                continue
            for metric, key in (("recall@1", "hits1"), ("recall@5", "hits5")):
                diffs = [x - y for x, y in zip(ra[key], rb[key])]
                mean, lo, hi = _mean_ci(diffs)
                out.append({"config": f"fixed {size}/{overlap}", "metric": metric,
                            "comparison": f"{a} - {b}", "mean": round(mean, 4),
                            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                            "n_questions": len(diffs)})
    return out


def _check(rows, archived) -> tuple[bool, list[dict]]:
    """bge and rerank20 must reproduce stage8/final; rerank20_ft is reported."""
    arch = {}
    for r in archived:
        if r.get("method") == "fixed":
            arch[(int(float(r["fixed_size"])), int(float(r["fixed_overlap"])),
                  r["arm"])] = r
    out, ok = [], True
    for r in rows:
        if r["arm"] not in (ARM_BGE, ARM_OTS, ARM_FT):
            continue
        ref = arch.get((r["fixed_size"], r["fixed_overlap"], r["arm"]))
        rec = {"arm": r["arm"], "config": f"fixed {r['fixed_size']}/{r['fixed_overlap']}",
               "gates_validity": r["arm"] != ARM_FT}
        if ref is None:
            rec["status"] = "no archived row"
            ok = ok and r["arm"] == ARM_FT
            out.append(rec)
            continue
        worst = 0.0
        for k in C.RECALL_KS:
            old, new = float(ref[f"recall@{k}"]), round(r[f"recall@{k}"], 4)
            rec[f"archived_recall@{k}"] = f"{old:.4f}"
            rec[f"now_recall@{k}"] = f"{new:.4f}"
            rec[f"delta_recall@{k}"] = f"{new - old:+.4f}"
            worst = max(worst, abs(new - old))
        chunks_ok = int(float(ref["n_chunks"])) == r["n_chunks"]
        rec["n_chunks_match"] = chunks_ok
        passed = chunks_ok and worst <= CHECK_TOLERANCE
        rec["status"] = "match" if passed else "differs"
        if r["arm"] != ARM_FT:
            ok = ok and passed
        out.append(rec)
    return ok, out


def _plot(paired, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [p for p in paired if p["metric"] == "recall@1"
            and p["comparison"] in (f"{ARM_CE} - {ARM_FT}", f"{ARM_RL} - {ARM_FT}")]
    if not rows:
        return
    configs = sorted({p["config"] for p in rows}, reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.36
    for offset, arm, colour, label in (
            (-width / 2, ARM_CE, "#7f7f7f", "+1 epoch listwise CE (control)"),
            (width / 2, ARM_RL, "#2a7ab0", "+1 epoch policy gradient (RL)")):
        xs, ys, lo, hi = [], [], [], []
        for i, cfg in enumerate(configs):
            p = next((p for p in rows if p["config"] == cfg
                      and p["comparison"] == f"{arm} - {ARM_FT}"), None)
            if p is None:
                continue
            xs.append(i + offset)
            ys.append(p["mean"])
            lo.append(p["mean"] - p["ci95_low"])
            hi.append(p["ci95_high"] - p["mean"])
        ax.bar(xs, ys, width, color=colour, label=label,
               yerr=[lo, hi], capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(configs)), configs)
    ax.set_ylabel("R@1 change vs Stage 8 (paired 95% CI)")
    ax.set_title("Stage 11: R@1 change from one more epoch, by objective")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[stage11] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 11: evaluate the CE and RL continuation arms.")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--configs", default=None,
                    help="final mode: size:overlap list from 15:0,6:0 "
                         "(default 15:0, the primary claim)")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="SMOKE ONLY: first N questions, outputs prefixed, no verdict")
    args = ap.parse_args()
    mode = "dev" if args.dev else "final"
    smoke = args.max_questions is not None

    from rag_chunk import rerank_rl as rr

    ft_dir = C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"
    arm_dirs = {ARM_CE: C.MODELS_DIR / C.STAGE11_MODEL_DIRNAME / "ce",
                ARM_RL: C.MODELS_DIR / C.STAGE11_MODEL_DIRNAME / "rl"}
    metas = {arm: rr.arm_meta(d) for arm, d in arm_dirs.items()}
    missing = [arm for arm, m in metas.items() if m is None]
    if missing:
        raise SystemExit(f"[stage11] no completed training for {missing} - run "
                         "scripts/27_train_reranker_rl.py first")
    live = metas[ARM_RL]["live_fraction"]

    archived = None
    if mode == "final":
        s8 = C.RESULTS_DIR / "stage8" / "final" / C.STAGE8_RESULTS_CSV
        if not s8.exists():
            raise SystemExit(f"[stage11] missing {s8} - the final mode checks "
                             "itself against the Stage 8 archive")
        with open(s8, newline="", encoding="utf-8") as fh:
            archived = list(csv.DictReader(fh))

    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()

    from rag_chunk.large_eval import append_checkpoint, load_checkpoint

    if mode == "final":
        n = int(C.N_NQ_DOCS_LARGE)
        C.apply(N_NQ_DOCS=n)
        C.apply(NQ_DIR=C.NQ_DIR / f"large_n{n}")
        from rag_chunk import nq_data
        docs, questions = nq_data.prepare_nq()
        configs = _parse_configs(args.configs)
    else:
        from rag_chunk import rerank_finetune as rf
        docs, questions = rf.load_bench("dev")
        configs = DEV_CONFIGS
    if smoke:
        questions = questions[:args.max_questions]
        print(f"[stage11] *** SMOKE MODE: {len(questions)} questions ***")
    print(f"[stage11] mode={mode}: {len(docs)} docs / {len(questions)} questions")

    from sentence_transformers import CrossEncoder
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = int(C.RERANK_MAX_LENGTH)
    model_paths = {ARM_FT: str(ft_dir), ARM_CE: str(arm_dirs[ARM_CE]),
                   ARM_RL: str(arm_dirs[ARM_RL])}
    if mode == "final":
        model_paths = {ARM_OTS: C.RERANKER_MODEL, **model_paths}
    models = {arm: CrossEncoder(path, max_length=max_len, device=device)
              for arm, path in model_paths.items()}

    def scorer(model):
        def run(pairs):
            if not pairs:
                return np.zeros(0, dtype="float32")
            return np.asarray(model.predict(pairs, batch_size=int(C.RERANK_BATCH_SIZE),
                                            show_progress_bar=False), dtype="float32")
        return run

    scorers = {arm: scorer(m) for arm, m in models.items()}

    def out(name: str) -> pathlib.Path:
        return C.RESULTS_LATEST_DIR / ((SMOKE_PREFIX + name) if smoke else name)

    checkpoint = out(f"stage11_checkpoint_{mode}.jsonl")
    meta = {"stage": "stage11", "mode": mode, "n_docs": len(docs),
            "n_questions": len(questions), "retrieval_model": C.RETRIEVAL_EMBED_MODEL,
            "models": {arm: path for arm, path in model_paths.items()},
            "arm_fingerprints": {arm: metas[arm]["fingerprint"] for arm in metas}}
    if args.fresh and checkpoint.exists():
        checkpoint.unlink()
    done = load_checkpoint(checkpoint, meta)
    if not done and not checkpoint.exists():
        append_checkpoint(checkpoint, {"meta": meta})
    rows = list(done)
    done_configs = {(r["fixed_size"], r["fixed_overlap"]) for r in done}
    for size, overlap in configs:
        if (size, overlap) in done_configs:
            print(f"[stage11] fixed {size}/{overlap} already in the checkpoint")
            continue
        print(f"[stage11] fixed {size}/{overlap}", flush=True)
        new_rows = _eval_config(docs, questions, size, overlap, scorers)
        append_checkpoint(checkpoint, {"rows": new_rows})
        rows += new_rows

    paired = _paired(rows)
    _write_rows(out(C.STAGE11_DEV_CSV if mode == "dev" else C.STAGE11_RESULTS_CSV),
                rows)
    _write_rows(out(C.STAGE11_PAIRED_CSV.replace(".csv", f"_{mode}.csv")), paired)

    primary = next(p for p in paired if p["config"] == "fixed 15/0"
                   and p["metric"] == "recall@1"
                   and p["comparison"] == f"{ARM_RL} - {ARM_CE}")
    for p in paired:
        print(f"[stage11] {p['config']} {p['metric']:9s} {p['comparison']:36s} "
              f"{p['mean']:+.4f} [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]")
    print(f"[stage11] RL live fraction during training: {live:.3f}")

    if smoke:
        print("[stage11] SMOKE MODE: no verdict.")
        return

    min_live = float(C.STAGE11_MIN_LIVE_FRACTION)
    if mode == "dev":
        if live < min_live:
            verdict, why = "NO-GO", (f"only {live:.1%} of RL groups produced a "
                                     "gradient - a failed optimisation")
        elif primary["mean"] >= float(C.STAGE11_GO_THRESHOLD):
            verdict, why = "GO", (f"dev R@1 RL - CE = {primary['mean']:+.4f}, not "
                                  "behind the control")
        else:
            verdict, why = "NO-GO", (f"dev R@1 RL - CE = {primary['mean']:+.4f}, "
                                     "behind the control")
        gate = {"verdict": verdict, "why": why, "rl_minus_ce_r1": primary["mean"],
                "ci95": [primary["ci95_low"], primary["ci95_high"]],
                "live_fraction": live, "n_questions": len(questions)}
        out("stage11_dev_gate.json").write_text(json.dumps(gate, indent=2),
                                               encoding="utf-8")
        print(f"\n[stage11] DEV GATE: {verdict} - {why}.")
        print("[stage11] the gate only decides whether the final run happens.")
        return

    check_ok, check_rows = _check(rows, archived)
    _write_rows(out(C.STAGE11_CHECK_CSV), check_rows)
    floor = float(C.STAGE11_PRACTICAL_FLOOR)
    m, lo, hi = primary["mean"], primary["ci95_low"], primary["ci95_high"]
    if not check_ok:
        verdict, why = "INVALID", ("the dense or off-the-shelf rows did not "
                                   "reproduce stage8/final")
    elif live < min_live:
        verdict, why = "FAILED-OPTIMISATION", (f"only {live:.1%} of RL groups "
                                               "produced a gradient")
    elif lo > 0 and m >= floor:
        verdict, why = "RL-BETTER", (f"RL beats CE by {m:+.4f} R@1, 95% CI "
                                     f"[{lo:+.4f}, {hi:+.4f}], above the {floor} floor")
    elif hi < 0:
        verdict, why = "CE-BETTER", (f"RL trails CE by {m:+.4f} R@1, 95% CI "
                                     f"[{lo:+.4f}, {hi:+.4f}]")
    else:
        verdict, why = "TIE", (f"RL - CE = {m:+.4f} R@1, 95% CI [{lo:+.4f}, "
                               f"{hi:+.4f}]: not a detectable difference")

    _plot(paired, out(C.STAGE11_DELTA_PNG))
    ft_row = next((c for c in check_rows if c["arm"] == ARM_FT
                   and c["config"] == "fixed 15/0"), None)
    lines = ["# Stage 11 - the reranker under an RL objective", "",
             f"Verdict: {verdict}. {why}.", "",
             f"- Stage 6 bench: {len(docs)} docs / {len(questions)} questions; "
             f"all arms rerank one shared BGE top-{DEPTH} pool per config",
             f"- pipeline check (bge, off-the-shelf vs stage8/final): "
             f"{'PASS' if check_ok else 'FAIL'}",
             f"- RL groups with a non-zero advantage during training: {live:.3f}",
             f"- practical floor (pre-registered): {floor} R@1", ""]
    if ft_row is not None:
        lines += [f"- Stage 8 weights now vs archived at fixed 15/0: R@1 "
                  f"{ft_row.get('now_recall@1')} vs {ft_row.get('archived_recall@1')}"
                  " (the Drive weights were retrained; not used as a gate)", ""]
    lines += ["| config | arm | R@1 | R@3 | R@5 |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| fixed {r['fixed_size']}/{r['fixed_overlap']} | {r['arm']} | "
                     f"{r['recall@1']:.4f} | {r['recall@3']:.4f} | {r['recall@5']:.4f} |")
    lines += ["", "| config | metric | comparison | mean | 95% CI |",
              "| --- | --- | --- | --- | --- |"]
    for p in paired:
        lines.append(f"| {p['config']} | {p['metric']} | {p['comparison']} | "
                     f"{p['mean']:+.4f} | [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}] |")
    out(C.STAGE11_SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    out("stage11_verdict.json").write_text(json.dumps(
        {"verdict": verdict, "why": why, "check_ok": check_ok, "live_fraction": live,
         "rl_minus_ce_r1": m, "ci95": [lo, hi]}, indent=2), encoding="utf-8")
    print(f"\n[stage11] VERDICT: {verdict} - {why}.")


if __name__ == "__main__":
    main()
