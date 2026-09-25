"""Stage 10 (step 2) - score the RL boundary policy on the Stage 6 bench.

Step 1 trained the policy; this script scores it. The rules below were
fixed in ``docs/stage10_rl_chunking.md`` before the first run, so the
thresholds were not chosen after seeing the numbers.

1. Matched size. A chunker that gains by making chunks bigger reproduces the
   Stage 1 size effect. Every comparison is against the archived
   row at the same target size and overlap, and a config whose average chunk
   size drifts more than ``STAGE10_SIZE_TOLERANCE`` sentences from that row is
   reported INVALID.
2. The threshold is the MDE. ``scripts/20_effect_size.py`` put the
   80%-power minimum detectable effect at 0.046 R@5 for n=1032, with the largest
   method spread observed at 0.023. A gap under the MDE is reported as
   directional but undetectable.
3. Baselines come from the archive. Two archived configs are
   recomputed here; if they do not reproduce ``stage6/final``, the RL rows
   cannot be used and the run is void.

Usage:
    python scripts/24_eval_rl_chunker.py --max-questions 50   # smoke
    python scripts/24_eval_rl_chunker.py                      # the real run
    python scripts/24_eval_rl_chunker.py --fresh
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

CHECKPOINT_FILE = "stage10_eval_checkpoint.jsonl"
CHECK_TOLERANCE = 0.005
SMOKE_PREFIX = "smoke_"
# Recomputed to prove the bench and pipeline still match the archive: one
# fixed-size row and one learned row, both from the Stage 6 bge arm.
CHECK_CONFIGS = (("fixed", 15, 0), ("transformer", 15, 1))


def _read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(value):
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _size_overlap(row: dict) -> tuple:
    """(target size, overlap) of a row, whichever chunking family it is from."""
    if str(row["method"]).strip() == "fixed":
        return (_num(row.get("fixed_size")), _num(row.get("fixed_overlap")))
    return (_num(row.get("semantic_target_size")),
            _num(row.get("semantic_overlap")))


def _write_rows(path: pathlib.Path, rows: list[dict]) -> None:
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
    print(f"[stage10] wrote {path} ({len(rows)} rows)")


def _eval_one(method, size, overlap, docs, questions, probs_by_model,
              label=None) -> dict:
    """One (method, size, overlap) row, built by the archived sweep's own path."""
    from rag_chunk import metrics, retrieval, sweep

    label = label or method
    if method == "fixed":
        index = retrieval.build_index_for_config(
            "fixed", docs, fixed_size=size, fixed_overlap=overlap)
        shape = {"fixed_size": size, "fixed_overlap": overlap}
    else:
        mn, mx = sweep._semantic_window(size)
        index = retrieval.build_index_for_config(
            "transformer", docs, semantic_policy="target",
            semantic_target_size=size, semantic_min_size=mn,
            semantic_max_size=mx, semantic_overlap=overlap,
            boundary_probs_by_id=probs_by_model[method])
        shape = {"semantic_policy": "target", "semantic_target_size": size,
                 "semantic_min_size": mn, "semantic_max_size": mx,
                 "semantic_overlap": overlap}
    rec = metrics.recall_at_k(index, questions, C.RECALL_KS)
    row = {"arm": "bge", "method": label,
           "embedding_model": C.RETRIEVAL_EMBED_MODEL,
           "retrieval_embedding_model": C.RETRIEVAL_EMBED_MODEL,
           "avg_chunk_size": index.avg_chunk_size(),
           "n_chunks": len(index.chunk_texts),
           "n_docs": len(docs), "n_questions": len(questions)}
    row.update(shape)
    for k in C.RECALL_KS:
        row[f"recall@{k}"] = rec["doc_constrained"][k]
    for k in C.RECALL_KS:
        row[f"recall_unconstrained@{k}"] = rec["unconstrained"][k]
    return row


def _write_check(archived, rows, path) -> bool:
    """Recomputed archived configs must reproduce stage6/final."""
    ks = sorted(C.RECALL_KS)
    by_key = {(str(r["method"]).strip(), _size_overlap(r)): r for r in archived}
    out, worst, unmatched = [], 0.0, 0
    for row in rows:
        key = (str(row["method"]).strip(), _size_overlap(row))
        ref = by_key.get(key)
        rec = {"method": row["method"], "size": key[1][0], "overlap": key[1][1]}
        if ref is None:
            unmatched += 1
            out.append(rec)
            continue
        for k in ks:
            old = float(ref[f"recall@{k}"])
            new = round(row[f"recall@{k}"], 4)
            delta = round(new - old, 4)
            worst = max(worst, abs(delta))
            rec[f"stage6_recall@{k}"] = f"{old:.4f}"
            rec[f"stage10_recall@{k}"] = f"{new:.4f}"
            rec[f"delta_recall@{k}"] = f"{delta:.4f}"
        out.append(rec)
    _write_rows(path, out)
    ok = unmatched == 0 and worst <= CHECK_TOLERANCE
    if unmatched:
        print(f"[stage10] WARN: {unmatched} check row(s) had no Stage 6 match")
    if ok and worst == 0.0:
        print("[stage10] check OK: the recomputed baselines reproduce "
              "stage6/final exactly.")
    elif ok:
        print(f"[stage10] check: max |delta| vs Stage 6 = {worst:.4f} "
              "(within one question).")
    else:
        print(f"[stage10] WARN: check FAILED (max |delta| = {worst:.4f}) - the "
              "RL rows cannot be used until this is explained.")
    return ok


def _matched(archived, rl_rows) -> list[dict]:
    """Per (size, overlap): the RL row against every archived method."""
    ks = sorted(C.RECALL_KS)
    baselines: dict[tuple, dict] = {}
    for r in archived:
        baselines.setdefault(_size_overlap(r), {})[str(r["method"]).strip()] = r

    out = []
    for row in sorted(rl_rows, key=_size_overlap):
        key = _size_overlap(row)
        peers = baselines.get(key, {})
        rec = {"target_size": key[0], "overlap": key[1],
               "rl_avg_chunk_size": round(row["avg_chunk_size"], 3)}
        for k in ks:
            rec[f"rl_recall@{k}"] = round(row[f"recall@{k}"], 4)
        best_name, best_r5 = None, None
        for name in ("fixed", "bilstm", "transformer"):
            peer = peers.get(name)
            if peer is None:
                continue
            r5 = float(peer[f"recall@{max(ks)}"])
            rec[f"{name}_recall@{max(ks)}"] = f"{r5:.4f}"
            rec[f"delta_vs_{name}@{max(ks)}"] = round(
                row[f"recall@{max(ks)}"] - r5, 4)
            rec[f"{name}_avg_chunk_size"] = f"{float(peer['avg_chunk_size']):.3f}"
            if best_r5 is None or r5 > best_r5:
                best_name, best_r5 = name, r5
        if best_r5 is not None:
            drift = abs(row["avg_chunk_size"]
                        - float(peers[best_name]["avg_chunk_size"]))
            rec["best_baseline"] = best_name
            rec[f"delta_vs_best@{max(ks)}"] = round(
                row[f"recall@{max(ks)}"] - best_r5, 4)
            rec["size_drift"] = round(drift, 3)
            rec["size_matched"] = drift <= C.STAGE10_SIZE_TOLERANCE
        out.append(rec)
    return out


def _verdict(matched: list[dict]) -> tuple[str, str]:
    k = max(sorted(C.RECALL_KS))
    primary = next(
        (r for r in matched
         if r["target_size"] == C.STAGE10_TARGET_SIZE
         and r["overlap"] == C.STAGE10_OVERLAP), None)
    if primary is None:
        return "INCOMPLETE", (
            f"the primary config (target {C.STAGE10_TARGET_SIZE}, overlap "
            f"{C.STAGE10_OVERLAP}) was not evaluated")
    delta = primary.get(f"delta_vs_best@{k}")
    if not primary.get("size_matched", False):
        return "INVALID", (
            f"average chunk size drifted {primary['size_drift']} sentences from "
            f"the matched baseline (tolerance {C.STAGE10_SIZE_TOLERANCE}); a "
            "recall gain here would come from chunk size")
    if delta is None:
        return "INCOMPLETE", "no archived baseline at the primary config"
    if delta >= C.STAGE10_MDE:
        return "OVERTURNS", (
            f"R@{k} beats the best matched baseline by {delta:+.4f}, above the "
            f"pre-registered MDE of {C.STAGE10_MDE}")
    if delta > 0:
        return "DIRECTIONAL", (
            f"R@{k} is {delta:+.4f} over the best matched baseline: positive "
            f"but below the {C.STAGE10_MDE} detection floor, so it does not "
            "support a claim")
    return "NULL", (
        f"R@{k} is {delta:+.4f} against the best matched baseline: optimising "
        "the retrieval metric directly does not beat fixed-size chunking, so "
        "the method tie still holds with the training objective changed")


def _plot(matched, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = max(sorted(C.RECALL_KS))
    rows = [r for r in matched if f"delta_vs_best@{k}" in r]
    if not rows:
        return
    labels = [f"{r['target_size']}/{r['overlap']}" for r in rows]
    deltas = [r[f"delta_vs_best@{k}"] for r in rows]
    colours = ["#2a7ab0" if d >= C.STAGE10_MDE else "#b0b0b0" for d in deltas]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, deltas, color=colours)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(C.STAGE10_MDE, color="#c0392b", linestyle="--", linewidth=1.0,
               label=f"80% MDE ({C.STAGE10_MDE})")
    ax.axhline(-C.STAGE10_MDE, color="#c0392b", linestyle="--", linewidth=1.0)
    ax.set_xlabel("target chunk size / overlap")
    ax.set_ylabel(f"RL - best matched baseline (R@{k})")
    ax.set_title("Stage 10: retrieval-trained chunking vs the archived baselines")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[stage10] wrote {path}")


def _summary(matched, verdict, why, check_ok, n_questions, path) -> None:
    k = max(sorted(C.RECALL_KS))
    lines = [
        "# Stage 10 - chunking trained on retrieval recall", "",
        f"Verdict: {verdict}. {why}.", "",
        f"- bench: {n_questions} questions, doc-constrained Recall@k",
        f"- reproduction check vs stage6/final: "
        f"{'PASS' if check_ok else 'FAIL'}",
        f"- pre-registered MDE: {C.STAGE10_MDE} R@{k} (80% power at n=1032)",
        f"- size tolerance: {C.STAGE10_SIZE_TOLERANCE} sentences", "",
        f"| size/overlap | RL R@{k} | best baseline | delta | size drift | matched |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    positive = 0
    for row in matched:
        delta = row.get(f"delta_vs_best@{k}")
        if delta is not None and delta > 0:
            positive += 1
        lines.append(
            f"| {row['target_size']}/{row['overlap']} | "
            f"{row.get(f'rl_recall@{k}')} | {row.get('best_baseline', '-')} | "
            f"{delta:+.4f} | {row.get('size_drift', '-')} | "
            f"{'yes' if row.get('size_matched') else 'NO'} |"
            if delta is not None else
            f"| {row['target_size']}/{row['overlap']} | "
            f"{row.get(f'rl_recall@{k}')} | - | - | - | - |")
    lines += ["", f"Positive deltas: {positive}/{len(matched)} configs.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[stage10] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 10 step 2: evaluate the RL boundary policy on the "
                    "Stage 6 bench against the archived baselines.")
    ap.add_argument("--configs", default="all",
                    help="'all' (the 5 target sizes x 2 overlaps) or e.g. "
                         "'15:0,15:1'")
    ap.add_argument("--fresh", action="store_true",
                    help="discard the checkpoint and start over")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="SMOKE ONLY: score just the first N questions. The "
                         "numbers are not comparable to the archives, the "
                         "reproduction check is skipped and every output is "
                         "written under a smoke_ prefix.")
    args = ap.parse_args()

    if args.configs == "all":
        configs = [(s, o) for s in C.TARGET_SIZE_GRID for o in C.OVERLAP_GRID]
    else:
        configs = []
        for spec in args.configs.split(","):
            size, overlap = spec.split(":")
            configs.append((int(size), int(overlap)))

    stage6_csv = C.RESULTS_DIR / "stage6" / "final" / C.STAGE6_RESULTS_CSV
    if not stage6_csv.exists():
        raise SystemExit(f"[stage10] missing {stage6_csv}; Stage 10 is scored "
                         "against the Stage 6 archive")
    archived = [r for r in _read_csv(stage6_csv)
                if str(r.get("arm", "")).strip() == "bge"]

    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()
    n = int(C.N_NQ_DOCS_LARGE)
    C.apply(N_NQ_DOCS=n)
    C.apply(NQ_DIR=C.NQ_DIR / f"large_n{n}")

    from rag_chunk import nq_data, rl_chunking as rl, sweep, training
    from rag_chunk.large_eval import append_checkpoint, load_checkpoint

    policy_path = C.model_path("rl")
    if not policy_path.exists():
        raise SystemExit(f"[stage10] no policy at {policy_path}; run "
                         "scripts/23_train_rl_chunker.py first")

    docs, questions = nq_data.prepare_nq()
    smoke = args.max_questions is not None
    if smoke:
        questions = questions[:args.max_questions]
        print(f"[stage10] *** SMOKE MODE: {len(questions)} questions; results "
              "are not comparable to the archives ***")
    print(f"[stage10] bench: {len(docs)} docs / {len(questions)} questions")

    def out(name: str) -> pathlib.Path:
        return C.RESULTS_LATEST_DIR / ((SMOKE_PREFIX + name) if smoke else name)

    policy = rl.load_policy()
    supervised = training.load_model("transformer")
    print("[stage10] precomputing boundary probabilities (one encode pass)")
    probs_by_model = sweep._precompute_boundary_probs(
        docs, {"rl": policy, "transformer": supervised})

    checkpoint = out(CHECKPOINT_FILE)
    meta = {"stage": "stage10", "n_docs": len(docs),
            "n_questions": len(questions),
            "retrieval_model": C.RETRIEVAL_EMBED_MODEL,
            "policy": str(policy_path)}
    if args.fresh and checkpoint.exists():
        checkpoint.unlink()
    done = load_checkpoint(checkpoint, meta)
    if not done and not checkpoint.exists():
        append_checkpoint(checkpoint, {"meta": meta})
    done_keys = {(str(r["method"]).strip(), _size_overlap(r)) for r in done}
    rows = list(done)
    if done:
        print(f"[stage10] resuming: {len(done)} row(s) already done")

    plan = [("rl", s, o) for s, o in configs]
    if not smoke:
        plan += [(m, s, o) for m, s, o in CHECK_CONFIGS]
    for i, (method, size, overlap) in enumerate(plan, 1):
        if (method, (size, overlap)) in done_keys:
            continue
        print(f"[stage10] ({i}/{len(plan)}) {method} size={size} "
              f"overlap={overlap}", flush=True)
        row = _eval_one(method, size, overlap, docs, questions, probs_by_model)
        append_checkpoint(checkpoint, {"rows": [row]})
        rows.append(row)

    rl_rows = [r for r in rows if str(r["method"]).strip() == "rl"]
    check_rows = [r for r in rows if str(r["method"]).strip() != "rl"]
    _write_rows(out(C.STAGE10_RESULTS_CSV), rows)

    check_ok = True
    if check_rows:
        check_ok = _write_check(archived, check_rows, out(C.STAGE10_CHECK_CSV))

    matched = _matched(archived, rl_rows)
    _write_rows(out(C.STAGE10_MATCHED_CSV), matched)
    verdict, why = _verdict(matched)
    _plot(matched, out(C.STAGE10_DELTA_PNG))
    _summary(matched, verdict, why, check_ok, len(questions),
             out(C.STAGE10_SUMMARY_MD))

    print(f"\n[stage10] VERDICT: {verdict} - {why}.")
    if smoke:
        print("[stage10] SMOKE MODE: not results, and the smoke_ outputs "
              "must not be archived.")
    elif check_ok:
        print("[stage10] archive with: python scripts/save_stage_results.py "
              "--stage stage10")
    else:
        print("[stage10] do not archive until the reproduction check passes.")


if __name__ == "__main__":
    main()
