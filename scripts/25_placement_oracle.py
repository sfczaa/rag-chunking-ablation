"""Stage 10 (oracle) - the ceiling for any placement of the cuts.

The first RL run failed to optimise, and its by-product explains why: inside the
decode window a uniformly random cut scored the same as the supervised model's
cut (training-reward difference +0.0050, 95% CI [-0.0034, +0.0134]). That
run could not show how good the best cut is. This script measures that
ceiling directly, with no learning involved.

For each evaluated document it scores complete chunkings of that document, all
drawn from the same target-size window the sweep and the RL policy use:

* ``fixed``      - fixed 15/0, the headline baseline;
* ``supervised`` - the Stage 2 transformer's greedy decode, the learned baseline;
* ``s0 .. sK-1`` - K chunkings with every cut drawn uniformly inside its window.

Every other document stays at its frozen fixed 15/0 chunking, so a question's
retrieval changes only through its own document's cuts. For each metric the
oracle takes, per question, the best of all candidates.

The oracle is a loose upper bound: it picks the chunking with the
question and its answer in hand, and may pick differently for two questions about
the same document. The pre-registered verdict (``docs/stage10_rl_chunking.md``)
leans on that asymmetry:

* if even this ceiling over fixed-size chunking sits confidently below the
  project's detection floor, no placement policy, RL or otherwise, can produce
  a detectable method effect at this size, and Stage 10 closes;
* if the ceiling clears the floor, headroom exists in principle, but how much of
  it a policy could capture without seeing the question is not measured here.

Cost model: the candidates of one document share many chunks (the first cut has
only ~9 possible positions), so each document's distinct chunk texts are encoded
in one batch; and the frozen corpus is scored once per question and merged with
each candidate's few new chunks rather than rebuilding the candidate pool.

Usage:
    python scripts/25_placement_oracle.py --smoke    # plumbing check, CPU-friendly
    python scripts/25_placement_oracle.py            # 400 documents, K = 16
    python scripts/25_placement_oracle.py --fresh
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

CHECKPOINT_FILE = "stage10_oracle_checkpoint.jsonl"
ARMS_CSV = "stage10_oracle_arms.csv"
DELTAS_CSV = "stage10_oracle_deltas.csv"
CURVE_CSV = "stage10_oracle_best_of_k.csv"
SUMMARY_MD = "stage10_oracle_summary.md"
CURVE_PNG = "stage10_oracle_best_of_k.png"
SMOKE_PREFIX = "smoke_"
DEFAULT_K = 16
DEFAULT_N_DOCS = 400
DEPTH = 10                     # same top-10 the RL reward used
CURVE_KS = (1, 2, 4, 8, 16)


def _hit(rank: int, k: int) -> float:
    return 1.0 if 0 < rank <= k else 0.0


METRICS = {
    "recall@1": lambda r: _hit(r, 1),
    "recall@5": lambda r: _hit(r, 5),
    "mrr@10": lambda r: 1.0 / r if r > 0 else 0.0,
}


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
    print(f"[oracle] wrote {path} ({len(rows)} rows)")


def _smoke_corpus(n_docs: int = 40):
    """A slice of the cached 200-doc NQ bench, for plumbing checks only."""
    from rag_chunk import nq_data

    docs = nq_data.load_docs()[:n_docs]
    if not docs:
        raise SystemExit("[oracle] no cached NQ bench; run scripts/1_prepare_data.py")
    keys = {d.get("title", d.get("id")) for d in docs}
    questions = [q for q in nq_data.load_questions() if q.get("doc_title") in keys]
    print("[oracle] SMOKE MODE: a slice of the validation bench. Plumbing check "
          "only; these numbers are not results.")
    return docs, questions


# --------------------------------------------------------------------------- #
# One document: every candidate chunking, scored against the frozen corpus
# --------------------------------------------------------------------------- #
def _first_hit_rank(base_top, new_sims, new_hits, depth: int) -> int:
    """Rank (1-based) of the first answer-bearing gold chunk in the merged top
    list, or 0. Base entries belong to other documents, so they can only push a
    hit down, never be one."""
    merged = [(float(s), False) for s in base_top]
    merged += [(float(s), bool(h)) for s, h in zip(new_sims, new_hits)]
    merged.sort(key=lambda item: -item[0])
    for rank, (_, is_hit) in enumerate(merged[:depth], start=1):
        if is_hit:
            return rank
    return 0


def _score_document(key, env, policy, k, rng, *, target, min_size, max_size,
                    overlap):
    """Ranks for every (candidate, question) of one document.

    Returns ``(record, texts_by_candidate)``; the record is what is checkpointed.
    """
    import numpy as np
    import torch

    from rag_chunk import chunking, embedding, metrics
    from rag_chunk.rl_chunking import rollout, spans_to_texts

    sents = env.sentences[key]
    n = len(sents)
    rows = env.q_rows_by_doc[key]
    qv = env.query_vecs[rows]
    answers = [env.answer_norm[r] for r in rows]
    first, last = env.base_rows[key]

    # The frozen corpus without this document, scored once for all candidates.
    base_sims = qv @ env.base_vecs.T
    base_sims[:, first:last] = -np.inf
    depth = min(DEPTH, env.base_vecs.shape[0] - (last - first))
    top = np.argpartition(-base_sims, depth - 1, axis=1)[:, :depth]
    base_top = np.take_along_axis(base_sims, top, axis=1)

    candidates: dict[str, tuple[list[str], list[int]]] = {}
    fixed = chunking.fixed_chunks(sents, target, overlap)
    candidates["fixed"] = ([chunking.chunk_text(ch) for ch in fixed],
                           [len(ch) for ch in fixed])

    common = dict(target_size=target, min_size=min_size, max_size=max_size,
                  overlap=overlap)
    if n >= 2:
        device = next(policy.parameters()).device
        with torch.no_grad():
            emb = torch.as_tensor(embedding.encode(sents, normalize=False),
                                  dtype=torch.float32, device=device)
            logits = policy(emb).cpu().numpy()
    else:
        logits = np.zeros(max(0, n - 1))
    spans, _ = rollout(logits, n, greedy=True, **common)
    candidates["supervised"] = (spans_to_texts(sents, spans),
                                [e - s for s, e in spans])

    uniform = np.zeros(max(0, n - 1))       # equal logits: every cut equally likely
    for j in range(k):
        spans, _ = rollout(uniform, n, greedy=False, rng=rng, temperature=1.0,
                           **common)
        candidates[f"s{j}"] = (spans_to_texts(sents, spans),
                               [e - s for s, e in spans])

    unique: list[str] = []
    index: dict[str, int] = {}
    for texts, _ in candidates.values():
        for text in texts:
            if text not in index:
                index[text] = len(unique)
                unique.append(text)
    vecs = embedding.encode_retrieval_passages(unique)
    norm = [metrics.normalize_text(t) for t in unique]
    new_sims_all = qv @ vecs.T if unique else np.zeros((len(rows), 0))

    record = {"doc": key, "n_sentences": n, "questions": len(rows),
              "candidates": {}}
    for name, (texts, sizes) in candidates.items():
        cols = [index[t] for t in texts]
        ranks = []
        for qi in range(len(rows)):
            hits = [answers[qi] in norm[c] for c in cols]
            ranks.append(_first_hit_rank(base_top[qi], new_sims_all[qi, cols],
                                         hits, depth))
        record["candidates"][name] = {"ranks": ranks, "n_chunks": len(texts),
                                      "sentences": int(sum(sizes))}
    return record, {name: texts for name, (texts, _) in candidates.items()}


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _summarise(records: list[dict], k: int) -> dict:
    samples = [f"s{j}" for j in range(k)]
    names = ["fixed", "supervised"] + samples
    per_q = [(rec, qi) for rec in records for qi in range(rec["questions"])]

    def val(rec, qi, name, metric):
        return METRICS[metric](rec["candidates"][name]["ranks"][qi])

    def avg(rec, name):
        c = rec["candidates"][name]
        return c["sentences"] / c["n_chunks"] if c["n_chunks"] else 0.0

    arms: dict[str, dict[str, list[float]]] = {}
    for metric in METRICS:
        arms[metric] = {
            "fixed": [val(r, q, "fixed", metric) for r, q in per_q],
            "supervised": [val(r, q, "supervised", metric) for r, q in per_q],
            "random_mean": [sum(val(r, q, s, metric) for s in samples) / k
                            for r, q in per_q],
            "oracle": [max(val(r, q, nm, metric) for nm in names)
                       for r, q in per_q],
        }

    # Average chunk size per arm, question-weighted and chunk-averaged like the
    # archived avg_chunk_size. The R@5 oracle breaks ties toward the candidate
    # closest in size to fixed, so any drift it reports is drift it needed.
    def totals(pairs):
        sent = sum(rec["candidates"][nm]["sentences"] for rec, nm in pairs)
        chunks = sum(rec["candidates"][nm]["n_chunks"] for rec, nm in pairs)
        return sent / chunks if chunks else float("nan")

    oracle_pick = []
    for rec, qi in per_q:
        best = max(val(rec, qi, nm, "recall@5") for nm in names)
        tied = [nm for nm in names if val(rec, qi, nm, "recall@5") == best]
        target = avg(rec, "fixed")
        pick = min(tied, key=lambda nm: (abs(avg(rec, nm) - target),
                                         names.index(nm)))
        oracle_pick.append((rec, pick))
    sizes = {
        "fixed": totals([(rec, "fixed") for rec, _ in per_q]),
        "supervised": totals([(rec, "supervised") for rec, _ in per_q]),
        "random_mean": sum(totals([(rec, s) for rec, _ in per_q])
                           for s in samples) / k,
        "oracle": totals(oracle_pick),
    }

    deltas = []
    for metric in METRICS:
        for a, b in (("oracle", "fixed"), ("oracle", "supervised"),
                     ("supervised", "fixed"), ("random_mean", "supervised")):
            diffs = [x - y for x, y in zip(arms[metric][a], arms[metric][b])]
            mean, lo, hi = _mean_ci(diffs)
            deltas.append({"metric": metric, "comparison": f"{a} - {b}",
                           "mean": round(mean, 4), "ci95_low": round(lo, 4),
                           "ci95_high": round(hi, 4), "n_questions": len(diffs)})

    curve = []
    for kk in CURVE_KS:
        if kk > k:
            continue
        row = {"k": kk}
        for metric in ("recall@5", "mrr@10"):
            best = [max(val(r, q, f"s{j}", metric) for j in range(kk))
                    for r, q in per_q]
            row[f"best_of_k_{metric}"] = round(sum(best) / len(best), 4)
        curve.append(row)

    return {"arms": arms, "sizes": sizes, "deltas": deltas, "curve": curve,
            "n_questions": len(per_q), "n_docs": len(records)}


def _verdict(summary: dict) -> tuple[str, str, dict]:
    ceiling = next(d for d in summary["deltas"]
                   if d["metric"] == "recall@5" and d["comparison"] == "oracle - fixed")
    drift = summary["sizes"]["oracle"] - summary["sizes"]["fixed"]
    mde, tol = C.STAGE10_MDE, C.STAGE10_SIZE_TOLERANCE
    m, lo, hi = ceiling["mean"], ceiling["ci95_low"], ceiling["ci95_high"]
    info = {"ceiling": m, "low": lo, "high": hi, "drift": drift}
    if hi < mde:
        return "CLOSED", (
            f"even the question-aware ceiling over fixed-size chunking is "
            f"{m:+.4f} R@5 (95% CI up to {hi:+.4f}), below the {mde} detection "
            f"floor: no in-window placement policy can produce a detectable "
            f"method effect at this size"), info
    if m < mde:
        return "INCONCLUSIVE", (
            f"the ceiling is {m:+.4f} R@5, below the {mde} floor, but its 95% CI "
            f"reaches {hi:+.4f}, so it cannot be ruled out"), info
    if abs(drift) > tol:
        return "SIZE-CONFOUNDED", (
            f"the ceiling is {m:+.4f} R@5, but the chunkings that reach it are "
            f"{drift:+.2f} sentences off fixed-size on average (tolerance {tol}), "
            f"so it cannot be attributed to placement"), info
    return "HEADROOM", (
        f"the ceiling is {m:+.4f} R@5 with size held within {tol} sentences: "
        f"placement headroom exists in principle, though this oracle sees the "
        f"question, so a real policy's share of it is unmeasured"), info


def _plot(summary: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summary["curve"]:
        return
    arms = summary["arms"]["recall@5"]
    fixed = sum(arms["fixed"]) / len(arms["fixed"])
    supervised = sum(arms["supervised"]) / len(arms["supervised"])
    ks = [row["k"] for row in summary["curve"]]
    best = [row["best_of_k_recall@5"] for row in summary["curve"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ks, best, marker="o", color="#2a7ab0",
            label="best of k uniform in-window chunkings")
    ax.axhline(fixed, color="black", linewidth=1.0, label=f"fixed 15/0 ({fixed:.3f})")
    ax.axhline(supervised, color="#7f7f7f", linewidth=1.0, linestyle=":",
               label=f"supervised Stage 2 ({supervised:.3f})")
    ax.axhline(fixed + C.STAGE10_MDE, color="#c0392b", linewidth=1.0,
               linestyle="--", label=f"fixed + detection floor ({C.STAGE10_MDE})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(x) for x in ks])
    ax.set_xlabel("k (candidate chunkings per question)")
    ax.set_ylabel("R@5 (question-aware best)")
    ax.set_title("Stage 10 oracle: ceiling for cut placement")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[oracle] wrote {path}")


def _write_summary(summary, verdict, why, info, k, path) -> None:
    arms, sizes = summary["arms"], summary["sizes"]
    lines = [
        "# Stage 10 oracle - ceiling for cut placement", "",
        f"Verdict: {verdict}. {why}.", "",
        f"- {summary['n_docs']} documents, {summary['n_questions']} questions, "
        f"K = {k} uniform in-window chunkings per document plus fixed and supervised",
        "- every other document frozen at fixed 15/0; top-10 over the whole corpus",
        f"- detection floor (pre-registered): {C.STAGE10_MDE} R@5; size tolerance "
        f"{C.STAGE10_SIZE_TOLERANCE} sentences",
        "- the oracle sees each question, so it is an optimistic ceiling", "",
        "| arm | R@1 | R@5 | MRR@10 | avg chunk size |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in ("fixed", "supervised", "random_mean", "oracle"):
        cells = [f"{sum(arms[m][arm]) / len(arms[m][arm]):.4f}"
                 for m in ("recall@1", "recall@5", "mrr@10")]
        lines.append(f"| {arm} | {' | '.join(cells)} | {sizes[arm]:.3f} |")
    lines += ["", "| metric | comparison | mean | 95% CI |", "| --- | --- | --- | --- |"]
    for d in summary["deltas"]:
        lines.append(f"| {d['metric']} | {d['comparison']} | {d['mean']:+.4f} | "
                     f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}] |")
    lines += ["", "| k | best-of-k R@5 | best-of-k MRR@10 |", "| --- | --- | --- |"]
    for row in summary["curve"]:
        lines.append(f"| {row['k']} | {row['best_of_k_recall@5']:.4f} | "
                     f"{row['best_of_k_mrr@10']:.4f} |")
    lines += ["", f"Oracle size drift vs fixed: {info['drift']:+.3f} sentences.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[oracle] wrote {path}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 10 oracle: the ceiling on what in-window cut "
                    "placement can buy at the matched size.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny plumbing run on the cached NQ bench; outputs are "
                         "prefixed so they cannot be mistaken for results")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="uniform in-window chunkings per document")
    ap.add_argument("--n-docs", type=int, default=DEFAULT_N_DOCS,
                    help="documents evaluated (a seeded random subset)")
    ap.add_argument("--seed", type=int, default=C.STAGE10_SEED)
    ap.add_argument("--fresh", action="store_true",
                    help="discard the checkpoint and start over")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()

    k, n_docs = int(args.k), int(args.n_docs)
    if args.smoke:
        k, n_docs = min(k, 4), min(n_docs, 6)

    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()

    import numpy as np

    from rag_chunk import rl_chunking as rl
    from rag_chunk import sweep
    from rag_chunk.large_eval import append_checkpoint, load_checkpoint

    if args.smoke:
        docs, questions = _smoke_corpus()
    else:
        from rag_chunk import rerank_finetune as rf

        docs, questions, _, _ = rf.prepare_train_and_dev()
        limit = int(C.STAGE10_N_TRAIN_DOCS)
        if len(docs) > limit:
            keep = {d.get("title", d.get("id")) for d in docs[:limit]}
            docs = [d for d in docs if d.get("title", d.get("id")) in keep]
            questions = [q for q in questions if q.get("doc_title") in keep]

    target, overlap = int(C.STAGE10_TARGET_SIZE), int(C.STAGE10_OVERLAP)
    min_size, max_size = sweep._semantic_window(target)

    env = rl.RecallEnv(docs, questions, reward_depth=DEPTH)
    keys = list(env.trainable_keys)
    order = np.random.default_rng(args.seed).permutation(len(keys))
    eval_keys = [keys[i] for i in order[:n_docs]]
    print(f"[oracle] evaluating {len(eval_keys)} of {len(keys)} answerable documents, "
          f"K = {k}, window [{min_size}, {max_size}], overlap {overlap}", flush=True)

    def out(name: str) -> pathlib.Path:
        return C.RESULTS_LATEST_DIR / ((SMOKE_PREFIX + name) if args.smoke else name)

    checkpoint = out(CHECKPOINT_FILE)
    meta = {"stage": "stage10-oracle", "k": k, "eval_docs": len(eval_keys),
            "seed": int(args.seed), "target": target, "overlap": overlap,
            "depth": DEPTH, "retrieval_model": C.RETRIEVAL_EMBED_MODEL,
            "corpus_docs": len(env.sentences), "corpus_questions": len(env.questions)}
    if args.fresh and checkpoint.exists():
        checkpoint.unlink()
        print(f"[oracle] --fresh: discarded {checkpoint.name}")
    done = load_checkpoint(checkpoint, meta)
    if not done and not checkpoint.exists():
        append_checkpoint(checkpoint, {"meta": meta})
    records = list(done)
    done_keys = {r["doc"] for r in done}
    if done:
        print(f"[oracle] resuming: {len(done)}/{len(eval_keys)} documents already done")

    policy = rl.build_policy(warm_start=True)
    policy.eval()

    todo = sum(1 for key in eval_keys if key not in done_keys)
    started = time.perf_counter()
    newly = 0
    checked = 0
    for position, key in enumerate(eval_keys):
        if key in done_keys:
            continue
        rng = np.random.default_rng([int(args.seed), position])   # resume-stable
        record, texts = _score_document(
            key, env, policy, k, rng, target=target, min_size=min_size,
            max_size=max_size, overlap=overlap)

        if args.smoke and checked < 3:
            # The merged ranking must agree with the RL environment's reward,
            # which rebuilds the full candidate pool instead of merging.
            for name, cand_texts in texts.items():
                want = env.reward({key: cand_texts}).get(key, 0.0)
                ranks = record["candidates"][name]["ranks"]
                got = sum(1.0 / r if r > 0 else 0.0 for r in ranks) / len(ranks)
                if abs(want - got) > 1e-6:
                    raise SystemExit(
                        f"[oracle] self-check FAILED on {key!r}/{name}: merged "
                        f"MRR {got:.6f} vs environment {want:.6f}")
            checked += 1
            print(f"[oracle] self-check OK on document {checked} "
                  f"({len(texts)} candidates agree with the RL reward)", flush=True)

        append_checkpoint(checkpoint, {"rows": [record]})
        records.append(record)
        newly += 1
        if newly % 10 == 0 or newly == todo:
            elapsed = (time.perf_counter() - started) / 60
            eta = elapsed / newly * (todo - newly)
            print(f"[oracle] {len(records)}/{len(eval_keys)} documents  "
                  f"{elapsed:.1f} min elapsed  ETA {eta:.1f} min", flush=True)

    summary = _summarise(records, k)
    verdict, why, info = _verdict(summary)

    arm_rows = []
    for arm in ("fixed", "supervised", "random_mean", "oracle"):
        row = {"arm": arm}
        for metric in METRICS:
            values = summary["arms"][metric][arm]
            row[metric] = round(sum(values) / len(values), 4)
        row["avg_chunk_size"] = round(summary["sizes"][arm], 3)
        row["n_questions"] = summary["n_questions"]
        arm_rows.append(row)
    _write_rows(out(ARMS_CSV), arm_rows)
    _write_rows(out(DELTAS_CSV), summary["deltas"])
    _write_rows(out(CURVE_CSV), summary["curve"])
    _plot(summary, out(CURVE_PNG))
    _write_summary(summary, verdict, why, info, k, out(SUMMARY_MD))

    print(f"\n[oracle] VERDICT: {verdict} - {why}.")
    if args.smoke:
        print("[oracle] SMOKE MODE: these are not results.")


if __name__ == "__main__":
    main()
