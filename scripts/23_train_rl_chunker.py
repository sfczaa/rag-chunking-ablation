"""Stage 10 (step 1) - train the boundary policy against retrieval recall.

The supervised boundary models (Stages 1-2) optimise a proxy: Wikipedia
section pseudo-labels under a weighted BCE loss. Every stage since has measured
them with doc-constrained Recall@k and found them tied with fixed-size chunking
at a matched size. This script keeps the Stage 2 architecture and weights and
changes only the objective - REINFORCE on the retrieval metric itself, with a
self-critical (greedy-decode) baseline. See ``rag_chunk/rl_chunking.py`` for the
formulation and ``docs/stage10_rl_chunking.md`` for the pre-registered criteria.

Data. Training and dev come from the NQ train split via the Stage 8 cache
(``rerank_finetune.prepare_train_and_dev``), so they are disjoint from every
evaluation bench in the project. Run ``scripts/16_build_rerank_train_data.py``
first if that cache does not exist yet.

Gate. After training, the dev delta against the supervised transformer at the
same target size decides whether the final Stage 6 bench run is worth the GPU
hour. The gate only decides cost: dev is ~400 questions, so its
2 SE is about 0.04 and a smaller dev delta does not show anything either way. The
claim threshold is the pre-registered MDE on the 1032-question bench, applied in
``scripts/24_eval_rl_chunker.py``.

Crash budget: the best weights are rewritten at every dev evaluation, so a lost
runtime costs at most ``--eval-every`` steps of training.

Usage:
    python scripts/23_train_rl_chunker.py --smoke      # plumbing check
    python scripts/23_train_rl_chunker.py              # the real run
    python scripts/23_train_rl_chunker.py --steps 200 --eval-every 25
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

SMOKE_PREFIX = "smoke_"


def _out_path(name: str, smoke: bool) -> pathlib.Path:
    return C.RESULTS_LATEST_DIR / ((SMOKE_PREFIX + name) if smoke else name)


def _weights_path(smoke: bool) -> pathlib.Path:
    if smoke:
        return C.MODELS_DIR / (SMOKE_PREFIX + C.MODEL_FILENAMES["rl"])
    return C.model_path("rl")


def _state_path(smoke: bool) -> pathlib.Path:
    name = "rl_boundary_train_state.pt"
    return C.MODELS_DIR / ((SMOKE_PREFIX + name) if smoke else name)


def _fingerprint(*parts) -> list:
    """Run identity. A resume must use the same hyper-parameters."""
    return [str(p) for p in parts]


def _plain(obj):
    """Numpy scalars as Python numbers, so the state loads with weights_only=True."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_plain(v) for v in obj)
    return obj.item() if type(obj).__module__ == "numpy" else obj


def _load_state(path: pathlib.Path, fingerprint: list) -> dict | None:
    """The training state from an interrupted run, or ``None`` to start over.

    A mismatched fingerprint refuses instead of mixing two runs; this is the
    same rule the Stage 6 checkpoint uses.
    """
    import pickle

    import torch

    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, RuntimeError, pickle.UnpicklingError) as exc:
        print(f"[stage10] ignoring unreadable {path.name}: {exc}")
        return None
    if state.get("fingerprint") != fingerprint:
        raise SystemExit(
            f"[stage10] {path.name} belongs to a different run.\n"
            f"[stage10]   checkpoint: {state.get('fingerprint')}\n"
            f"[stage10]   this run  : {fingerprint}\n"
            "[stage10] Re-run with --fresh to discard it and start over.")
    return state


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
        for row in rows:
            writer.writerow(row)


def _fixed_baseline(docs, questions, size, overlap) -> dict:
    """Matched fixed-size row, scored by the same path as every archived row."""
    from rag_chunk import metrics, retrieval

    index = retrieval.build_index_for_config(
        "fixed", docs, fixed_size=size, fixed_overlap=overlap)
    rec = metrics.recall_at_k(index, questions, C.RECALL_KS)
    row = {"method": "fixed", "fixed_size": size, "fixed_overlap": overlap,
           "avg_chunk_size": index.avg_chunk_size(),
           "n_chunks": len(index.chunk_texts),
           "n_docs": len(docs), "n_questions": len(questions)}
    for k in C.RECALL_KS:
        row[f"recall@{k}"] = rec["doc_constrained"][k]
    return row


def _self_check(trials: int = 200, draws: int = 4000) -> None:
    """The two properties the policy gradient depends on.

    Both failed without an error during development, so they are checked
    here: a greedy decode that drifts from the archived chunker makes
    the self-critical baseline incomparable, and a sampler that does not match
    the distribution the loss differentiates produces a gradient for a policy
    that was never rolled out.
    """
    import numpy as np

    from rag_chunk import chunking
    from rag_chunk.rl_chunking import rollout

    rng = np.random.default_rng(0)
    for _ in range(trials):
        n = int(rng.integers(1, 60))
        sents = [f"s{i}" for i in range(n)]
        logits = rng.normal(scale=3.0, size=max(0, n - 1))
        target = int(rng.integers(6, 16))
        mn, mx = max(2, target - 4), target + 4
        overlap = int(rng.integers(0, 2))
        ref = chunking.semantic_target_chunks(
            sents, 1.0 / (1.0 + np.exp(-logits)), target, mn, mx, overlap)
        spans, _ = rollout(logits, n, target_size=target, min_size=mn,
                           max_size=mx, overlap=overlap, greedy=True)
        if [sents[s:e] for s, e in spans] != ref:
            raise SystemExit("[stage10] self-check FAILED: greedy decode no "
                             "longer reproduces chunking.semantic_target_chunks")

    n, target, overlap = 60, 15, 0
    mn, mx = max(2, target - 4), target + 4
    logits = rng.normal(scale=2.0, size=n - 1)
    srng = np.random.default_rng(1)
    counts: dict[int, int] = {}
    window = None
    for _ in range(draws):
        _, choices = rollout(logits, n, target_size=target, min_size=mn,
                             max_size=mx, overlap=overlap, greedy=False,
                             rng=srng, temperature=1.0)
        lo, hi, chosen = choices[0]
        window = (lo, hi)
        counts[chosen] = counts.get(chosen, 0) + 1
    lo, hi = window
    scores = logits[lo:hi + 1]
    want = np.exp(scores - scores.max())
    want /= want.sum()
    got = np.array([counts.get(lo + j, 0) / draws for j in range(hi - lo + 1)])
    worst = float(np.max(np.abs(got - want)))
    if worst > 0.03:
        raise SystemExit(f"[stage10] self-check FAILED: the sampler and the "
                         f"policy-gradient distribution disagree by {worst:.3f}")
    print(f"[stage10] self-check OK (greedy decode exact; sampler within "
          f"{worst:.4f} of the policy)")


def _smoke_corpus(n_docs: int = 40):
    """A small slice of the cached NQ bench, for plumbing checks only.

    This trains on documents drawn from the validation bench, so its numbers
    are not results. It exists so the loop, the reward and the
    evaluation path can be exercised without a GPU session, which is also why
    the slice is small: the run has to finish on a CPU in a few minutes.
    Outputs and weights are written under a ``smoke_`` prefix and cannot be
    archived.
    """
    from rag_chunk import nq_data

    docs = nq_data.load_docs()[:n_docs]
    questions = nq_data.load_questions()
    if not docs:
        raise SystemExit("[stage10] no cached NQ bench; run scripts/1_prepare_data.py")
    keys = [d.get("title", d.get("id")) for d in docs]
    cut = max(2, len(docs) // 2)
    train_keys, dev_keys = set(keys[:cut]), set(keys[cut:])

    def split(selected):
        sub_docs = [d for d in docs if d.get("title", d.get("id")) in selected]
        sub_q = [q for q in questions if q.get("doc_title") in selected]
        return sub_docs, sub_q

    train_docs, train_q = split(train_keys)
    dev_docs, dev_q = split(dev_keys)
    print("[stage10] SMOKE MODE: training on a slice of the validation bench. "
          "Plumbing check only; these numbers are not results.")
    return train_docs, train_q, dev_docs, dev_q


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 10 step 1: REINFORCE the boundary policy on "
                    "doc-constrained retrieval recall.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny plumbing run on the cached NQ bench; outputs and "
                         "weights are prefixed so they cannot be archived")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--docs-per-step", type=int, default=None)
    ap.add_argument("--n-train-docs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--target-size", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--entropy-coef", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="discard the training checkpoint and start over")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="must match the eval pipeline")
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else C.STAGE10_STEPS
    docs_per_step = (args.docs_per_step if args.docs_per_step is not None
                     else C.STAGE10_DOCS_PER_STEP)
    n_train_docs = (args.n_train_docs if args.n_train_docs is not None
                    else C.STAGE10_N_TRAIN_DOCS)
    lr = args.lr if args.lr is not None else C.STAGE10_LR
    target_size = (args.target_size if args.target_size is not None
                   else C.STAGE10_TARGET_SIZE)
    overlap = args.overlap if args.overlap is not None else C.STAGE10_OVERLAP
    temperature = (args.temperature if args.temperature is not None
                   else C.STAGE10_TEMPERATURE)
    entropy_coef = (args.entropy_coef if args.entropy_coef is not None
                    else C.STAGE10_ENTROPY_COEF)
    eval_every = (args.eval_every if args.eval_every is not None
                  else C.STAGE10_EVAL_EVERY)
    seed = args.seed if args.seed is not None else C.STAGE10_SEED

    if args.smoke:
        steps = min(steps, 12)
        docs_per_step = min(docs_per_step, 8)
        eval_every = min(eval_every, 6)

    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()

    import torch

    from rag_chunk import rl_chunking as rl

    if args.smoke:
        _self_check()
        train_docs, train_q, dev_docs, dev_q = _smoke_corpus()
    else:
        from rag_chunk import rerank_finetune as rf

        train_docs, train_q, dev_docs, dev_q = rf.prepare_train_and_dev()
        if n_train_docs and len(train_docs) > n_train_docs:
            keep = {d.get("title", d.get("id")) for d in train_docs[:n_train_docs]}
            train_docs = [d for d in train_docs
                          if d.get("title", d.get("id")) in keep]
            train_q = [q for q in train_q if q.get("doc_title") in keep]

    print(f"[stage10] train {len(train_docs)} docs / {len(train_q)} questions; "
          f"dev {len(dev_docs)} docs / {len(dev_q)} questions")
    print(f"[stage10] target size {target_size}, overlap {overlap}, "
          f"{steps} steps x {docs_per_step} docs, lr {lr}")

    env = rl.RecallEnv(train_docs, train_q)
    print("[stage10] caching train sentence embeddings")
    train_emb = rl.cache_sentence_embeddings(env.sentences)
    dev_sentences = {d.get("title", d.get("id")): d["sentences"] for d in dev_docs}
    print("[stage10] caching dev sentence embeddings")
    dev_emb = rl.cache_sentence_embeddings(dev_sentences)

    policy = rl.build_policy()

    log_path = _out_path(C.STAGE10_TRAIN_LOG_CSV, args.smoke)
    dev_path = _out_path(C.STAGE10_DEV_CSV, args.smoke)
    weights_path = _weights_path(args.smoke)
    state_path = _state_path(args.smoke)
    # `steps` is not part of the identity: a 100-step run that stopped is a
    # valid starting point for a 400-step one, so a run can be extended later
    # on a free GPU quota. Every other setting changes what the earlier steps
    # meant, so it is part of the identity.
    fingerprint = _fingerprint(docs_per_step, n_train_docs, lr, target_size,
                               overlap, temperature, entropy_coef, seed,
                               len(train_docs), len(dev_docs))

    if args.fresh and state_path.exists():
        state_path.unlink()
        print(f"[stage10] --fresh: discarded {state_path.name}")
    state = _load_state(state_path, fingerprint)

    if state is not None:
        start_step = int(state["step"])
        dev_rows = list(state["dev_rows"])
        log_rows = list(state["log_rows"])
        best = dict(state["best"])
        baseline_r5 = float(state["baseline_r5"])
        fixed_r5 = float(state["fixed_r5"])
        policy.load_state_dict(state["policy"])
        resume_state = {"optimiser": state["optimiser"], "rng": state["rng"],
                        "torch_rng": state.get("torch_rng"),
                        "cuda_rng": state.get("cuda_rng")}
        print(f"[stage10] resuming at step {start_step}/{steps} "
              f"(best dev R@5 {best['recall']:.4f} at step {best['step']})")
    else:
        start_step = 0
        resume_state = None
        print("[stage10] dev baselines at the matched size")
        dev_rows = [_fixed_baseline(dev_docs, dev_q, target_size, overlap)]
        supervised = rl.build_policy(warm_start=True)
        base_row = rl.evaluate_policy(supervised, dev_docs, dev_q, dev_emb,
                                      target_size=target_size, overlap=overlap,
                                      method_label="transformer")
        base_row["step"] = None
        dev_rows.append(base_row)
        del supervised
        baseline_r5 = base_row[f"recall@{max(C.RECALL_KS)}"]
        fixed_r5 = dev_rows[0][f"recall@{max(C.RECALL_KS)}"]
        print(f"[stage10] dev R@5 - supervised {baseline_r5:.4f}, "
              f"fixed {fixed_r5:.4f}")
        best = {"recall": -1.0, "step": None}
        log_rows = []

    def eval_fn(step: int) -> dict:
        row = rl.evaluate_policy(policy, dev_docs, dev_q, dev_emb,
                                 target_size=target_size, overlap=overlap)
        row["step"] = step
        dev_rows.append(row)
        _write_rows(dev_path, dev_rows)
        recall = row[f"recall@{max(C.RECALL_KS)}"]
        if recall > best["recall"]:
            best.update(recall=recall, step=step)
            torch.save(policy.state_dict(), weights_path)
            print(f"[stage10] step {step}: dev R@5 {recall:.4f} (best, saved)",
                  flush=True)
        else:
            print(f"[stage10] step {step}: dev R@5 {recall:.4f}", flush=True)
        return {f"dev_recall@{k}": row[f"recall@{k}"] for k in C.RECALL_KS} | {
            "dev_avg_chunk_size": row["avg_chunk_size"]}

    def log_fn(row: dict) -> None:
        log_rows.append(row)
        if row["step"] % 10 == 0 or row["step"] == 1:
            print(f"[stage10] step {row['step']}/{steps} "
                  f"reward(greedy) {row['reward_greedy']:.4f} "
                  f"adv {row['advantage']:+.4f} "
                  f"entropy {row['entropy']:.3f}", flush=True)
            _write_rows(log_path, log_rows)

    def checkpoint_fn(step, optimiser, rng) -> None:
        """Everything needed to restart this run exactly where it stopped."""
        tmp = state_path.with_suffix(".tmp")
        torch.save({"fingerprint": fingerprint, "step": int(step),
                    "policy": policy.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                    "best": _plain(best), "log_rows": _plain(log_rows),
                    "dev_rows": _plain(dev_rows),
                    "baseline_r5": float(baseline_r5),
                    "fixed_r5": float(fixed_r5)}, tmp)
        tmp.replace(state_path)          # atomic: a killed save cannot truncate

    started = time.perf_counter()
    rl.train(policy, env, train_emb, steps=steps, docs_per_step=docs_per_step,
             lr=lr, target_size=target_size, overlap=overlap,
             temperature=temperature, entropy_coef=entropy_coef,
             grad_clip=C.STAGE10_GRAD_CLIP, seed=seed, eval_every=eval_every,
             eval_fn=eval_fn, log_fn=log_fn, start_step=start_step,
             resume_state=resume_state, checkpoint_fn=checkpoint_fn)
    eval_fn(steps)
    _write_rows(log_path, log_rows)
    minutes = (time.perf_counter() - started) / 60

    delta = best["recall"] - baseline_r5
    verdict = "GO" if delta >= C.STAGE10_GO_THRESHOLD else "NO-GO"

    # A NO-GO only means something if the policy actually moved. The warm start
    # is close to uniform inside the decode window (the Stage 2 model barely
    # separates adjacent boundary positions), so "no change" is the failure mode
    # to watch for: it means the optimiser did not work, and says nothing about
    # the objective.
    import math

    uniform = math.log(C.STAGE10_TARGET_SIZE + 4 - max(2, C.STAGE10_TARGET_SIZE - 4) + 1)
    # Averaged over a window: step-to-step entropy swings by ~0.2 nats purely
    # from which documents the batch drew, so first-vs-last would report motion
    # that is not there (or miss motion that is).
    window = max(1, min(10, len(log_rows) // 2)) if log_rows else 1
    values = [r["entropy"] for r in log_rows] or [float("nan")]
    first_entropy = sum(values[:window]) / window
    last_entropy = sum(values[-window:]) / window
    moved = abs(last_entropy - first_entropy)
    mean_advantage = (sum(abs(r["advantage"]) for r in log_rows) / len(log_rows)
                      if log_rows else 0.0)

    print(f"\n[stage10] done in {minutes:.1f} min")
    print(f"[stage10] policy entropy {first_entropy:.4f} -> {last_entropy:.4f} "
          f"nats/decision (uniform over the window = {uniform:.4f}); "
          f"mean |advantage| {mean_advantage:.4f}")
    if moved < 0.05:
        print("[stage10] WARN: the policy barely moved. Treat this run as a "
              "FAILED OPTIMISATION. A NULL verdict only "
              "counts when the policy changed. Raise --lr or "
              "--steps and rerun before writing anything up.")
    print(f"[stage10] best dev R@5 {best['recall']:.4f} at step {best['step']} "
          f"(weights: {weights_path})")
    if state_path.exists():
        print(f"[stage10] training state kept at {state_path.name}: rerun with "
              f"a larger --steps to extend this run, or --fresh to start over.")
    print(f"[stage10] dev delta vs supervised transformer: {delta:+.4f} "
          f"(gate {C.STAGE10_GO_THRESHOLD:+.3f}) -> {verdict}")
    print("[stage10] the dev gate only decides cost: at "
          f"{len(dev_q)} questions its 2 SE is about 0.04.")
    if args.smoke:
        print("[stage10] SMOKE MODE: these are not results and the "
              "smoke_ outputs must not be archived.")
    elif verdict == "GO":
        print("[stage10] next: python scripts/24_eval_rl_chunker.py")
    else:
        print("[stage10] NO-GO: record it in "
              "docs/stage10_rl_chunking.md rather than re-rolling the gate.")


if __name__ == "__main__":
    main()
