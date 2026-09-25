"""Stage 10 - optimise the boundary model against retrieval recall directly.

Every learned chunker in this project (Stage 1 BiLSTM, Stage 2 Transformer) was
trained on Wikipedia section pseudo-labels with a weighted BCE loss and then
scored with doc-constrained Recall@k. Stages 1-7 established that those two
objectives are not the same thing: at a matched chunk size the learned cutters
tie with fixed-size chunking, and the size+method regression puts the method
effect ~18x below the size effect.

This module removes the proxy. The boundary network keeps its architecture and
is warm-started from the Stage 2 weights; only the objective changes, from
"predict the section break" to "place the cut that makes the answer retrievable".
A null result then says something about the objective and not the architecture.

Formulation
-----------
The environment is the sweep's own target-size decode
(``chunking.semantic_target_chunks``): walking a document, every chunk must end
somewhere inside a ``[min_size, max_size]`` window, and the model picks where.
So one decision is a categorical choice over ~9 candidate boundaries instead of
``n-1`` independent binary decisions. This has two consequences:

* the credit assignment is over a handful of actions per chunk, which keeps the
  REINFORCE variance manageable at this corpus size, and
* average chunk size is held in the baseline's band by the window. A recall
  reward with a free action space would learn to grow chunks and reproduce
  the Stage 1 size effect, which is already known.

Reward: for each question, ``1 / rank`` of the first retrieved chunk that both
contains the answer string and comes from the gold document, within the top
``reward_depth`` of the whole corpus (0 if absent). That is a dense stand-in for
doc-constrained Recall@k, which is what the stage is finally judged on.

Baseline: self-critical. Each step scores the sampled chunking against the
greedy chunking of the same documents, so the advantage is
``reward(sample) - reward(greedy)`` with no learned critic.

Cost model: only the sampled documents are re-chunked and re-encoded each step;
the rest of the corpus stays at a frozen fixed-size chunking and its BGE vectors
are encoded once. That keeps a step at ~1-2 s instead of the ~3 min a full
corpus re-encode would cost. It is an approximation used only during training: the
distractor pool is chunked by the reference config rather than by the current
policy. The Stage 10 evaluation re-chunks every document with the policy and
carries no such shortcut.
"""

from __future__ import annotations

import collections

import numpy as np

import config as C


# --------------------------------------------------------------------------- #
# Decoding: the sweep's target-size walk, re-expressed as a sequence of choices
# --------------------------------------------------------------------------- #
def rollout(
    logits,
    n_sentences: int,
    *,
    target_size: int,
    min_size: int,
    max_size: int,
    overlap: int,
    greedy: bool = True,
    rng=None,
    temperature: float = 1.0,
):
    """Chunk one document, returning ``(spans, choices)``.

    ``spans`` is a list of ``(start, end)`` sentence slices; ``choices`` is a
    list of ``(lo, hi, chosen)`` boundary-window decisions, which is what the
    policy gradient needs. ``logits[i]`` is the model's raw score for cutting
    after sentence ``i`` (length ``n_sentences - 1``).

    Sampling draws from ``softmax(logits[window] / temperature)``, which is the
    same distribution :func:`choice_logprob` differentiates; if the two
    differ, the gradient is for a policy that was never sampled.
    Greedy decode ranks by ``sigmoid(logit)`` instead, purely so it reproduces
    :func:`chunking.semantic_target_chunks` bit for bit, including its tie-break
    toward ``target_size``. Sigmoid is monotone, so the two orderings agree; the
    greedy rollout is the deterministic chunker every earlier stage evaluated,
    which is what makes the self-critical baseline and the final comparison
    commensurable.
    """
    n = int(n_sentences)
    if n == 0:
        return [], []
    if n == 1:
        return [(0, 1)], []
    logits = np.asarray(logits, dtype="float64")
    probs = 1.0 / (1.0 + np.exp(-logits))

    min_size = max(1, int(min_size))
    max_size = max(min_size, int(max_size))
    target_size = min(max(int(target_size), min_size), max_size)
    overlap = max(0, min(int(overlap), min_size - 1))

    spans: list[tuple[int, int]] = []
    choices: list[tuple[int, int, int]] = []
    start = 0
    while start < n:
        if n - start <= max_size:
            spans.append((start, n))            # tail fits; stop cutting
            break
        lo = start + min_size - 1               # cut index giving length min_size
        hi = min(start + max_size - 1, n - 2)   # ... up to max_size / last boundary
        if greedy:
            best_i, best_key = lo, None
            for i in range(lo, hi + 1):
                key = (float(probs[i]), -abs((i - start + 1) - target_size))
                if best_key is None or key > best_key:
                    best_key, best_i = key, i
            chosen = best_i
        else:
            scores = logits[lo:hi + 1] / max(temperature, 1e-6)
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            chosen = lo + int(rng.choice(len(weights), p=weights))
        choices.append((lo, hi, chosen))
        end = chosen + 1
        spans.append((start, end))
        start = end - overlap
    return spans, choices


def spans_to_texts(sentences: list[str], spans) -> list[str]:
    from rag_chunk import chunking

    return [chunking.chunk_text(list(sentences[s:e])) for s, e in spans]


def choice_logprob(logits, choices, temperature: float = 1.0):
    """``(sum log pi(a|s), mean entropy)`` for one document's decisions.

    ``logits`` is the live ``(n-1,)`` tensor from the policy, so the returned
    log-probability carries gradients; ``choices`` comes from :func:`rollout`,
    which sampled from a detached copy of the same distribution.

    The log-probability is a sum: every cut in a document contributes to the
    one reward that document earns, so the joint action is what REINFORCE must
    credit. The entropy is a mean, so that its coefficient means nats per
    decision and does not scale with document length.
    """
    import torch.nn.functional as F

    zero = logits.sum() * 0.0
    if not choices:
        return zero, zero
    logp_total = None
    ent_total = None
    counted = 0
    for lo, hi, chosen in choices:
        window = logits[lo:hi + 1]
        if window.numel() <= 1:
            continue
        logp = F.log_softmax(window / max(temperature, 1e-6), dim=0)
        step_logp = logp[chosen - lo]
        step_ent = -(logp.exp() * logp).sum()
        logp_total = step_logp if logp_total is None else logp_total + step_logp
        ent_total = step_ent if ent_total is None else ent_total + step_ent
        counted += 1
    if logp_total is None:
        return zero, zero
    return logp_total, ent_total / counted


# --------------------------------------------------------------------------- #
# Environment: retrieval reward over a corpus whose distractors stay frozen
# --------------------------------------------------------------------------- #
class RecallEnv:
    """Doc-constrained retrieval reward for re-chunked documents.

    The corpus is encoded once under a reference fixed-size chunking. Scoring a
    set of re-chunked documents swaps out just those documents' rows, so a step
    costs one small encode instead of a full corpus pass.
    """

    def __init__(self, docs: list[dict], questions: list[dict], *,
                 ref_size: int | None = None, ref_overlap: int | None = None,
                 reward_depth: int | None = None) -> None:
        from rag_chunk import chunking, embedding, metrics

        ref_size = int(C.STAGE10_REF_SIZE if ref_size is None else ref_size)
        ref_overlap = int(C.STAGE10_REF_OVERLAP if ref_overlap is None
                          else ref_overlap)
        self.reward_depth = int(C.STAGE10_REWARD_DEPTH if reward_depth is None
                                else reward_depth)

        self.sentences: dict[str, list[str]] = {}
        for d in docs:
            key = d.get("title", d.get("id"))
            self.sentences[key] = d["sentences"]

        texts: list[str] = []
        doc_ids: list[str] = []
        self.base_rows: dict[str, tuple[int, int]] = {}
        for key, sents in self.sentences.items():
            first = len(texts)
            for ch in chunking.fixed_chunks(sents, ref_size, ref_overlap):
                texts.append(chunking.chunk_text(ch))
                doc_ids.append(key)
            self.base_rows[key] = (first, len(texts))
        self.base_texts = texts
        self.base_norm = [metrics.normalize_text(t) for t in texts]
        self.base_doc_ids = doc_ids
        print(f"[stage10] encoding the frozen corpus: {len(texts)} chunks "
              f"(fixed {ref_size}/{ref_overlap})", flush=True)
        self.base_vecs = embedding.encode_retrieval_passages(texts)

        by_doc = collections.defaultdict(list)
        flat: list[dict] = []
        for q in questions:
            key = q.get("doc_title")
            if key in self.sentences:
                by_doc[key].append(len(flat))
                flat.append(q)
        self.questions = flat
        self.answer_norm = [metrics.normalize_text(q["answer"]) for q in flat]
        self.q_rows_by_doc = {k: v for k, v in by_doc.items() if v}
        self.trainable_keys = sorted(self.q_rows_by_doc)
        print(f"[stage10] corpus: {len(self.sentences)} docs, {len(flat)} "
              f"questions, {len(self.trainable_keys)} answerable docs", flush=True)
        self.query_vecs = embedding.encode_retrieval_queries(
            [q["question"] for q in flat])

    def reward(self, updates: dict[str, list[str]]) -> dict[str, float]:
        """Mean reward per document for a batch of re-chunked documents.

        ``updates`` maps a document key to its new chunk texts. Only questions
        whose gold document is in ``updates`` are scored; every other document
        contributes its frozen chunks as distractors.
        """
        from rag_chunk import embedding, metrics

        keys = list(updates)
        if not keys:
            return {}
        new_texts: list[str] = []
        new_doc_ids: list[str] = []
        for key in keys:
            new_texts.extend(updates[key])
            new_doc_ids.extend([key] * len(updates[key]))
        new_vecs = embedding.encode_retrieval_passages(new_texts)
        new_norm = [metrics.normalize_text(t) for t in new_texts]

        keep = np.ones(len(self.base_texts), dtype=bool)
        for key in keys:
            first, last = self.base_rows[key]
            keep[first:last] = False
        keep_idx = np.flatnonzero(keep)
        cand_vecs = np.vstack([self.base_vecs[keep_idx], new_vecs])
        n_keep = len(keep_idx)

        rows = [r for key in keys for r in self.q_rows_by_doc.get(key, ())]
        sims = self.query_vecs[rows] @ cand_vecs.T
        depth = min(self.reward_depth, cand_vecs.shape[0])
        part = np.argpartition(-sims, depth - 1, axis=1)[:, :depth]
        ordered = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1),
            axis=1)

        per_doc = collections.defaultdict(list)
        for local, row in enumerate(rows):
            gold = self.questions[row].get("doc_title")
            answer = self.answer_norm[row]
            value = 0.0
            for rank, j in enumerate(ordered[local], start=1):
                if j < n_keep:
                    src = keep_idx[j]
                    text, doc = self.base_norm[src], self.base_doc_ids[src]
                else:
                    text, doc = new_norm[j - n_keep], new_doc_ids[j - n_keep]
                if doc == gold and answer in text:
                    value = 1.0 / rank
                    break
            per_doc[gold].append(value)
        return {k: float(np.mean(v)) for k, v in per_doc.items()}


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def build_policy(device: str | None = None, warm_start: bool = True):
    """The Stage 2 architecture, warm-started from the Stage 2 weights.

    The warm start keeps the comparison controlled: the RL run and the supervised
    baseline are the same network from the same initialisation, so any gap is
    attributable to the objective.
    """
    import torch

    from models import TransformerBoundary

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = TransformerBoundary(
        C.EMBED_DIM, C.TRANSFORMER_LAYERS, C.TRANSFORMER_HEADS,
        C.TRANSFORMER_FF_DIM, C.TRANSFORMER_DROPOUT).to(device)
    if warm_start:
        path = C.model_path("transformer")
        if not path.exists():
            raise SystemExit(
                f"[stage10] no Stage 2 weights at {path}. Train the supervised "
                f"transformer first (Phase 7) - Stage 10 fine-tunes it.")
        policy.load_state_dict(torch.load(path, map_location=device,
                                          weights_only=True))
        print(f"[stage10] warm start from {path}")
    return policy


def load_policy(device: str | None = None):
    """Rebuild the policy and load the trained Stage 10 weights."""
    import torch

    policy = build_policy(device=device, warm_start=False)
    device = next(policy.parameters()).device
    policy.load_state_dict(torch.load(C.model_path("rl"), map_location=device,
                                      weights_only=True))
    policy.eval()
    return policy


def cache_sentence_embeddings(sentences_by_key: dict) -> dict:
    """Frozen boundary-model sentence vectors, one encode pass per document."""
    from rag_chunk import embedding

    out = {}
    total = len(sentences_by_key)
    for i, (key, sents) in enumerate(sentences_by_key.items(), 1):
        out[key] = embedding.encode(sents, normalize=False).astype("float16")
        if i % 200 == 0 or i == total:
            print(f"[stage10] sentence embeddings: {i}/{total}", flush=True)
    return out


def boundary_probs_by_id(policy, sentences_by_key: dict, sent_emb: dict) -> dict:
    """``{doc key: per-boundary probabilities}`` for the evaluation path."""
    import torch

    device = next(policy.parameters()).device
    policy.eval()
    out = {}
    with torch.no_grad():
        for key, sents in sentences_by_key.items():
            if len(sents) <= 1:
                out[key] = np.zeros(max(0, len(sents) - 1), dtype="float32")
                continue
            emb = torch.as_tensor(np.asarray(sent_emb[key], dtype="float32"),
                                  device=device)
            out[key] = policy.predict_proba(emb).cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(policy, env: RecallEnv, sent_emb: dict, *, steps: int,
          docs_per_step: int, lr: float, target_size: int, overlap: int,
          temperature: float, entropy_coef: float, grad_clip: float,
          seed: int, eval_every: int = 0, eval_fn=None, log_fn=None,
          start_step: int = 0, resume_state: dict | None = None,
          checkpoint_fn=None):
    """REINFORCE with a self-critical baseline. Returns the training log rows.

    ``eval_fn(step) -> dict`` runs before the first step and then every
    ``eval_every`` steps, so the caller can score a held-out bench with the real
    evaluation path; whatever it returns is merged into that step's log row.

    Resuming: a Colab runtime dies more often than this run takes, so
    ``checkpoint_fn(step, optimiser, rng)`` is called right after every dev
    evaluation and the caller persists whatever it needs. ``start_step`` and
    ``resume_state`` (``optimiser`` / ``rng`` entries) put the loop back exactly
    where it stopped - restoring the optimiser moments matters here because
    AdamW's second moment is what keeps the tiny REINFORCE gradients moving at
    all, and restarting it from zero wastes the first steps after a resume.
    """
    import torch

    from rag_chunk import sweep

    device = next(policy.parameters()).device
    min_size, max_size = sweep._semantic_window(int(target_size))
    optimiser = torch.optim.AdamW(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    if resume_state:
        if resume_state.get("optimiser") is not None:
            optimiser.load_state_dict(resume_state["optimiser"])
        if resume_state.get("rng") is not None:
            rng.bit_generator.state = resume_state["rng"]
        # the policy trains with dropout, so torch's own generator is part of
        # the run state - without it a resumed run diverges from an unbroken one
        if resume_state.get("torch_rng") is not None:
            torch.set_rng_state(resume_state["torch_rng"])
        cuda_rng = resume_state.get("cuda_rng")
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
    keys = list(env.trainable_keys)
    log: list[dict] = []

    for step in range(int(start_step) + 1, int(steps) + 1):
        metrics_row = {}
        if eval_every and eval_fn is not None and (step - 1) % eval_every == 0:
            metrics_row = eval_fn(step - 1)
            if checkpoint_fn is not None:
                checkpoint_fn(step - 1, optimiser, rng)

        batch = [keys[i] for i in rng.choice(
            len(keys), size=min(docs_per_step, len(keys)), replace=False)]

        policy.train()
        logits_by_key = {}
        logits_np = {}
        for key in batch:
            emb = torch.as_tensor(np.asarray(sent_emb[key], dtype="float32"),
                                  device=device)
            logits = policy(emb)
            logits_by_key[key] = logits
            logits_np[key] = logits.detach().cpu().numpy()

        greedy_texts, sample_texts, sample_choices = {}, {}, {}
        for key in batch:
            sents = env.sentences[key]
            common = dict(target_size=target_size, min_size=min_size,
                          max_size=max_size, overlap=overlap)
            spans_g, _ = rollout(logits_np[key], len(sents), greedy=True,
                                 **common)
            spans_s, choices = rollout(logits_np[key], len(sents),
                                       greedy=False, rng=rng,
                                       temperature=temperature, **common)
            greedy_texts[key] = spans_to_texts(sents, spans_g)
            sample_texts[key] = spans_to_texts(sents, spans_s)
            sample_choices[key] = choices

        reward_greedy = env.reward(greedy_texts)
        reward_sample = env.reward(sample_texts)

        loss = None
        advantages, entropies = [], []
        for key in batch:
            advantage = reward_sample.get(key, 0.0) - reward_greedy.get(key, 0.0)
            advantages.append(advantage)
            logp, ent = choice_logprob(logits_by_key[key], sample_choices[key],
                                       temperature)
            entropies.append(float(ent.detach()))
            term = -advantage * logp - entropy_coef * ent
            loss = term if loss is None else loss + term
        loss = loss / max(len(batch), 1)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimiser.step()

        row = {
            "step": step,
            "loss": float(loss.detach()),
            "reward_greedy": float(np.mean(list(reward_greedy.values()) or [0.0])),
            "reward_sample": float(np.mean(list(reward_sample.values()) or [0.0])),
            "advantage": float(np.mean(advantages or [0.0])),
            "entropy": float(np.mean(entropies or [0.0])),
            "grad_norm": float(grad_norm),
        }
        row.update(metrics_row)
        log.append(row)
        if log_fn is not None:
            log_fn(row)

    # Checkpoint the finished run too, so `--steps N` later can pick up where
    # this one stopped instead of paying for the same steps twice.
    if checkpoint_fn is not None:
        checkpoint_fn(int(steps), optimiser, rng)
    return log


def evaluate_policy(policy, docs, questions, sent_emb, *, target_size, overlap,
                    method_label="rl"):
    """Doc-constrained Recall@k for the policy on a bench, via the sweep path.

    Uses ``retrieval.build_index_for_config`` with precomputed probabilities, so
    the RL policy is scored by exactly the code that produced every archived
    learned-chunker row. ``method="transformer"`` there selects the learned code
    path; the row is labelled ``method_label`` for the results CSV.
    """
    from rag_chunk import metrics, retrieval, sweep

    min_size, max_size = sweep._semantic_window(int(target_size))
    sentences_by_key = {d.get("title", d.get("id")): d["sentences"] for d in docs}
    probs = boundary_probs_by_id(policy, sentences_by_key, sent_emb)
    index = retrieval.build_index_for_config(
        "transformer", docs,
        semantic_policy="target", semantic_target_size=int(target_size),
        semantic_min_size=min_size, semantic_max_size=max_size,
        semantic_overlap=int(overlap), boundary_probs_by_id=probs)
    rec = metrics.recall_at_k(index, questions, C.RECALL_KS)
    row = {
        "method": method_label,
        "semantic_policy": "target",
        "semantic_target_size": int(target_size),
        "semantic_min_size": min_size,
        "semantic_max_size": max_size,
        "semantic_overlap": int(overlap),
        "avg_chunk_size": index.avg_chunk_size(),
        "n_chunks": len(index.chunk_texts),
        "n_docs": len(docs),
        "n_questions": len(questions),
    }
    for k in C.RECALL_KS:
        row[f"recall@{k}"] = rec["doc_constrained"][k]
    for k in C.RECALL_KS:
        row[f"recall_unconstrained@{k}"] = rec["unconstrained"][k]
    return row
