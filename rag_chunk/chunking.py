"""Chunking strategies: learned (BiLSTM) vs fixed-size baseline.

A "chunk" is a list of consecutive sentences; ``chunk_text`` joins it into the
string that actually gets embedded and indexed.
"""

from __future__ import annotations

import numpy as np

import config as C


def boundaries_to_chunks(sentences: list[str], cut_after: np.ndarray) -> list[list[str]]:
    """Split ``sentences`` wherever ``cut_after[i]`` is True (cut after sent i).

    ``cut_after`` has length ``len(sentences) - 1``.
    """
    if not sentences:
        return []
    chunks: list[list[str]] = []
    cur = [sentences[0]]
    for i in range(len(sentences) - 1):
        if cut_after[i]:
            chunks.append(cur)
            cur = []
        cur.append(sentences[i + 1])
    chunks.append(cur)
    return chunks


def _strip_redundant_tail(chunks: list[list[str]], overlap: int) -> list[list[str]]:
    """Drop a final chunk that overlap produced but its predecessor already covers.

    With ``overlap > 0`` the last window can be a pure tail of ``<= overlap``
    sentences (e.g. size=10, overlap=1 on a 10-sentence doc -> a phantom
    1-sentence chunk). Such a chunk sits entirely inside its predecessor, so it
    adds a near-duplicate vector and quietly deflates ``avg_chunk_size`` /
    inflates ``n_chunks`` — both inputs to the sweep's fair ranking. Remove it.
    """
    if overlap > 0 and len(chunks) >= 2 and len(chunks[-1]) <= overlap:
        chunks = chunks[:-1]
    return chunks


def probs_to_constrained_chunks(
    sentences: list[str],
    probs: np.ndarray,
    threshold: float | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    overlap: int | None = None,
) -> list[list[str]]:
    """Convert boundary probabilities into chunks with size constraints.

    The raw BiLSTM probabilities are noisy: a low threshold over-splits, while a
    high threshold can produce huge chunks. These constraints keep the learned
    splitter in a useful retrieval range.
    """
    if not sentences:
        return []
    if threshold is None:
        threshold = C.BOUNDARY_THRESHOLD
    if min_size is None:
        min_size = C.BILSTM_MIN_CHUNK_SIZE
    if max_size is None:
        max_size = C.BILSTM_MAX_CHUNK_SIZE
    if overlap is None:
        overlap = C.BILSTM_OVERLAP

    min_size = max(1, int(min_size))
    max_size = max(min_size, int(max_size))
    overlap = max(0, min(int(overlap), max_size - 1))

    chunks: list[list[str]] = []
    start = 0
    for i in range(len(sentences) - 1):
        size = i - start + 1
        should_cut = (
            (float(probs[i]) >= threshold and size >= min_size)
            or size >= max_size
        )
        if should_cut:
            end = i + 1
            chunks.append(list(sentences[start:end]))
            start = max(0, end - overlap)

    if start < len(sentences):
        chunks.append(list(sentences[start:]))
    return _strip_redundant_tail(chunks, overlap)


def semantic_target_chunks(
    sentences: list[str],
    boundary_probs,
    target_size: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    overlap: int | None = None,
) -> list[list[str]]:
    """Target-size semantic cutting.

    Walk the document; within each chunk's ``[min_size, max_size]`` window let
    the model pick the boundary it is most confident about (ties broken toward
    ``target_size``), cut there, then re-enter ``overlap`` sentences. This keeps
    learned chunks size-comparable to the fixed-size baseline, so the sweep can
    ask *"at the same size, does learned cutting pick better boundaries?"*

    ``boundary_probs`` is a length ``len(sentences) - 1`` sequence where
    ``boundary_probs[i]`` is P(cut after sentence ``i``); a numpy array or plain
    list both work. Defaults come from the ``SEMANTIC_*`` config at call time.
    """
    if target_size is None:
        target_size = C.SEMANTIC_TARGET_CHUNK_SIZE
    if min_size is None:
        min_size = C.SEMANTIC_MIN_CHUNK_SIZE
    if max_size is None:
        max_size = C.SEMANTIC_MAX_CHUNK_SIZE
    if overlap is None:
        overlap = C.SEMANTIC_OVERLAP

    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [list(sentences)]

    # Normalise so target sits inside [min, max] and overlap can never stall:
    # every produced chunk is >= min_size > overlap, so `start` strictly advances.
    min_size = max(1, int(min_size))
    max_size = max(min_size, int(max_size))
    target_size = min(max(int(target_size), min_size), max_size)
    overlap = max(0, min(int(overlap), min_size - 1))

    chunks: list[list[str]] = []
    start = 0
    while start < n:
        if n - start <= max_size:
            chunks.append(list(sentences[start:n]))     # tail fits; stop cutting
            break
        lo = start + min_size - 1                        # cut index giving length min_size
        hi = min(start + max_size - 1, n - 2)            # ... up to max_size / last boundary
        best_i, best_key = lo, None
        for i in range(lo, hi + 1):
            # primary: model confidence; tie-break: closeness to target length.
            key = (float(boundary_probs[i]), -abs((i - start + 1) - target_size))
            if best_key is None or key > best_key:
                best_key, best_i = key, i
        end = best_i + 1                                  # chunk = sentences[start:end]
        chunks.append(list(sentences[start:end]))
        start = end - overlap                             # re-enter `overlap` sentences
    return chunks


def predict_boundary_probs(sentences: list[str], embeddings, model) -> np.ndarray:
    """Per-boundary probabilities from the model for one document (length n-1).

    Empty array for documents with < 2 sentences. Read the model's device at
    call time so it works on CPU or GPU.
    """
    import torch

    n = len(sentences)
    if n <= 1:
        return np.zeros(max(0, n - 1), dtype="float32")
    device = next(model.parameters()).device
    emb = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    model.eval()
    return model.predict_proba(emb).cpu().numpy()


def bilstm_chunks(
    sentences: list[str],
    embeddings: np.ndarray,
    model,
    threshold: float | None = None,
) -> list[list[str]]:
    """Chunk a document using the trained BiLSTM boundary detector (threshold
    policy — the original Phase 4/5 behaviour).

    ``threshold`` defaults to the *current* ``C.BOUNDARY_THRESHOLD`` (read at
    call time, so ``C.apply(...)`` overrides take effect).
    """
    if threshold is None:
        threshold = C.BOUNDARY_THRESHOLD
    n = len(sentences)
    if n <= 1:
        return [list(sentences)] if n == 1 else []
    probs = predict_boundary_probs(sentences, embeddings, model)
    return probs_to_constrained_chunks(sentences, probs, threshold)


def chunks_from_probs(
    sentences: list[str],
    boundary_probs,
    policy: str | None = None,
    threshold: float | None = None,
    target_size: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    overlap: int | None = None,
) -> list[list[str]]:
    """Turn *precomputed* boundary probabilities into chunks under ``policy``.

    Split out from :func:`learned_chunks` so the sweep can run the model forward
    **once** per document (see :func:`predict_boundary_probs`) and then re-chunk
    it under many size/overlap configs cheaply — pure-Python slicing, no
    re-encoding.

    * ``"threshold"`` -> :func:`probs_to_constrained_chunks` (size-clamped
      probability cuts; falls back to the ``BILSTM_*`` knobs).
    * ``"target"``    -> :func:`semantic_target_chunks` (falls back to the
      ``SEMANTIC_*`` knobs).
    """
    if policy is None:
        policy = C.SEMANTIC_CHUNK_POLICY
    n = len(sentences)
    if n <= 1:
        return [list(sentences)] if n == 1 else []
    if policy == "threshold":
        return probs_to_constrained_chunks(
            sentences, boundary_probs, threshold, min_size, max_size, overlap)
    if policy == "target":
        return semantic_target_chunks(
            sentences, boundary_probs, target_size, min_size, max_size, overlap)
    raise ValueError(
        f"unknown chunk policy {policy!r} (expected 'target' or 'threshold')")


def learned_chunks(
    sentences: list[str],
    embeddings,
    model,
    policy: str | None = None,
    threshold: float | None = None,
    target_size: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    overlap: int | None = None,
) -> list[list[str]]:
    """Model-driven chunking: run the boundary model once, then dispatch on
    ``policy`` (``C.SEMANTIC_CHUNK_POLICY`` by default) via
    :func:`chunks_from_probs`.

    Size/overlap args default to the policy's config knobs; the sweep passes
    them explicitly per config, so the same trained model drives every learned
    configuration.
    """
    n = len(sentences)
    if n <= 1:
        return [list(sentences)] if n == 1 else []
    probs = predict_boundary_probs(sentences, embeddings, model)
    return chunks_from_probs(
        sentences, probs, policy, threshold,
        target_size, min_size, max_size, overlap)


def fixed_chunks(
    sentences: list[str],
    size: int | None = None,
    overlap: int | None = None,
) -> list[list[str]]:
    """Baseline: fixed-size chunks with optional sentence overlap.

    ``size`` and ``overlap`` default to the current config values, read at call
    time so ``C.apply(...)`` overrides take effect.
    """
    if size is None:
        size = C.FIXED_CHUNK_SIZE
    if overlap is None:
        overlap = C.FIXED_CHUNK_OVERLAP

    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    step = size - overlap
    chunks = [list(sentences[i : i + size]) for i in range(0, len(sentences), step)]
    return _strip_redundant_tail(chunks, overlap)


def chunk_text(chunk: list[str]) -> str:
    return " ".join(chunk)
