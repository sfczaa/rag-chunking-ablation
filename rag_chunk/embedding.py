"""Phase 2 — offline sentence embeddings + a reusable encoder.

The same ``all-MiniLM-L6-v2`` encoder is used everywhere:
  * Phase 2 pre-computes per-article sentence embeddings once (so training
    spends its time on the BiLSTM, not on re-embedding every epoch);
  * Phase 4 reuses it to embed chunks and questions.

Stored sentence embeddings are kept **un-normalised** (raw encoder output) so
they feed the BiLSTM directly; L2-normalisation is applied later, only for the
cosine/inner-product FAISS retrieval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config as C

_MODELS = {}


def _model_name(role: str, model_name: str | None = None) -> str:
    if model_name is not None:
        return model_name
    if role == "boundary":
        return getattr(C, "BOUNDARY_EMBED_MODEL", C.EMBED_MODEL)
    if role == "retrieval":
        return getattr(C, "RETRIEVAL_EMBED_MODEL", C.EMBED_MODEL)
    raise ValueError(f"unknown embedding role {role!r}")


def _batch_size(role: str) -> int:
    if role == "retrieval":
        return getattr(C, "RETRIEVAL_EMBED_BATCH", C.EMBED_BATCH)
    return C.EMBED_BATCH


def _empty_dim(role: str, model_name: str | None = None) -> int:
    if role == "boundary":
        return getattr(C, "BOUNDARY_EMBED_DIM", C.EMBED_DIM)
    configured = getattr(C, "RETRIEVAL_EMBED_DIM", None)
    if configured is not None:
        return int(configured)
    model = get_embedder(role=role, model_name=model_name)
    dim = model.get_sentence_embedding_dimension()
    return int(dim) if dim is not None else C.EMBED_DIM


def _prepare_texts(texts: list[str], role: str, is_query: bool,
                   model_name: str | None = None) -> list[str]:
    if role != "retrieval" or not is_query:
        return texts
    instruction = C.retrieval_query_instruction(_model_name(role, model_name))
    if not instruction:
        return texts
    return [instruction + text for text in texts]


def get_embedder(role: str = "boundary", model_name: str | None = None):
    """Singleton ``SentenceTransformer`` per role/model (GPU if available)."""
    name = _model_name(role, model_name)
    key = (role, name)
    if key not in _MODELS:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODELS[key] = SentenceTransformer(name, device=device)
        print(f"[embed] loaded {role} model {name} on {device}")
    return _MODELS[key]


def encode(
    texts: list[str],
    normalize: bool = False,
    *,
    role: str = "boundary",
    is_query: bool = False,
    model_name: str | None = None,
) -> np.ndarray:
    """Encode a list of strings -> ``(len(texts), EMBED_DIM)`` float32 array."""
    if not texts:
        return np.zeros((0, _empty_dim(role, model_name)), dtype="float32")
    model = get_embedder(role=role, model_name=model_name)
    texts = _prepare_texts(texts, role=role, is_query=is_query, model_name=model_name)
    vecs = model.encode(
        texts,
        batch_size=_batch_size(role),
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype("float32")


def encode_retrieval_passages(texts: list[str]) -> np.ndarray:
    """Encode chunk/passages with the active retrieval model."""
    return encode(
        texts,
        normalize=C.RETRIEVAL_EMBED_NORMALIZE,
        role="retrieval",
        is_query=False,
    )


def encode_retrieval_queries(texts: list[str]) -> np.ndarray:
    """Encode retrieval queries, applying the BGE query instruction when active."""
    return encode(
        texts,
        normalize=C.RETRIEVAL_EMBED_NORMALIZE,
        role="retrieval",
        is_query=True,
    )


def _emb_path(article_id: str) -> Path:
    return C.EMBEDDINGS_DIR / f"{article_id}.npy"


def embed_offline() -> int:
    """Embed every article's sentences and cache to disk. Returns #new files."""
    from rag_chunk import wiki_data

    C.ensure_dirs()
    n_done = 0
    records = list(wiki_data.iter_all_records())
    for i, rec in enumerate(records, 1):
        out = _emb_path(rec["id"])
        if out.exists():
            continue
        np.save(out, encode(rec["sentences"], normalize=False))
        n_done += 1
        if i % 200 == 0:
            print(f"[embed] {i}/{len(records)} articles")
    print(f"[embed] wrote {n_done} new embedding files")
    return n_done


def load_article_embeddings(article_id: str) -> np.ndarray:
    return np.load(_emb_path(article_id))
