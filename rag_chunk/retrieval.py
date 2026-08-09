"""FAISS dense retrieval over chunks (inner-product on L2-normalised vectors,
i.e. cosine similarity), per the spec's ``IndexFlatIP``.

A ``ChunkIndex`` bundles the FAISS index with the chunk texts, their sizes
(in sentences) and the **source document id** of each chunk, so Phase 5 can
retrieve, report average chunk size, **and** tell whether a retrieved chunk
actually came from the question's gold document (doc-constrained Recall@k).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import config as C


class ChunkIndex:
    def __init__(self, chunk_texts: list[str] | None = None,
                 chunk_sizes: list[int] | None = None,
                 chunk_doc_ids: list[str] | None = None) -> None:
        self.chunk_texts = chunk_texts or []
        self.chunk_sizes = chunk_sizes or []   # #sentences per chunk
        self.chunk_doc_ids = chunk_doc_ids or []   # source doc id per chunk
        self.index = None

    def build(self) -> "ChunkIndex":
        import faiss

        from rag_chunk import embedding

        vecs = embedding.encode_retrieval_passages(self.chunk_texts)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        print(f"[index] built FAISS IndexFlatIP with {self.index.ntotal} chunks")
        return self

    def search_texts(self, queries: list[str], k: int) -> list[list[str]]:
        """Top-k chunk *texts* for each query."""
        from rag_chunk import embedding

        if self.index is None:
            raise RuntimeError("index not built/loaded")
        qv = embedding.encode_retrieval_queries(queries)
        k = min(k, len(self.chunk_texts))
        _, idx = self.index.search(qv, k)
        return [[self.chunk_texts[j] for j in row if j >= 0] for row in idx]

    def search_chunks(self, queries: list[str], k: int) -> list[list[dict]]:
        """Top-k chunks for each query as ``{'text', 'doc_id'}`` dicts, so the
        caller can check which source document a retrieved chunk came from."""
        from rag_chunk import embedding

        if self.index is None:
            raise RuntimeError("index not built/loaded")
        qv = embedding.encode_retrieval_queries(queries)
        k = min(k, len(self.chunk_texts))
        _, idx = self.index.search(qv, k)
        out: list[list[dict]] = []
        for row in idx:
            out.append([
                {"text": self.chunk_texts[j],
                 "doc_id": self.chunk_doc_ids[j] if j < len(self.chunk_doc_ids) else None}
                for j in row if j >= 0
            ])
        return out

    def search_ids(self, queries: list[str], k: int) -> list[list[int]]:
        """Top-k chunk *ids* (row indices into ``chunk_texts``) for each query.
        Used by the Stage 4 hybrid sweep to fuse dense and BM25 rankings over
        one shared id space."""
        from rag_chunk import embedding

        if self.index is None:
            raise RuntimeError("index not built/loaded")
        qv = embedding.encode_retrieval_queries(queries)
        k = min(k, len(self.chunk_texts))
        _, idx = self.index.search(qv, k)
        return [[int(j) for j in row if j >= 0] for row in idx]

    def avg_chunk_size(self) -> float:
        return sum(self.chunk_sizes) / len(self.chunk_sizes) if self.chunk_sizes else 0.0

    # -- persistence --------------------------------------------------------- #
    def save(self, prefix: Path) -> None:
        import faiss

        prefix.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(prefix.with_suffix(".faiss")))
        prefix.with_suffix(".json").write_text(
            json.dumps({"chunk_texts": self.chunk_texts,
                        "chunk_sizes": self.chunk_sizes,
                        "chunk_doc_ids": self.chunk_doc_ids},
                       ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, prefix: Path) -> "ChunkIndex":
        import faiss

        meta = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
        obj = cls(meta["chunk_texts"], meta["chunk_sizes"], meta.get("chunk_doc_ids"))
        obj.index = faiss.read_index(str(prefix.with_suffix(".faiss")))
        return obj

    @classmethod
    def exists(cls, prefix: Path) -> bool:
        return prefix.with_suffix(".faiss").exists() and prefix.with_suffix(".json").exists()


def index_prefix(name: str) -> Path:
    """Path prefix for a named **main-evaluation** index (e.g. 'bilstm' / 'fixed').

    Lives under ``nq/indices/main/`` so the Phase 5 pair never mixes with the
    sweep's per-config indices (which are in-memory by default).
    """
    return C.retrieval_index_main_dir() / f"index_{name}"


# --------------------------------------------------------------------------- #
# Cache validity: a manifest records the config the indices were built with, so
# Phase 5 rebuilds them when anything that affects chunking/retrieval changes
# (threshold, fixed chunk size, embedding model, model weights, corpus size).
# --------------------------------------------------------------------------- #
def _manifest_path() -> Path:
    return C.retrieval_index_main_dir() / "index_manifest.json"


def _file_sha1(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _corpus_hash(docs: list, questions: list) -> str:
    """Stable SHA-1 of the *content* of the NQ corpus + queries, so a rebuilt
    corpus that happens to have the same doc/question counts (e.g. after an
    ``NQ_CONFIG`` change) still invalidates the cached indices."""
    h = hashlib.sha1()
    for d in docs:
        h.update(b"D|")
        h.update(str(d.get("title", "")).encode("utf-8"))
        h.update(b"|")
        h.update("␟".join(d.get("sentences", [])).encode("utf-8"))  # actually-indexed text
        h.update(b"\n")
    for q in questions:
        h.update(b"Q|")
        h.update(str(q.get("question", "")).encode("utf-8"))
        h.update(b"|")
        h.update(str(q.get("answer", "")).encode("utf-8"))
        h.update(b"|")
        h.update(str(q.get("doc_title", "")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def current_signature(docs: list, questions: list,
                      threshold: float | None = None) -> dict:
    """Config fingerprint that the cached **main** indices must match to be reused.

    Records everything that affects the Phase 5 ``bilstm`` + ``fixed`` indices
    *and* the chunking-policy knobs, so changing any of them forces a rebuild
    instead of silently evaluating new settings against a stale FAISS cache.

    ``threshold`` defaults to the *current* ``C.BOUNDARY_THRESHOLD`` (read at
    call time, so ``C.apply(...)`` overrides take effect). Bumping
    ``manifest_version`` invalidates every previously cached index on purpose.
    """
    if threshold is None:
        threshold = C.BOUNDARY_THRESHOLD
    return {
        "manifest_version": 3,
        "method": "bilstm+fixed",
        "model_type": C.MODEL_TYPE,
        "embedding_model": C.RETRIEVAL_EMBED_MODEL,
        "boundary_embedding_model": C.BOUNDARY_EMBED_MODEL,
        "boundary_embed_dim": C.BOUNDARY_EMBED_DIM,
        "retrieval_embedding_model": C.RETRIEVAL_EMBED_MODEL,
        "retrieval_embed_dim": C.RETRIEVAL_EMBED_DIM,
        "retrieval_embed_normalize": C.RETRIEVAL_EMBED_NORMALIZE,
        "retrieval_query_instruction": C.retrieval_query_instruction(),
        "model_sha1": _file_sha1(C.model_path()),
        "n_docs": len(docs),
        "n_questions": len(questions),
        "corpus_hash": _corpus_hash(docs, questions),
        "fixed_chunk_size": C.FIXED_CHUNK_SIZE,
        "fixed_chunk_overlap": C.FIXED_CHUNK_OVERLAP,
        "semantic_chunk_policy": C.SEMANTIC_CHUNK_POLICY,
        "semantic_target_chunk_size": C.SEMANTIC_TARGET_CHUNK_SIZE,
        "semantic_min_chunk_size": C.SEMANTIC_MIN_CHUNK_SIZE,
        "semantic_max_chunk_size": C.SEMANTIC_MAX_CHUNK_SIZE,
        "semantic_overlap": C.SEMANTIC_OVERLAP,
        "boundary_threshold": float(threshold),
        "bilstm_min_chunk_size": C.BILSTM_MIN_CHUNK_SIZE,
        "bilstm_max_chunk_size": C.BILSTM_MAX_CHUNK_SIZE,
        "bilstm_overlap": C.BILSTM_OVERLAP,
    }


def load_manifest() -> dict | None:
    p = _manifest_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_indexes(model, threshold: float | None = None, save: bool = True) -> dict:
    """Phase 4: chunk every NQ doc both ways, embed chunks, build both indices.

    ``threshold`` defaults to the *current* ``C.BOUNDARY_THRESHOLD`` (read at
    call time). Returns
    ``{'bilstm': ChunkIndex, 'fixed': ChunkIndex, 'docs', 'questions'}``.
    """
    from rag_chunk import chunking, embedding, nq_data

    if threshold is None:
        threshold = C.BOUNDARY_THRESHOLD
    docs, questions = nq_data.prepare_nq()
    bi_texts, bi_sizes, bi_docs = [], [], []
    fx_texts, fx_sizes, fx_docs = [], [], []
    for d in docs:
        sents = d["sentences"]
        doc_id = d["title"]            # matches questions' "doc_title"
        # sentence embeddings (un-normalised) feed the BiLSTM boundary detector
        sent_emb = embedding.encode(sents, normalize=False)
        for ch in chunking.bilstm_chunks(sents, sent_emb, model, threshold):
            bi_texts.append(chunking.chunk_text(ch))
            bi_sizes.append(len(ch))
            bi_docs.append(doc_id)
        for ch in chunking.fixed_chunks(sents):
            fx_texts.append(chunking.chunk_text(ch))
            fx_sizes.append(len(ch))
            fx_docs.append(doc_id)

    bilstm = ChunkIndex(bi_texts, bi_sizes, bi_docs).build()
    fixed = ChunkIndex(fx_texts, fx_sizes, fx_docs).build()
    if save:
        bilstm.save(index_prefix("bilstm"))
        fixed.save(index_prefix("fixed"))
        _manifest_path().write_text(
            json.dumps(current_signature(docs, questions, threshold)),
            encoding="utf-8",
        )
    return {"bilstm": bilstm, "fixed": fixed, "docs": docs, "questions": questions}


# --------------------------------------------------------------------------- #
# Flexible single-config builder (used by the Phase 6 sweep)
# --------------------------------------------------------------------------- #
def build_index_for_config(
    method: str,
    docs: list[dict],
    model=None,
    fixed_size: int | None = None,
    fixed_overlap: int | None = None,
    semantic_policy: str | None = None,
    semantic_target_size: int | None = None,
    semantic_min_size: int | None = None,
    semantic_max_size: int | None = None,
    semantic_overlap: int | None = None,
    threshold: float | None = None,
    boundary_probs_by_id: dict | None = None,
    save_prefix: Path | None = None,
) -> "ChunkIndex":
    """Chunk every doc under one ``(method, config)`` and build a single index.

    ``method`` is ``"fixed"``, ``"bilstm"``, or ``"transformer"``. The two
    learned methods share one code path — the ``model`` (or precomputed probs)
    you pass in decides which boundary model is used — and need either ``model``
    *or* a precomputed ``boundary_probs_by_id`` map (doc id -> per-boundary
    probabilities); the sweep passes the latter so the model forward runs once
    per doc rather than once per config.

    By default **nothing is persisted** — the sweep builds these in memory and
    discards them so per-config FAISS files never pile up under ``nq/``. Pass
    ``save_prefix`` (e.g. ``C.NQ_INDEX_SWEEP_DIR / 'fixed_s10_o1'``) to opt into
    writing ``.faiss``/``.json``.
    """
    from rag_chunk import chunking, embedding

    texts: list[str] = []
    sizes: list[int] = []
    doc_ids: list[str] = []
    for d in docs:
        sents = d["sentences"]
        doc_key = d.get("title", d.get("id"))          # matches questions' doc_title
        if method == "fixed":
            doc_chunks = chunking.fixed_chunks(sents, fixed_size, fixed_overlap)
        elif method in ("bilstm", "transformer"):
            if boundary_probs_by_id is not None:
                probs = boundary_probs_by_id[doc_key]
            elif model is not None:
                probs = chunking.predict_boundary_probs(
                    sents, embedding.encode(sents, normalize=False), model)
            else:
                raise ValueError(
                    f"method {method!r} needs `model` or `boundary_probs_by_id`")
            doc_chunks = chunking.chunks_from_probs(
                sents, probs,
                policy=semantic_policy, threshold=threshold,
                target_size=semantic_target_size,
                min_size=semantic_min_size, max_size=semantic_max_size,
                overlap=semantic_overlap,
            )
        else:
            raise ValueError(
                f"unknown method {method!r} (expected 'fixed', 'bilstm', 'transformer')")
        for ch in doc_chunks:
            texts.append(chunking.chunk_text(ch))
            sizes.append(len(ch))
            doc_ids.append(doc_key)

    index = ChunkIndex(texts, sizes, doc_ids).build()
    if save_prefix is not None:
        index.save(Path(save_prefix))
    return index
