"""rag_chunk — shared logic for the semantic-chunking RAG project.

Submodules:
  wiki_data   Phase 1  Wikipedia -> (sentences, labels)
  embedding   Phase 2  offline sentence embeddings + reusable encoder
  training    Phase 3  BiLSTM training loop
  chunking             learned vs fixed-size chunking
  nq_data     Phase 4  Natural Questions corpus + questions
  retrieval   Phase 4  FAISS index build / search
  metrics     Phase 5  Recall@k, Boundary F1, threshold sweep + prob diagnostics
  evaluation  Phase 5  end-to-end comparison + figure/table
  sweep       Phase 6  chunking sweep optimizer (best config / fair table / plots)
  calibration Phase 7  Transformer boundary-threshold calibration (Stage 2.1)
"""

__all__ = [
    "wiki_data", "embedding", "training", "chunking",
    "nq_data", "retrieval", "metrics", "evaluation", "sweep", "calibration",
]
