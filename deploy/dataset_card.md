---
license: mit
language:
  - en
tags:
  - retrieval
  - rag
  - benchmark
---

# RAG chunking demo assets (Natural Questions bench + prebuilt FAISS indices)

Prebuilt, read-only assets that let an interactive retrieval demo start instantly
without embedding a corpus at boot.

## Contents

| path | what it is |
|---|---|
| `data/nq/large_n1000/docs.jsonl` | 1000 Natural Questions Wikipedia documents, pre-split into sentences (`id`, `title`, `sentences`) |
| `data/nq/large_n1000/questions.jsonl` | 1032 NQ **validation** questions with short answers and gold document titles |
| `data/nq/large_n1000/indices/demo/index_fixed.{faiss,json}` | FAISS index over the corpus chunked at a fixed 15 sentences / overlap 0 — 19507 chunks, mean 14.65 sentences |
| `data/nq/large_n1000/indices/demo/index_bilstm.{faiss,json}` | FAISS index over the same corpus chunked by a learned BiLSTM boundary model at target size 15 — 19306 chunks, mean 14.80 sentences |

Both indices are `IndexFlatIP` over `BAAI/bge-base-en-v1.5` embeddings,
L2-normalised so inner product equals cosine similarity. The `.json` sidecar
holds each chunk's text, sentence count and source document id, so a retrieved
chunk can be traced back to its gold document.

The directory layout mirrors the study's data root, so a consumer can point its
data-root environment variable at a snapshot of this repo and resolve every path
unchanged.

## Provenance

These are the exact indices behind the study's large-scale evaluation — their
chunk counts and mean chunk sizes match the archived results row for row. Study,
code and full write-up: https://github.com/sfczaa/rag-chunking-ablation

## Licensing

Metadata and index structures: MIT. The underlying Wikipedia passage text comes
from Natural Questions and remains under CC BY-SA 3.0.
