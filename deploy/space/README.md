---
title: Retrieval-aware RAG Chunking
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Does chunking method or chunk size drive RAG recall? Try it.
models:
  - sfczaa/bge-reranker-base-nq-ft
  - BAAI/bge-base-en-v1.5
  - BAAI/bge-reranker-base
datasets:
  - sfczaa/rag-chunking-ablation-demo-assets
---

# Retrieval-aware RAG chunking — interactive demo

Side-by-side retrieval over a 1000-document Wikipedia bench with 1032 Natural
Questions. Pick a bench question (or type your own), choose a ranking arm, and
compare two chunking strategies on an identical BGE top-20 candidate pool:

- **fixed 15 sentences / overlap 0** — the deployment configuration;
- **BiLSTM boundary model, target size 15** — a learned semantic chunker.

Ranking arms:

| arm | what it does |
|---|---|
| `BGE dense` | the raw dense order, no reranking |
| `+ off-the-shelf rerank20` | reorders the pool with `BAAI/bge-reranker-base` |
| `+ fine-tuned rerank20` | reorders it with the same model fine-tuned on NQ train |

Bench questions show their gold answer and document, so retrieved chunks are
badged (`gold doc`, `answer hit`), the answer string is highlighted, and each
chunk shows how far reranking moved it (`dense #7 → #1`).

## What the study behind it found

- **Chunk size dominates recall; chunking method does not.** Across 30 configs
  (3 methods × 5 sizes × 2 overlaps, n=1032), the size effect over 6→15
  sentences is +0.064 R@5 (p ≈ 3e-16) — roughly **18×** the largest
  chunking-method coefficient, which is not significant. The largest observed
  between-method gap (0.023) sits *below* the 0.032 detection floor at this
  sample size, so the tie is measured, not merely unproven.
- **The embedder was the big lever** (MiniLM → BGE lifted all 30 matched
  configs), while BM25/RRF hybrid retrieval did not help.
- **Fine-tuning the reranker was the one intervention that moved the size-15
  sweet spot** (+0.107 R@1 in-domain). Cross-dataset it only recovers parity
  with plain dense retrieval — reported honestly as damage control, not lift.

Full write-up, code and archived results:
**https://github.com/sfczaa/rag-chunking-ablation**

## Notes

- Runs on ZeroGPU: a GPU is attached only while a query is being served, so the
  first query after an idle period waits briefly in a queue.
- The FAISS indices and bench corpus are prebuilt and downloaded at startup;
  nothing is embedded at boot and no experiment is ever re-run here.
- Reported per-side timings are retrieval + reranking only.
