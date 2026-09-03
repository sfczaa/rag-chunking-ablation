# RAG Chunking Ablation

A controlled study of whether RAG retrieval depends more on **chunking method** or **chunk size**.

The project compares fixed-size, BiLSTM, and Transformer chunking on Natural Questions (NQ), using doc-constrained Recall@k. Each stage changes one variable and checks that the previous result still reproduces.

[Live demo](https://huggingface.co/spaces/sfczaa/rag-chunking-ablation-demo) · [Fine-tuned reranker](https://huggingface.co/sfczaa/bge-reranker-base-nq-ft) · [Benchmark assets](https://huggingface.co/datasets/sfczaa/rag-chunking-ablation-demo-assets)

## Key results

- **Chunk size matters more than chunking method.** At matched sizes and overlaps, fixed-size, BiLSTM, and Transformer chunking are effectively tied.
- **The retrieval embedder was the main improvement.** Switching MiniLM to BGE improved all 30 matched configurations.
- **Hybrid retrieval was not consistently better.** BM25/RRF and an off-the-shelf reranker did not beat dense retrieval reliably.
- **Fine-tuning helped in-domain.** The fine-tuned reranker improved NQ results, but transferred to TriviaQA as parity rather than a lift.

The detailed claims are backed by archived CSVs and figures in [`artifacts/results/`](artifacts/results).

## Reproduce

The main path is Google Colab with a GPU:

1. Upload the project folder to Google Drive.
2. Open [`notebooks/RAG_chunk_optimize_colab.ipynb`](notebooks/RAG_chunk_optimize_colab.ipynb).
3. Run the smoke test first, then run the numbered stages.

For a local smoke test:

```bash
pip install -r requirements.txt
export RAG_DATA_ROOT=./artifacts
python scripts/0_smoke_test.py
```

Long stages are checkpointed and resumable. Results are written to `artifacts/results/latest/` and archived under `artifacts/results/<stage>/final/`.

## Results

![Best doc-constrained Recall@5 per stage](artifacts/results/portfolio/best_r5_evolution.png)

The stage write-ups include the exact settings, reproduction checks, and negative results:

- [Stage 1 — Chunking sweep](docs/stage1_chunking_optimizer.md)
- [Stage 2 — Transformer boundary model](docs/stage2_transformer_boundary.md)
- [Stage 3 — BGE retrieval embedding](docs/stage3_bge_retrieval.md)
- [Stage 4 — Hybrid retrieval](docs/stage4_hybrid_retrieval.md)
- [Stage 5 — Reranking](docs/stage5_reranker.md)
- [Stage 6 — Larger evaluation](docs/stage6_large_eval.md)
- [Stage 7 — Cross-dataset evaluation](docs/stage7_cross_dataset.md)
- [Stage 8 — Reranker fine-tuning](docs/stage8_reranker_finetune.md)

## Repository layout

```text
rag_chunk/       chunking, retrieval, reranking, metrics, and sweeps
models/          BiLSTM and Transformer boundary models
scripts/         numbered pipeline and analysis entry points
notebooks/       Colab notebooks
docs/            stage write-ups
deploy/          Hugging Face Space payload
artifacts/       archived results and runtime data
config.py        paths and experiment settings
```

## License

[MIT](LICENSE)
