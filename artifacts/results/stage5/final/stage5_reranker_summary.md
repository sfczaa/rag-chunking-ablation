# Stage 5 - cross-encoder reranking (BGE top-k + pretrained reranker)

- Boundary/chunking embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Dense retrieval embedding model: `BAAI/bge-base-en-v1.5`
- Query instruction: `Represent this sentence for searching relevant passages: `
- Reranker: `BAAI/bge-reranker-base` (off-the-shelf, no fine-tuning; batch=32, max_length=512)
- Candidate-pool depths: [20, 50]
- Stage 3 baseline compared from: `<project-root>/artifacts/results/stage3/final`
- Stage 3 bge-arm check: max |recall delta| = 0.0000, n_chunks mismatches = 0, unmatched = 0

Outputs in this folder:
- `rerank_sweep_results.csv`: one row per (arm, chunking config), incl. pool_recall@depth (the rerank ceiling) and rerank_seconds
- `rerank_best_config.json`: best config per arm + overall
- `rerank_matched.csv`: arms side by side per config with rerank-vs-bge deltas
- `stage3_vs_stage5_bge_check.csv`: Stage 5 bge arm vs archived Stage 3 rows
- `rerank_recall_vs_chunk_size.png` and `rerank_comparison.png`: plots

Best config per arm:
- bge: `fixed`, `fixed_size=15,overlap=0` — R@1=0.6502 R@3=0.8473 R@5=0.9212
- rerank20: `fixed`, `fixed_size=15,overlap=1` — R@1=0.6404 R@3=0.8621 R@5=0.9310
- rerank50: `fixed`, `fixed_size=15,overlap=1` — R@1=0.6404 R@3=0.8473 R@5=0.9163
