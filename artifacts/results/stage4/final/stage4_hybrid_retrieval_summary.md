# Stage 4 - hybrid retrieval ablation (BM25 + BGE + RRF)

- Boundary/chunking embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Dense retrieval embedding model: `BAAI/bge-base-en-v1.5`
- Query instruction: `Represent this sentence for searching relevant passages: `
- BM25: Okapi, k1=1.5, b=0.75, lower-case alphanumeric tokens, Lucene idf
- RRF: k=60, fuse depth=50 per retriever
- Stage 3 baseline compared from: `<project-root>/artifacts/results/stage3/final`
- Stage 3 bge-arm check: max |recall delta| = 0.0000, n_chunks mismatches = 0, unmatched = 0

Outputs in this folder:
- `hybrid_sweep_results.csv`: one row per (retriever, chunking config)
- `hybrid_best_config.json`: best config per retriever + overall
- `hybrid_retriever_matched.csv`: bge / bm25 / rrf side by side per config
- `stage3_vs_stage4_bge_check.csv`: Stage 4 bge arm vs archived Stage 3 rows
- `hybrid_recall_vs_chunk_size.png` and `hybrid_retriever_comparison.png`: plots

Best config per retriever:
- bge: `fixed`, `fixed_size=15,overlap=0` — R@1=0.6502 R@3=0.8473 R@5=0.9212
- bm25: `bilstm`, `target=15,min=11,max=19,overlap=0` — R@1=0.5025 R@3=0.7094 R@5=0.8030
- rrf: `fixed`, `fixed_size=15,overlap=0` — R@1=0.6404 R@3=0.7980 R@5=0.8867
