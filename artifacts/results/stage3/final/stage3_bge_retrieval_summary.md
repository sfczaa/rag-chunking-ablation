# Stage 3 - BGE retrieval embedding ablation

- Boundary/chunking embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval embedding model: `BAAI/bge-base-en-v1.5`
- Retrieval embeddings normalized: `True`
- Query instruction: `Represent this sentence for searching relevant passages: `
- Stage 2 archive compared from: `<project-root>/artifacts/results/stage2/final`

Outputs in this folder:
- `sweep_results.csv`: Stage 3 BGE sweep rows
- `best_config.json`: best Stage 3 BGE config
- `fair_comparison_table.csv`: matching-config Stage 2 MiniLM vs Stage 3 BGE table
- `stage2_vs_stage3_matched.csv`: the same matched table under an unambiguous name
- `recall_vs_chunk_size.png` and `model_comparison.png`: Stage 3 BGE plots

Best Stage 3 config:
- method: `fixed`
- chunk config: `fixed_size=15,overlap=0`
