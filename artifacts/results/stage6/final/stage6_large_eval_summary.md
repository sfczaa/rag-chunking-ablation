# Stage 6 - larger-scale robustness evaluation

- Eval set: **1000 docs / 1032 questions** (requested ~1000 docs; the stream stops at the N-th usable document, so the counts are reported, not assumed)
- Boundary/chunking embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Dense retrieval embedding model: `BAAI/bge-base-en-v1.5`
- Reranker: `BAAI/bge-reranker-base` (off-the-shelf, top-20 only, no fine-tuning)
- bge arm: full 30-config grid; rerank20 arm: 5 selected configs

Best bge config: `fixed`, `fixed_size=15,overlap=0` — R@1=0.6279 R@3=0.8159 R@5=0.8808

## rerank20 vs bge on the selected configs

| config | pool_recall@20 | ΔR@1 | ΔR@3 | ΔR@5 |
|---|---|---|---|---|
| fixed fixed_size=6,overlap=0 | 0.9264 | +0.0359 | +0.0174 | +0.0203 |
| fixed fixed_size=15,overlap=0 | 0.9641 | +0.0010 | +0.0000 | -0.0048 |
| bilstm target=15,min=11,max=19,overlap=0 | 0.9632 | -0.0029 | +0.0203 | +0.0116 |
| transformer target=15,min=11,max=19,overlap=0 | 0.9574 | -0.0116 | +0.0039 | +0.0097 |
| fixed fixed_size=15,overlap=1 | 0.9593 | +0.0010 | +0.0174 | +0.0194 |

## Direction checks (do the Stage 1-5 conclusions replicate?)

- **yes** — 1. chunk size matters more than chunking method: pearson_r(avg_chunk_size, recall@5) over bge configs — observed 0.9515 (rule: r >= 0.5; small-eval reference: Stage 2/3 (n=203): r ~ +0.77)
- **yes** — 1. chunk size matters more than chunking method: recall@5 size effect (largest vs smallest size) minus mean method spread at matched (size, overlap) — observed 0.0523 (rule: size effect > method spread; small-eval reference: Stage 2 (n=203): size ~ +0.07 vs method <= 0.014)
- **yes** — 2. BGE-only remains a strong baseline: best bge recall@5 over the grid — observed 0.8808 (rule: >= 0.80 (loose heuristic; the corpus is ~5x larger, so some absolute drop vs the 200-doc eval is expected); small-eval reference: Stage 3 (200 docs): 0.921)
- **yes** — 3. rerank20 helps mainly at small chunks: recall@1 delta at fixed size 6 (vs mean delta at the size-15 configs) — observed 0.0359 (rule: delta(size 6) >= 2 SE (0.0308) and > mean size-15 delta (-0.0031); small-eval reference: Stage 5 (n=203): +0.113 at size 6 vs ~+0.002 mean at size 15)
- **yes** — 4. rerank20 does not clearly improve the size-15 sweet spot: max recall@1 delta over the size-15 configs — observed 0.001 (rule: max size-15 delta < 2 SE (0.0308); small-eval reference: Stage 5 (n=203): size-15 deltas within 1 SE ~ 0.034)

**5/5 direction checks replicate.**

