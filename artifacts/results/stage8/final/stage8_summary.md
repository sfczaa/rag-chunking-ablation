# Stage 8 - fine-tuned reranker evaluation

- Mode: **final** — 1000 docs / 1032 questions
- Base reranker: `BAAI/bge-reranker-base` (off-the-shelf arm)
- Arms share one identical BGE top-20 pool per config.

- Built-in check vs stage6/final: **OK (exact)**
- Go/no-go gate (fixed 15/0, ΔR@1 ft−ots = +0.1066, threshold 0.02): **GO**

## Per-config results

| config | pool@20 | bge R@1 | ots R@1 | ft R@1 | ft−ots ΔR@1 | ft−bge ΔR@1 | ft R@5 |
|---|---|---|---|---|---|---|---|
| fixed fixed_size=6,overlap=0 | 0.9264 | 0.5087 | 0.5446 | 0.6502 | +0.1056 | +0.1415 | 0.8585 |
| fixed fixed_size=15,overlap=0 | 0.9641 | 0.6279 | 0.6289 | 0.7355 | +0.1066 | +0.1076 | 0.9215 |
| fixed fixed_size=15,overlap=1 | 0.9593 | 0.6260 | 0.6269 | 0.7200 | +0.0930 | +0.0940 | 0.9215 |
| bilstm target=15,min=11,max=19,overlap=0 | 0.9632 | 0.6298 | 0.6269 | 0.7141 | +0.0872 | +0.0843 | 0.9244 |
| transformer target=15,min=11,max=19,overlap=0 | 0.9574 | 0.6366 | 0.6250 | 0.7316 | +0.1066 | +0.0950 | 0.9099 |
