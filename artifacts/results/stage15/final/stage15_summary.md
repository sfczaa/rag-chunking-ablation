# Stage 15 - a larger cross-encoder under the Stage 8 recipe

Verdict: LARGE-BETTER. large beats Stage 8 by +0.0213 R@1, 95% CI [+0.0009, +0.0417], at or above the 0.02 floor.

- Stage 6 bench: 1000 docs / 1032 questions; all arms rerank the same BGE top-20 pool
- pipeline check (bge, off-the-shelf vs stage8/final, within 0.005, same chunk and question counts): PASS
- primary comparison: fixed 15/0, R@1, rerank20_s15_large - rerank20_ft, paired over 1032 questions; threshold 0.02
- base model BAAI/bge-reranker-large at 55611d7bca2a7133960a6d3b71e083071bbfc312
- training: 1018 steps, 64.3 min, final mean epoch loss 0.3147, non-finite losses 0

| arm | R@1 | R@3 | R@5 | pool@20 | s per question, load included |
| --- | --- | --- | --- | --- | --- |
| bge | 0.6279 | 0.8159 | 0.8808 | 0.9641 | - |
| rerank20 | 0.6289 | 0.8159 | 0.8760 | 0.9641 | 0.632 |
| rerank20_ft | 0.7345 | 0.8750 | 0.9186 | 0.9641 | 0.631 |
| rerank20_large | 0.5940 | 0.8052 | 0.8808 | 0.9641 | 1.867 |
| rerank20_s15_large | 0.7558 | 0.8915 | 0.9293 | 0.9641 | 1.867 |

| metric | comparison | mean | 95% CI |
| --- | --- | --- | --- |
| recall@1 | rerank20_s15_large - rerank20_ft | +0.0213 | [+0.0009, +0.0417] |
| recall@5 | rerank20_s15_large - rerank20_ft | +0.0107 | [-0.0015, +0.0228] |
| recall@1 | rerank20_large - rerank20 | -0.0349 | [-0.0631, -0.0067] |
| recall@5 | rerank20_large - rerank20 | +0.0048 | [-0.0125, +0.0222] |
| recall@1 | rerank20_s15_large - rerank20_large | +0.1618 | [+0.1347, +0.1890] |
| recall@5 | rerank20_s15_large - rerank20_large | +0.0484 | [+0.0331, +0.0638] |
