# Stage 13 - the RL reranker at a matched training budget

Verdict: INCONCLUSIVE-BUDGET. the RL arm accumulated 1769 live groups, short of the 1977 the budget condition asks for.

- Stage 6 bench: 1000 docs / 1032 questions; all arms rerank the same BGE top-20 pool per config
- pipeline check (bge, off-the-shelf vs stage8/final, within 0.005, same chunk and question counts): PASS
- primary comparison: fixed 15/0, R@1, rl4 - ce4, paired over 1032 questions; threshold 0.02
- RL budget: 1769 live groups against the 1977 the condition asks for; live fraction 0.224
- Stage 8 weights now vs archived at fixed 15/0: R@1 0.7345 vs 0.7355 (retrained weights, not used for validity)

| config | arm | R@1 | R@3 | R@5 | pool@20 |
| --- | --- | --- | --- | --- | --- |
| fixed 15/0 | bge | 0.6279 | 0.8159 | 0.8808 | 0.9641 |
| fixed 15/0 | rerank20 | 0.6289 | 0.8159 | 0.8760 | 0.9641 |
| fixed 15/0 | rerank20_ft | 0.7345 | 0.8750 | 0.9186 | 0.9641 |
| fixed 15/0 | rerank20_s13_ce4 | 0.7171 | 0.8760 | 0.9264 | 0.9641 |
| fixed 15/0 | rerank20_s13_rl4 | 0.7219 | 0.8886 | 0.9302 | 0.9641 |
| fixed 15/0 | rerank20_s13_ce8 | 0.7161 | 0.8576 | 0.9050 | 0.9641 |
| fixed 15/0 | rerank20_s13_rl8 | 0.7267 | 0.8837 | 0.9273 | 0.9641 |

| config | metric | comparison | mean | 95% CI |
| --- | --- | --- | --- | --- |
| fixed 15/0 | recall@1 | rerank20_s13_rl4 - rerank20_s13_ce4 | +0.0048 | [-0.0109, +0.0206] |
| fixed 15/0 | recall@5 | rerank20_s13_rl4 - rerank20_s13_ce4 | +0.0039 | [-0.0062, +0.0139] |
| fixed 15/0 | recall@1 | rerank20_s13_rl8 - rerank20_s13_ce8 | +0.0107 | [-0.0060, +0.0273] |
| fixed 15/0 | recall@5 | rerank20_s13_rl8 - rerank20_s13_ce8 | +0.0223 | [+0.0096, +0.0350] |
| fixed 15/0 | recall@1 | rerank20_s13_ce4 - rerank20_ft | -0.0174 | [-0.0333, -0.0016] |
| fixed 15/0 | recall@5 | rerank20_s13_ce4 - rerank20_ft | +0.0078 | [-0.0033, +0.0188] |
| fixed 15/0 | recall@1 | rerank20_s13_rl4 - rerank20_ft | -0.0126 | [-0.0284, +0.0032] |
| fixed 15/0 | recall@5 | rerank20_s13_rl4 - rerank20_ft | +0.0116 | [+0.0020, +0.0213] |
| fixed 15/0 | recall@1 | rerank20_s13_ce8 - rerank20_ft | -0.0184 | [-0.0353, -0.0016] |
| fixed 15/0 | recall@5 | rerank20_s13_ce8 - rerank20_ft | -0.0136 | [-0.0270, -0.0002] |
| fixed 15/0 | recall@1 | rerank20_s13_rl8 - rerank20_ft | -0.0078 | [-0.0245, +0.0090] |
| fixed 15/0 | recall@5 | rerank20_s13_rl8 - rerank20_ft | +0.0087 | [-0.0015, +0.0189] |
