# Stage 12 - one extra CE epoch vs the Stage 8 reranker

Verdict: TIE. CE - Stage 8 = +0.0019 R@1, 95% CI [-0.0120, +0.0159]: not a detectable difference.

- Stage 6 bench: 1000 docs / 1032 questions; all arms rerank the same BGE top-20 pool per config
- pipeline check (bge, off-the-shelf vs stage8/final, within 0.005, same chunk and question counts): PASS
- primary comparison: fixed 15/0, R@1, CE - Stage 8, paired over 1032 questions; threshold 0.02
- CE arm: 495 steps on 1977 groups, started from b4badac446f6:1112201932
- Stage 8 weights now vs archived at fixed 15/0: R@1 0.7345 vs 0.7355 (retrained weights, not used for validity)

| config | arm | R@1 | R@3 | R@5 | pool@20 |
| --- | --- | --- | --- | --- | --- |
| fixed 15/0 | bge | 0.6279 | 0.8159 | 0.8808 | 0.9641 |
| fixed 15/0 | rerank20 | 0.6289 | 0.8159 | 0.8760 | 0.9641 |
| fixed 15/0 | rerank20_ft | 0.7345 | 0.8750 | 0.9186 | 0.9641 |
| fixed 15/0 | rerank20_s11_ce | 0.7364 | 0.8808 | 0.9254 | 0.9641 |

| config | metric | comparison | mean | 95% CI |
| --- | --- | --- | --- | --- |
| fixed 15/0 | recall@1 | rerank20_s11_ce - rerank20_ft | +0.0019 | [-0.0120, +0.0159] |
| fixed 15/0 | recall@5 | rerank20_s11_ce - rerank20_ft | +0.0068 | [-0.0023, +0.0159] |
