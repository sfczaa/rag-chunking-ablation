# Stage 14 - more training data for the fine-tuned reranker

Verdict: TIE. 10k - Stage 8 = -0.0048 R@1, 95% CI [-0.0204, +0.0107]: not a detectable difference.

- Stage 6 bench: 1000 docs / 1032 questions; all arms rerank the same BGE top-20 pool per config
- pipeline check (bge, off-the-shelf vs stage8/final, within 0.005, same chunk and question counts): PASS
- primary comparison: fixed 15/0, R@1, 10k - Stage 8, paired over 1032 questions; threshold 0.02
- training groups: Stage 8 2034, 4k 4011, 10k 10334 (size condition 9000)
- R@1 rises monotonically from 2k to 4k to 10k: no
- Stage 8 weights now vs archived at fixed 15/0: R@1 0.7345 vs 0.7355 (retrained weights, not used for validity)

| config | arm | R@1 | R@3 | R@5 | pool@20 |
| --- | --- | --- | --- | --- | --- |
| fixed 15/0 | bge | 0.6279 | 0.8159 | 0.8808 | 0.9641 |
| fixed 15/0 | rerank20 | 0.6289 | 0.8159 | 0.8760 | 0.9641 |
| fixed 15/0 | rerank20_ft | 0.7345 | 0.8750 | 0.9186 | 0.9641 |
| fixed 15/0 | rerank20_s14_4k | 0.7355 | 0.8905 | 0.9234 | 0.9641 |
| fixed 15/0 | rerank20_s14_10k | 0.7297 | 0.8798 | 0.9234 | 0.9641 |

| config | metric | comparison | mean | 95% CI |
| --- | --- | --- | --- | --- |
| fixed 15/0 | recall@1 | rerank20_s14_10k - rerank20_ft | -0.0048 | [-0.0204, +0.0107] |
| fixed 15/0 | recall@5 | rerank20_s14_10k - rerank20_ft | +0.0048 | [-0.0064, +0.0161] |
| fixed 15/0 | recall@1 | rerank20_s14_4k - rerank20_ft | +0.0010 | [-0.0131, +0.0151] |
| fixed 15/0 | recall@5 | rerank20_s14_4k - rerank20_ft | +0.0048 | [-0.0050, +0.0147] |
| fixed 15/0 | recall@1 | rerank20_s14_10k - rerank20_s14_4k | -0.0058 | [-0.0200, +0.0084] |
| fixed 15/0 | recall@5 | rerank20_s14_10k - rerank20_s14_4k | +0.0000 | [-0.0093, +0.0093] |
