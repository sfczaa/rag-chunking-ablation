# Stage 16 - the Stage 15 comparison on a holdout NQ bench

Verdict: BELOW-FLOOR. large beats Stage 8 by +0.0193 R@1, 95% CI [+0.0072, +0.0314], below the 0.02 floor.

- holdout bench: 2852 docs / 3063 questions from 7830 rows; title overlap with Stage 6 0, question overlap 0
- titles also in the Stage 8 training documents: 188
- weights: match the pre-registered hashes
- primary comparison: fixed 15/0, R@1, rerank20_s15_large - rerank20_ft, paired over 3063 questions; threshold 0.02

| arm | R@1 | R@3 | R@5 | pool@20 | s per question, load included |
| --- | --- | --- | --- | --- | --- |
| bge | 0.6069 | 0.7989 | 0.8528 | 0.9406 | - |
| rerank20_ft | 0.7231 | 0.8652 | 0.8981 | 0.9406 | 0.615 |
| rerank20_s15_large | 0.7424 | 0.8766 | 0.9070 | 0.9406 | 1.880 |

| bench | metric | comparison | mean | 95% CI | n |
| --- | --- | --- | --- | --- | --- |
| holdout | recall@1 | rerank20_s15_large - rerank20_ft | +0.0193 | [+0.0072, +0.0314] | 3063 |
| holdout | recall@5 | rerank20_s15_large - rerank20_ft | +0.0088 | [+0.0025, +0.0151] | 3063 |
| stage6 + holdout | recall@1 | rerank20_s15_large - rerank20_ft | +0.0198 | [+0.0094, +0.0302] | 4095 |
| stage6 + holdout | recall@5 | rerank20_s15_large - rerank20_ft | +0.0093 | [+0.0037, +0.0149] | 4095 |
