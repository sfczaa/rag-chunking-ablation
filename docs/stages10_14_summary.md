# Stages 10 to 14: follow-ups to the fine-tuned reranker

Stage 8 fine-tuned `BAAI/bge-reranker-base` on 2034 NQ-train groups. On the
Stage 6 bench (NQ validation, 1032 questions, fixed 15/0) it raised R@1 from
0.6289 (off-the-shelf) to 0.7355, while the BGE top-20 pool held the answer for
0.9641 of questions. Stages 10 to 14 tried to improve on that result. This page
collects their verdicts; each stage document has the full design and numbers.

Every stage wrote its primary comparison, threshold and verdict rules before its
GPU run, and ran once. The reranker stages (11 to 14) all use the 0.02 R@1
threshold of the Stage 8 go/no-go gate, which was set before any of their data
existed. In those stages a gain counts only if its 95% interval excludes 0 and
its mean reaches 0.02.

## Verdicts

| Stage | What changed | Primary comparison | Result | Verdict |
| --- | --- | --- | --- | --- |
| [10](stage10_rl_chunking.md) | Chunker trained on a retrieval reward | Dev R@5, RL policy minus supervised chunker, gate 0.01 | Policy did not move from the warm start | NO-GO (failed optimisation) |
| [11](stage11_reranker_rl.md) | Reranker objective: policy gradient against listwise CE, 1 epoch | Dev gate: R@1, RL minus CE, must be at least 0 | -0.0246 [-0.0438, -0.0054], n = 406 | NO-GO |
| [12](stage12_ce_epoch.md) | One more CE epoch on 1977 new groups | R@1, CE minus Stage 8 | +0.0019 [-0.0120, +0.0159] | TIE |
| [13](stage13_rl_budget.md) | RL given a matched training budget, 4 and 8 epochs | R@1, `rl4` minus `ce4` | +0.0048 [-0.0109, +0.0206], not claimed | INCONCLUSIVE-BUDGET |
| [14](stage14_data_scale.md) | Stage 8 recipe on 4011 and 10,334 groups | R@1, 10k minus Stage 8 | -0.0048 [-0.0204, +0.0107] | TIE |

Stages 12 to 14 are scored on the Stage 6 bench with n = 1032. In each of them
the dense and off-the-shelf rows reproduced `stage8/final` exactly, so the
pipeline checks passed.

## Three directions around the Stage 8 recipe

Objective (Stages 11 and 13). A Plackett-Luce policy gradient rewarded by the
positive's reciprocal rank finished behind its cross-entropy control after one
epoch. Stage 13 asked whether that was a budget effect. The RL arm reached 1769
of the 1977 live groups the budget condition required, so no claim is made; at
eight epochs `rl8` minus `ce8` was +0.0107 R@1 with an interval that includes 0.

Training length (Stages 12 and 13). One more CE epoch did not beat Stage 8 on
the claim bench, although the Stage 11 dev run had suggested +0.0222. Four and
eight CE epochs finished below Stage 8 on R@1 while training-batch reciprocal
rank reached 0.987 to 0.998.

Data (Stage 14). Doubling and then quintupling the training groups, mined with
the same rule, left R@1 where it was: 0.7345, 0.7355 and 0.7297 at 2034, 4011 and
10,334 groups.

Every continued or retrained reranker against the Stage 8 weights, Stage 6
bench, fixed 15/0 (reported comparisons, outside the primary verdicts):

| Reranker | Training groups | R@1 | R@1 minus Stage 8, 95% CI |
| --- | --- | --- | --- |
| Stage 8 weights | 2034 | 0.7345 | - |
| + one CE epoch (Stage 12) | 2034 + 1977 | 0.7364 | +0.0019 [-0.0120, +0.0159] |
| `ce4` (Stage 13) | 2034 + 1977 | 0.7171 | -0.0174 [-0.0333, -0.0016] |
| `rl4` (Stage 13) | 2034 + 1977 | 0.7219 | -0.0126 [-0.0284, +0.0032] |
| `ce8` (Stage 13) | 2034 + 1977 | 0.7161 | -0.0184 [-0.0353, -0.0016] |
| `rl8` (Stage 13) | 2034 + 1977 | 0.7267 | -0.0078 [-0.0245, +0.0090] |
| `4k` (Stage 14) | 4011 | 0.7355 | +0.0010 [-0.0131, +0.0151] |
| `10k` (Stage 14) | 10,334 | 0.7297 | -0.0048 [-0.0204, +0.0107] |

The largest gain is +0.0019, a tenth of the threshold. The Stage 8 weights were
retrained with the same function and seed after the originals were lost, and
score 0.7345 against the archived 0.7355; all comparisons above use the retrained
weights.

Twice a dev-bench gain of about +0.015 to +0.022 R@1 (Stage 11's CE arm, Stage
14's 4k and 10k arms) shrank to within noise on the 1032-question bench.

## The chunker (Stage 10)

Stage 10 trained the Transformer boundary model on an MRR@10 retrieval reward
instead of section labels. The policy stayed near uniform over its window, so the
run says nothing about the objective. A pre-registered placement oracle found
+0.0605 R@5 of headroom over fixed 15/0, but only when the cut is chosen with the
question in view; chosen from a document's other questions (82 held-out questions), the gain was
+0.0090 with an interval that includes 0. This agrees with Stages 1 to 7: at a
matched size, for a chunker that cuts before it sees the question, where the cut
falls did not change recall by a detectable amount.

## Scope and what was not tested

All results are on NQ, and no transfer to another corpus is claimed. The gap between pool recall (0.9641) and R@1 (about 0.73) is still a
ranking gap. Harder negatives, a larger reranker, a different candidate pool
depth and a different chunk width were not tried here. Stage 9 (rerank depth with
the fine-tuned reranker) has a script and a notebook but was never run, so it has
no result.
