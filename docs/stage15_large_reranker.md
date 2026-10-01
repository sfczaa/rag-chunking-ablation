# Stage 15 - a larger cross-encoder under the Stage 8 recipe

Status: pre-registered and run on 2026-09-27. Verdict: LARGE-BETTER (+0.0213
R@1, 95% CI [+0.0009, +0.0417], n = 1032). The design criteria and thresholds
below were fixed before any GPU run for this stage.

Question: trained on the same 2034 Stage 8 groups with the same recipe, does
`BAAI/bge-reranker-large` rank better on the Stage 6 bench than the Stage 8
reranker built on `BAAI/bge-reranker-base`?

## Motivation

At fixed 15/0 on the Stage 6 bench the BGE top-20 pool contains the answer for
0.9641 of questions, and the Stage 8 reranker ranks it first for 0.7345. Stages 11
to 14 changed the objective, the training length and the training-set size of the
base model, and none beat the Stage 8 weights by the 0.02 R@1 threshold (see
[stages10_14_summary.md](stages10_14_summary.md)). This stage changes the model
size and keeps the data and recipe fixed.

## Arms

| Arm | Model | Training |
| --- | --- | --- |
| `rerank20_ft` | `bge-reranker-base` | Stage 8 weights, 2034 groups |
| `rerank20_s15_large` | `bge-reranker-large` at revision `55611d7bca2a7133960a6d3b71e083071bbfc312` | the same 2034 groups and recipe |
| `rerank20_large` | `bge-reranker-large`, same revision | none, reported only |

Also scored for the pipeline check: `bge` (dense order) and `rerank20`
(off-the-shelf base).

The recipe is the Stage 8 one: listwise softmax cross-entropy over one positive and
seven hard negatives, 2 epochs, learning rate 2e-5 with 10% warmup and linear
decay, weight decay 0.01, four groups per optimizer step, fp16, gradient clipping
at 1.0, seed 42, max length 512. To fit a 16 GB T4, each step's four groups run
as two passes of two groups with the losses weighted by group count, and gradient
checkpointing is on. On CPU with dropout off, one step's gradient matched a single
pass within 2e-8; with dropout the masks differ, so the run is not bit-identical
to a single pass.

The Stage 8 weights on Drive are the retrained ones (same function and recipe,
2026-07-25). They score 0.7345 R@1 at fixed 15/0 against the archived 0.7355,
and Stages 12 to 14 compared against the same weights.

## Pre-registered criteria

1. Pipeline validity. At fixed 15/0 the `bge` and `rerank20` rows must match
   `artifacts/results/stage8/final/stage8_ft_eval_results.csv` within 0.005 on
   R@1, R@3 and R@5, with the same chunk count and the same 1032 questions.
   Otherwise the verdict is `INVALID`.
2. Training condition. The training run must finish with at most 10 non-finite
   losses; the trainer stops otherwise, and the verdict is `FAILED-TRAINING` with
   no claim about capacity.
3. The claim. Stage 6 bench, n = 1032, fixed 15/0, R@1, `rerank20_s15_large` minus
   `rerank20_ft`, paired over questions, mean with a 95% interval of mean +/- 1.96
   standard errors:
   - `LARGE-BETTER`: the lower bound is above 0 and the mean is at least 0.02.
   - `LARGE-WORSE`: the upper bound is below 0.
   - `TIE`: anything else, including a significant gain below 0.02.

   The verdict uses unrounded values; tables show four decimals. The 0.02 floor is
   the one used by the Stage 8 gate and Stages 11 to 14.
4. No dev bench and no direction gate. The Stage 6 bench runs once the training
   condition is met.
5. Reported, not claimed: `rerank20_large` minus `rerank20` (capacity without
   fine-tuning), `rerank20_s15_large` minus `rerank20_large` (the gain from
   fine-tuning the large model), R@3 and R@5 for every arm, the paired R@5
   comparisons, peak GPU memory in training, and reranking seconds per question
   (model loading included).
6. One run. No re-seeding and no change to the learning rate, epochs or data to
   move a verdict. A crashed step resumes from its checkpoint or score cache, which
   is the same run. The pass size (`STAGE15_MICRO_GROUPS`) may be lowered from 2 to
   1 before training starts if the GPU runs out of memory; it is a memory setting,
   not part of the recipe.
7. Scope. Training data and bench are both NQ, so the result applies to NQ only. A
   `LARGE-BETTER` verdict does not by itself change the deployed demo.

## Pre-run expectations

The paired standard error of two rerankers of this quality was about 0.007 in
Stages 12 and 14, so the interval half-width should be about 0.015 and a true gain
of 0.02 or more would usually be detected. The recipe was tuned for the base model
and is not re-tuned, so a `TIE` would not exclude a gain under a recipe tuned for
the larger model.

## Storage and checkpoints

A full training state of the large model is about 6.8 GB and the final weights
about 2.2 GB. The trainer saves one resumable checkpoint, at step 600 of 1018, and
the preflight requires 12 GB free under the data root. Each reranker's scores are
saved once computed, so a restarted evaluation skips finished rerankers.

## Outputs

Under `artifacts/results/latest/`:

- `stage15_eval_results.csv` - one row per arm
- `stage15_paired_deltas.csv` - the paired comparisons
- `stage15_check_vs_stage8.csv` - the pipeline check
- `stage15_summary.md` and `stage15_verdict.json`
- `stage15_train_progress.jsonl` and `stage15_scores/` - progress log and score cache

Weights: `artifacts/models/bge_reranker_stage15/large/final/`.

## How to run

1. `notebooks/RAG_chunk_optimize_stage15_check_colab.ipynb`: preflight, a training
   smoke run on eight groups that stops and resumes once, and an evaluation smoke
   run on 40 questions. Nothing is kept.
2. `notebooks/RAG_chunk_optimize_stage15_colab.ipynb`: training
   (`MODE = 'fresh'`, then `'resume'` after any interruption) and the evaluation.

## Results

### Run 1 (2026-09-27): LARGE-BETTER

Run on a Colab GPU from `notebooks/RAG_chunk_optimize_stage15_colab.ipynb` under
one account, fixed 15/0 only.

Training: 1018 steps in 64.3 minutes, final mean epoch loss 0.3147, no non-finite
losses, peak GPU memory 10.45 GiB with two groups per pass. The checkpoint at
step 600 was saved and not needed.

Pipeline validity: PASS. The `bge` and `rerank20` rows reproduce `stage8/final`
exactly, with the same chunk count (19,507) and the same 1032 questions. Pool
recall@20 was 0.9641.

| Arm | R@1 | R@3 | R@5 | s per question, load included |
| --- | --- | --- | --- | --- |
| dense BGE | 0.6279 | 0.8159 | 0.8808 | - |
| off-the-shelf base | 0.6289 | 0.8159 | 0.8760 | 0.632 |
| Stage 8 weights (base) | 0.7345 | 0.8750 | 0.9186 | 0.631 |
| off-the-shelf large | 0.5940 | 0.8052 | 0.8808 | 1.867 |
| `rerank20_s15_large` | 0.7558 | 0.8915 | 0.9293 | 1.867 |

| Paired comparison | mean | 95% CI |
| --- | --- | --- |
| large fine-tuned - Stage 8, R@1 | +0.0213 | [+0.0009, +0.0417] |
| large fine-tuned - Stage 8, R@5 | +0.0107 | [-0.0015, +0.0228] |
| off-the-shelf large - off-the-shelf base, R@1 | -0.0349 | [-0.0631, -0.0067] |
| off-the-shelf large - off-the-shelf base, R@5 | +0.0048 | [-0.0125, +0.0222] |
| large fine-tuned - off-the-shelf large, R@1 | +0.1618 | [+0.1347, +0.1890] |
| large fine-tuned - off-the-shelf large, R@5 | +0.0484 | [+0.0331, +0.0638] |

By criterion 3 the verdict is LARGE-BETTER: the lower bound is above 0 and the mean
is at least 0.02.

Reading. The margin is small on both counts: the mean clears the 0.02 floor by
0.0013 and the lower bound clears 0 by 0.0009. The paired standard error was
0.0104, larger than the 0.007 expected before the run. Of the 1032 questions,
69 are ranked first only by the large model and 47 only by the Stage 8 model. R@5
moves +0.0107 with an interval that includes 0. Without fine-tuning the large
model ranks worse than the base model; its gain comes from fine-tuning on the
same 2034 groups. Reranking takes about three times as long per question.

This is one run with one seed on NQ. Stages 11 to 14 tested other changes against
the same bench and threshold, so a result this close to the floor is best read
as a small capacity gain that a second pre-registered run would need to confirm.

[Stage 16](stage16_holdout_bench.md) rescored both sets of weights on 3063 new
validation questions: +0.0193 R@1, 95% CI [+0.0072, +0.0314], BELOW-FLOOR. The
gain replicates; its size stays at about the threshold.
