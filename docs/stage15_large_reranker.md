# Stage 15 - a larger cross-encoder under the Stage 8 recipe

Status: pre-registered on 2026-09-27, not yet run. The design criteria and
thresholds below were fixed before any GPU run for this stage.

Question: trained on the same 2034 Stage 8 groups with the same recipe, does
`BAAI/bge-reranker-large` rank better on the Stage 6 bench than the Stage 8
reranker built on `BAAI/bge-reranker-base`?

## Motivation

At fixed 15/0 on the Stage 6 bench the BGE top-20 pool contains the answer for
0.9641 of questions, and the Stage 8 reranker ranks it first for 0.7345. Stages 11
to 14 changed the objective, the training length and the training-set size of the
base model, and none beat the Stage 8 weights by the 0.02 R@1 threshold (see
[stages10_14_summary.md](stages10_14_summary.md)). Model capacity is the remaining
variable of the recipe that has not been changed. This stage changes it and keeps
the data and the recipe fixed.

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
at 1.0, seed 42, max length 512. To keep the large model within a 16 GB T4, two
implementation settings differ, neither of which changes the recipe:

- Each optimizer step of four groups runs as forward/backward passes of two
  groups, and each pass's loss is weighted by its share of the step. The step's
  gradient is the same mean over four groups; dropout masks and float order differ
  from a single pass, so the run is not bit-identical to one. On CPU with a small
  model and dropout off, one step's gradient matched a single pass within 2e-8.
- Gradient checkpointing recomputes activations in the backward pass.

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
and is not re-tuned here. A `TIE` would therefore say that capacity, under this
recipe and these 2034 groups, does not move R@1 by the threshold; it would not
exclude a gain under a recipe tuned for the larger model.

## Storage and checkpoints

One full training state of the large model (weights plus optimizer) is about
6.8 GB. The trainer saves one resumable checkpoint, at step 600 of 1018, so the
shared folder never holds two states at once; the final weights are about 2.2 GB.
The preflight requires 12 GB free under the data root. Each evaluated reranker's
scores are saved as soon as they are computed, so a restarted evaluation reuses
finished rerankers.

## Outputs

Under `artifacts/results/latest/`:

- `stage15_eval_results.csv` - one row per arm
- `stage15_paired_deltas.csv` - the paired comparisons
- `stage15_check_vs_stage8.csv` - the pipeline check
- `stage15_summary.md` and `stage15_verdict.json`
- `stage15_train_progress.jsonl` and `stage15_scores/` - progress log and score cache

Weights: `artifacts/models/bge_reranker_stage15/large/final/`.

## How to run

Code is checked out from the public repository inside the Colab runtime; the
shared Drive folder holds only data, weights and results.

1. `notebooks/RAG_chunk_optimize_stage15_check_colab.ipynb`: preflight, a training
   smoke run on eight groups that stops and resumes once, and an evaluation smoke
   run on 40 questions. Nothing is kept.
2. `notebooks/RAG_chunk_optimize_stage15_colab.ipynb`: training
   (`MODE = 'fresh'`, then `'resume'` after any interruption) and the evaluation.

## Results

Not yet run.
