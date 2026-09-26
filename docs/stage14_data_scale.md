# Stage 14 - reranker training data scale

Status: pre-registered on 2026-09-17, run on 2026-09-23. The design criteria and
thresholds were fixed before any GPU run for this stage. Explanatory wording
above the Results section was revised after the run; the criteria, thresholds
and verdict rules are unchanged in substance, and the git history keeps every
version.

Question: trained from the base model with the Stage 8 recipe, does a reranker
fine-tuned on about 10,000 groups rank better on the Stage 6 bench than the Stage 8
reranker trained on 2034?

## Motivation

Stages 11 to 13 changed the objective or the amount of training on a fixed set of
about 2000 groups. None of the continued models beat the Stage 8 weights. The
training rankings were already strong: the Stage 8 model ranks the positive first
on about 84% of the Stage 11 groups. Longer training raised training greedy RR to
0.987 to 1.000 while bench R@1 fell below Stage 8, which is consistent with
overfitting. Both objectives favour the positive candidate in this single-positive setting,
but a shared optimal ranking does not imply equal optimisation or generalisation.
This stage tests the effect of adding groups under the existing recipe.

## Arms

Every trained arm uses the training function of `scripts/17_train_reranker.py`
unchanged: base model `BAAI/bge-reranker-base`, listwise softmax cross-entropy
over one positive and seven hard negatives, 2 epochs, learning rate 2e-5 with 10%
warmup and linear decay, weight decay 0.01, four groups per step, fp16, gradient
clipping at 1.0, seed 42, max length 512. Only the training groups differ, and the
sets are nested.

| Arm | Training groups | Source |
| --- | --- | --- |
| `rerank20_ft` | 2034 | Stage 8 groups; the existing Stage 8 weights, same function and recipe |
| `rerank20_s14_4k` | 4011 | Stage 8 groups + Stage 11 groups (1977) |
| `rerank20_s14_10k` | 4011 + about 6000 | the 4k set + three new shards |

The 2k point is not retrained: the Stage 8 weights on Drive were produced by the
same function with the same recipe on 2026-07-25.

## New training data

`scripts/32_build_stage14_data.py` continues the NQ train stream after the Stage 8
and Stage 11 windows (skipping 7466 + 13,103 = 20,569 rows), excludes every Stage 8
train and dev title and every Stage 11 title (4400 in total), and keeps the next
6000 usable documents with the per-row rule of Stage 8 and Stage 11.

The documents are split in stream order into three shards of 2000, and each shard
is mined separately with Stage 8's own `mine_training_groups` (fixed 15/0 chunks,
BGE top-20 pool, one positive and seven hard negatives). Mining per 2000-document
shard keeps the distractor pool the same size as in Stages 8 and 11, so hard
negatives are no harder; mining all 6000 together would change the negatives along
with the amount of data.

Nothing evaluated is trained on: the Stage 8 dev bench titles are excluded from
the new documents, and the Stage 6 bench comes from the NQ validation split.

## Pre-registered criteria

1. Pipeline validity. At fixed 15/0 the `bge` and `rerank20` rows must match
   `artifacts/results/stage8/final/stage8_ft_eval_results.csv` within 0.005 on
   R@1, R@3 and R@5, with the same chunk count and the same 1032 questions.
   Otherwise the verdict is `INVALID`.
2. Size condition. The 10k arm's training set must hold at least 9000 groups.
   Otherwise the verdict is `INCONCLUSIVE-SIZE`: the arm is not the data scale the
   stage is about, and no claim is made.
3. The claim. Stage 6 bench, n = 1032, fixed 15/0, R@1, `rerank20_s14_10k` minus
   `rerank20_ft`, paired over questions, mean with a 95% interval of mean +/- 1.96
   standard errors:
   - `MORE-DATA-BETTER`: the lower bound is above 0 and the mean is at least 0.02.
   - `MORE-DATA-WORSE`: the upper bound is below 0.
   - `TIE`: anything else, including a significant gain below 0.02.

   The verdict uses unrounded values; tables show four decimals. The 0.02 floor is
   the one used by the Stage 8 gate and Stages 11 to 13.
4. No direction gate. The dev bench is scored once for provenance and to catch a
   broken arm; the Stage 6 bench runs whatever it says.
5. Reported, not claimed: `rerank20_s14_4k` minus `rerank20_ft`, `rerank20_s14_10k`
   minus `rerank20_s14_4k`, whether R@1 rises monotonically from 2k to 4k to 10k,
   R@3 and R@5 for every arm, and the paired R@5 comparisons.
6. One run. No re-seeding, no re-mining and no change to the recipe or the shard
   split to move a verdict. A crashed step resumes from its cache or checkpoint,
   which is the same run.
7. Scope. Training data and bench are both NQ, so the result applies to NQ only.

## Pre-run expectations

Stage 8 took the off-the-shelf reranker from R@1 0.6289 to 0.7355 at fixed 15/0
with 2034 groups. Fine-tuning gains usually grow more slowly than the data, so a
fivefold increase could add little. The paired standard error for two rerankers of
this quality was about 0.007 in Stage 12, so a true gain of 0.02 or more would be
detectable. A `TIE` would say that, for this recipe and this bench, data volume in
this range is not the limit either.

## Cost

| Step | Runtime | Estimate |
| --- | --- | --- |
| Stream 6000 documents (Stage 11 streamed 2000 in 8.2 minutes) | CPU | about 25 minutes |
| Mine three shards of about 38,000 chunks each | GPU | 45 to 60 minutes, the least certain figure |
| Train the 4k arm, about 2006 steps (Stage 8: 1018 steps in 17.1 minutes) | GPU | about 34 minutes |
| Train the 10k arm, about 5000 steps | GPU | about 84 minutes |
| Dev bench, 3 rerankers plus the index | GPU | about 15 minutes |
| Stage 6 bench, fixed 15/0, 4 rerankers plus the index | GPU | about 45 minutes |

Each step caches or checkpoints its output: the stream writes its documents when
it finishes, each shard's groups are written when that shard is mined, each
trained arm writes a completion marker, and the evaluation is checkpointed per
config. An interrupted arm is retrained from the start: resuming from its epoch
checkpoint would restart the learning-rate schedule, which is a different recipe.
A lost runtime therefore costs the stream, the shard or the arm in progress.

## Outputs

Written to `artifacts/results/latest/`:

- `stage14_dev_results.csv` and `stage14_paired_deltas_dev.csv` - provenance only
- `stage14_eval_results.csv` - Stage 6 bench, every arm
- `stage14_paired_deltas_final.csv` - the paired comparisons with 95% intervals
- `stage14_check_vs_stage8.csv` - the reproduction check
- `stage14_summary.md` and `stage14_verdict.json`

Training data under `artifacts/data/nq_train_stage14/`, weights under
`models/bge_reranker_stage14/{4k,10k}/`.

## Amendment before any run (2026-09-17)

Nothing had been streamed, mined, trained or evaluated when this was made. The run
may be continued by several Colab accounts in one shared Drive folder, and the plan
above had two gaps for that: an interrupted arm was retrained from the start (up to
about 84 minutes lost for the 10k arm), and nothing stopped two runtimes from
writing the same files. So:

- Training uses a resumable copy of the `scripts/17` loop. Every 500 steps it saves
  the full training state (weights, optimizer, schedule, gradient scaler, random
  states, and the current epoch's data order and position), and a resumed arm
  continues from that step with the same schedule. A local CPU test with a small
  BERT model found the final weights identical, bit for bit, to `scripts/17` with
  no interruption and after interruptions inside an epoch, across an epoch boundary
  and with a checkpoint at every step. An interruption now costs at most 500 steps
  plus the checkpoint being written; each checkpoint writes about 3.3 GB to Drive.
- Every formal step checks that the mounted folder holds the shared-root id in
  `artifacts/RUN_ROOT_ID.json`, probes write, read, replace and delete in the
  directories it writes to, and takes one Stage 14 run lock
  (`artifacts/stage14_run.lock.json`). A lock left by a runtime that died is never
  removed automatically. Training also records the run identity (trainer code hash,
  training-set hashes, recipe) and stops if a later start differs.
- `scripts/35_preflight_stage14.py` runs those checks and reports progress. The
  notebooks are a data notebook (CPU), a check notebook with smoke runs that is safe
  to run in full, and the run notebook with a fresh or resume mode. Training
  progress, including which account saved or resumed each checkpoint, is appended
  to `results/latest/stage14_train_progress.jsonl`.

The question, arms, data, recipe, criteria, thresholds and verdict rules are
unchanged. The sentence in the Cost section saying an interrupted arm is retrained
from the start is superseded.

## Results

### Run 1 (2026-09-23): TIE

Run on Colab from `notebooks/RAG_chunk_optimize_stage14_colab.ipynb` under one
account, fixed 15/0 only.

New data. The stream scanned 36,948 rows of the NQ train split after the skip and
kept 6000 documents with 6684 questions in 22.2 minutes; 6240 rows were dropped by
the title list or the per-row rule. Mining the three shards took 53.4 minutes and
produced 2269, 2086 and 1968 groups, so the 10k arm trained on 4011 + 6323 =
10,334 groups. Train pool recall@20 per shard was 0.955, 0.960 and 0.967, close to
the bench's 0.964, so the new negatives are about as hard as the old ones.

Training. The 4k arm ran 2006 steps in 40.5 minutes to a final mean epoch loss of
0.5049; the 10k arm ran 5168 steps in 100.9 minutes to 0.5180.

Size condition: met, 10,334 groups against the 9000 required.

Pipeline validity: PASS. The `bge` and off-the-shelf `rerank20` rows reproduce
`stage8/final` exactly (every recall delta 0.0000), with the same chunk count
(19,507) and the same 1032 questions. Pool recall@20 was 0.9641.

| Arm | Training groups | R@1 | R@3 | R@5 |
| --- | --- | --- | --- | --- |
| dense BGE | - | 0.6279 | 0.8159 | 0.8808 |
| off-the-shelf rerank20 | - | 0.6289 | 0.8159 | 0.8760 |
| Stage 8 weights | 2034 | 0.7345 | 0.8750 | 0.9186 |
| `rerank20_s14_4k` | 4011 | 0.7355 | 0.8905 | 0.9234 |
| `rerank20_s14_10k` | 10,334 | 0.7297 | 0.8798 | 0.9234 |

| Paired comparison | mean | 95% CI |
| --- | --- | --- |
| 10k - Stage 8, R@1 | -0.0048 | [-0.0204, +0.0107] |
| 10k - Stage 8, R@5 | +0.0048 | [-0.0064, +0.0161] |
| 4k - Stage 8, R@1 | +0.0010 | [-0.0131, +0.0151] |
| 4k - Stage 8, R@5 | +0.0048 | [-0.0050, +0.0147] |
| 10k - 4k, R@1 | -0.0058 | [-0.0200, +0.0084] |
| 10k - 4k, R@5 | +0.0000 | [-0.0093, +0.0093] |

By criterion 3 the verdict is TIE: the primary interval includes 0 and the mean is
below the 0.02 threshold, with the point estimate on the negative side.

R@1 does not rise monotonically with data: 0.7345 at 2034 groups, 0.7355 at 4011,
0.7297 at 10,334. All paired R@1 intervals include zero.

The Stage 8 row scores 0.7345 against its archived 0.7355 at R@1 (-0.0010) and
0.9186 against 0.9215 at R@5 (-0.0029), the same gap Stage 12 measured on the
retrained weights. It is reported and does not affect validity.

Dev bench, provenance only, 400 documents and 406 questions: R@1 was 0.7808 for
Stage 8, 0.7980 for 4k and 0.7956 for 10k, so 4k minus Stage 8 was +0.0172
[-0.0059, +0.0404] and 10k minus Stage 8 was +0.0148 [-0.0098, +0.0394]. Neither
held up on the larger bench, the same shrinkage Stage 12 saw.

Reading. Five times the Stage 8 training data leaves the bench score where it was.
The first 2034 groups are worth +0.1056 R@1 over the off-the-shelf reranker, and
the next 8300, mined by the same rule from the same distribution, are worth
nothing measurable. What movement there is sits below rank 1: both new arms gain
+0.0048 at R@5 and the 4k arm gains +0.0155 at R@3 on the arm table, while the
pre-registered R@5 intervals still include 0.

For this recipe and mining rule, increasing the training set from about 2000 to
10,000 groups did not produce a measurable NQ gain. Stages 11 to 13 changed the
objective and training budget on a fixed set and also did not beat the Stage 8
weights by the 0.02 threshold. These results do not exclude gains from harder
negatives, a larger reranker, a different chunk width or another corpus.

| Step | Runtime | Actual |
| --- | --- | --- |
| Stream 6000 documents | CPU | 22.2 minutes |
| Mine three shards | GPU | 53.4 minutes |
| Train the 4k arm, 2006 steps | GPU | 40.5 minutes |
| Train the 10k arm, 5168 steps | GPU | 100.9 minutes |
| Dev bench, 3 rerankers plus the index | GPU | 18 minutes |
| Stage 6 bench, 4 rerankers plus the index | GPU | 55 minutes |

The resumable loop saved 4 checkpoints for the 4k arm and 10 for the 10k arm, and
the run lock was taken and released by every formal step. Nothing had to be
resumed, and no step ran twice.
