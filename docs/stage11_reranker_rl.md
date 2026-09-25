# Stage 11: the reranker under an RL objective

Status: run 1 executed on 2026-09-15. NO-GO at the dev gate: the RL arm finished
behind its cross-entropy control, so the Stage 6 bench was not run. The design,
thresholds and verdict rules below were fixed before any training or evaluation.
The results are in "Results (executed)" at the end.

Question: when both start from the same weights and train on the same data for
the same compute, does a policy-gradient objective on the ranking reward rank
better than the listwise cross-entropy used in Stage 8?

## Why the reranker

At fixed 15/0 on the Stage 6 bench, the BGE top-20 pool contains the answer for
96.4% of questions, but the Stage 8 fine-tuned reranker ranks it first for only
73.6%. This is the largest measured gap left in the project. Stage 10 found little
headroom in chunk placement chosen without the question. Its oracle found headroom
only when it could see the question, and a reranker sees the question.

## Design

Both arms continue from the Stage 8 fine-tuned weights for one epoch, with the
same groups in the same order, the same dropout stream, learning rate 2e-5 with
10% warmup and linear decay, weight decay 0.01, four groups per step, fp16,
gradient clipping at 1.0 and seed 11. Only the loss differs.

- CE (control): Stage 8's listwise softmax cross-entropy over each group of one
  answer-bearing chunk and seven hard negatives.
- RL: the scores over a group define a Plackett-Luce policy over rankings at
  temperature 1. Eight rankings are sampled per group, and each gets the
  reciprocal rank of the positive as its reward. A sample's advantage is its
  reward minus the mean reward of the group's other samples. This is the
  group-relative estimator known as GRPO, without per-group standard-deviation
  rescaling. The gradient goes through the log-probability of each ranking's
  prefix up to the positive, because the reward depends only on that prefix.

Local checks before any GPU time:

| Property | Result |
| --- | --- |
| Sampler vs exact Plackett-Luce, 24 permutations | max frequency error 0.0006 |
| Full-ranking and prefix log-probabilities vs enumeration | error below 1e-15 |
| Policy-gradient estimate vs exact gradient of expected reciprocal rank | cosine 1.0000, relative error 0.007 |
| Group whose samples all agree | gradient exactly zero |

Starting weights. The Stage 8 weights on Drive were retrained on 2026-07-25 with
the Stage 8 recipe after the originals were lost. Both arms start from these
weights, so the RL vs CE comparison is unaffected. The gap between these weights
and the archived Stage 8 row is reported but not used as a gate.

## Training data

A probe of the Stage 8 model on 64 of its own training groups found the positive
ranked first in 59 (92%), with a cross-entropy of 0.26. Another epoch on those
groups would give either objective very little to learn, and a tie would say
nothing about the objectives.

Stage 11 therefore trains on the next 2000 NQ-train documents after Stage 8's
window. `scripts/26_build_stage11_data.py` skips the 7466 rows Stage 8's stream
consumed, excludes all 2400 Stage 8 train and dev titles, and mines groups with
Stage 8's rule: fixed 15/0 chunks, the BGE top-20 pool, one positive and seven
hard negatives. No evaluation data is used for training:

| Role | Source |
| --- | --- |
| Training | fresh NQ-train documents, disjoint from every Stage 8 title |
| Gate | Stage 8 dev bench, NQ train split, titles excluded from the fresh window |
| Final comparison | Stage 6 bench, NQ validation split |

## Temperature

Fixed at 1, the model's own score scale, without tuning. The risk was Stage 10's
failure mode, where scores are so sharp that sampled rankings never differ and no
gradient flows. A probe on training groups only:

| Temperature | Groups with non-zero advantage | Among groups the model ranks wrong |
| --- | --- | --- |
| 0.5 | 14.8% | 86.8% |
| 1 | 26.4% | 99.7% |
| 2 | 51.8% | 100% |
| 4 | 91.3% | 100% |

At temperature 1 the signal reaches almost every group the model ranks wrong. A
higher temperature adds signal mostly on groups that are already right, and
flattens the policy away from the deterministic ranking the metric scores. The
probe had only five wrong groups and the new data will contain more, so the share
of groups with a non-zero advantage (the live fraction) is logged during training
and checked below.

## Pre-registered criteria

1. Pipeline validity. The dense and off-the-shelf rows at fixed 15/0, and at fixed
   6/0 when it runs, are deterministic reruns of archived rows. They must match
   `stage8/final` within 0.005 with identical chunk counts, or the run is
   `INVALID`.
2. Optimiser check. If fewer than 5% of RL groups had a non-zero advantage across
   training, the verdict is `FAILED-OPTIMISATION`, and never `TIE`.
3. Dev gate. On the Stage 8 dev bench at fixed 15/0, the final run happens only if
   the optimiser check passes and RL minus CE on R@1 is at least 0. The gate only
   decides whether to spend GPU time on the final run. A `NO-GO` is recorded as a
   `NO-GO`.
4. Primary comparison. Stage 6 bench, n = 1032, fixed 15/0, R@1, RL minus CE,
   paired over questions with a 95% interval:
   - `RL-BETTER`: the interval excludes 0 and the mean is at least 0.02.
   - `CE-BETTER`: the interval is entirely below 0.
   - `TIE`: anything else.
   The 0.02 threshold matches Stage 8's gate. A significant gain below it is
   reported with its numbers and does not count as `RL-BETTER`.
5. Reported only: R@5, and each arm against the Stage 8 weights, which shows the
   effect of one epoch on new data. Fixed 6/0 is optional (see the amendment
   below) and is reported the same way when it runs.
6. One run. No re-seeding, and no re-tuning of the temperature, learning rate or
   sample count to change a verdict.
7. Scope. A positive NQ result says nothing about other corpora.

## How to run

```bash
# 0. fresh data: streaming runs fine on a CPU runtime; mining wants the GPU
python scripts/26_build_stage11_data.py --stream-only
python scripts/26_build_stage11_data.py

# 1. both arms (each saves a completion marker; a rerun skips finished arms)
python scripts/27_train_reranker_rl.py --smoke
python scripts/27_train_reranker_rl.py

# 2. dev gate, then the claim only on GO
python scripts/28_eval_reranker_rl.py --dev
python scripts/28_eval_reranker_rl.py                       # the claim, fixed 15/0
python scripts/28_eval_reranker_rl.py --configs 15:0,6:0    # optional secondary
```

Costs on a T4. Reranking times come from the Stage 8 archive (35 pairs per second
at fixed 15/0, 60 at fixed 6/0). BGE encoding rates are estimated from the Stage
10 runs and are less certain.

| Step | Estimate |
| --- | --- |
| Mining: BGE over about 40,800 chunks | 15 to 20 minutes |
| Both arms: Stage 8 ran 1018 steps in 17.1 minutes | about 20 minutes |
| Dev gate: 3 rerankers at 3.8 minutes each, plus the index | about 20 minutes |
| Final comparison, fixed 15/0: 4 rerankers at 9.7 minutes each, plus the index | about 50 minutes |
| Optional fixed 6/0: 4 rerankers at 5.7 minutes each, plus the index | about 30 minutes |

The streaming time was not known in advance and is printed every 500 rows.

If the runtime is lost, the stream, arm or evaluation config in progress is lost.
Finished arms and finished configs are kept.

## Amendment before any training or evaluation (2026-09-15)

When this amendment was made, no arm had been trained and nothing had been
evaluated; the data stream had finished and mining was running. Costing the plan
with the archived T4 timings put the full GO path near three hours of GPU on a
free Colab quota, almost an hour of it on steps that do not affect the verdict.
So:

- the final run defaults to the primary config, fixed 15/0, and fixed 6/0 became
  an optional second run that resumes from the same checkpoint;
- the 40-question smoke run before the final evaluation was dropped. It rebuilt
  the dense indexes only to repeat a scoring path the dev gate had already run.

The primary comparison, its thresholds, the gate and every validity check are
unchanged. The off-the-shelf reproduction check covers fixed 15/0, and fixed 6/0
when it runs.

## Results (executed)

### Run 1 (2026-09-15): NO-GO at the dev gate

Data. The stream ran on a CPU runtime in 8.2 minutes: 13,103 rows scanned, 2,502
skipped as Stage 8 titles, 2000 documents and 2091 questions kept, disjoint from
all 2400 Stage 8 train and dev titles. Mining kept 1977 groups (85 questions had
no positive in the pool, 29 had too few negatives); pool recall@20 was 0.959 over
38,453 chunks.

Training. 495 steps per arm, 7.9 minutes for CE and 7.5 for RL on a T4. The RL arm
had a non-zero advantage on 31.9% of groups (mean absolute advantage 0.047), so the
optimiser check passed.

Dev gate. Stage 8 dev bench, 406 questions, fixed 15/0, all arms reranking one
shared BGE top-20 pool (pool recall 0.973):

| Arm | R@1 | R@3 | R@5 |
| --- | --- | --- | --- |
| dense BGE | 0.6527 | 0.8571 | 0.8966 |
| Stage 8 weights | 0.7808 | 0.9212 | 0.9483 |
| + one CE epoch | 0.8030 | 0.9236 | 0.9433 |
| + one RL epoch | 0.7783 | 0.9261 | 0.9384 |

| Paired comparison | R@1, 95% CI | R@5, 95% CI |
| --- | --- | --- |
| RL - CE | -0.0246 [-0.0438, -0.0054] | -0.0049 [-0.0117, +0.0019] |
| RL - Stage 8 | -0.0025 [-0.0246, +0.0197] | -0.0099 [-0.0195, -0.0002] |
| CE - Stage 8 | +0.0222 [+0.0001, +0.0442] | -0.0049 [-0.0117, +0.0019] |

RL minus CE on R@1 is below 0, so by criterion 3 the verdict is NO-GO. The Stage 6
bench was not run and nothing was archived.

Interpretation. The dev gate only decides cost, so these numbers are not the Stage
11 result. The direction is still clear: the RL objective produced a gradient on a
third of the groups, finished behind its control with a dev interval that excludes
zero, and left the Stage 8 model's R@1 unchanged. The CE epoch on the same groups
raised R@1, with an interval that only just excludes zero.

Two possible reasons RL is behind, neither tested here. Cross-entropy gets a
gradient from every group and keeps increasing the margin even when the positive
is already first, while the RL estimator gets nothing from a group whose sampled
rankings agree, which was two groups in three. At fixed compute it learns from
less. Also, the reward is computed over eight candidates while the metric ranks
twenty, and a wider margin helps more with more candidates.

Two notes on the logs. Greedy reciprocal rank on the training batches fell from
about 0.88 to 0.82 in both arms. It is measured on each step's own batch with
dropout on and the same data order in both arms, so it follows batch difficulty
and does not measure learning. The `lr_scheduler.step()` warning on the first step
of each arm comes from the fp16 gradient scaler skipping an overflowed step, and
is the same in both arms.

Open question. Whether one more CE epoch on new data improves on Stage 8 is a
separate question from the one pre-registered here. The dev result suggests it,
and answering it needs a separate pre-registered run on the Stage 6 bench.
