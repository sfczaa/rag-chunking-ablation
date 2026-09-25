# Stage 13 - was the RL reranker simply undertrained?

Status: pre-registered on 2026-09-16, run on 2026-09-17. Verdict:
INCONCLUSIVE-BUDGET (the RL arm reached 1769 of the 1977 live groups the budget
condition asks for). Everything above the Results section was written before
any GPU run for this stage.

Question: Stage 11 trained the policy-gradient arm and its cross-entropy control
for the same number of steps, and RL finished behind (dev R@1 RL - CE = -0.0246,
95% CI [-0.0438, -0.0054], n = 406). The RL estimator produced a gradient on only
31.9% of groups, so at equal steps it learned from roughly a third of the data.
Does RL catch up when it is given enough training to see as much usable signal as
CE, and more?

## Why equal steps was not a fair budget

The RL loss is silent on a group whose sampled rankings agree on the reward: the
advantage is zero and no gradient flows. Stage 11 measured that on 68.1% of
groups. Cross-entropy takes a gradient from every group. So Stage 11's
step-matched design gave the two objectives different amounts of usable signal,
and its result cannot separate "the objective is worse" from "the objective saw
less data".

Stage 13 fixes the budget rather than the step count, and adds a second, larger
budget so the answer does not rest on one point.

## Arms

All arms continue from the same Stage 8 weights (`models/bge_reranker_ft/final`)
over the same 1977 Stage 11 groups (file SHA-1
`4f67bb1fd344b30915f30403ddeea1a064b064b6`), with learning rate 2e-5, 10% warmup
and linear decay, weight decay 0.01, four groups per step, fp16, gradient clipping
at 1.0, seed 11. RL samples 8 rankings per group at temperature 1. Only the loss
and the number of epochs differ.

| Arm | Training | Role |
| --- | --- | --- |
| `rerank20_ft` | none | Stage 8 weights, the common starting point |
| `rerank20_s11_ce` | 1 CE epoch | the Stage 11 arm that Stage 12 scored |
| `ce4` | 4 CE epochs | budget-matched control |
| `rl4` | 4 RL epochs | the primary arm |
| `ce8` | `ce4` continued for 4 more epochs | larger budget control |
| `rl8` | `rl4` continued for 4 more epochs | larger budget |

The 8-epoch arms are the 4-epoch checkpoints continued, so their schedule restarts
at the halfway point. Both objectives get identical treatment, so the comparison
stays matched; this is also how Stage 11 itself continued from Stage 8.

## The budget condition

CE sees 1977 usable groups in one epoch. The RL arm's usable groups are the ones
with a non-zero advantage, which the trainer already counts as the live fraction.
`rl4` satisfies the budget condition when its cumulative live groups reach at least
1977. At Stage 11's measured live fraction of 0.319 the expected count after 4
epochs is about 2523, so the condition should hold with margin, but it is checked
against the logged count rather than assumed.

If `rl4` ends below 1977 cumulative live groups, the verdict is
`INCONCLUSIVE-BUDGET`: the arm did not reach the budget the stage is about, and no
claim is made about the objective.

## Pre-registered criteria

1. Pipeline validity. At fixed 15/0 the `bge` and `rerank20` rows must match
   `artifacts/results/stage8/final/stage8_ft_eval_results.csv` within 0.005 on
   R@1, R@3 and R@5, with the same chunk count and the same 1032 questions.
   Otherwise the verdict is `INVALID`.
2. Optimiser guard. If fewer than 5% of RL groups produced a non-zero advantage
   across training, the verdict is `FAILED-OPTIMISATION`, never a tie.
3. Budget condition, as defined above, or `INCONCLUSIVE-BUDGET`.
4. The claim. Stage 6 bench, n = 1032, fixed 15/0, R@1, `rl4` minus `ce4`, paired
   over questions, mean with a 95% interval of mean +/- 1.96 standard errors:
   - `RL-BETTER`: the lower bound is above 0 and the mean is at least 0.02.
   - `CE-BETTER`: the upper bound is below 0.
   - `TIE`: anything else, including a significant gain below 0.02.

   The verdict uses unrounded values; tables show four decimals. The 0.02 floor is
   the one used by the Stage 8 gate, Stage 11 and Stage 12, set before any of this
   data existed.
5. No direction gate. The Stage 6 bench runs whatever the dev numbers say. Stage 11
   stopped at a dev gate and the dev signal then failed to replicate in Stage 12,
   so this stage answers its question on the claim bench in both directions. The
   dev bench is scored once for provenance and to catch a broken arm, and it
   decides nothing.
6. Reported, not claimed: the larger budget pair `rl8` minus `ce8`; whether more
   CE training helps at all (`ce4` and `ce8` against `rerank20_s11_ce` and against
   the Stage 8 weights); R@5 for every comparison; per-arm R@1, R@3, R@5.
7. One run. No re-seeding and no re-tuning of the temperature, learning rate,
   sample count or epoch counts to move a verdict. A crashed run resumes from its
   checkpoint, which is the same run.
8. Scope. Both objectives train on NQ and the bench is NQ, so the result applies
   to NQ only.

## What to expect before running

From Stage 11's dev numbers, RL trails CE by 0.0246 R@1 at one epoch. Stage 12
showed the CE side gains little from its first epoch on the claim bench
(+0.0019 against Stage 8). So there are three plausible outcomes, and the design
distinguishes them: RL catches up once the budget matches, which would make
Stage 11's result a budget artifact; RL stays behind at both budgets, which makes
the objective itself the explanation; or both objectives move together, which
would say the extra training, not the objective, is what matters.

## Cost

On a T4, from the Stage 11 and Stage 12 measurements (CE 7.9 min per epoch, RL 7.5
min per epoch, reranking 35 pairs per second at fixed 15/0).

| Step | Estimate |
| --- | --- |
| Train `ce4` and `rl4` | about 65 minutes |
| Continue to `ce8` and `rl8` | about 65 minutes |
| Dev bench, 4 new arms plus the index | about 25 minutes |
| Stage 6 bench, fixed 15/0, 6 rerankers plus the index | about 70 minutes |

Two Colab sessions fit this: training plus dev in the first, the claim bench in the
second. Each arm writes a completion marker, and the evaluation is checkpointed per
config, so a lost runtime costs the arm or config in progress.

## Outputs

Written to `artifacts/results/latest/`:

- `stage13_train_log_{ce4,rl4,ce8,rl8}.csv` - per-step training logs
- `stage13_dev_results.csv` - dev bench rows, provenance only
- `stage13_eval_results.csv` - Stage 6 bench, every arm
- `stage13_paired_deltas.csv` - the paired comparisons with 95% intervals
- `stage13_check_vs_stage8.csv` - the reproduction check
- `stage13_summary.md` and `stage13_verdict.json`

## Results

### Run 1 (2026-09-17): INCONCLUSIVE-BUDGET

Run on a Colab T4 from `notebooks/RAG_chunk_optimize_stage13_colab.ipynb`: all four
arms trained, then the dev bench and the Stage 6 bench at fixed 15/0.

Pipeline validity: PASS. The `bge` and `rerank20` rows reproduce `stage8/final`
exactly, with the same chunk count (19,507) and the same 1032 questions.

Budget condition: not met. `rl4` accumulated 1769 live groups against the 1977 the
condition asks for. Its live fraction over four epochs was 0.224, below the 0.319
Stage 11 measured over one epoch, so the pre-registered estimate of about 2523 was
too high: the fraction falls as training makes the sampled rankings agree more
often. By criterion 3 the verdict is INCONCLUSIVE-BUDGET, and no claim is made
about the objective.

Training, 1980 steps per arm on a T4:

| Arm | Starts from | Minutes | Live fraction | Live groups, cumulative | Greedy RR, first to last tenth |
| --- | --- | --- | --- | --- | --- |
| `ce4` | Stage 8 | 29.2 | - | - | 0.8611 to 0.9871 |
| `rl4` | Stage 8 | 28.6 | 0.224 | 1769 | 0.8611 to 0.9387 |
| `ce8` | `ce4` | 28.7 | - | - | 0.9963 to 0.9981 |
| `rl8` | `rl4` | 28.4 | 0.099 | 2552 | 0.9462 to 0.9709 |

Stage 6 bench, fixed 15/0 (pool recall@20 0.9641):

| Arm | R@1 | R@3 | R@5 |
| --- | --- | --- | --- |
| dense BGE | 0.6279 | 0.8159 | 0.8808 |
| off-the-shelf rerank20 | 0.6289 | 0.8159 | 0.8760 |
| Stage 8 weights | 0.7345 | 0.8750 | 0.9186 |
| `ce4` | 0.7171 | 0.8760 | 0.9264 |
| `rl4` | 0.7219 | 0.8886 | 0.9302 |
| `ce8` | 0.7161 | 0.8576 | 0.9050 |
| `rl8` | 0.7267 | 0.8837 | 0.9273 |

Paired comparisons over the 1032 questions. The first row is the pre-registered
comparison, not claimed because the budget condition failed; the rest are reported
only.

| Comparison | R@1, 95% CI | R@5, 95% CI |
| --- | --- | --- |
| `rl4` - `ce4` | +0.0048 [-0.0109, +0.0206] | +0.0039 [-0.0062, +0.0139] |
| `rl8` - `ce8` | +0.0107 [-0.0060, +0.0273] | +0.0223 [+0.0096, +0.0350] |
| `ce4` - Stage 8 | -0.0174 [-0.0333, -0.0016] | +0.0078 [-0.0033, +0.0188] |
| `rl4` - Stage 8 | -0.0126 [-0.0284, +0.0032] | +0.0116 [+0.0020, +0.0213] |
| `ce8` - Stage 8 | -0.0184 [-0.0353, -0.0016] | -0.0136 [-0.0270, -0.0002] |
| `rl8` - Stage 8 | -0.0078 [-0.0245, +0.0090] | +0.0087 [-0.0015, +0.0189] |

The dev bench (406 questions, provenance only) gave `rl4` - `ce4` = 0.0000 R@1,
95% CI [-0.0216, +0.0216].

Reading. None of this is the pre-registered claim.

- The budget condition failed because the live fraction decays with training. The
  larger-budget RL arm did pass the 1977 mark (2552 cumulative), and against its
  epoch-matched control it is not behind: `rl8` - `ce8` is +0.0107 R@1 with an
  interval that includes 0, and +0.0223 R@5 with an interval above 0.
- Longer training on these 1977 groups does not beat the Stage 8 weights. Both CE
  budgets finish below Stage 8 on R@1 with intervals entirely below 0, while their
  greedy RR on the training batches reaches 0.987 to 0.998, which fits the model
  memorising the groups. The RL arms fall less far, and their R@1 intervals include
  0.
- With Stages 11 and 12: the dev deficit RL showed in Stage 11 at one epoch is not
  present at four or eight epochs, but neither objective improves on Stage 8 with
  this data. The limit appears to be the 1977 groups rather than the objective.
  That last point is an interpretation and was not tested here.

The `lr_scheduler.step()` warning on the first step of each arm is the fp16
gradient scaler skipping an overflowed step, as in Stage 11.
