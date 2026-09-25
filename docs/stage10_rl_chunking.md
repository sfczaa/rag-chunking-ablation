# Stage 10: training the chunker on retrieval recall

Status: RL run 1 (2026-09-14) failed to optimise. The pre-registered placement
oracle (2026-09-15) returned HEADROOM, and post-hoc checks suggest that the
headroom depends on seeing the question. Stage 10 is open pending one decision.
The thresholds and verdict rules below were fixed before each run and have not
changed. The results are in "Results (executed)" at the end.

Question: so far, chunking method has not mattered at a matched chunk size. Does
that still hold when the chunker is trained on the retrieval metric itself?

## Why this stage exists

Stages 1 and 2 trained boundary models on Wikipedia section pseudo-labels with a
weighted BCE loss, then scored them with doc-constrained Recall@k. Stages 3-7
found the same result each time: at a matched chunk size, fixed-size, BiLSTM and
Transformer chunking perform the same.

One objection to that conclusion is that the learned models were never optimised
for retrieval. Section breaks are a proxy, and `scripts/20_effect_size.py`
measured how weak the method signal is (largest method coefficient 0.0036, not
significant; size range effect +0.064). Stage 10 addresses the objection with the
same architecture, the same warm start from the Stage 2 weights, the same
evaluation path, and a different objective.

If a chunker trained directly on recall still ties with fixed-size chunking, the
tie comes from the task and not from the training signal, which makes the
project's main conclusion stronger.

## Formulation

The environment is the sweep's own target-size decode
(`chunking.semantic_target_chunks`): walking a document, each chunk must end
inside a `[target-4, target+4]` window, and the model chooses where. Each decision
is a choice among about nine candidate boundaries rather than `n-1` independent
binary decisions, for two reasons:

- Variance. With roughly one question per document, the reward is a single
  near-binary event per document. Nine actions per chunk is a credit assignment
  problem this corpus size can support; `n-1` binary decisions is not.
- Size control. The window keeps average chunk size in the baseline's band. With a
  free action space, a recall reward would learn to grow chunks and reproduce the
  Stage 1 size effect, which is already known.

Reward. For each question, `1/rank` of the first retrieved chunk that contains the
answer string and comes from the gold document, within the top 10 of the whole
corpus, and 0 otherwise. It is a dense stand-in for doc-constrained Recall@k, the
metric the stage is finally judged on.

Baseline. Self-critical: the advantage is `reward(sampled chunking) -
reward(greedy chunking)` for the same documents at the same step. The greedy
rollout is exactly the deterministic chunker every earlier stage evaluated, so the
comparison is consistent and no learned critic is needed.

Two properties the gradient depends on are checked in code, because both failed
without any error during development: the greedy decode must reproduce
`chunking.semantic_target_chunks` exactly, and the sampler must draw from the
same distribution the loss differentiates. `scripts/23_train_rl_chunker.py
--smoke` checks both before training.

Training-time approximation. Only the sampled documents are re-chunked and
re-encoded at each step; the rest of the corpus stays at a frozen fixed 15/0
chunking and is encoded once. This keeps a step at a couple of seconds instead of
the ~3 minutes a full re-encode of the corpus takes. As a result, the distractor
pool uses the reference chunking rather than the current policy. The Stage 10
evaluation re-chunks every document with the policy and does not use this
shortcut.

## Data

Training and dev reuse the Stage 8 cache built from the NQ train split
(`rerank_finetune.prepare_train_and_dev`), so they are disjoint from every
evaluation bench in the project. The Stage 6 bench is used once, at the end. Run
`scripts/16_build_rerank_train_data.py` first if that cache does not exist.

## Pre-registered criteria

1. Matched size. Every comparison is against the archived Stage 6 row at the same
   target size and overlap. If the average chunk size drifts more than
   `STAGE10_SIZE_TOLERANCE` = 0.3 sentences from that row, the config is reported
   as INVALID. A gain from larger chunks is the Stage 1 result again.
2. The threshold is the MDE. `scripts/20_effect_size.py` put the 80%-power minimum
   detectable effect at 0.046 R@5 for n = 1032, while the largest method spread
   observed so far is 0.023. Verdicts:
   - `OVERTURNS`: R@5 beats the best matched baseline by >= 0.046.
   - `DIRECTIONAL`: positive but below 0.046. Reported as directional but
     undetectable, and not claimed.
   - `NULL`: no better than the matched baseline. The main conclusion stands, and
     the objection about the training objective no longer applies.
   - `INVALID`: the size tolerance was exceeded.
3. Baselines come from the archive. Two archived configs (`fixed 15/0`,
   `transformer 15/1`) are recomputed in the same session. If they do not match
   `stage6/final` within 0.005, the run is void.
4. One reported run. The dev gate may be used to stop a run early. It may not be
   used to choose the best of several seeds and report that as the result. A NO-GO
   is recorded as a NO-GO.
5. Both metrics. The reward is built from R@5, so R@1 and the unconstrained
   variants are reported alongside it. A gain that appears only in the optimised
   metric is reported as such.
6. In-domain only. A positive NQ result says nothing about other corpora until the
   TriviaQA direction check (`scripts/15_cross_dataset_eval.py`'s bench)
   reproduces it, as with Stage 8's transfer caveat.
7. The policy must move. The warm start is within 0.003 nats of uniform inside the
   decode window; the Stage 2 model barely separates adjacent boundary positions,
   which is itself consistent with the main conclusion. The failure to rule out is
   an optimiser that does not move the policy: if policy entropy has not changed
   by 0.05 nats per decision over the run, the result is a failed optimisation. It
   then says nothing about the objective, and it is rerun with a larger learning
   rate or more steps instead of being written up. The training script prints
   this and warns.

The dev gate only decides whether the final bench run is worth the GPU time. Dev
has about 400 questions, so its 2 SE is roughly 0.04, and a dev delta below that
does not show anything in either direction.

## How to run

```bash
# 0. one-off: the NQ train-split cache (shared with Stage 8)
python scripts/16_build_rerank_train_data.py

# 1. plumbing check, no GPU needed, numbers are meaningless by construction
python scripts/23_train_rl_chunker.py --smoke

# 2. train the policy (dev gate at the end)
python scripts/23_train_rl_chunker.py

# 3. only on GO: the Stage 6 bench, with the archived-baseline check
python scripts/24_eval_rl_chunker.py --max-questions 50   # smoke first
python scripts/24_eval_rl_chunker.py

# 4. only if the check passed
python scripts/save_stage_results.py --stage stage10
```

Notebook: `notebooks/RAG_chunk_optimize_stage10_colab.ipynb`.

If the Colab runtime is lost: the best weights are rewritten at every dev
evaluation, so at most `STAGE10_EVAL_EVERY` steps of training are lost. The bench
evaluation is checkpointed per config and resumes where it stopped.

## Oracle ceiling (pre-registered 2026-09-14, before it ran)

Run 1 compared a random cut with the supervised cut as a by-product. It could not
show how good the best possible cut is, and Stage 10 depends on that number: if no
placement inside the window can give a detectable gain, no chunker, RL-trained or
not, can change the conclusion at this size. `scripts/25_placement_oracle.py`
measures that ceiling without any learning.

Setup. The same 1000 NQ-train documents the RL run used, with every document
frozen at fixed 15/0 as distractors. A seeded subset (seed 42) of 400 answerable
documents is evaluated. For each one, 18 complete chunkings are scored: fixed
15/0, the supervised Stage 2 greedy decode, and 16 chunkings with every cut drawn
uniformly inside its `[11, 19]` window, overlap 0. Retrieval is top-10 over the
whole corpus. The Stage 6 bench is not used.

Metric. For each question, the oracle takes the best of the 18 candidates. The
primary quantity is the R@5 ceiling over fixed-size chunking: the mean of
`oracle - fixed` across questions, with a 95% CI. R@1, MRR@10 and a best-of-k
curve (k = 1, 2, 4, 8, 16) are reported alongside; the curve shows whether 16
samples come close to saturating the search.

Size guard. Among candidates tied for the best R@5, the oracle takes the one
closest in average chunk size to fixed, so any size drift it reports was needed to
reach the ceiling. Drift is the oracle's chunk-averaged size minus fixed's.

Verdicts, against the same threshold as the main criteria (0.046 R@5, the
smallest method effect the n = 1032 bench can detect):

- `CLOSED`: the CI's upper bound is below 0.046. No in-window placement policy can
  produce a detectable method effect at size 15, and Stage 10 closes.
- `INCONCLUSIVE`: the mean is below 0.046 but the CI reaches it.
- `SIZE-CONFOUNDED`: the mean is at least 0.046 but drift exceeds 0.3 sentences.
- `HEADROOM`: the mean is at least 0.046 with drift within 0.3.

How to read it. The oracle picks each chunking with the question and answer
available, and may pick differently for two questions about the same document, so
it overstates what a real chunker can reach. `CLOSED` is therefore strong
evidence. `HEADROOM` is not a claim: it allows at most one more RL run with a
lower-variance estimator (several samples per document with a leave-one-out
baseline), judged by the original criteria.

One run with K = 16 and 400 documents, as fixed here. It is not rerun with a
different K or subset to change the verdict.

## Results (executed)

### Run 1 (2026-09-14): failed optimisation

100 steps x 32 documents at `--lr 5e-5` on a Colab GPU, 78.9 minutes. Training
corpus 1000 NQ-train documents / 1116 questions, frozen distractor pool 21,335
chunks; dev bench 400 documents / 406 questions. The dev gate returned NO-GO on
both conditions: the best dev R@5 was the warm start itself, and the policy did
not move.

| Quantity | Value |
| --- | --- |
| Entropy per decision, first / last 10 steps | 2.1943 / 2.1941 |
| Uniform over the 9-boundary window (ln 9) | 2.1972 |
| Greedy reward, steps 1-50 vs 51-100 | 0.7409 vs 0.7381 (t = -0.21) |
| Dev R@5 at steps 0 / 50 / 100 | 0.9187 / 0.8966 / 0.9064 (2 SE about 0.03) |
| Gradient norm, median / max (clip 1.0) | 1.90 / 106.75, clipped in 78 of 100 steps |

Step 0 reproduced the supervised transformer's dev row exactly, as expected with
identical weights. By criterion 7 the run says nothing about the objective and is
not reported as a NULL. The Stage 6 bench was not used and nothing was archived.

Reading the logged entropy. Per-step values fall on four levels (about 1.988,
2.057, 2.126, 2.194). This comes from batch composition and not from policy
change: each step averages over 32 documents, a document short enough to need no
cut contributes zero entropy, and each level is ln 9 lowered by exactly
ln 9 / 32 = 0.0687 per such document. With those documents removed, every step is
within 0.0041 nats of uniform. The training script's windowed movement check has
the same problem; it reached the right verdict here, but it should average only
over documents that made decisions.

By-product (descriptive, not pre-registered). Because the policy stayed close to
uniform, every sampled rollout was a uniformly random cut inside each window,
while the greedy rollout was, apart from a few flipped cuts (the dev chunk count
moved by 18 of 7,937), the supervised Stage 2 cut. Across the run's 3,200 document
rollouts the two scored the same on the training reward:

- reward(random in-window cut) - reward(supervised cut): mean +0.0050,
  95% CI [-0.0034, +0.0134], positive in 53 of 100 steps.

In a narrow sense: on NQ-train, against a frozen fixed 15/0 distractor pool, under
an MRR@10 reward, the supervised model's choice of where to cut inside a
+/-4-sentence window is no better than a random choice. This is what the main
conclusion predicts, and it also explains the failed optimisation: with no
reliable reward difference between cut positions, the policy gradient is mostly
noise, as the heavy-tailed gradient norms show. It does not bound how much the
best possible cut could gain, because it compares random with supervised and not
with an optimum. Steps share documents, so the interval is approximate.

### Oracle run (2026-09-15): HEADROOM, as pre-registered

400 NQ-train documents, 446 questions, K = 16, seed 42, 69.9 minutes on a Colab
GPU. The smoke check's three self-checks against the RL reward path also passed on
the GPU. Nothing was archived and the Stage 6 bench was not used.

| Arm | R@1 | R@5 | MRR@10 | Avg chunk size |
| --- | --- | --- | --- | --- |
| fixed 15/0 | 0.6502 | 0.8879 | 0.7498 | 14.686 |
| supervised Stage 2 | 0.6525 | 0.8901 | 0.7507 | 14.854 |
| random in-window, mean of 16 | 0.6446 | 0.8858 | 0.7465 | 14.852 |
| question-aware oracle | 0.8475 | 0.9484 | 0.8919 | 14.686 |

| Comparison | R@5 | 95% CI |
| --- | --- | --- |
| oracle - fixed (primary) | +0.0605 | [+0.0384, +0.0827] |
| oracle - supervised | +0.0583 | [+0.0365, +0.0801] |
| supervised - fixed | +0.0022 | [-0.0214, +0.0259] |
| random mean - supervised | -0.0043 | [-0.0206, +0.0119] |

Best-of-k R@5 at k = 1, 2, 4, 8, 16 is 0.8789, 0.9126, 0.9283, 0.9439, 0.9484. It
is still rising at k = 16, so the true question-aware ceiling is somewhat higher
than reported.

The ceiling's mean is above the 0.046 threshold with zero size drift, so by the
pre-registered rule the verdict is HEADROOM. Under that rule it allows at most one
more RL run with a lower-variance estimator, and it is not a claim. The table also
agrees with the main conclusion in two ways: supervised and fixed tie again on
this corpus, and a random in-window chunking ties with both.

Post-hoc diagnostics (not pre-registered). Both use the run's checkpoint, with no
further GPU time, and ask how much of the ceiling a chunker could reach without
seeing the question.

- Where the ceiling comes from. The whole +0.0605 comes from 27 of 446 questions on
  which fixed misses at R@5 and some candidate hits. On those, a median of 7 of the
  16 random chunkings hit. Only 88 questions (19.7%) have an R@5 outcome that
  depends on placement at all; for the rest every chunking gives the same outcome.
  For the questions that do depend on placement, where the cut falls is close to a
  coin flip, and no arm gets it right systematically. That is why fixed,
  supervised and random tie on average.
- Held-out questions. On the 36 documents with two or more questions, each
  document's chunking is chosen using its other questions and scored on the
  held-out one:

| Metric | Chosen without the question, minus fixed | Question-aware oracle minus fixed, same 82 questions |
| --- | --- | --- |
| R@5 | +0.0090 [-0.0349, +0.0528] | +0.0732 [+0.0165, +0.1299] |
| MRR@10 | +0.0390 [-0.0065, +0.0846] | +0.1608 [+0.0994, +0.2222] |

About an eighth of the R@5 ceiling remains when the question is hidden, and what
remains is not distinguishable from zero. The sample is small and the bias can go
either way: choosing from one or two other questions is a weak signal, which
understates what a richer policy could learn, while related questions about the
same document would overstate generalisation. Together with RL run 1's near-zero
advantage, the evidence suggests the headroom is real per question but mostly out
of reach for a chunker that has to cut before the question is asked.

Open decision. The allowed RL rerun is still available. Based on these
diagnostics, its expected gain is well below the detection threshold.
