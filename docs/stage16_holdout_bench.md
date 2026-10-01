# Stage 16 - the Stage 15 comparison on a holdout NQ bench

Status: pre-registered on 2026-09-27, run on 2026-10-01. Verdict: BELOW-FLOOR
(+0.0193 R@1, 95% CI [+0.0072, +0.0314], n = 3063). The design criteria and
thresholds below were fixed before any GPU run for this stage.

Question: on NQ validation questions that no earlier stage has scored, does the
Stage 15 reranker (`bge-reranker-large`) still beat the Stage 8 reranker?

## Motivation

Stage 15 passed its rule by a small margin: +0.0213 R@1, 95% CI [+0.0009,
+0.0417], on the 1032-question Stage 6 bench. Most of that interval's width comes
from which questions are in the bench, and the same bench had already been used
by Stages 11 to 14. Retraining with another seed and scoring the same questions
would not address that, so this stage keeps both sets of weights and changes the
questions.

## Bench

The NQ validation split is streamed with the rules of `nq_data.prepare_nq`
(first short answer, non-HTML tokens, at least two sentences), skipping every row
whose title is in the Stage 6 bench or whose question text is a Stage 6 question.
The bench is the first 3000 new titles in stream order, or all of them if the
split ends first, and the questions met before the last of them is added. It is
built once by `scripts/39_build_holdout_bench.py` and cached under
`artifacts/data/nq/holdout/`.

Sentences are split with the vendored Punkt code now in the repository; the
Stage 6 bench was split with the NLTK package. Both benches are scored with the
same chunking, retrieval and reranking code.

## Arms

| Arm | Weights |
| --- | --- |
| `bge` | dense order, no reranker |
| `rerank20_ft` | Stage 8 weights, `model.safetensors` sha256 `0f638a21a29c40ea3ec15c78c5a12537bc274610480e8cee0f46edbc648593a0` |
| `rerank20_s15_large` | Stage 15 weights, `model.safetensors` sha256 `e7fff83c1ee8faf36bb11aabafb786c4f1c2c9438067987780554fd015fc51b3` |

Nothing is trained. Both rerankers reorder one shared BGE top-20 pool at fixed
15/0.

## Pre-registered criteria

1. Validity. Both weight files must match the hashes above, and the bench must
   share no title and no question text with the Stage 6 bench. Otherwise the
   verdict is `INVALID`.
2. Size. The bench must hold at least 1500 questions, or the verdict is
   `INCONCLUSIVE-SIZE`.
3. The claim. Holdout bench, fixed 15/0, R@1, `rerank20_s15_large` minus
   `rerank20_ft`, paired over questions, mean with a 95% interval of mean +/- 1.96
   standard errors:
   - `CONFIRMED`: the lower bound is above 0 and the mean is at least 0.02, the
     Stage 15 rule.
   - `BELOW-FLOOR`: the lower bound is above 0 and the mean is below 0.02.
   - `LARGE-WORSE`: the upper bound is below 0.
   - `NOT-CONFIRMED`: the interval includes 0.

   The verdict uses unrounded values; tables show four decimals.
4. Reported, not claimed: R@5 and the per-arm table on the holdout bench, the
   pooled comparison over the Stage 6 and holdout questions, the number of
   holdout titles that also appear in the Stage 8 training documents, and seconds
   per question.
5. One run. The bench is not rebuilt or resampled to move a verdict. A crashed
   step resumes from the bench cache and the score cache, which is the same run.
6. Scope. NQ only.

## Pre-run expectations

Scaling Stage 15's paired standard error (0.0104 at 1032 questions) to the
holdout size, and taking the Stage 15 estimate as the true gain:

| Holdout questions | Standard error | P(lower bound > 0) | P(`CONFIRMED`) |
| --- | --- | --- | --- |
| 1500 | 0.0086 | 0.69 | 0.56 |
| 2000 | 0.0075 | 0.81 | 0.57 |
| 3000 | 0.0061 | 0.94 | 0.58 |

With a true gain of 0.01 instead, P(lower bound > 0) is 0.21 to 0.37 over the
same sizes. A gain this close to the floor cannot be placed above or below 0.02
reliably at any of these sizes, so `BELOW-FLOOR` and `CONFIRMED` are both
consistent with a real gain near 0.02. The distinction this stage can make is
between a gain that replicates on new questions and one that does not.

## Outputs

Under `artifacts/results/latest/`: `stage16_eval_results.csv`,
`stage16_paired_deltas.csv`, `stage16_summary.md`, `stage16_verdict.json`, the
checkpoint `stage16_checkpoint_final.jsonl` and the score cache `stage16_scores/`.

## How to run

`notebooks/RAG_chunk_optimize_stage16_colab.ipynb`: build the bench, a 40-question
smoke run, then the evaluation.

## Results

### Run 1 (2026-10-01): BELOW-FLOOR

Run on a Colab GPU from `notebooks/RAG_chunk_optimize_stage16_colab.ipynb` at
commit `543b000`, one account, one session.

Bench: the stream read all 7830 rows of the validation split and ran out after
2852 new documents with 3063 questions; 1226 rows were skipped for Stage 6 titles
and none for Stage 6 questions. 188 of the holdout titles also appear in the
Stage 8 training documents.

Validity: PASS. Both weight files match the pre-registered hashes, and the bench
shares no title and no question with the Stage 6 bench. Size: 3063 questions
against the 1500 required.

| Arm | R@1 | R@3 | R@5 | s per question, load included |
| --- | --- | --- | --- | --- |
| dense BGE | 0.6069 | 0.7989 | 0.8528 | - |
| Stage 8 weights (base) | 0.7231 | 0.8652 | 0.8981 | 0.615 |
| Stage 15 weights (large) | 0.7424 | 0.8766 | 0.9070 | 1.880 |

Pool recall@20 was 0.9406.

| Comparison, large - Stage 8 | Questions | mean | 95% CI |
| --- | --- | --- | --- |
| holdout, R@1 | 3063 | +0.0193 | [+0.0072, +0.0314] |
| holdout, R@5 | 3063 | +0.0088 | [+0.0025, +0.0151] |
| Stage 6 + holdout, R@1 | 4095 | +0.0198 | [+0.0094, +0.0302] |
| Stage 6 + holdout, R@5 | 4095 | +0.0093 | [+0.0037, +0.0149] |

By criterion 3 the verdict is BELOW-FLOOR: the lower bound is above 0 and the
mean is below 0.02.

Reading. The Stage 15 gain replicates on questions no earlier stage had scored:
+0.0193 R@1 against +0.0213 on the Stage 6 bench, with an interval that now
excludes 0 by a clear margin. Of the 3063 questions, 209 are ranked first only by
the large model and 150 only by the Stage 8 model. R@5 also improves, with an
interval above 0. The size of the gain sits at the 0.02 threshold on both benches,
as the pre-run expectations anticipated, so the data support a real gain of about
0.02 R@1 and do not settle whether it is above or below 0.02. The large model
takes about three times as long per question.
