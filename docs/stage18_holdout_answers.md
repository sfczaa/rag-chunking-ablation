# Stage 18 - the Stage 17 comparisons on the holdout bench

Status: pre-registered on 2026-10-08, not run. The criteria below were fixed before
any reader output on the holdout bench existed.

Question: on NQ validation questions that no reader has answered, do the two Stage 17
comparisons, chunk size and boundary placement, show a difference in answer
accuracy?

## Motivation

Stage 17 scored 1031 Stage 6 bench questions with a fixed reader on top of the
Stage 8 reranker ([stage17_answer_quality.md](stage17_answer_quality.md)):

| Comparison | mean | 95% CI | Verdict |
| --- | --- | --- | --- |
| `fixed15_ft` - `fixed6_ft` | +0.0155 | [-0.0040, +0.0351] | SIZE-NOT-DETECTED |
| `bilstm15_ft` - `fixed15_ft` | +0.0175 | [-0.0001, +0.0351] | INCONCLUSIVE |

Both intervals end near 0. The Stage 16 holdout bench is about three times larger,
and no reader output exists for it.

## Bench

The Stage 16 holdout bench under `artifacts/data/nq/holdout/`: 2852 documents and
3063 questions, with no title or question shared with the Stage 6 bench. One stored
answer is empty after answer normalization and is dropped from every arm, which
leaves 3062 questions. A gold chunk exists for 3049 of them, and 2320 stored answers
are at most five NQ tokens long. Stage 16 measured dense and reranked retrieval at
fixed 15/0 on this bench; fixed 6/0 and bilstm 15/0 have not been run on it.

## Arms

`closed_book`, `gold`, `fixed15_ft`, `fixed6_ft` and `bilstm15_ft`, defined as in
Stage 17. `fixed15_large` is left out; Stage 17 reported it only.

Unchanged from Stage 17, including its 2026-10-05 amendment: the reader and its
revision, the prompt text, the 4096-token passage cap, k = 5, greedy decoding with
at most 64 new tokens, 512-token prefill chunks, the prediction rule, the
containment, exact-match and F1 definitions, and the hashes of the Stage 8 reranker
and the BiLSTM boundary model.

## Pre-registered criteria

1. Validity. The two weight files must match their hashes and the reader must load
   at its pinned revision. The bench must load with 2852 documents and 3063
   questions, share no title and no question with the Stage 6 bench, and give 3062
   scored questions and 3049 gold chunks. The dense `bge` row and the Stage 8
   reranker row at fixed 15/0 must match `stage16/final` within 0.005 on R@1, R@3 and
   R@5, with 54,177 chunks. Fixed 6/0 must give 133,298 chunks, counted before the
   run, and every retrieval arm must hold a top-five list for each of the 3063
   questions. The bilstm 15/0 chunk count has no reference and is reported. Otherwise
   the verdict is `INVALID`.
2. Reader check. As Stage 17 criterion 2, over the 3049 questions with a gold chunk.
3. Claim 1, chunk size. `fixed15_ft` minus `fixed6_ft` accuracy, paired over the 3062
   questions, with the Stage 17 criterion 3 rule: `SIZE-CARRIES`, `SIZE-SMALL`,
   `SIZE-REVERSED` or `SIZE-NOT-DETECTED`, floor 0.02.
4. Claim 2, boundary placement. `bilstm15_ft` minus `fixed15_ft` accuracy, paired over
   the 3062 questions, with the Stage 17 criterion 4 rule: `BILSTM-BETTER`,
   `BILSTM-WORSE`, `EQUIVALENT` or `INCONCLUSIVE`, margin 0.03.
5. Verdicts use unrounded values; tables show four decimals. The two claims are
   separate questions, each judged at its nominal level with no correction across
   them; no joint confidence level is claimed for the pair.
6. Reported only: the Stage 17 per-arm tables for this bench; both comparisons
   pooled over the Stage 6 and holdout questions (4093 questions); accuracy by stored
   answer length; passage cuts.
7. One run. Nothing is changed to move a verdict, and a stopped run resumes from its
   saved outputs. No setting is tried on holdout questions beforehand: the check
   notebook runs the reader on NQ-train rows only and the holdout bench only on the
   retrieval path.
8. Scope. One reader, greedy decoding, k = 5, NQ only.

## Pre-run expectations

With n = 3062, and the share of questions the two arms answer differently around
the Stage 17 values (0.103 for claim 1, 0.083 for claim 2):

| Claim 1, share answered differently | Standard error | P(lower bound > 0), true gap 0.0155 | 0.02 | 0.03 | P(`SIZE-CARRIES`), true gap 0.0155 | 0.02 | 0.03 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.08 | 0.0051 | 0.86 | 0.98 | 1.00 | 0.19 | 0.50 | 0.98 |
| 0.10 | 0.0057 | 0.78 | 0.94 | 1.00 | 0.22 | 0.50 | 0.96 |
| 0.12 | 0.0063 | 0.70 | 0.89 | 1.00 | 0.24 | 0.50 | 0.95 |

| Claim 2, share answered differently | Standard error | P(`BILSTM-BETTER`), true gap 0.0175 | P(`EQUIVALENT`), true gap 0 | true gap 0.01 | true gap 0.0175 |
| --- | --- | --- | --- | --- | --- |
| 0.07 | 0.0048 | 0.96 | 0.95 | 0.45 | 0.04 |
| 0.083 | 0.0052 | 0.92 | 0.95 | 0.52 | 0.08 |
| 0.10 | 0.0057 | 0.87 | 0.95 | 0.58 | 0.13 |

If the claim 1 gap is the Stage 17 estimate, the most likely verdict is
`SIZE-SMALL`: a gap above 0 that stays below the 0.02 floor. For claim 2, at these
standard errors the 90% interval stays inside the margin whenever the 95% interval
includes 0, so `INCONCLUSIVE` is unlikely: a gap of the Stage 17 size would most
likely give `BILSTM-BETTER`, no gap `EQUIVALENT`, and a gap near 0.01 either of the
two with about equal probability.

## Run safety and outputs

As in Stage 17, with the `stage18_` prefix, run version `stage18-v1` and the lock
`stage18_run.lock.json`. The reader inputs are saved after each chunking config, so
a restart rebuilds only the config in progress. The Stage 16 score cache for the
Stage 8 reranker at fixed 15/0, computed from the same weight file whose hash Stage
16 checked, is reused when its scored pairs match and recomputed otherwise. Outputs
go under `artifacts/results/latest/`; the contexts and the generations stay on
Drive. The Stage 17 scoring code is used unchanged.

## How to run

1. `notebooks/RAG_chunk_optimize_stage18_check_colab.ipynb`: preflight, the
   retrieval path on 40 holdout questions without the reader, and the reader smoke on
   NQ-train rows. Nothing is kept.
2. `notebooks/RAG_chunk_optimize_stage18_colab.ipynb`: reader inputs (`MODE = 'fresh'`
   for the first run and `'resume'` after any interruption), reader outputs and
   scores.

## Results

Not run yet.
