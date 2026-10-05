# Stage 17 - answer accuracy with a reader on top of retrieval

Status: pre-registered on 2026-10-03, amended on 2026-10-05 before any bench output
(see the amendment), not run. The criteria below were fixed before any reader output
on the bench existed.

Question: when an instruction-tuned reader answers from the top five reranked
chunks, do the retrieval differences found on the Stage 6 bench carry over to
answer accuracy?

## Motivation

Stages 1 to 16 measured retrieval only: whether the answer string is in a top-k
chunk from the question's own document. Two retrieval results on the Stage 6
bench (NQ validation, 1000 documents, 1032 questions), with the BGE top-20 pool
reordered by the Stage 8 reranker, frame this stage:

| Config | R@1 | R@5 | Avg chunk size |
| --- | --- | --- | --- |
| fixed 15/0 | 0.7355 | 0.9215 | 14.649 |
| fixed 6/0 | 0.6502 | 0.8585 | 5.950 |
| bilstm 15/0 | 0.7141 | 0.9244 | 14.801 |

These are the archived Stage 8 rows (`artifacts/results/stage8/final/`). The
retrained Stage 8 weights used since Stage 12 score 0.7345 R@1 and 0.9186 R@5 at
fixed 15/0; this stage rescores all three configs with them.

- Chunk size: fixed 15/0 has the answer in the top five for 0.063 more of the
  questions than fixed 6/0.
- Boundary placement: at matched size, BiLSTM and fixed chunking are level at R@5.

A reader may gain less than R@5 suggests, because a larger chunk also gives it more
text to read, and the rank of the answer-bearing chunk inside the top five may
matter. This stage measures both effects once, with one reader and one prompt
fixed in advance.

## Reader

`Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`: Apache-2.0, not gated, 4.0B
parameters, instruction-tuned without a thinking mode, run in fp16 on a 16 GB T4.

It was chosen on licence and memory before any output existed.
`Qwen2.5-3B-Instruct`, the first candidate, is under the Qwen Research licence.
Llama 3.2 and Gemma 2 require an access request.

Fallback. If the preflight finds a non-finite logit or an empty output on its
NQ-train rows, the reader becomes `microsoft/Phi-3.5-mini-instruct` at revision
`2fe192450127e6a83f7441aef6e3ca586c338b77` (MIT), and the switch is recorded here
before any bench question is generated. No other change of reader is allowed.

## Bench and arms

The Stage 6 bench. One stored answer, ")", is empty after answer normalization; that
question is dropped from every arm, which leaves 1031 questions.

Each retrieval arm reorders the BGE top-20 pool of its config with a cross-encoder
and passes the top five chunks to the reader, rank 1 first.

| Arm | Chunking | Reranker | Role |
| --- | --- | --- | --- |
| `fixed15_ft` | fixed 15/0 | Stage 8 weights | claims 1 and 2 |
| `fixed6_ft` | fixed 6/0 | Stage 8 weights | claim 1 |
| `bilstm15_ft` | BiLSTM, target 15, min 11, max 19, overlap 0 | Stage 8 weights | claim 2 |
| `fixed15_large` | fixed 15/0 | Stage 15 weights | reported only |
| `closed_book` | no passage | none | reader check |
| `gold` | one passage: the first fixed 15/0 chunk of the question's document that contains the answer | none | reader check |

Weights, checked by sha256 before use:

| File | sha256 |
| --- | --- |
| Stage 8 reranker `model.safetensors` | `0f638a21a29c40ea3ec15c78c5a12537bc274610480e8cee0f46edbc648593a0` |
| Stage 15 reranker `model.safetensors` | `e7fff83c1ee8faf36bb11aabafb786c4f1c2c9438067987780554fd015fc51b3` |
| BiLSTM boundary model `bilstm_best.pt` | `04242d41034e018acf23dbcf0a0f7089dbe6509d573440fa62035549c9c2f222` |

The two reranker files are the ones Stage 16 checked. A gold chunk exists for 1028
of the 1031 questions; for the other 3 the answer string crosses a chunk boundary.

## Prompt and decoding

Each passage is cut to its first 4096 reader tokens. Most chunks are far shorter:
at fixed 15/0 the median is 387 tokens, the 99th percentile 2475 and the longest
22,064. The number of cut passages per arm is reported, with the number of
questions whose answer string survives only in a cut part.

User message for the retrieval arms and `gold`:

```text
Answer the question using the passages below. Reply with the answer only, as a short phrase, without explanation.

[1] {title of the chunk's document}
{chunk text}

[2] {title}
{chunk text}

(up to [5])

Question: {question}
```

User message for `closed_book`:

```text
Answer the question. Reply with the answer only, as a short phrase, without explanation.

Question: {question}
```

The message goes through the reader's chat template with the generation prompt
and no system message. Decoding is greedy with at most 64 new tokens, one question
per forward pass so no padding is involved, in fp16.

## Scoring

The prediction is the reader output with any `<think>...</think>` block removed,
cut at the first line break and stripped. Normalization follows SQuAD: lower case,
punctuation removed, the articles a, an and the removed, whitespace collapsed.

- Accuracy, the primary metric: a prediction is correct when the normalized stored
  answer appears as a contiguous token sequence in the normalized prediction.
- Exact match and token F1 (SQuAD definitions): reported only.

The stored answer is the first annotated short answer, the string the retrieval
recall already matches, so accuracy and R@k use one gold string per question.
Answers from the other NQ annotators are not used. 263 of the 1032 stored answers
are longer than five NQ tokens; accuracy is reported for both groups.

## Pre-registered criteria

1. Validity. The three weight files must match the hashes above and the reader the
   pinned revision; the bench must load with 1000 documents and 1032 questions. The
   dense `bge` rows at fixed 15/0, fixed 6/0 and bilstm 15/0 must match
   `stage8/final` within 0.005 on R@1, R@3 and R@5 with the same chunk counts
   (19,507, 48,029 and 19,306). The retrieval rows of `fixed15_ft` and
   `fixed15_large` must match `stage15/final` within 0.005. Otherwise the verdict
   is `INVALID`.
2. Reader check. `gold` minus `closed_book` accuracy over the 1028 questions with
   a gold chunk must have a 95% lower bound above 0. Otherwise the verdict is
   `INVALID-READER` and no claim verdict is issued.
3. Claim 1, chunk size. `fixed15_ft` minus `fixed6_ft` accuracy, paired over the
   1031 questions, mean with a 95% interval of mean +/- 1.96 standard errors:
   - `SIZE-CARRIES`: the lower bound is above 0 and the mean is at least 0.02.
   - `SIZE-SMALL`: the lower bound is above 0 and the mean is below 0.02.
   - `SIZE-REVERSED`: the upper bound is below 0.
   - `SIZE-NOT-DETECTED`: the interval includes 0.
4. Claim 2, boundary placement. `bilstm15_ft` minus `fixed15_ft` accuracy, paired
   over the 1031 questions:
   - `BILSTM-BETTER`: the 95% lower bound is above 0.
   - `BILSTM-WORSE`: the 95% upper bound is below 0.
   - `EQUIVALENT`: the 95% interval includes 0 and the 90% interval, mean +/- 1.645
     standard errors, lies inside [-0.03, +0.03].
   - `INCONCLUSIVE`: anything else.
5. Verdicts use unrounded values; tables show four decimals. The two claims are
   separate questions, each judged at its own level, with no correction across
   them.
6. Reported only: `fixed15_large` minus `fixed15_ft`; accuracy, exact match and F1
   for every arm, with R@1 and R@5 for the retrieval arms; accuracy split by
   whether the answer is in the top five and by stored answer length (at most five
   NQ tokens, or more); for claim 1 the accuracy gap divided by the R@5 gap; mean
   prediction length in words; prompt tokens; passage cuts; seconds per question.
7. One run. The reader, prompt, passage cap, k and decoding are not changed to move
   a verdict. A stopped run resumes from its saved generations, which is the same
   run. No setting is tried on bench questions beforehand: the preflight runs the
   reader on NQ-train rows only, and touches bench questions only on the retrieval
   path, where no reader output is produced.
8. Scope. One reader, greedy decoding, k = 5, NQ only. Claim 1 holds k fixed, so
   the 15/0 arm also gives the reader more text: averaged over all bench chunks,
   five 15/0 chunks hold about 2360 reader tokens and five 6/0 chunks about 960. A
   comparison at matched context length is outside this stage.

## Pre-run expectations

If accuracy were R@5 times the reader's accuracy when the answer is in the context
(a), plus 1 - R@5 times its accuracy when it is not (b), the claim 1 gap would be
about 0.063 x (a - b): 0.025 to 0.038 for a - b between 0.4 and 0.6. A longer
context could move a in either direction.

The paired standard error depends on the share of questions that the two arms
answer differently, which is unknown before the run. With n = 1031:

| Discordant share | Standard error | P(lower bound > 0), true gap 0.02 | 0.03 | 0.04 |
| --- | --- | --- | --- | --- |
| 0.10 | 0.0098 | 0.53 | 0.86 | 0.98 |
| 0.15 | 0.0121 | 0.38 | 0.70 | 0.92 |
| 0.20 | 0.0139 | 0.30 | 0.58 | 0.82 |
| 0.25 | 0.0156 | 0.25 | 0.49 | 0.73 |

P(`SIZE-CARRIES`) is within 0.03 of the same values. For claim 2, the probability
of `EQUIVALENT`:

| Discordant share | Standard error | margin 0.02, true gap 0 | margin 0.03, true gap 0 | margin 0.03, true gap 0.01 |
| --- | --- | --- | --- | --- |
| 0.08 | 0.0088 | 0.47 | 0.92 | 0.73 |
| 0.10 | 0.0098 | 0.30 | 0.84 | 0.64 |
| 0.15 | 0.0121 | 0.01 | 0.60 | 0.46 |
| 0.20 | 0.0139 | 0.00 | 0.39 | 0.31 |

The 0.03 margin is set by what 1031 questions can resolve: with a margin of 0.02
the chance of `EQUIVALENT` stays below one half in every row of the table, even
when the two arms truly agree. A narrower margin needs more questions.

The retrieval results that motivate both claims were measured on these questions.
The reader outputs are new, but they are not independent of those results. The
3063-question holdout bench of Stage 16 is not used here and remains available for
a confirmation run under its own pre-registration.

## Run safety

Retrieval runs first and writes, per arm and question, the top five chunk
identifiers and texts to `stage17_contexts.jsonl`; the Stage 15 score cache is
reused at fixed 15/0 when the hash of the scored pairs matches. The rerankers are
released before the reader is loaded. If the retrieval rows fail criterion 1,
nothing is generated.

Generation runs arm by arm in bench order. Every 32 questions are appended to
`stage17_generations/<arm>.jsonl` with the prompt hash, so a lost runtime loses at
most 32 generations and a restart skips finished rows after checking their prompt
hashes again. The shared-folder identity
check and the run lock of Stages 14 to 16 apply, with run version `stage17-v1`.
Each row records the GPU name, and the run stops if it differs from the first
row's. The first run records the reader and its revision, the prompt, the passage
cap, the decoding settings, the bench questions and answers, the verdict thresholds,
a hash of the scoring code and the weight hashes in `stage17_run_identity.json`; a
resume with any difference stops.

## Outputs

Under `artifacts/results/latest/`:

- `stage17_run_identity.json` - the settings a resume must share
- `stage17_contexts.jsonl` - the reader inputs and retrieval rows, kept on Drive only
- `stage17_scores/` - the reranker score cache
- `stage17_generations/` - one JSONL per arm with the raw output and the prediction
- `stage17_eval_results.csv` - one row per arm
- `stage17_paired_deltas.csv` - the paired comparisons
- `stage17_check_vs_archive.csv` - the validity check
- `stage17_summary.md` and `stage17_verdict.json`

## How to run

1. `notebooks/RAG_chunk_optimize_stage17_check_colab.ipynb`: the preflight
   (`scripts/44_preflight_stage17.py`: shared folder, write access, bench, weights,
   archives, the pinned reader revision, GPU), the retrieval path on 40 bench
   questions without the reader (`scripts/41_stage17_contexts.py --smoke`), and the
   reader on 16 NQ-train groups from `stage8_train_groups.jsonl`
   (`scripts/42_stage17_generate.py --smoke`): closed book, the positive alone, and
   the positive with four negatives, all labelled with the group's title because
   negatives carry none. The reader smoke checks every generation step for a
   non-finite logit and every output for an empty prediction, the fallback
   condition. Nothing is kept.
2. `notebooks/RAG_chunk_optimize_stage17_colab.ipynb`: reader inputs
   (`scripts/41_stage17_contexts.py`, `MODE = 'fresh'` for the first run and
   `'resume'` after any interruption), reader outputs
   (`scripts/42_stage17_generate.py`) and scores (`scripts/43_stage17_score.py`).

## Amendment before any bench output (2026-10-05)

The reader smoke ran out of T4 memory on an NQ-train prompt of about 7,800 tokens.
The eleven shorter prompts before it had finite logits and non-empty outputs, so
the fallback condition was not met. With no padding, the model drops the attention
mask when the query and key lengths match, so SDPA took its grouped-query path,
which on this GPU falls back to a kernel that holds the full attention matrix. No
bench question had been generated. So:

- the prompt is prefilled in chunks of 512 tokens (`prefill_chunk_size`) and decoding
  continues from the filled cache; on a small test model, chunked and single-pass
  greedy outputs were identical;
- the reader smoke also runs one prompt at the longest length the passage cap allows,
  five passages of 4096 tokens, and prints the peak and free GPU memory;
- the CUDA allocator uses expandable segments;
- the run identity records the chunk size, and each generation records the torch and
  transformers versions.

A review of the code before the run led to further changes, also before any bench
output:

- the prediction rule is stated in full: after any think block is removed, the text
  is stripped, cut at the first line break and stripped again, which is what the code
  did;
- the run identity also records the bench questions and answers, the two verdict
  thresholds and a hash of the scoring code;
- a weight-hash or bench-size failure now writes the INVALID verdict without
  running retrieval, and an archive check needs exactly one reference row per arm;
- the gold arm counts answers lost to the passage cap, and the reported R@1 and R@5
  use the 1031 scored questions; the archive check still uses all 1032.

The reader, the prompt text, the passage cap, k, greedy decoding and every criterion
are unchanged.

## Results

Not run yet.
