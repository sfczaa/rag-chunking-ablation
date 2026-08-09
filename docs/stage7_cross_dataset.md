# Stage 7 - Cross-dataset robustness check (TriviaQA rc.wikipedia)

**Status: complete.** Run on Colab (T4) 2026-07-07: check mode reproduced the
archived Stage 3 rows exactly (30/30, all deltas 0.0000), the TriviaQA eval
ran on **472 docs / 300 questions**, and **4/4 direction checks replicate**.
Archived to `artifacts/results/stage7/final/` (8 files). Results below.

Stage 7 asks one question: **is "chunk size dominates Recall, chunking method
ties" an artifact of Natural Questions + its Wikipedia pages, or does it
survive a change of QA dataset?** Every conclusion so far (Stages 1-6,
including the n=1032 replication) was measured on NQ. This is the strongest
remaining robustness check available without adding any new model.

## Dataset choice: TriviaQA `rc.wikipedia`

TriviaQA `rc.wikipedia` (validation split), streamed the same way
`nq_data.py` streams NQ.

**Why not HotpotQA** (considered first because of its clean supporting-facts
annotation): HotpotQA's context "documents" are the *introductory paragraphs*
of 2017 Wikipedia pages — typically ~4 sentences each, *shorter than the
smallest size on our chunk grid (6)*. All 30 configs would chunk such
documents nearly identically, so the size-vs-method question would be
untestable (a degenerate sweep). Repairing that would require fetching
current full articles, putting 9 years of drift between the questions and
the text they are scored against. TriviaQA `rc.wikipedia` instead bundles
**full Wikipedia pages** (`entity_pages.wiki_context`) inside the dataset:
the 6-15 sentence grid stays meaningful with zero fetching and zero drift.

## Gold-document definition (the crux for doc-constrained recall)

Each TriviaQA question ships 1-2 `entity_pages` — full Wikipedia pages about
entities in the question. The loader (`rag_chunk/cross_dataset.py`):

1. splits every entity page into sentences (a page needs >= 2 sentences to be
   usable — same rule as the NQ loader);
2. takes the answer candidates in a fixed order — `answer.value` first, then
   `answer.aliases` in dataset order — and normalizes each with the *same*
   `normalize_text` the metric uses;
3. keeps the **first candidate that appears as a substring of some page's
   sentence-joined normalized text**. That candidate becomes the question's
   single `answer` string — so the Recall@k computation stays literally
   identical to NQ's — and the **gold set is every entity page containing
   it** (`doc_titles`, 1-2 titles);
4. drops the question if no candidate appears in any of its pages (counted
   and reported, not silent).

Because the answer is verified against the *sentence-joined* text — exactly
the text chunks are built from (`chunking.chunk_text` joins with single
spaces; `normalize_text` collapses whitespace) — the NQ honesty guarantee is
restored: a doc-constrained miss means chunking split the answer span or
retrieval missed the chunk, never that the answer was absent.

**Distant-supervision caveat, stated up front:** a gold entity page is
*known to contain the answer string*, but unlike NQ's annotated gold
documents it is not human-verified to support the answer. For a **recall**
metric this is sound — we measure whether an answer-bearing chunk of a
designated gold page is retrieved — but it is a weaker gold notion than
NQ's, and every place Stage 7 numbers appear must say so.

Multi-gold plumbing: `metrics.recall_from_retrieved` now accepts an optional
`doc_titles` list per question and falls back to the single `doc_title`
unchanged. The check mode below proves the fallback is behaviour-identical
for NQ.

## Scope

Changed (the *only* change): the QA dataset — NQ → TriviaQA `rc.wikipedia`.

Held fixed (identical to Stage 3):

- Chunking logic and grids: fixed / BiLSTM / Transformer × sizes
  `{6,8,10,12,15}` × overlap `{0,1}` = 30 configs.
- Boundary/chunking embeddings (`all-MiniLM-L6-v2`) and the Stage 2
  BiLSTM/Transformer weights, loaded as-is.
- Dense retrieval: `BAAI/bge-base-en-v1.5` + BGE query instruction,
  L2-normalised cosine.
- Doc-constrained Recall@k as the headline metric.

Not included: BM25/RRF, reranking (any depth), fine-tuning of anything,
HotpotQA, new chunking models. **bge arm only** — this is a robustness check
of claim 1, not a leaderboard run.

## Corpus construction

- Stream validation questions until `STAGE7_N_QUESTIONS` (default 300) are
  kept. Actual doc/question counts are reported in every output, not assumed.
- Corpus = every usable entity page of every *kept* question, gold or not
  (a non-gold entity page of a kept question is a natural distractor),
  deduplicated by title.
- **Comparability guard:** the loader aborts if the median corpus document is
  shorter than 2× the largest grid size (30 sentences). That is the
  degenerate-sweep trap that disqualified raw HotpotQA — checked against
  real data at load time instead of assumed away.
- The corpus caches under `data/triviaqa/n<N>/`; the NQ caches are never
  touched.

## Validation

1. **Check mode first** (`python scripts/15_cross_dataset_eval.py --check`):
   re-runs the 30-config bge-only sweep **on the cached NQ 200-doc corpus**
   and every row must reproduce `stage3/final/sweep_results.csv` exactly
   (`stage7_check_vs_stage3.csv`: every `delta_n_chunks` 0, every recall
   delta 0.0000; the console prints a verdict). This proves (a) the Stage 7
   code path *is* the Stage 3 pipeline and (b) the multi-gold metric
   extension did not change single-gold behaviour. No conclusion may be
   drawn from TriviaQA rows before this prints check OK.
2. **TriviaQA mode** re-tests the direction claims with explicit rules
   (`stage7_direction_check.csv`; SE from the mean bge R@5 at the actual
   question count):

| # | Claim | Rule |
|---|---|---|
| 1a | size > method | Pearson r(avg chunk size, R@5) >= 0.5 over the 30 configs |
| 1b | size > method | R@5 size effect (largest vs smallest size) > mean method spread at matched (size, overlap) |
| 2 | methods tie at matched size | max method spread across the size-15 (size, overlap) cells < 2 SE |
| 3 | best chunk size is large | nominal size of the best-R@5 config (project ranking) ∈ {12, 15} |

A claim failing to replicate is a **finding, not an error**: the summary must
then say which explanation the data supports — dataset structure (e.g. page
length distribution), the weaker distant-supervision gold, question style
(trivia vs search queries), or a pipeline assumption — rather than re-tuning
until it passes.

## Run

```bash
# 1. sanity mode first: NQ 200-doc corpus, must reproduce stage3/final exactly.
python scripts/15_cross_dataset_eval.py --check

# 2. the TriviaQA eval (resume-safe: re-run the same command after an
#    interruption; --fresh discards the checkpoint).
python scripts/15_cross_dataset_eval.py

# 3. archive only after reviewing the direction checks.
python scripts/save_stage_results.py --stage stage7
```

The script refuses to start if `results/stage3/final/` is missing or was
built with a different retrieval model. Both modes checkpoint per config to
`results/latest/stage7_checkpoint_<mode>.jsonl`.

## Outputs

Written to `artifacts/results/latest/`, archived to
`artifacts/results/stage7/final/` after review:

- `stage7_cross_dataset_results.csv` — 30 bge rows with `n_docs` /
  `n_questions` / `dataset` columns.
- `stage7_matched_summary.csv` — per (size, overlap) cell: the three methods'
  R@5 side by side with the method spread (claims 1b/2 at a glance).
- `stage7_direction_check.csv` — the table above with observed values and
  yes/no verdicts.
- `stage7_dataset_summary.md` — dataset/gold definition, loader filter
  statistics, corpus length distribution, best config, verdicts.
- `stage7_recall_vs_chunk_size.png` — R@5 vs avg chunk size, NQ (Stage 3,
  n=203) and TriviaQA side by side, same visual language as Stage 6.
- `stage7_check_vs_stage3.csv` — from the check-mode run.
- `stage7_checkpoint_{check,trivia}.jsonl` — resume checkpoints (kept for
  provenance).

## Results (Colab run, 2026-07-07)

Check mode: **check OK** — all 30 rows identical to `stage3/final`
(every recall delta 0.0000, every chunk count equal), so the code path
including the multi-gold metric extension is the Stage 3 pipeline.

Eval set: 472 docs / 300 questions. Loader accountability: 302 rows scanned,
2 dropped (no answer candidate in their pages), 1 page under 2 sentences;
**118/300 questions have two gold pages** (the multi-gold extension is load-
bearing, not theoretical); document length median 152 / mean 210 sentences
(comparability guard ≥ 30: passed). 1 SE ≈ 0.018 at the mean R@5.

Direction checks — **4/4 replicate**:

| # | Claim | Observed | Rule | NQ reference |
|---|---|---|---|---|
| 1a | size > method | Pearson r(size, R@5) = **0.8005** | ≥ 0.5 | 0.77 (MiniLM n=203) / 0.95 (n=1032); the Stage 3 BGE rows in the figure give 0.86 |
| 1b | size > method | size effect **+0.0361** vs mean method spread 0.0153 (difference +0.0208) | effect > spread | +0.07 vs ≤ 0.014 (n=203); 0.052 (n=1032) |
| 2 | methods tie at size 15 | max size-15 spread = **0.0200** | < 2 SE (0.0355) | 0.020 at 15/0 (~1 SE 0.019) |
| 3 | best size is large | best config nominal size = **15** | ∈ {12, 15} | 15 at both NQ scales |

Best bge config: **bilstm target 15, overlap 1** — R@1 0.6933 / R@3 0.8667 /
R@5 0.9200 (fixed 15/1: 0.9067; transformer 15/1: 0.9000).

Findings:

1. **The headline transfers.** Recall still tracks chunk size (r = 0.80) with
   the method colours intermixed along the trend
   (`stage7_recall_vs_chunk_size.png`), and the best config is again size 15.
2. **The "winner" flips — which reinforces the tie.** On NQ the best config
   was fixed 15/0; on TriviaQA it is BiLSTM t15/1, by a margin (0.020) well
   inside 2 SE (0.036). A method that genuinely won would not swap places
   across datasets; noise would.
3. **The size effect is flatter here** (+0.036 R@5 from size 6→15, vs +0.062
   on the Stage 3 NQ rows): trivia questions against entity pages are already
   easy at small chunks (R@5 ≈ 0.87 at size ~6), leaving less headroom.
   Direction unchanged, magnitude dataset-dependent — reported as is.
4. Caveat repeated: TriviaQA gold documents are distant-supervised (page
   contains the answer string), a weaker gold notion than NQ's annotated
   documents. The doc-constrained recall numbers are internally consistent
   but not directly comparable in absolute terms across the two datasets.
