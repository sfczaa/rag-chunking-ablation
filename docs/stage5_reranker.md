# Stage 5 - Cross-Encoder Reranking (BGE top-k + pretrained reranker)

**Status: complete.** Full-grid Colab run on 2026-07-04; the re-run `bge` arm
reproduced the archived Stage 3 baseline exactly (30/30 configs, every delta
0.0000) and the results below are archived to `artifacts/results/stage5/final/`.
Stage 5 keeps the entire Stage 3/4 setup —
same dataset size, same chunking logic, same boundary models and weights, same
BGE dense retriever — and asks one new question: **does reordering the BGE
top-k candidates with an off-the-shelf cross-encoder improve Recall@1 and
Recall@3?** Recall@1 (0.650 at the Stage 3/4 best config) is where the dense
baseline has the most headroom; Recall@5 is already near its ceiling (0.921).

## Scope

Stage 5 changes only how the BGE candidate pool is *ordered* before the top-k
cut. Three arms are compared per chunking config:

1. **`bge`** — dense retrieval, exactly the Stage 3/4 path (BGE embeddings,
   L2-normalised, `IndexFlatIP`, BGE query instruction prefix).
2. **`rerank20`** — BGE retrieves the top-20 candidates; the pretrained
   cross-encoder `BAAI/bge-reranker-base` scores each (question, chunk) pair
   and the pool is reordered by that score (ties break on the original dense
   rank, so reranking is fully deterministic).
3. **`rerank50`** — the same with a top-50 candidate pool.

Held fixed (identical to Stage 3/4):

- NQ corpus (`N_NQ_DOCS=200`) and questions; doc-constrained Recall@{1,3,5}.
- Fixed / BiLSTM / Transformer sweep grids: sizes `{6,8,10,12,15}` ×
  overlap `{0,1}` (target-size policy for the learned methods).
- Boundary/chunking embeddings (`all-MiniLM-L6-v2`) and existing Stage 2
  BiLSTM/Transformer weights, loaded as-is.
- Dense retrieval model `BAAI/bge-base-en-v1.5` with its query instruction.

Per chunking config the chunks are built **once** and the dense index embeds
them once; the top-50 pool extends the top-20 pool (same dense ranking), so the
cross-encoder scores each (question, chunk) pair exactly once — pairs for ranks
1-20 are timed separately from ranks 21-50, keeping the per-depth latency
honest. All arms are scored by the same `metrics.recall_from_retrieved` code
path.

Every row also reports the **candidate-pool ceiling** `pool_recall@{20,50}` —
whether the answer-bearing chunk was present in the BGE pool at all. Reranking
can only promote chunks the pool already contains, so any Recall@k claim is
read against that ceiling.

Not included in this stage: BM25/RRF changes, weighted fusion, RL, BGE-M3, a
larger dataset, a deeper Transformer, new chunking objectives or models, and
**no reranker training or fine-tuning** — the published weights are used as-is.

## Config

Added to `config.py`:

| Knob | Default | Meaning |
|---|---|---|
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | pretrained cross-encoder (via `sentence_transformers.CrossEncoder` — no new pip dependency) |
| `RERANK_DEPTHS` | `(20, 50)` | BGE candidate-pool sizes; one rerank arm per depth |
| `RERANK_BATCH_SIZE` | 32 | (question, chunk) pairs per cross-encoder batch |
| `RERANK_MAX_LENGTH` | 512 | cross-encoder token cap (longer chunks truncate) |

## Run

Stage 5 requires the Stage 3 archive (`artifacts/results/stage3/final/`) as
the bge-arm baseline, and expects `results/latest/` to contain only files
already archived (normally the Stage 4 outputs, archived by
`save_stage_results.py --stage stage4`):

```bash
python scripts/11_sweep_reranker.py
```

For a faster sanity run:

```bash
python scripts/11_sweep_reranker.py --quick
```

The script refuses to start if the Stage 3 archive is missing or was built
with a different dense retrieval model, and refuses to delete any file from
`results/latest/` without a byte-identical copy in a stage archive — so
nothing unarchived is ever lost, and the Stage 3/4 finals are never touched.
The reranker weights are downloaded/loaded **before** `latest/` is cleared, so
a failed download leaves everything in place.

After the run, `stage3_vs_stage5_bge_check.csv` compares the Stage 5 `bge` arm
against the archived Stage 3 rows config-by-config. Because that arm re-runs
the exact Stage 3 pipeline, every `delta_n_chunks` must be 0 and every recall
delta should be 0.0000. The console prints a verdict; only archive once it
passes:

```bash
python scripts/save_stage_results.py --stage stage5
```

## Outputs

Written to `artifacts/results/latest/`:

- `rerank_sweep_results.csv` - one row per (arm × method × chunk config);
  30 configs × 3 arms = 90 rows on the full grid. Includes
  `pool_recall@{20,50}` (the rerank ceiling), `rerank_seconds` (cumulative
  cross-encoder time for that depth) and `n_pairs`.
- `rerank_best_config.json` - best config per arm plus the overall best
  (same ranking as Stages 1-4: R@5, then R@3, R@1, then smaller chunks).
- `rerank_matched.csv` - bge / rerank20 / rerank50 Recall@k side by side per
  chunking config, with `rerank{20,50}_minus_bge@k` deltas and pool recalls.
- `stage3_vs_stage5_bge_check.csv` - the Stage 3 reproduction check described
  above.
- `rerank_recall_vs_chunk_size.png` - **Recall@1** (the goal metric) vs
  average chunk size, coloured by arm with per-arm trend lines.
- `rerank_comparison.png` - grouped Recall@k bars for the best config of each
  arm.
- `stage5_reranker_summary.md` - run metadata, check verdict, pool ceilings,
  mean rerank-vs-bge deltas, rerank cost, and best configs.

The console summary prints, before any improvement claim: the mean candidate-
pool recall at each depth, the mean rerank-vs-bge delta at each k, and the
mean rerank cost per config.

## Results (Colab run, 2026-07-04)

The Stage 3 reproduction check passed: 30/30 configs with `delta_n_chunks = 0`
and every recall delta 0.0000 (`stage3_vs_stage5_bge_check.csv`), so both
rerank arms are measured against a verified-identical baseline.

**Candidate-pool ceiling (reported before any improvement claim):** mean
`pool_recall@20` = **0.9650**, `pool_recall@50` = **0.9851**. The answer chunk
is almost always in the pool — ranking, not pool recall, is the binding
constraint, so reranking had room to work at both depths.

Mean paired deltas over the 30 configs:

| Comparison | ΔR@1 | ΔR@3 | ΔR@5 | R@1 sign test |
|---|---|---|---|---|
| rerank20 − bge | **+0.0440** | +0.0278 | +0.0177 | 26 better / 3 worse / 1 tie, *P* ≈ 1.5×10⁻⁵ |
| rerank50 − bge | +0.0235 | +0.0187 | +0.0126 | 20 better / 8 worse / 2 ties |
| rerank50 − rerank20 | −0.0205 | — | — | 0 better / 28 worse / 2 ties, *P* ≈ 7×10⁻⁹ |

Best config per arm (ranked by R@5 then R@3/R@1, as in Stages 1-4):

| Arm | Best config | R@1 | R@3 | R@5 |
|---|---|---|---|---|
| bge | fixed size 15, overlap 0 | 0.650 | 0.847 | 0.921 |
| rerank20 | fixed size 15, overlap 1 | 0.640 | 0.862 | **0.931** |
| rerank50 | fixed size 15, overlap 1 | 0.640 | 0.847 | 0.916 |

Findings:

1. **Reranking the top-20 genuinely improves R@1 on average** — +0.044 (~9 of
   203 questions), consistent across 26/30 configs. This is the first
   post-Stage-3 change with a one-sided, significant effect in the *positive*
   direction (Stage 4's RRF was one-sided negative).
2. **The gain concentrates at small chunks and vanishes at the sweet spot.**
   Mean ΔR@1 by size: **+0.075** (size ~6), +0.042 (8), +0.059 (10),
   +0.043 (12), **+0.002** (15). At the Stage 3 best config (fixed 15/0) the
   deltas are +0.010 R@1 / −0.010 R@3 / −0.025 R@5 — inside 1 SE ≈ 0.034.
   The rerank20 trend line in `rerank_recall_vs_chunk_size.png` is nearly
   flat: reranking *flattens the chunk-size effect* rather than raising the
   ceiling.
3. **A deeper pool strictly hurts.** rerank50 never beats rerank20 on R@1
   (28 worse, 2 tied out of 30) despite its higher pool ceiling: candidates
   past rank 20 are almost all distractors (+0.02 pool recall for 30 more
   candidates), and the cross-encoder sometimes promotes one over the true
   chunk.
4. **Cost.** ≈ 24.7 ms per (question, chunk) pair on a T4 GPU — mean 100 s per
   config at depth 20 (≈ 0.49 s/query) and 241 s at depth 50 (≈ 1.18 s/query).

**Verdict:** with a strong dense retriever *and* well-sized chunks, an
off-the-shelf cross-encoder is a wash at the best config; its real value here
is rescuing small-chunk configs, and the pool should stay shallow (20, not 50).
