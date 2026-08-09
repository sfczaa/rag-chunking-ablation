# Stage 4 - Hybrid Retrieval Ablation (BM25 + BGE + RRF)

**Status: complete — run on Colab 2026-07-03; the Stage 3 reproduction check
passed exactly (30/30 configs, all deltas 0). See Results below.**
Stage 4 keeps the entire Stage 3 setup — same
dataset size, same chunking logic, same boundary models and weights, same BGE
dense retrieval — and asks one new question: **over the identical chunks, does
lexical BM25 or a BGE+BM25 Reciprocal Rank Fusion beat BGE-only retrieval?**

## Scope

Stage 4 changes only how chunks are *ranked* for a query. Three retrievers are
compared per chunking config:

1. **`bge`** — dense retrieval, exactly the Stage 3 path (BGE embeddings,
   L2-normalised, `IndexFlatIP`, BGE query instruction prefix).
2. **`bm25`** — Okapi BM25 over the same chunk texts (pure-numpy implementation
   in `rag_chunk/hybrid.py`; lower-case alphanumeric tokens, Lucene idf
   `log(1 + (N - df + 0.5)/(df + 0.5))`, `k1=1.5`, `b=0.75`; no stemming or
   stopwords — a standard, fully deterministic baseline with no new dependency).
3. **`rrf`** — Reciprocal Rank Fusion of the two rankings:
   `score(d) = Σ 1/(RRF_K + rank_retriever(d))` over the top
   `HYBRID_FUSE_DEPTH` candidates from each retriever (`RRF_K=60`, depth 50).

Held fixed (identical to Stage 3):

- NQ corpus (`N_NQ_DOCS=200`) and questions; doc-constrained Recall@{1,3,5}.
- Fixed / BiLSTM / Transformer sweep grids: sizes `{6,8,10,12,15}` ×
  overlap `{0,1}` (target-size policy for the learned methods).
- Boundary/chunking embeddings (`all-MiniLM-L6-v2`) and existing Stage 2
  BiLSTM/Transformer weights, loaded as-is.
- Dense retrieval model `BAAI/bge-base-en-v1.5` with its query instruction.

Per chunking config the chunks are built **once**; the dense index embeds them
once, BM25 indexes the same texts over the same chunk-id space, and RRF fuses
the two rankings — so within a config the only difference between the three
result rows is the retriever. All three are scored by the same
`metrics.recall_from_retrieved` code path.

Not included in this stage: rerankers, RL, BGE-M3, a larger dataset, a deeper
Transformer, or new chunking objectives.

## Config

Added to `config.py`:

| Knob | Default | Meaning |
|---|---|---|
| `BM25_K1` / `BM25_B` | 1.5 / 0.75 | Okapi BM25 term-saturation / length-normalisation |
| `RRF_K` | 60 | RRF constant (standard value from the RRF paper) |
| `HYBRID_FUSE_DEPTH` | 50 | top-N candidates taken from each retriever before fusion |

## Run

Stage 4 requires Stage 3 to be archived first, because
`artifacts/results/latest/` is overwritten by each new sweep. The archive step
is **copy-only** — it does not rerun or modify Stage 3:

```bash
python scripts/save_stage_results.py --stage stage3
python scripts/10_sweep_hybrid_retrieval.py
```

For a faster sanity run:

```bash
python scripts/10_sweep_hybrid_retrieval.py --quick
```

The script refuses to start if `artifacts/results/stage3/final/` is missing,
if the archive was built with a different dense retrieval model, or if
`results/latest/` contains any file without a byte-identical archived copy —
so nothing unarchived is ever deleted, and `stage3/final` is never touched.

After the run, `stage3_vs_stage4_bge_check.csv` compares the Stage 4 `bge` arm
against the archived Stage 3 rows config-by-config. Because that arm re-runs
the exact Stage 3 pipeline, every `delta_n_chunks` must be 0 and every recall
delta should be 0.0000. The console prints a verdict; only archive once it
passes:

```bash
python scripts/save_stage_results.py --stage stage4
```

## Outputs

Written to `artifacts/results/latest/`:

- `hybrid_sweep_results.csv` - one row per (retriever × method × chunk config);
  30 configs × 3 retrievers = 90 rows on the full grid.
- `hybrid_best_config.json` - best config per retriever plus the overall best.
- `hybrid_retriever_matched.csv` - bge / bm25 / rrf Recall@k side by side per
  chunking config, with `rrf_minus_bge@k` and `rrf_minus_bm25@k` deltas.
- `stage3_vs_stage4_bge_check.csv` - the Stage 3 reproduction check described
  above.
- `hybrid_recall_vs_chunk_size.png` - Recall@5 vs average chunk size, coloured
  by retriever with per-retriever trend lines.
- `hybrid_retriever_comparison.png` - grouped Recall@k bars for the best config
  of each retriever.
- `stage4_hybrid_retrieval_summary.md` - run metadata, check verdict, and best
  configs.

Do not treat Stage 4 as complete until those files exist from an actual run and
the Stage 3 check passes.

## Results (Colab run, 2026-07-03)

The check passed exactly: all 30 configs have `delta_n_chunks = 0` and every
recall delta is 0.0000, so the `bge` arm is byte-for-byte the Stage 3 baseline.

| Retriever | Best Recall@5 | Best config |
|---|---|---|
| bge | **0.921** | fixed size 15, overlap 0 |
| rrf | 0.887 | fixed size 15, overlap 0 |
| bm25 | 0.803 | bilstm target 15, overlap 0 |

**Hybrid does not help on this benchmark.** Across the 30 matched configs at
R@5, `rrf − bge` has mean −0.021 (range −0.054…+0.035; 24 negative / 2 zero /
4 positive — sign test P ≈ 2×10⁻⁴, each delta within ~1 SE ≈ 0.027), while
`rrf − bm25` is +0.100 on average and positive 30/30. BM25 trails BGE by
~0.12 R@5 on average, and equal-weight RRF lets that weaker lexical ranking
dilute the dense one. At R@1 rrf vs bge is a wash (mean −0.007, 15 worse /
13 better / 2 ties). All three retrievers still improve with chunk size — the
"size > method" finding holds for the third stage running.
