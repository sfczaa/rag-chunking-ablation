# Retrieval-aware RAG Chunking

**Does *how* you split documents matter for RAG retrieval, or only *how big* the
pieces are?**

Smarter chunking is widely assumed to improve retrieval. This is a controlled
ablation that tests the assumption: a BiLSTM and a Transformer are trained to
predict topic boundaries, their predictions are turned into chunks that stay
size-comparable to a fixed-size baseline, and all three are then swept across
chunk sizes and overlaps and scored on **Natural Questions** by doc-constrained
Recall@k. Each stage changes exactly one variable and must reproduce the
previous stage's numbers before its own are read.

**[▶ Try the live demo](https://huggingface.co/spaces/sfczaa/rag-chunking-ablation-demo)**
· [Fine-tuned reranker](https://huggingface.co/sfczaa/bge-reranker-base-nq-ft)
· [Benchmark assets](https://huggingface.co/datasets/sfczaa/rag-chunking-ablation-demo-assets)

## Key findings

- **Chunk size dominates; the chunking method does not.** Over 30 configurations
  (3 methods × 5 sizes × 2 overlaps, n=1032 questions), moving from 6 to 15
  sentences is worth **+0.064 Recall@5** (p ≈ 3e-16) — roughly **18×** the
  largest chunking-method coefficient, which is not significant. Even *overlap*,
  a nuisance knob, outweighs the choice of method.
- **The tie is measured, not merely unproven.** At this sample size the smallest
  detectable between-method gap is 0.032, and the largest gap actually observed
  at any matched (size, overlap) cell is 0.023 — below the detection floor.
- **It replicates.** The result holds at 5× scale (200 → 1000 documents) and
  survives a change of dataset (NQ → TriviaQA).
- **The embedder was the real lever.** Swapping only the retrieval embedder
  (MiniLM → BGE), chunks held identical, improved **all 30/30** matched
  configurations. Hybrid BM25 + RRF did not help.
- **Fine-tuning the reranker was the one intervention that moved the sweet
  spot**: +0.107 Recall@1 in-domain. Cross-dataset it only recovers parity with
  plain dense retrieval — reported as damage control, not lift.

Negative and null results are reported as such. Every number below is backed by
an archived CSV under [`artifacts/results/`](artifacts/results), and each stage
links to its own write-up.

---

## How it works

| Step | What happens | Code |
|---|---|---|
| Corpus | Wikipedia articles become sentences with *real* section boundaries, parsed from raw wikitext rather than the cleaned dump | `rag_chunk/wiki_data.py` |
| Boundary models | A BiLSTM and a Transformer predict which sentences begin a new topic | `models/` · `rag_chunk/training.py` |
| Chunking | Those predictions become chunks by **target-size cutting**, which keeps learned chunks size-comparable to the fixed-size baseline | `rag_chunk/chunking.py` |
| Retrieval | Chunks and queries are embedded and indexed in FAISS (`IndexFlatIP` over L2-normalised vectors, so inner product is cosine) | `rag_chunk/retrieval.py` |
| Reranking | The dense top-k is reordered by a cross-encoder, off-the-shelf or fine-tuned on in-domain hard negatives | `rag_chunk/rerank.py` · `rerank_finetune.py` |
| Scoring | Doc-constrained Recall@k across a swept grid of sizes, overlaps and methods | `rag_chunk/metrics.py` · `sweep.py` |

Three choices do the work of keeping the comparison fair:

- **Target-size cutting.** A learned chunker that simply produces bigger chunks
  would "win" for the wrong reason. Learned cuts are constrained to a size
  window around the same target as the baseline, so the comparison is about
  *boundary quality*, not chunk length.
- **Doc-constrained Recall@k.** A short answer — a year, a common surname — can
  string-match a chunk from an unrelated document and inflate recall, unevenly
  across methods. A hit only counts when the matching chunk came from the
  question's gold document. The permissive number is still reported as
  `unconstrained` for reference.
- **Reproduce before concluding.** Every stage re-runs the previous stage's arm
  and must reproduce its archived numbers exactly before its own new arm is
  read. Several stages report 30/30 or 35/35 rows identical; where a check
  failed, that is written down instead of smoothed over.

---

## Results

### The whole project in one figure

![Best doc-constrained Recall@5 per stage](artifacts/results/portfolio/best_r5_evolution.png)

Seven stages on this axis, one controlled change each. Only two changes ever
moved the ceiling: the retrieval **embedder** (Stage 3, MiniLM → BGE, +0.054
R@5) and **in-domain fine-tuning of the reranker** (Stage 8, +0.044 R@5 and
+0.107 R@1 over the off-the-shelf reranker at the best config). Swapping the
chunking *method* (Stages 1–2), adding BM25/RRF hybrid retrieval (Stage 4),
and the *off-the-shelf* reranker (Stage 5, +0.010 R@5, within noise) never
did; the colored per-method dots stay clustered at every stage. Stage 6
re-runs the identical pipeline at 5× evaluation scale and everything
replicates — its lower absolute number reflects the 5× larger distractor pool
(new eval set), not a regression; Stage 8 shares that same eval set, so the
Stage 6 → 8 segment is directly comparable. Stage 7 repeats the bge-only
sweep on a *second dataset* (TriviaQA rc.wikipedia) and the size-over-method
headline replicates there too (4/4 direction checks — different dataset, so
it is not drawn on this axis).
Regenerate with `python scripts/14_evolution_plot.py` (reads only the archived
`stage*/final/` CSVs; writes `artifacts/results/portfolio/`).

### Stage by stage

Each row changes exactly one variable and had to reproduce the previous stage's
numbers before its own were read. Full write-ups, including the reproduction
checks and the negative results in detail, are linked in the last column.

| Stage | The one variable changed | What happened | Detail |
|---|---|---|---|
| **1–2** | Chunking method: fixed vs BiLSTM vs Transformer | At a matched size the three are statistically indistinguishable — best R@5 0.867 vs 0.857, a +0.010 gap against an SE of ±0.034 | [1](docs/stage1_chunking_optimizer.md) · [2](docs/stage2_transformer_boundary.md) |
| **3** | Retrieval embedder: MiniLM → BGE | **This is where the recall was.** Better on **30/30** matched configs (mean +0.054 R@5), ceiling 0.867 → 0.921 | [3](docs/stage3_bge_retrieval.md) |
| **4** | Ranking: + BM25, + RRF fusion | Hybrid does not help. RRF trails BGE-only on 24/30 configs (mean −0.021 R@5): fusing in a lexical ranking that is ~0.12 R@5 weaker dilutes the dense one | [4](docs/stage4_hybrid_retrieval.md) |
| **5** | + an off-the-shelf cross-encoder | Helps only with a shallow pool — reranking the top-20 adds +0.044 R@1 (better on 26/30, sign test *P* ≈ 1.5×10⁻⁵), while the top-50 never beats it | [5](docs/stage5_reranker.md) |
| **6** | Scale: 200 → 1000 documents | Every conclusion replicates, and the central one strengthens: r(size, R@5) rises 0.77 → **0.95** | [6](docs/stage6_large_eval.md) |
| **7** | Dataset: NQ → TriviaQA | The headline transfers: r = 0.80, methods still tied at size 15 (spread 0.020 < 2 SE = 0.036), best config still size 15 — **4/4** direction checks replicate | [7](docs/stage7_cross_dataset.md) |
| **8** | Reranker weights: → fine-tuned on NQ train | **The first intervention that moves the size-15 sweet spot.** ΔR@1 = +0.087…+0.107 across all five configs (2 SE = 0.030); at fixed 15/0, R@1 goes 0.629 → **0.736**, closing about a third of the gap to the 0.964 pool ceiling | [8](docs/stage8_reranker_finetune.md) |
| **8b** | The same reranker, evaluated on TriviaQA | Partial transfer only. Out of domain the *off-the-shelf* reranker actively hurts, and fine-tuning recovers to parity with plain dense retrieval rather than beating it — damage control, not lift | [transfer](docs/stage8_transfer.md) |

*Archive gap, noted rather than hidden:* Stage 1's own CSVs were overwritten
before being snapshotted and are unrecoverable. Its numbers survive exactly as
the fixed + BiLSTM rows of Stage 2's sweep, which re-ran the same deterministic
grid unchanged — see
[`artifacts/results/stage1/final/README.md`](artifacts/results/stage1/final/README.md).

Two follow-up analyses quantify the null rather than just asserting it: an OLS
fit of `R@5 ~ size + overlap + C(method)` over the 30 archived configurations,
and a minimum-detectable-effect calculation showing the observed method gap
sits below the detection floor at this sample size. Both read only the archived
CSVs and rerun no experiment —
[`artifacts/results/portfolio/effect_size_report.md`](artifacts/results/portfolio/effect_size_report.md).

## Implementation notes

- **Real section labels, not heuristics.** The cleaned Wikipedia dump drops
  `== Section ==` markers, so boundaries cannot be recovered from it. Article
  titles are streamed from the dump, the **raw wikitext** is fetched from the
  MediaWiki API (batched, cached, resumable) and parsed with `mwparserfromhell`;
  a sentence is labelled positive when it starts a new section.
- **Answer matching that cannot silently lie.** Document text and gold
  short-answer strings are reconstructed from the *same* token list with the
  same join rule, so a chunk covering the answer span substring-matches it after
  normalisation. A miss therefore means the answer's chunk really was not
  retrieved, rather than a formatting mismatch.
- **Class imbalance.** Boundary sentences are ~6% of the data; training uses a
  weighted BCE with `pos_weight = #neg/#pos` computed from the training split.
- **Caches invalidate themselves.** The corpus and every FAISS index store a
  manifest of the settings they were built with — including the embedding model,
  a hash of the trained weights and a hash of the corpus *content* — so a
  changed setting rebuilds the index instead of quietly scoring new settings
  against a stale one.

## Reproducing

Everything runs on a free Colab T4; no paid APIs are involved. The notebooks in
[`notebooks/`](notebooks) drive the pipeline end to end, and
`scripts/0_smoke_test.py` runs the whole chain at tiny scale first as a plumbing
check. The numbered scripts in [`scripts/`](scripts) are thin entry points — data
preparation, boundary training, the chunking sweeps, the embedder swap, hybrid
retrieval, reranking, the 5× scale run, the cross-dataset check, reranker
fine-tuning, and the analyses. `scripts/19_demo.py` serves the same demo as the
hosted Space locally, and [`deploy/`](deploy) holds the Space payload.

Two things are worth knowing before a rerun. Each stage writes to
`artifacts/results/latest/` and is snapshotted into `artifacts/results/<stage>/final/`
by `scripts/save_stage_results.py`; and the long stages are checkpointed and
resume-safe, because several take hours. There is no separate test suite — the
smoke test and the per-stage exact-reproduction checks are the verification
gates.

## Repository layout

```
├── rag_chunk/     chunking, boundary models, retrieval, reranking, metrics, sweeps
├── models/        BiLSTM and Transformer boundary architectures
├── scripts/       numbered entry points, one per pipeline step and analysis
├── notebooks/     Colab notebooks that drive the study
├── deploy/        Hugging Face Space payload and publishing scripts
├── docs/          per-stage write-ups
├── artifacts/
│   └── results/   archived, read-only evidence for every claim below
└── config.py      every path and tunable, read at call time
```

Data caches and model weights are not tracked; `artifacts/results/` is, because
the archived CSVs and figures are what the conclusions rest on.

## License

[MIT](LICENSE).
