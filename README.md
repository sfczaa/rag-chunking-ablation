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

Negative and null results are reported as such; the two figures below are the
short version, and every number links to an archived CSV under
[`artifacts/results/`](artifacts/results).

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

Evaluated on **200 NQ documents / 203 questions**, doc-constrained Recall@k, every
learned config size-matched to fixed-size (the Stage 2 sweep, MiniLM embeddings).

> **Headline: at a matched chunk size the three methods are statistically
> indistinguishable — Recall@k is driven by chunk _size_, not chunk _method_.**

| Method | Best Recall@5 | Best config |
|---|---|---|
| Fixed-size | 0.857 | size 15, overlap 1 |
| BiLSTM | 0.867 | target 15, overlap 0 |
| Transformer | 0.852 | target 12, overlap 0 |

- Best learned − best fixed = **+0.010 R@5**, but the standard error of that
  difference is **±0.034** (n = 203) → *z* ≈ 0.3, **not significant** (p > 0.05).
- Recall tracks size, not method: **Pearson r(avg chunk size, R@5) = +0.77**; going
  from ~8 to ~12 sentences adds **≈ +0.04 R@5**, whereas the method spread *at a
  matched size* is only 0.015–0.045 and has no stable winner (at size 8 the
  Transformer trails fixed; the leading method changes with size). A nuisance knob
  (overlap) swings one method's R@5 by up to 0.054 — larger than the method effect
  itself. A regression of R@5 on size + method dummies makes it explicit: **size
  contributes +0.07 across the swept range while both method effects stay ≤ 0.014
  (≈ 5× smaller)**. See
  [`artifacts/results/stage2/final/recall_vs_size_scatter.png`](artifacts/results/stage2/final/recall_vs_size_scatter.png).

**Why this is the ceiling (not a bug).** The training objective (Wikipedia
*section* pseudo-labels, weighted BCE) is misaligned with the evaluation
(retrieval Recall@k), and the **frozen MiniLM sentence embeddings carry only a weak
topic-boundary signal** (boundary precision ≈ 0.13, only ~1.4× the base rate). The
high-leverage gains live in the *embedder* and a *retrieval-aligned* objective —
not a bigger boundary network.

**Stage 2.1 — calibration caught a real failure.** The Transformer first reported
Boundary F1 ≈ 0. The added validation threshold sweep + probability diagnostics
showed its sigmoid outputs had collapsed to a near-constant (~0.55): unit-amplitude
positional encoding was swamping the small frozen MiniLM vectors (~10–40×). An
input `LayerNorm` restores the content↔position balance, after which the model
becomes discriminative and competitive (table above). Details:
[`docs/stage2_transformer_boundary.md`](docs/stage2_transformer_boundary.md).

### Stage 3 — swapping the retrieval embedder (MiniLM → BGE)

A controlled ablation on top of the Stage 2 sweep: **only the retrieval embedding
model changes** (`all-MiniLM-L6-v2` → `BAAI/bge-base-en-v1.5`) while chunks, chunk
sizes, corpus and questions are held identical — every matched row has
Δ avg-chunk-size ≈ 0 and Δ n-chunks = 0. This is the direct test of the Stage 2
prediction that the high-leverage knob is the *embedder*, not the boundary network.

> **Headline: the embedder is where the recall is. BGE beats MiniLM on all 30 / 30
> matched configs (mean +0.054 R@5, +0.080 R@1) and lifts the ceiling from 0.867 to
> 0.921 R@5 — a large, consistent effect, in contrast to the within-noise Stage 2
> _method_ differences.**

| Retrieval embedder | Best Recall@5 | Best config | Mean Δ vs MiniLM |
|---|---|---|---|
| MiniLM (Stage 2) | 0.867 | bilstm target 15, overlap 0 | — |
| BGE-base (Stage 3) | **0.921** | fixed size 15, overlap 0 | +0.054 R@5, +0.080 R@1 |

- **Consistent, not cherry-picked.** All 30 matched configs improve on R@1, R@3 and
  R@5 (30/30 each); a sign test alone gives *P* ≈ 9×10⁻¹⁰ under the no-effect null.
  At the best config BGE adds **+0.113 R@5** — ≈ 23 of 203 questions — a paired
  *z* ≳ 3.4 (contrast Stage 2's *z* ≈ 0.3).
- **The "size > method" finding survives the swap.** *Within* BGE the three methods
  still tie at matched size (at size 15 / overlap 0: fixed 0.921, transformer 0.916,
  bilstm 0.902 — inside one SE ≈ 0.019) and larger chunks still win. So the Stage 2
  conclusion is robust to a much stronger embedder; the embedder is a *separate,
  bigger* lever stacked on top.
- **Same fair-comparison harness.** Identical chunks / metric / questions; BGE
  queries use its instruction prefix and L2-normalised cosine. Full matched table:
  `stage2_vs_stage3_matched.csv`; details in
  [`docs/stage3_bge_retrieval.md`](docs/stage3_bge_retrieval.md).

### Stage 4 — hybrid retrieval (BGE vs BM25 vs RRF over identical chunks)

A retriever ablation on top of Stage 3: chunks, sweep configs, corpus and
questions all held identical — only the *ranker* changes. Three retrievers per
config: BGE dense (the exact Stage 3 path), pure-numpy Okapi BM25 over the same
chunk texts, and Reciprocal Rank Fusion of the two (`RRF_K=60`, depth 50). The
re-run BGE arm reproduces the archived Stage 3 baseline **exactly** (30/30
configs: Δ n-chunks = 0, every recall delta 0.0000), so all three retrievers are
scored against a verified-identical baseline.

> **Headline: hybrid does not help here. BGE-only keeps the ceiling (0.921 R@5);
> equal-weight RRF trails it slightly but consistently (mean −0.021 R@5, worse on
> 24 / 30 configs), because fusing in a much weaker lexical ranking — BM25 trails
> BGE by ~0.12 R@5 on average — dilutes the dense one.**

| Retriever | Best Recall@5 | Best config |
|---|---|---|
| BGE-only | **0.921** | fixed size 15, overlap 0 |
| RRF (BGE + BM25) | 0.887 | fixed size 15, overlap 0 |
| BM25-only | 0.803 | bilstm target 15, overlap 0 |

- **Per-config deltas are small but one-sided at R@5.** rrf − bge: mean −0.021,
  range −0.054…+0.035 — each within ~1 SE ≈ 0.027, but 24 of 30 configs are
  negative (sign test *P* ≈ 2×10⁻⁴), so the direction is consistent even though
  the size is marginal. At R@1 it is a wash (mean −0.007; 15 worse / 13 better /
  2 ties).
- **RRF beats BM25 everywhere** (30/30, mean +0.100 R@5): fusion recovers most
  of the dense quality — it is "safe" relative to BM25 — it just adds nothing on
  top of a strong dense retriever on this benchmark.
- **"Size > method" survives a third time.** All three retrievers gain with
  larger chunks (`hybrid_recall_vs_chunk_size.png`) and every retriever's best
  config sits at size 15.
- Artifacts: `hybrid_sweep_results.csv` (90 rows), `hybrid_retriever_matched.csv`,
  `stage3_vs_stage4_bge_check.csv`; details in
  [`docs/stage4_hybrid_retrieval.md`](docs/stage4_hybrid_retrieval.md).

### Stage 5 — cross-encoder reranking (BGE top-k + bge-reranker-base)

A ranking ablation on top of Stage 3/4: chunks, sweep configs, corpus, questions
and the BGE retriever all held identical — the only change is that the BGE top-k
candidate pool is *reordered* by the off-the-shelf cross-encoder
`BAAI/bge-reranker-base` (no training or fine-tuning) before the top-5 cut.
Three arms per config: `bge` (the exact Stage 3 path — re-verified to reproduce
the archived baseline exactly, 30/30), `rerank20`, and `rerank50`. The
candidate-pool ceiling is reported before any improvement claim: the answer
chunk is in the BGE top-20 for **96.5%** of questions on average (98.5% at
top-50), so ranking — not pool recall — is the binding constraint.

> **Headline: reranking is the first post-Stage-3 change that actually helps —
> but only with a *shallow* pool. Reranking the BGE top-20 adds a mean
> +0.044 R@1 (better on 26 / 30 configs, sign test *P* ≈ 1.5×10⁻⁵); widening
> the pool to top-50 only feeds the reranker distractors — it never beats
> top-20 on R@1 (0 / 30, *P* ≈ 7×10⁻⁹).**

| Arm | Best Recall@5 | Best config | Mean Δ vs BGE-only (R@1 / R@3 / R@5) |
|---|---|---|---|
| BGE-only | 0.921 | fixed size 15, overlap 0 | — |
| BGE top-20 + CE | **0.931** | fixed size 15, overlap 1 | +0.044 / +0.028 / +0.018 |
| BGE top-50 + CE | 0.916 | fixed size 15, overlap 1 | +0.024 / +0.019 / +0.013 |

- **The gain lives at small chunk sizes.** Mean rerank20−bge R@1 falls from
  **+0.075 at size ~6 to +0.002 at size 15**: the cross-encoder largely rescues
  badly-chunked (small) configs but adds nothing at the size-15 sweet spot,
  where the deltas sit inside noise (at the Stage 3 best config: +0.010 R@1,
  −0.010 R@3, −0.025 R@5; 1 SE ≈ 0.034). In
  `rerank_recall_vs_chunk_size.png` the rerank20 trend line is nearly flat —
  reranking *flattens the chunk-size effect* on R@1 rather than raising the
  ceiling.
- **Deeper pool ≠ better.** rerank50 sees strictly more candidates yet is
  ≤ rerank20 on R@1 in every one of the 30 configs (28 worse, 2 tied): a
  candidate past rank 20 is far more likely a distractor than the answer chunk
  (the pool ceiling gains only +0.02), and the cross-encoder sometimes promotes
  it over the true one.
- **Cost.** ≈ 24.7 ms per (question, chunk) pair on a T4 → ≈ 0.49 s per query
  at depth 20 and ≈ 1.18 s at depth 50 — real latency for gains that are
  noise-level at the best config; worth paying only when stuck with small
  chunks.
- Artifacts: `rerank_sweep_results.csv` (90 rows), `rerank_matched.csv`,
  `stage3_vs_stage5_bge_check.csv` (30/30 exact); details in
  [`docs/stage5_reranker.md`](docs/stage5_reranker.md).

### Stage 6 — larger-scale robustness check (1000 docs / 1032 questions)

Everything held identical to Stages 3–5 (dataset source, chunking grids,
boundary models/weights, BGE retriever, off-the-shelf reranker at top-20 only);
the **only** change is corpus scale, 200 → 1000 NQ docs (203 → 1032 questions),
which halves the sampling noise (1 SE ≈ 0.034 → 0.015). Before the large run, a
check mode re-ran the full pipeline at N=200 and reproduced the archived
Stage 5 rows **exactly** (35/35 rows, all deltas 0.0000) — so any difference at
scale is the data, not the code.

> **Headline: all four Stage 1–5 conclusions replicate at 5× scale — and the
> central one gets *stronger*: Pearson r(chunk size, R@5) rises from 0.77 to
> 0.95. The reranker's small-chunk rescue shrinks from +0.113 to +0.036 R@1
> (still > 2 SE), and its gain at the size-15 sweet spot stays zero.**

| # | Claim (from n=203) | Observed at n=1032 | Replicates |
|---|---|---|---|
| 1 | chunk size matters more than chunking method | r(size, R@5) = **0.95**; size effect 0.052 > method spread | yes |
| 2 | BGE-only stays a strong baseline | best R@5 = **0.881** (fixed 15/0; 0.921 at 200 docs — a drop is expected with 5× more distractor docs) | yes |
| 3 | rerank20 helps mainly at small chunks | ΔR@1 = **+0.036** at fixed 6/0 (2 SE = 0.031) vs ≈ 0 at size 15 | yes |
| 4 | rerank20 adds nothing at the size-15 sweet spot | max size-15 ΔR@1 = **+0.001** | yes |

- Best config overall is unchanged: **fixed size 15, overlap 0** —
  R@1 0.628 / R@3 0.816 / R@5 0.881 with BGE alone; reranking it moves R@1 by
  +0.001. The size-15 pool ceilings stay ≥ 0.95 (`pool_recall@20`
  0.957–0.964), so the candidate pool is still not the bottleneck.
- `stage6_size_vs_recall.png` shows the small and large evals side by side —
  same upward size trend, methods intermixed along it at both scales;
  `stage6_rerank_delta.png` shows the rerank20 gain collapsing toward zero
  everywhere except the deliberately-bad small-chunk config.
- Artifacts: `stage6_large_eval_results.csv` (35 rows with actual corpus
  counts), `stage6_matched_summary.csv`, `stage6_direction_check.csv`,
  `stage6_check_vs_stage5.csv` (35/35 exact), 2 figures; details in
  [`docs/stage6_large_eval.md`](docs/stage6_large_eval.md).

### Stage 7 — cross-dataset robustness check (TriviaQA rc.wikipedia)

The last remaining threat to the headline claim was dataset specificity:
every number so far came from NQ + its Wikipedia pages. Stage 7 re-runs the
identical 30-config **bge-only** sweep (same grids, boundary weights, BGE
retriever, doc-constrained metric) on **TriviaQA `rc.wikipedia`** — 300 kept
questions against 472 full Wikipedia entity pages bundled in the dataset (no
fetching). Gold documents are the question's entity pages whose text contains
the answer string — *distant supervision*, a weaker gold notion than NQ's
annotated documents, stated wherever these numbers appear; 118/300 questions
have two gold pages, handled by a backwards-compatible multi-gold extension
of the metric. Before the run, a check mode re-ran the sweep on the cached NQ
corpus and reproduced the archived Stage 3 rows **exactly** (30/30, all
deltas 0.0000) — so the code path, including the metric extension, is
verifiably the Stage 3 pipeline.

> **Headline: the size-over-method conclusion transfers. On TriviaQA,
> r(size, R@5) = 0.80, the methods stay tied at matched size (max size-15
> spread 0.020 < 2 SE = 0.036), and the best config is again a size-15 one —
> 4/4 direction checks replicate.**

| Eval | Best config | Best R@5 | r(size, R@5) | Size effect (6→15) |
|---|---|---|---|---|
| NQ (Stage 3, 203 q) | fixed 15/0 | 0.9212 | 0.86 | +0.062 |
| TriviaQA (Stage 7, 300 q) | bilstm t15/1 | 0.9200 | 0.80 | +0.036 |

- **The "winner" flips — which reinforces the tie.** Fixed won on NQ; BiLSTM
  edges ahead on TriviaQA by 0.013–0.020, well inside 2 SE. A method that
  genuinely won would not swap places across datasets; noise would.
- **The size effect is flatter here** (+0.036 vs +0.062 R@5 from size 6 to
  15): trivia questions against entity pages are already easy at small chunks
  (R@5 ≈ 0.87 at size ~6), leaving less headroom. Direction unchanged,
  magnitude dataset-dependent — reported as is.
- **Loader accountability:** 302 rows scanned → 300 kept (2 dropped with no
  answer candidate in their pages); median document 152 sentences, so the
  6–15 sentence grid stays meaningful (the degenerate-sweep guard that
  disqualified raw HotpotQA paragraphs — see the docs for why HotpotQA was
  rejected).
- Artifacts: `stage7_cross_dataset_results.csv` (30 rows),
  `stage7_matched_summary.csv`, `stage7_direction_check.csv`,
  `stage7_check_vs_stage3.csv` (30/30 exact), 1 figure; details in
  [`docs/stage7_cross_dataset.md`](docs/stage7_cross_dataset.md).

### Stage 8 — fine-tuning the cross-encoder reranker (NQ train split)

Stage 6 left a precise diagnosis: at the size-15 sweet spot the answer chunk
is in the BGE top-20 pool for ~96% of questions (`pool_recall@20`
0.957–0.964) yet R@1 ≈ 0.63, and the off-the-shelf reranker adds +0.001 —
**ranking, not pool recall, is the bottleneck**. Stage 8 tests the one model
change with theoretical headroom: fine-tune `BAAI/bge-reranker-base` on the
NQ **train** split (every eval bench uses the validation split). 2161 train
questions were mined into 2034 (1 positive + 7 hard-negative) groups at the
deployment chunking (fixed 15/0, BGE top-20 pool); training was 2 epochs of
listwise cross-entropy, 16.6 min on a T4. A held-out dev bench (400 docs /
406 questions) gated the expensive final eval: dev ΔR@1 (ft − ots) = +0.047 ≥
the pre-registered +0.02 threshold → **GO**. In the final run the `bge` and
`rerank20` arms re-ran the exact Stage 6 pipeline and reproduced
`stage6/final` **exactly** (10/10 rows, all deltas 0.0000) before the
fine-tuned arm was read.

> **Headline: the first intervention that works at the size-15 sweet spot.
> Fine-tuned − off-the-shelf ΔR@1 = +0.087…+0.107 across all five configs
> (2 SE = 0.030) — at fixed 15/0, R@1 goes 0.629 → 0.736, closing about a
> third of the gap to the 0.964 pool ceiling. R@3/R@5 rise too.**

| Config (1032 q) | bge R@1 | ots rerank R@1 | **ft rerank R@1** | ft − ots |
|---|---|---|---|---|
| fixed 6/0 | 0.509 | 0.545 | 0.650 | +0.106 |
| fixed 15/0 | 0.628 | 0.629 | **0.736** | **+0.107** |
| fixed 15/1 | 0.626 | 0.627 | 0.720 | +0.093 |
| bilstm 15/0 | 0.630 | 0.627 | 0.714 | +0.087 |
| transformer 15/0 | 0.637 | 0.625 | 0.732 | +0.107 |

- **The headline conclusions survive intact**: even with the fine-tuned
  reranker, chunk size still dominates (ft R@1 0.650 at size 6 vs 0.736 at
  size 15) and the size-15 methods still tie (ft R@5 spread 0.9099–0.9244,
  ≈ 2 SE). Fine-tuning lifts the whole curve; it does not change its shape.
- **Honest caveats:** the gain is *in-domain* (trained on NQ train, evaluated
  on NQ validation) — no transfer claim to other datasets is made. Questions
  are split-disjoint, but popular Wikipedia pages can appear in both splits'
  corpora (standard for NQ). Reranking costs ~0.56 s/question at depth 20 on
  a T4, unchanged by fine-tuning.
- Artifacts: `stage8_ft_eval_results.csv` (15 rows),
  `stage8_matched_summary.csv`, `stage8_dev_results.csv` + gate verdict,
  `stage8_check_vs_stage6.csv` (10/10 exact), 1 figure; model +
  `training_meta.json` under `models/bge_reranker_ft/`; details in
  [`docs/stage8_reranker_finetune.md`](docs/stage8_reranker_finetune.md).

### Post-Stage 8 robustness — effect sizes + cross-dataset reranker transfer

Two follow-ups that harden the conclusions without re-running the sweep.

**How big is the "method ties" null?** An OLS fit `R@5 ~ size + overlap +
C(method)` over the 30 archived Stage 6 dense configs puts the size effect at
+0.064 across 6→15 sentences (p ≈ 3e-16) — about **18×** the largest
chunking-method coefficient (0.0036, not significant, p = 0.23–0.87); overlap,
a nuisance knob, outweighs the method choice too. And the tie is not
underpowering: at n = 1032 the smallest between-method gap detectable at 95% is
0.032, yet the largest observed gap at any matched (size, overlap) cell is
0.023 — below that floor. So "size dominates, method ties" is a *measured*
null, not a lack of power. (`scripts/20_effect_size.py` →
`artifacts/results/portfolio/effect_size_*`.)

**Does the fine-tuned reranker's gain transfer?** Re-running the Stage 8
protocol on the TriviaQA bench (300 q) instead of NQ: the off-the-shelf
reranker *hurts* out-of-domain (ots − bge = −0.03…−0.07 R@1), and fine-tuning
recovers to **parity with the dense baseline** (ft − bge ≈ 0 at size 15) rather
than beating it as it did in-domain (+0.107). The +0.057 win over the
off-the-shelf reranker (fixed 15/0, just over 2 SE = 0.054, positive at 4/5
configs) is therefore **damage control, not net lift** — an honest, partial
transfer. The headline is untouched: chunking method still ties under the
fine-tuned reranker cross-dataset. Details in
[`docs/stage8_transfer.md`](docs/stage8_transfer.md).

---

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

## Stages

| Stage | One variable changed | Write-up |
|---|---|---|
| 1 | Fixed vs BiLSTM chunking under MiniLM | [stage1](docs/stage1_chunking_optimizer.md) |
| 2 | Adds a Transformer boundary model | [stage2](docs/stage2_transformer_boundary.md) |
| 3 | Retrieval embedder: MiniLM → BGE | [stage3](docs/stage3_bge_retrieval.md) |
| 4 | Ranking: BGE vs BM25 vs RRF over identical chunks | [stage4](docs/stage4_hybrid_retrieval.md) |
| 5 | Adds an off-the-shelf cross-encoder reranker | [stage5](docs/stage5_reranker.md) |
| 6 | Scale: 200 → 1000 documents | [stage6](docs/stage6_large_eval.md) |
| 7 | Dataset: NQ → TriviaQA | [stage7](docs/stage7_cross_dataset.md) |
| 8 | Reranker weights: off-the-shelf → fine-tuned | [stage8](docs/stage8_reranker_finetune.md) |
| 8b | Where that fine-tuning does and does not transfer | [stage8 transfer](docs/stage8_transfer.md) |

*Archive gap, noted for honesty:* Stage 1's own CSVs were overwritten before
being snapshotted and are unrecoverable. Its numbers survive exactly as the
fixed + BiLSTM rows of Stage 2's sweep, which re-ran the same deterministic grid
unchanged — see [`artifacts/results/stage1/final/README.md`](artifacts/results/stage1/final/README.md).

## License

[MIT](LICENSE).
