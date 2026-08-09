# Stage 8 addendum - cross-dataset transfer of the fine-tuned reranker

**Status: complete.** Run on Colab (T4) 2026-07-25. Stage 8 fine-tuned
`BAAI/bge-reranker-base` on the NQ **train** split and measured ft-ots
ΔR@1 = +0.087…+0.107 *in-domain* (NQ val). Its stated caveat was that no
transfer claim was made. This addendum answers that caveat directly: it re-runs
the **exact Stage 8 final protocol** — same 5 configs, same three arms
(`bge` / off-the-shelf `rerank20` / fine-tuned `rerank20_ft`) sharing one BGE
top-20 pool, same fine-tuned checkpoint — but on the **Stage 7 TriviaQA
rc.wikipedia bench** (472 docs / 300 questions) instead of NQ. Only the eval
dataset changes, so any difference is a transfer effect, not a protocol
difference. Archived to `artifacts/results/stage8/final/`.

## The one number the `ft − ots` verdict hides

The script's automatic verdict compares the fine-tuned reranker to the
**off-the-shelf** one (`ft − ots`) and, at fixed 15/0, reports
+0.0567 > 2 SE (0.0538) → "TRANSFERS". That is true but incomplete. The
honest reading also needs `ft − bge` — the fine-tuned reranker versus **no
reranking at all** (the raw dense order):

| config (300 q) | bge R@1 | ots R@1 | ft R@1 | **ft − ots** | **ft − bge** | ots − bge |
|---|---|---|---|---|---|---|
| fixed 6/0 | 0.6200 | 0.5800 | 0.6400 | +0.0600 | +0.0200 | −0.0400 |
| **fixed 15/0** | 0.7067 | 0.6533 | 0.7100 | **+0.0567** | **+0.0033** | −0.0534 |
| fixed 15/1 | 0.7033 | 0.6667 | 0.7033 | +0.0367 | 0.0000 | −0.0367 |
| bilstm 15/0 | 0.7000 | 0.6300 | 0.7033 | +0.0733 | +0.0033 | −0.0700 |
| transformer 15/0 | 0.6767 | 0.6467 | 0.7167 | +0.0700 | +0.0400 | −0.0300 |

2 SE at n=300 ≈ **0.0538** (wide — small eval set). In-domain (NQ) reference at
fixed 15/0: ft − ots = **+0.1066**, and ft − bge = **+0.107** (the off-the-shelf
reranker was neutral there, ots − bge ≈ +0.001).

## Findings

1. **The off-the-shelf reranker generalises poorly — it *hurts* on TriviaQA.**
   `ots − bge` is negative at every config (−0.03…−0.07 R@1): reranking the
   dense pool with the un-tuned cross-encoder makes the top-1 *worse* than
   plain dense retrieval. (In-domain it was merely neutral.)
2. **Fine-tuning transfers as damage control, not as net lift.** The
   fine-tuned reranker recovers to **parity with the dense baseline**
   (`ft − bge` ≈ 0.000–0.004 at size 15, +0.02–0.04 at the extremes), so its
   +0.057 win over the off-the-shelf reranker is real but is mostly *undoing
   the off-the-shelf reranker's out-of-domain degradation* — not adding new
   value over doing no reranking. Contrast in-domain, where `ft − bge` = +0.107.
3. **Direction is consistent; magnitude is ~half.** All 5 configs give a
   positive `ft − ots` (+0.037…+0.073); 4/5 clear 2 SE (only fixed 15/1 sits
   inside the band). The transfer magnitude is about half the in-domain one
   (+0.057 vs +0.107 at fixed 15/0). The consistent sign across configs is more
   convincing than the primary config's razor-thin crossing of 2 SE.
4. **The headline is unaffected.** Under the fine-tuned reranker, cross-dataset,
   the size-15 chunking methods still tie: the `ft − ots` spread across
   fixed/bilstm/transformer at size 15/0 is 0.0567–0.0733 (range 0.0166) ≪ 2 SE.
   "Size dominates, method ties" is untouched — this only probes the reranking
   lever.

## Honest caveats

- TriviaQA gold is **distant-supervised** (answer-string match on the entity
  pages), weaker than NQ's annotated gold; absolute recall is **not comparable**
  to the NQ bench.
- n = 300 makes the 2 SE band ~0.054, so the fixed-15/0 "TRANSFERS" label rests
  on a +0.0567 vs 0.0538 margin — report it as *partial / damage-control
  transfer with a consistent positive direction*, not a clean win.
- The fine-tuned checkpoint used here was retrained from the same NQ-train data,
  recipe and seed (42) as the original Stage 8 run (its ~1.1 GB weights had been
  lost from Drive); the in-domain column is the original Stage 8 archive. They
  match within training noise.

## Artifacts

`artifacts/results/stage8/final/`: `stage8_transfer_results.csv` (15 rows),
`stage8_transfer_matched.csv` (5 rows, with the in-domain column),
`stage8_transfer_delta.png`, `stage8_transfer_summary.md`, and the run
checkpoint. Produced by `scripts/21_reranker_transfer.py`.
