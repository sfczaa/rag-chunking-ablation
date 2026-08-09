# Stage 2 — Transformer Boundary Model

**Status: implemented (Stage 2 + 2.1).** Adds a Transformer boundary detector as
a **second** learned chunking model alongside the BiLSTM, for a fair three-way
comparison against the fixed-size baseline. Stage 2.1 adds a pairwise boundary
head and calibrates the Transformer's Boundary-F1 threshold (so it no longer reads
0 at the BiLSTM-tuned 0.8). Stage 1 is untouched.

Out of scope (deliberately, this stage): BGE embedding migration, CrossEncoder
reranker, BM25 / RRF, RL, and large-dataset expansion. The embedding model stays
MiniLM (`EMBED_DIM = 384`).

## What it adds

- `models/transformer_boundary.py::TransformerBoundary` — a 2-layer Transformer
  encoder over the same MiniLM sentence embeddings, with sinusoidal positional
  encoding (self-attention is order-agnostic) and the **same interface** as the
  BiLSTM (`forward` / `predict_proba` / `predict_boundaries`), so chunking,
  retrieval and the sweep treat both models identically. Output length is
  `len(sentences) - 1`, one logit per inter-sentence boundary.
- **Input LayerNorm (fixes a collapse).** The frozen MiniLM vectors are small per
  component (RMS ~0.05) while the sinusoidal PE has amplitude ~1, so adding raw PE
  swamped the content ~10–40× and the encoder collapsed to a *constant* predictor:
  the first cut reported Boundary F1 ≈ 0 / a degenerate "cut-everywhere" baseline
  (precision = positive-rate, recall = 1.0). A `LayerNorm` on the input rescales
  each sentence vector to unit per-component variance before PE is added, so the
  topic-change signal survives independent of the embedder's output magnitude.
- **Pairwise boundary head.** Each boundary's logit is read from a richer view of
  the two adjacent encoded states — `[h_i ; h_{i+1} ; |h_i − h_{i+1}| ; h_i · h_{i+1}]`
  → `Linear(4d → 1)`. The absolute-difference and element-wise product give the
  classifier an explicit *dissimilarity* signal (the natural cue for a topic
  change) rather than asking a plain `concat` to recover it. *(This changes the
  saved weight shape, so retrain the Transformer before sweeping — see below.)*
- It uses the **same target-size semantic cutting policy** as the BiLSTM
  (`chunking.semantic_target_chunks`) — only the boundary scorer differs.
- `MODEL_TYPE` now supports `"bilstm"` and `"transformer"`. The default stays
  `"bilstm"` so Phase 4/5 and the Stage 1 sweep are unchanged.

Config (in `config.py`):

```python
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4        # must divide EMBED_DIM (384 % 4 == 0)
TRANSFORMER_FF_DIM = 512
TRANSFORMER_DROPOUT = 0.1
```

## Weights

Each model has its own weight file (they coexist):

- BiLSTM: `models/bilstm_best.pt` (unchanged)
- Transformer: `models/transformer_best.pt`

## Train the Transformer

Reuses the **same** cached MiniLM embeddings + Wikipedia section labels from
Phases 1–3 — no Phase 1/2 rerun.

```bash
python scripts/7_train_transformer.py            # or --epochs N
```

or in code / the notebook (Phase 7):

```python
from rag_chunk import training
training.train_model(model_type="transformer")   # -> models/transformer_best.pt
```

## Stage 2.1 — boundary-threshold calibration

The shared `BOUNDARY_THRESHOLD = 0.8` was tuned for the BiLSTM. The Transformer's
sigmoid outputs sit in a different range, so at 0.8 it could report **Boundary
F1 = 0** even when the learned boundaries were fine — a calibration artifact, not
a model failure.

Phase 7 training now calibrates the threshold automatically (transformer only):

1. collect per-boundary probabilities over the **validation** split (one forward
   pass per article),
2. sweep thresholds `0.01 … 0.99` and pick the one with the **max boundary F1**
   (ties → higher threshold),
3. re-report the held-out **test** F1 at that threshold and stash it in
   `C.TRANSFORMER_BOUNDARY_THRESHOLD`,
4. write three diagnostics to `artifacts/results/latest/`:
   - `transformer_threshold_f1.csv` — precision/recall/F1 at every threshold,
   - `transformer_boundary_diagnostics.json` — probability distribution (overall,
     true-boundary vs non-boundary, percentiles, histogram) + the chosen threshold
     and a reference showing what the old 0.8 scored,
   - `transformer_boundary_threshold_f1.png` — F1 / precision / recall vs
     threshold, with the calibrated and old-fixed thresholds marked.

`train_model(model_type="transformer")` returns these under `stats["calibration"]`,
and `stats["test_boundary_f1"]` is now read at the calibrated threshold.

> **Scope:** this is the Boundary-F1 **diagnostic** only. The retrieval sweep
> (Phases 6 & 8) cuts with the target-size **argmax** policy
> (`semantic_target_chunks`) and never reads a probability threshold, so Stage 1
> and the Stage 2 retrieval comparison are **unchanged** by calibration. The
> BiLSTM path (`metrics.boundary_f1` at `BOUNDARY_THRESHOLD`) is untouched.

## Sweep Fixed vs BiLSTM vs Transformer

```bash
python scripts/8_sweep_with_transformer.py            # full grid
python scripts/8_sweep_with_transformer.py --quick    # fast grid
python scripts/8_sweep_with_transformer.py --save-run  # also snapshot to results/runs/
```

or in code / the notebook (Phase 8):

```python
from rag_chunk import sweep, training
bilstm = training.load_model("bilstm")
transformer = training.load_model("transformer")
rows = sweep.run_sweep(bilstm, transformer_model=transformer)
```

Fairness conditions (all three methods): same MiniLM embeddings, same cached NQ
docs/questions, same Recall@k metric, same target-size/min/max/overlap policy,
compared by similar average chunk size.

`scripts/6_sweep_chunking.py` is unchanged — it still runs the Stage 1
fixed + BiLSTM sweep.

## Outputs

Written to `artifacts/results/latest/` (same files as Stage 1, now including the
transformer rows / series): `sweep_results.csv`, `best_config.json`,
`fair_comparison_table.csv`, `recall_vs_chunk_size.png`, `model_comparison.png`,
`recall_vs_size_scatter.png`.

- `recall_vs_chunk_size.png` gains Transformer overlap=0/1 series.
- `model_comparison.png` shows the best fixed vs best BiLSTM vs best Transformer.
- `recall_vs_size_scatter.png` — Recall@k vs avg chunk size, coloured by method,
  with the size-trend line + Pearson *r* and the 1-SE noise floor. This is the
  figure behind the headline finding (size dominates, methods overlap; see the
  [README Results](../README.md#results) for the numbers).
- `best_config.json` is chosen across all three methods.
- `fair_comparison_table.csv` keeps its Stage 1 schema (BiLSTM-vs-fixed); the
  transformer's size-matched comparison is read from the plots and `best_config`.

Phase 7 (Stage 2.1) additionally writes the boundary-calibration diagnostics to
the same folder: `transformer_threshold_f1.csv`,
`transformer_boundary_diagnostics.json`, `transformer_boundary_threshold_f1.png`.

Archive a finished run:

```bash
python scripts/save_stage_results.py --stage stage2   # -> results/stage2/final/
```

`artifacts/data/` and `artifacts/results/runs/` are never moved or deleted.

Stage 3 starts only after this archive exists. It keeps the same fixed/BiLSTM/
Transformer chunking outputs and swaps only the retrieval embedding model to BGE;
see [`stage3_bge_retrieval.md`](stage3_bge_retrieval.md).
