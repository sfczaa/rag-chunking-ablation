# Stage 1 — Chunking Sweep Optimizer

**Status: implemented.**

Stage 1 turns the project from a single fixed-vs-learned comparison into a fair
*optimizer*. It answers one question:

> At the same approximate chunk size and overlap, does learned **target-size**
> semantic cutting choose better boundaries than fixed-size cutting?

It does **not** assume learned chunking wins — it controls for chunk size and
overlap, then measures which strategy retrieves answer-bearing chunks better and
exports the best configuration it finds.

## What it adds

- **Target-size semantic cutting** (`rag_chunk/chunking.py::semantic_target_chunks`):
  within each chunk's `[min, max]` window the BiLSTM picks the boundary it is most
  confident about (ties broken toward the target size), keeping learned chunks
  size-comparable to the fixed-size baseline.
- **Fixed-size chunking with overlap** (`fixed_chunks`), so both methods are
  compared under matching overlap settings.
- **Sweep** (`rag_chunk/sweep.py`): builds an in-memory FAISS index per config,
  scores NQ doc-constrained Recall@k, and ranks configs. The boundary model runs
  once per document and is reused across every learned config; per-config indices
  are discarded after scoring (nothing is cached under `nq/` unless you opt in).
- **Cache manifest** (`rag_chunk/retrieval.py::current_signature`) records the
  full chunking/model/embedding configuration so a stale FAISS cache is never
  silently reused.

## How to run

```bash
python scripts/6_sweep_chunking.py            # full grid
python scripts/6_sweep_chunking.py --quick    # smaller, fast grid
python scripts/6_sweep_chunking.py --save-run  # also snapshot to results/runs/<ts>_sweep/
```

Or in the notebook: **Phase 6 — Chunking sweep optimizer**.

The sweep reuses the cached trained model and NQ corpus, so it does not require
re-running Phases 1–3.

## Outputs

Written to `artifacts/results/latest/` (overwritten by each new run):

| File | Contents |
|---|---|
| `sweep_results.csv` | one row per swept config: method, size/overlap (or target/min/max/overlap), doc-constrained + unconstrained Recall@k, avg chunk size, chunk count |
| `best_config.json` | the validation-best config (full row) + the ranking metric |
| `fair_comparison_table.csv` | each learned row paired with the closest-avg-size fixed row at the same overlap, with Recall@1/3/5 deltas |
| `recall_vs_chunk_size.png` | doc-constrained Recall@5 vs avg chunk size, one line per (method × overlap) |
| `model_comparison.png` | best fixed vs best learned, grouped Recall@k |

## Archiving a finished run

`latest/` is the working folder. To preserve a run as a Stage 1 snapshot:

```bash
python scripts/save_stage_results.py --stage stage1
```

This copies everything from `results/latest/` into `results/stage1/final/`
(copy only — nothing in `latest/`, `runs/`, or `data/` is moved or deleted).

## Key config knobs (`config.py`)

| Knob | Default | Meaning |
|---|---|---|
| `SEMANTIC_CHUNK_POLICY` | `"target"` | `"target"` (size-matched) or `"threshold"` |
| `SEMANTIC_TARGET/MIN/MAX_CHUNK_SIZE` | 10 / 6 / 12 | target-size cutting window |
| `SEMANTIC_OVERLAP` | 1 | overlap between learned chunks |
| `FIXED_CHUNK_SIZE` / `FIXED_CHUNK_OVERLAP` | 10 / 1 | fixed baseline |
| `FIXED_SIZE_GRID` / `TARGET_SIZE_GRID` / `OVERLAP_GRID` | `[6,8,10,12,15]` / `[6,8,10,12,15]` / `[0,1]` | sweep grids |

## Extension points left for later stages

`MODEL_TYPE` (`bilstm` / `transformer`), `BOUNDARY_EMBED_MODEL`, and
`RETRIEVAL_EMBED_MODEL` flow through the config, cache manifest, and sweep rows,
so later stages can plug in without reworking the optimizer. See
[`stage2_transformer_boundary.md`](stage2_transformer_boundary.md) and
[`stage3_bge_retrieval.md`](stage3_bge_retrieval.md).
