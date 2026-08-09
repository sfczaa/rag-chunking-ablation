# Stage 3 - BGE Retrieval Embedding Ablation

**Status: implemented as code.** Stage 3 compares the archived Stage 2 MiniLM
retrieval results against BGE retrieval embeddings while keeping chunking fair.

## Scope

Stage 3 changes only the retrieval embedding path:

- Boundary/chunking embeddings stay on `sentence-transformers/all-MiniLM-L6-v2`.
- Existing BiLSTM and Transformer weights are loaded as-is.
- Fixed-size, BiLSTM, and Transformer chunking logic stay unchanged.
- Transformer calibration and `TRANSFORMER_BOUNDARY_THRESHOLD` stay unchanged.
- FAISS chunk embeddings and query embeddings use the active
  `RETRIEVAL_EMBED_MODEL`.
- BGE query encoding applies the standard query instruction prefix; passage/chunk
  encoding does not.
- Retrieval embeddings are normalized before `IndexFlatIP`.
- BGE indices, when saved, use a model-specific path under
  `artifacts/data/nq/indices/retrieval_<model>/`.

Not included in this stage: BM25, RRF, rerankers, RL, BGE-M3, a larger dataset,
or a deeper Transformer.

## Config

`config.py` separates the two embedding roles:

| Knob | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | historical MiniLM default kept for backward compatibility |
| `BOUNDARY_EMBED_MODEL` | `EMBED_MODEL` | sentence embeddings used by BiLSTM/Transformer boundary models |
| `RETRIEVAL_EMBED_MODEL` | `EMBED_MODEL` | chunk/query embeddings used by FAISS retrieval |
| `RETRIEVAL_EMBED_NORMALIZE` | `True` | normalize dense retrieval vectors before inner-product search |
| `BGE_QUERY_INSTRUCTION` | BGE search instruction | prefix applied to BGE queries only |

## Run

Stage 3 requires Stage 2 results to be archived first because
`artifacts/results/latest/` is overwritten by each new sweep.

```bash
python scripts/save_stage_results.py --stage stage2
python scripts/9_sweep_bge_retrieval.py
```

For a faster sanity run:

```bash
python scripts/9_sweep_bge_retrieval.py --quick
```

If Colab is slow with BGE base:

```bash
python scripts/9_sweep_bge_retrieval.py --retrieval-model BAAI/bge-small-en-v1.5
```

After a successful full run, archive Stage 3 if the outputs look right:

```bash
python scripts/save_stage_results.py --stage stage3
```

## Outputs

Written to `artifacts/results/latest/`:

- `sweep_results.csv` - Stage 3 BGE sweep rows for fixed, BiLSTM, and Transformer.
- `best_config.json` - best Stage 3 BGE config.
- `fair_comparison_table.csv` - matching-config comparison of archived Stage 2
  MiniLM retrieval rows vs Stage 3 BGE retrieval rows.
- `stage2_vs_stage3_matched.csv` - identical matched MiniLM-vs-BGE table under an
  unambiguous name (the `fair_comparison_table.csv` name is reused across stages).
- `recall_vs_chunk_size.png` - Stage 3 BGE recall vs average chunk size.
- `model_comparison.png` - best fixed vs BiLSTM vs Transformer under BGE retrieval.
- `stage3_bge_retrieval_summary.md` - run metadata and output notes.

Do not treat Stage 3 as complete until those files exist from an actual run.
