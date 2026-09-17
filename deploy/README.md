# Deployment — Hugging Face ZeroGPU Space

Everything needed to host the interactive demo (`scripts/19_demo.py`'s sibling,
rebuilt for the Hub) as a ZeroGPU Space. Three Hub repos are involved:

| repo | type | what it holds |
|---|---|---|
| `sfczaa/bge-reranker-base-nq-ft` | model | the Stage 8 fine-tuned cross-encoder (card: `model_card.md`) |
| `sfczaa/rag-chunking-ablation-demo-assets` | dataset | bench corpus + prebuilt FAISS indices (card: `dataset_card.md`) |
| `sfczaa/rag-chunking-ablation-demo` | space | the Gradio app (`space/`) |

## Status

All three repositories are public. The Space pins the dataset and model
revisions in `space/demo_settings.py`.

## Layout

```
deploy/
  space/            payload unique to the Space
    app.py            Gradio entry point (ZeroGPU-aware)
    requirements.txt  torch deliberately omitted — the ZeroGPU image supplies it
    README.md         Space card (YAML frontmatter pins the Gradio SDK version)
  create_space.py   assemble + create + push the Space
  smoke_space.py    local CPU test of the exact payload
  upload_model.py   push the fine-tuned reranker
  upload_dataset.py push corpus + indices
  model_card.md     README shipped with the model repo
  dataset_card.md   README shipped with the dataset repo
```

`config.py` and `rag_chunk/` are **not** duplicated here. `create_space.py`
copies them from the repo root at assembly time, so the deployed app cannot
drift from the study's code.

## Deploying

Assets live outside the repo (they are gitignored), so pass their location:

```bash
# 1. verify the payload end to end on CPU (must print ALL CHECKS PASSED)
python deploy/smoke_space.py \
  --data-root "<data root>" \
  --ft-model  "<data root>/models/bge_reranker_ft/final"

# 2. create and push the Space (add --public when it should be world-readable)
python deploy/create_space.py
```

The smoke test asserts the shipped indices are the archived ones (19507 chunks /
14.649 mean sentences for fixed 15/0; 19306 / 14.801 for bilstm t15/0). A
mismatch means the assets were rebuilt rather than reused, and the demo would no
longer show the study's own retrieval.

## Access

The asset repositories are public, so the Space needs no token. Private asset
repositories would need a read-only `HF_TOKEN` Space secret.

## Design notes that must not regress

- `import spaces` is the **first** import in `app.py` — it patches torch.
- `RAG_DATA_ROOT` is set **before** `import config`, so every path resolves out
  of the downloaded snapshot.
- Models are placed on the device at **module scope** (ZeroGPU requirement);
  only inference runs inside `@spaces.GPU`, and nothing infers at import time.
- Indices are **loaded, never built**. A missing index raises instead of
  silently embedding ~40k chunks at boot.
- The bench is read straight from JSONL, bypassing the dataset-streaming loader,
  so the Space can never try to re-download Natural Questions.
