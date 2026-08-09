"""Fast end-to-end plumbing check.

Runs all five phases at a tiny scale (~30 Wikipedia articles, 1 epoch, ~10 NQ
docs) inside a separate ``<root>_smoke`` folder, so you can confirm the whole
pipeline works on Colab in a few minutes **before** launching the full run.

The original config (data root + scale knobs) is restored afterwards, so you
can run the real phases in the same kernel without re-importing anything.
"""

from __future__ import annotations

import config as C

_SAVED_KEYS = (
    "DATA_ROOT", "N_WIKI_ARTICLES", "MAX_EPOCHS", "EARLY_STOP_PATIENCE",
    "N_NQ_DOCS", "MIN_SENTENCES_PER_ARTICLE",
)


def run_smoke() -> dict:
    saved = {k: getattr(C, k) for k in _SAVED_KEYS}
    try:
        C.use_smoke()
        from rag_chunk import embedding, evaluation, training, wiki_data

        print("\n=== Phase 1 (smoke): prepare data ===")
        wiki_data.prepare_dataset()
        print("\n=== Phase 2 (smoke): offline embedding ===")
        embedding.embed_offline()
        print("\n=== Phase 3 (smoke): train BiLSTM ===")
        training.train_model()
        print("\n=== Phase 4+5 (smoke): build indices + evaluate ===")
        model = training.load_model()
        results = evaluation.evaluate_all(model, rebuild=True)
        print("\n[smoke] OK — the full pipeline runs end to end. "
              "Numbers here are meaningless (tiny data); now run the real phases.")
        return results
    finally:
        C.apply(**{k: saved[k] for k in _SAVED_KEYS if k != "DATA_ROOT"})
        C.set_data_root(saved["DATA_ROOT"])
        print(f"[smoke] restored real config; DATA_ROOT -> {C.DATA_ROOT}")
