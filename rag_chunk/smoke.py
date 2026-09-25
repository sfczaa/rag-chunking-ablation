"""Small end-to-end pipeline check.

Runs five phases with about 30 Wikipedia articles, one epoch, and 10 NQ
documents in a separate ``<root>_smoke`` directory. Restores the original
data-root and scale settings afterward.
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
        print("\n[smoke] OK - phases 1-5 ran on tiny data. "
              "These numbers are not meaningful.")
        return results
    finally:
        C.apply(**{k: saved[k] for k in _SAVED_KEYS if k != "DATA_ROOT"})
        C.set_data_root(saved["DATA_ROOT"])
        print(f"[smoke] restored real config; DATA_ROOT -> {C.DATA_ROOT}")
