"""Phase 8 (Stage 2) — sweep Fixed vs BiLSTM vs Transformer chunking.

Runs the same fair sweep as Phase 6 (``scripts/6_sweep_chunking.py``) but adds the
Transformer boundary model as a third method, so all three are compared under the
same MiniLM embeddings, the same cached NQ docs/questions, the same Recall@k
metric and the same target-size/min/max/overlap policy. Artifacts are written to
``artifacts/results/latest/`` (archive a finished run with
``python scripts/save_stage_results.py --stage stage2``).

Requires both trained models (``scripts/3_train.py`` for BiLSTM,
``scripts/7_train_transformer.py`` for the Transformer).

Usage:
    python scripts/8_sweep_with_transformer.py            # full grid
    python scripts/8_sweep_with_transformer.py --quick    # smaller, fast grid
    python scripts/8_sweep_with_transformer.py --save-run  # also snapshot to results/runs/
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep Fixed + BiLSTM + Transformer chunking (Stage 2)")
    ap.add_argument("--quick", action="store_true",
                    help="use the smaller QUICK_* grids for a fast sanity sweep")
    ap.add_argument("--save-run", action="store_true",
                    help="also snapshot artifacts to results/runs/<timestamp>_sweep/")
    ap.add_argument("--save-sweep-index", action="store_true",
                    help="persist each per-config FAISS index under nq/indices/sweep/ "
                         "(default: build in memory and discard)")
    args = ap.parse_args()

    # Heavy imports happen only after arg parsing, so `--help` stays dependency-free.
    from rag_chunk import sweep, training

    bilstm = training.load_model("bilstm")
    transformer = training.load_model("transformer")
    sweep.run_sweep(
        bilstm,
        transformer_model=transformer,
        quick=args.quick,
        save_run=args.save_run,
        save_sweep_index=args.save_sweep_index,
    )


if __name__ == "__main__":
    main()
