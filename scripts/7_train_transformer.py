"""Phase 7 (Stage 2) — train the Transformer boundary detector.

Trains a 2-layer Transformer boundary model on the SAME cached MiniLM sentence
embeddings + Wikipedia section labels used for the BiLSTM, so no Phase 1/2 rerun
is needed. Best weights are saved to ``models/transformer_best.pt``; the BiLSTM
weights are untouched.

Usage:  python scripts/7_train_transformer.py [--epochs N]
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train the Transformer boundary model (Stage 2)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="max training epochs (default: config MAX_EPOCHS)")
    args = ap.parse_args()

    # Heavy import only after arg parsing, so `--help` stays dependency-free.
    from rag_chunk import training

    stats = training.train_model(model_type="transformer", max_epochs=args.epochs)
    print(f"[train] transformer: best val loss={stats['best_val_loss']:.4f}  "
          f"test boundary F1={stats['test_boundary_f1']['f1']:.4f}\n"
          f"[train] weights -> {stats['model_path']}")


if __name__ == "__main__":
    main()
