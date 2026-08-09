"""Phase 3 — train the BiLSTM boundary detector (saves best weights to Drive).
Usage:  python scripts/3_train.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag_chunk import training  # noqa: E402


def main() -> None:
    training.train_model()


if __name__ == "__main__":
    main()
