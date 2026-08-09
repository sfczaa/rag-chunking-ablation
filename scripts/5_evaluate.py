"""Phase 5 — evaluate Recall@k + Boundary F1, write comparison table & figure.
Usage:  python scripts/5_evaluate.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag_chunk import evaluation, training  # noqa: E402


def main() -> None:
    model = training.load_model()
    evaluation.evaluate_all(model)


if __name__ == "__main__":
    main()
