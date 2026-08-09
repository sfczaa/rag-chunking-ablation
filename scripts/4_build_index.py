"""Phase 4 — chunk NQ docs (BiLSTM + fixed-size) and build both FAISS indices.
Usage:  python scripts/4_build_index.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from rag_chunk import retrieval, training  # noqa: E402


def main() -> None:
    C.ensure_dirs()
    model = training.load_model()
    retrieval.build_indexes(model)


if __name__ == "__main__":
    main()
