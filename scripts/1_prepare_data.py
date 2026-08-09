"""Phase 1 — download Wikipedia (titles), fetch wikitext, build (sentences,
labels) splits.  Usage:  python scripts/1_prepare_data.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from rag_chunk import wiki_data  # noqa: E402


def main() -> None:
    C.ensure_dirs()
    print(C.summary())
    wiki_data.prepare_dataset(C.N_WIKI_ARTICLES)


if __name__ == "__main__":
    main()
