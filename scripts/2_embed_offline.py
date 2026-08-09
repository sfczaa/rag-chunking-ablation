"""Phase 2 — pre-compute and cache per-article sentence embeddings.
Usage:  python scripts/2_embed_offline.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from rag_chunk import embedding  # noqa: E402


def main() -> None:
    C.ensure_dirs()
    embedding.embed_offline()


if __name__ == "__main__":
    main()
