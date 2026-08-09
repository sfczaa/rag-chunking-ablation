"""Smoke test — run all 5 phases at tiny scale to validate the pipeline plumbing
in a separate `*_smoke` artifact folder.  Usage:  python scripts/0_smoke_test.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag_chunk import smoke  # noqa: E402


def main() -> None:
    smoke.run_smoke()


if __name__ == "__main__":
    main()
