"""Phase 6 — chunking sweep optimizer (CLI wrapper).

Sweeps fixed-size and learned target-size chunking across chunk sizes and
overlaps, scores each on NQ Recall@k, and writes the optimizer artifacts under
``artifacts/results/latest/``.

Usage:
    python scripts/6_sweep_chunking.py            # full grid
    python scripts/6_sweep_chunking.py --quick    # smaller, fast grid
    python scripts/6_sweep_chunking.py --save-run  # also snapshot to results/runs/
    python scripts/6_sweep_chunking.py --save-sweep-index  # persist temp indices
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG chunking sweep optimizer (Phase 6)")
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

    model = training.load_model()
    sweep.run_sweep(
        model,
        quick=args.quick,
        save_run=args.save_run,
        save_sweep_index=args.save_sweep_index,
    )


if __name__ == "__main__":
    main()
