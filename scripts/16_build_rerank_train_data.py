"""Stage 8 (step 1) - build the reranker fine-tuning data from the NQ train split.

One streaming pass collects the training corpus (STAGE8_N_TRAIN_DOCS docs +
their questions) and the dev bench (the NEXT STAGE8_N_DEV_DOCS docs + their
questions) — disjoint from each other and from every eval bench (which all
use the validation split). The training corpus is then chunked with the
deployment config (fixed 15/0), each question's BGE top-20 pool is retrieved,
and (1 positive + STAGE8_NUM_NEGATIVES hard negatives) groups are mined.

Everything caches under data/nq_train/ — the eval caches are never touched,
and nothing is written to results/latest/ (this is data, not results).

Needs a GPU session for the BGE embedding of ~40k chunks (a few minutes on a
T4); no boundary models are involved (fixed chunking only).

Usage:
    python scripts/16_build_rerank_train_data.py
    python scripts/16_build_rerank_train_data.py --force   # rebuild caches
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 8 step 1: stream the NQ train split, build the "
                    "train corpus + dev bench, and mine reranker groups.")
    ap.add_argument("--force", action="store_true",
                    help="rebuild the corpus/dev caches and re-mine")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="dense retrieval model used for mining; must match "
                         "the eval pipeline")
    args = ap.parse_args()

    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()

    from rag_chunk import rerank_finetune as rf

    print(f"[stage8-data] train split: {C.NQ_DATASET} ({C.NQ_CONFIG})")
    print(f"[stage8-data] mining: fixed {C.STAGE8_TRAIN_CHUNK_SIZE}/"
          f"{C.STAGE8_TRAIN_CHUNK_OVERLAP} chunks, BGE top-"
          f"{C.STAGE8_MINE_DEPTH}, {C.STAGE8_NUM_NEGATIVES} negatives/group")

    train_docs, train_q, dev_docs, dev_q = rf.prepare_train_and_dev(
        force=args.force)

    if not args.force and rf.groups_path().exists():
        groups = rf.load_groups()
        print(f"[stage8-data] groups already mined: {len(groups)} "
              f"({rf.groups_path()}) — use --force to re-mine")
        return

    groups, stats = rf.mine_training_groups(train_docs, train_q)
    rf.write_groups(groups, stats)

    depth = int(C.STAGE8_MINE_DEPTH)
    print("\n[stage8-data] mining stats:")
    print(f"  questions:                {stats['questions']}")
    print(f"  kept groups:              {stats['kept']}")
    print(f"  dropped (no positive):    {stats['dropped_no_positive']} "
          "(pool ceiling misses — reranking cannot rescue these at eval "
          "either)")
    print(f"  dropped (few negatives):  {stats['dropped_too_few_negatives']}")
    print(f"  train pool_recall@{depth}:    "
          f"{stats[f'train_pool_recall@{depth}']:.4f}")
    print(f"  corpus chunks:            {stats['n_chunks']} "
          f"(avg size {stats['avg_chunk_size']:.2f})")
    print(f"  dev bench:                {len(dev_docs)} docs / "
          f"{len(dev_q)} questions")
    print("\n[stage8-data] next: python scripts/17_train_reranker.py")


if __name__ == "__main__":
    main()
