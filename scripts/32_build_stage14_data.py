"""Stage 14 (step 0) - about 6000 more reranker training groups.

Continues the NQ train stream after the Stage 8 and Stage 11 windows, excludes
every Stage 8 train and dev title and every Stage 11 title, and keeps the next
STAGE14_N_NEW_DOCS usable documents with the Stage 8 per-row rule. The documents
are split in stream order into shards of STAGE14_SHARD_DOCS, and each shard is
mined on its own with Stage 8's mine_training_groups, so the distractor pool, and
with it the hardness of the negatives, matches Stages 8 and 11.

Streaming needs no GPU (--stream-only); mining encodes each shard with BGE. The
documents are cached when the stream finishes, and each shard's groups are written
when that shard is mined, so a rerun skips finished work.

Usage:
    python scripts/32_build_stage14_data.py --stream-only   # CPU runtime
    python scripts/32_build_stage14_data.py                 # stream if needed, then mine
    python scripts/32_build_stage14_data.py --smoke         # 6 docs, 3 shards, plumbing only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402

SMOKE_PREFIX = "smoke_"


def _load_stage11_data():
    spec = importlib.util.spec_from_file_location("stage11_data",
                                                  SCRIPTS / "26_build_stage11_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S11 = _load_stage11_data()


def data_dir(smoke: bool = False) -> pathlib.Path:
    name = C.STAGE14_DATA_DIRNAME
    return C.DATA_DIR / ((SMOKE_PREFIX + name) if smoke else name)


def shard_path(k: int, smoke: bool = False) -> pathlib.Path:
    return data_dir(smoke) / f"shard{k}_groups.jsonl"


def split_shards(docs: list[dict], questions: list[dict], shard_docs: int):
    """Documents in stream order, ``shard_docs`` per shard; each question follows
    its document. Returns a list of ``(docs, questions)``."""
    shards = []
    for start in range(0, len(docs), shard_docs):
        part = docs[start:start + shard_docs]
        titles = {d["title"] for d in part}
        shards.append((part, [q for q in questions if q["doc_title"] in titles]))
    return shards


def excluded_titles() -> tuple[set, int]:
    """All Stage 8 train/dev titles and Stage 11 titles, and the rows both windows
    consumed."""
    from rag_chunk import rerank_finetune as rf

    s8_meta = rf.load_meta()
    s8_train, _ = rf.load_bench("train")
    s8_dev, _ = rf.load_bench("dev")
    s11_dir = C.DATA_DIR / C.STAGE11_DATA_DIRNAME
    s11_meta_path = s11_dir / "stage11_data_meta.json"
    if s8_meta is None or not s8_train or not s8_dev or not s11_meta_path.exists():
        raise SystemExit("[stage14-data] the Stage 8 or Stage 11 cache is missing - run "
                         "scripts/16_build_rerank_train_data.py and "
                         "scripts/26_build_stage11_data.py first")
    s11_meta = json.loads(s11_meta_path.read_text(encoding="utf-8"))
    s11_docs = S11._read_jsonl(s11_dir / "train_docs.jsonl")
    titles = ({d["title"] for d in s8_train} | {d["title"] for d in s8_dev}
              | {d["title"] for d in s11_docs})
    skip = int(s8_meta["rows_scanned"]) + int(s11_meta["stream"]["rows_scanned"])
    if s11_meta["window"]["skip_rows"] != int(s8_meta["rows_scanned"]):
        raise SystemExit("[stage14-data] the Stage 11 window did not start where Stage 8 "
                         "ended - the skip count would be wrong")
    return titles, skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 14 step 0: new NQ-train reranker groups, mined per shard.")
    ap.add_argument("--stream-only", action="store_true",
                    help="collect and cache the documents, skip mining (no GPU)")
    ap.add_argument("--smoke", action="store_true",
                    help="6 documents from the start of the stream, no skip or exclusion, "
                         "3 shards, written under a smoke_ folder")
    ap.add_argument("--account", default="unlabelled",
                    help="operational label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()
    C.apply(RETRIEVAL_EMBED_MODEL="BAAI/bge-base-en-v1.5", RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()
    if args.smoke:
        build(args)
        return

    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE14_ROOT_ID)
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE14_LOCK_FILE,
                             run_version=C.STAGE14_RUN_VERSION, account=args.account,
                             task="stream" if args.stream_only else "data",
                             clear_stale=args.clear_stale_lock):
        run_guard.probe_directory(data_dir())
        build(args)


def build(args) -> None:
    from rag_chunk import rerank_finetune as rf

    if args.smoke:
        exclude, skip = set(), 0
        n_docs, shard_docs = 6, 2
    else:
        exclude, skip = excluded_titles()
        n_docs, shard_docs = int(C.STAGE14_N_NEW_DOCS), int(C.STAGE14_SHARD_DOCS)

    out = data_dir(args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    docs_path, q_path = out / "train_docs.jsonl", out / "train_questions.jsonl"
    meta_path = out / "stage14_data_meta.json"
    window = {"dataset": C.NQ_DATASET, "config": C.NQ_CONFIG, "split": "train",
              "skip_rows": skip, "n_docs": n_docs, "shard_docs": shard_docs,
              "excluded_titles": len(exclude),
              "excluded_titles_sha1": S11._titles_sha1(exclude)}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if meta.get("window") == window and docs_path.exists() and q_path.exists():
        docs, questions = S11._read_jsonl(docs_path), S11._read_jsonl(q_path)
        print(f"[stage14-data] cached: {len(docs)} docs / {len(questions)} questions")
    else:
        print(f"[stage14-data] streaming the NQ train split: skipping {skip} rows, "
              f"excluding {len(exclude)} titles, keeping {n_docs} documents", flush=True)
        started = time.perf_counter()
        stream = rf._load_train_stream()
        if skip:
            stream = stream.skip(skip)
        docs, questions, stats = S11.collect_window(stream, exclude_titles=exclude,
                                                    n_docs=n_docs)
        if len(docs) < n_docs:
            raise SystemExit(f"[stage14-data] stream ended early: {len(docs)}/{n_docs}")
        leaked = {d["title"] for d in docs} & exclude
        if leaked:                                  # checked before anything is cached
            raise SystemExit(f"[stage14-data] {len(leaked)} excluded title(s) leaked into "
                             "the new window - refusing to continue")
        for i, d in enumerate(docs):
            d["id"] = f"s14_doc_{i}"
        S11._write_jsonl(docs_path, docs)
        S11._write_jsonl(q_path, questions)
        for k in range(len(split_shards(docs, questions, shard_docs))):
            shard_path(k, args.smoke).unlink(missing_ok=True)   # mined from another window
        stats["minutes"] = round((time.perf_counter() - started) / 60, 1)
        meta = {"window": window, "stream": stats, "n_docs": len(docs),
                "n_questions": len(questions), "shards": {}}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[stage14-data] kept {len(docs)} docs / {len(questions)} questions after "
              f"{stats['rows_scanned']} rows ({stats['excluded_rows']} excluded) in "
              f"{stats['minutes']} min", flush=True)

    if not args.smoke:
        leaked = {d["title"] for d in docs} & exclude
        if leaked:
            raise SystemExit(f"[stage14-data] {len(leaked)} excluded title(s) leaked into "
                             "the new window - refusing to continue")
        print(f"[stage14-data] disjoint from all {len(exclude)} Stage 8 and Stage 11 titles")

    if args.stream_only:
        print("[stage14-data] --stream-only: done. Mine on a GPU runtime with "
              "`python scripts/32_build_stage14_data.py`.")
        return

    shards = split_shards(docs, questions, shard_docs)
    meta.setdefault("shards", {})
    for k, (shard_docs_k, shard_q) in enumerate(shards):
        path = shard_path(k, args.smoke)
        if path.exists() and str(k) in meta["shards"]:
            print(f"[stage14-data] shard {k} already mined: "
                  f"{meta['shards'][str(k)]['kept']} groups")
            continue
        print(f"[stage14-data] mining shard {k}: {len(shard_docs_k)} docs / "
              f"{len(shard_q)} questions", flush=True)
        started = time.perf_counter()
        groups, stats = rf.mine_training_groups(shard_docs_k, shard_q)
        stats["minutes"] = round((time.perf_counter() - started) / 60, 1)
        S11._write_jsonl(path, groups)
        meta["shards"][str(k)] = stats
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[stage14-data] shard {k}: {len(groups)} groups, pool recall@20 "
              f"{stats.get('train_pool_recall@20')}, {stats['minutes']} min", flush=True)

    total = sum(int(s["kept"]) for s in meta["shards"].values())
    print(f"[stage14-data] new groups across {len(shards)} shards: {total}")
    if args.smoke:
        print("[stage14-data] SMOKE MODE: plumbing only; the smoke_ folder is never "
              "used for training.")
    else:
        print("[stage14-data] next: python scripts/33_train_data_scale.py")


if __name__ == "__main__":
    main()
