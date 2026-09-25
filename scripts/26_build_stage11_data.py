"""Stage 11 (step 0) - fresh reranker training groups the Stage 8 model never saw.

Stage 8's reranker already ranks the positive first in 92% of its own training
groups (a probe on 64 of them), so continuing on those groups would give either
objective almost nothing to learn and make Stage 11 a tie by construction. This
builds a new training set from the same source with the same rules:

* stream the NQ train split, skipping the rows Stage 8's window consumed
  (``rows_scanned`` in its cache meta) and excluding every Stage 8 train and
  dev title, so the new documents are disjoint from both;
* keep the next ``STAGE11_N_TRAIN_DOCS`` usable documents and their questions,
  by the collection rule of ``rerank_finetune.prepare_train_and_dev``;
* mine groups with Stage 8's own ``mine_training_groups`` (fixed 15/0 chunks,
  BGE top-20 pool, one positive and seven hard negatives).

The Stage 8 dev bench remains the Stage 11 gate - its titles are excluded here -
and the final bench comes from the validation split, so nothing evaluated is
ever trained on.

Streaming needs no GPU and can run on a CPU runtime (``--stream-only``). Mining
encodes about 40k chunks with BGE and wants the GPU. Both steps cache under
``data/nq_train_stage11/`` and are skipped when the cache matches. The stream is
not checkpointed: a lost runtime restarts it.

Usage:
    python scripts/26_build_stage11_data.py --stream-only   # CPU runtime is fine
    python scripts/26_build_stage11_data.py                 # stream if needed, then mine
    python scripts/26_build_stage11_data.py --smoke         # 5 docs, prefixed, plumbing only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

SMOKE_PREFIX = "smoke_"


def data_dir(smoke: bool = False) -> pathlib.Path:
    name = C.STAGE11_DATA_DIRNAME
    return C.DATA_DIR / ((SMOKE_PREFIX + name) if smoke else name)


def collect_window(rows, *, exclude_titles, n_docs: int, progress_every: int = 500):
    """Collect the next ``n_docs`` usable documents from a row stream.

    The per-row rule is Stage 8's: a row needs a short answer and a document
    with at least two sentences; later questions about an already-kept document
    are kept too; the first new document past a full window ends the scan.
    Rows whose title is in ``exclude_titles`` are skipped and counted.
    Returns ``(docs, questions, stats)``.
    """
    from rag_chunk import nq_data, wiki_data

    docs: dict[str, dict] = {}
    questions: list[dict] = []
    stats = {"rows_scanned": 0, "excluded_rows": 0, "schema_errors": 0}
    started = time.perf_counter()
    for row in rows:
        stats["rows_scanned"] += 1
        if progress_every and stats["rows_scanned"] % progress_every == 0:
            print(f"[stage11-data] scanned {stats['rows_scanned']} rows: "
                  f"{len(docs)}/{n_docs} docs, {len(questions)} questions "
                  f"({(time.perf_counter() - started) / 60:.1f} min)", flush=True)
        try:
            span = nq_data._first_short_answer(row)
            if span is None:
                continue
            start, end = span
            title = row["document"]["title"]
            if title in exclude_titles:
                stats["excluded_rows"] += 1
                continue
            toks, is_html = nq_data._doc_tokens(row)
            answer = nq_data._join_tokens(toks, is_html, start, end).strip()
            if not answer:
                continue
            if title not in docs:
                sents = wiki_data.split_sentences(nq_data._join_tokens(toks, is_html))
                if len(sents) < 2:
                    continue
                if len(docs) >= n_docs:
                    break                          # window full -> stop
                docs[title] = {"id": f"s11_doc_{len(docs)}", "title": title,
                               "sentences": sents}
            questions.append({"question": row["question"]["text"],
                              "answer": answer, "doc_title": title})
        except (KeyError, TypeError, IndexError) as exc:
            stats["schema_errors"] += 1
            if stats["schema_errors"] <= 3:
                print(f"[stage11-data] WARN skipped a row: {exc!r}", flush=True)
    return list(docs.values()), questions, stats


def _write_jsonl(path: pathlib.Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _titles_sha1(titles) -> str:
    return hashlib.sha1("\n".join(sorted(titles)).encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 11 step 0: fresh NQ-train reranker groups, disjoint "
                    "from everything Stage 8 trained or gated on.")
    ap.add_argument("--stream-only", action="store_true",
                    help="collect and cache the documents, skip mining (no GPU)")
    ap.add_argument("--smoke", action="store_true",
                    help="5 documents from the start of the stream, no skip or "
                         "exclusion, written under a smoke_ folder; plumbing only")
    ap.add_argument("--force", action="store_true",
                    help="rebuild the cache even if it matches")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()

    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()

    from rag_chunk import rerank_finetune as rf

    s8_meta = rf.load_meta()
    s8_train, _ = rf.load_bench("train")
    s8_dev, _ = rf.load_bench("dev")
    if s8_meta is None or not s8_train or not s8_dev:
        raise SystemExit("[stage11-data] the Stage 8 cache is missing - run "
                         "scripts/16_build_rerank_train_data.py first")
    stage8_titles = {d["title"] for d in s8_train} | {d["title"] for d in s8_dev}

    n_docs = 5 if args.smoke else int(C.STAGE11_N_TRAIN_DOCS)
    skip = 0 if args.smoke else int(s8_meta["rows_scanned"])
    exclude = set() if args.smoke else stage8_titles
    out = data_dir(args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    docs_path, q_path = out / "train_docs.jsonl", out / "train_questions.jsonl"
    groups_path, meta_path = out / C.STAGE11_GROUPS_JSONL, out / "stage11_data_meta.json"

    window = {"dataset": C.NQ_DATASET, "config": C.NQ_CONFIG, "split": "train",
              "skip_rows": skip, "n_docs": n_docs,
              "excluded_titles": len(exclude), "excluded_titles_sha1": _titles_sha1(exclude)}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if (not args.force and meta.get("window") == window
            and docs_path.exists() and q_path.exists()):
        docs, questions = _read_jsonl(docs_path), _read_jsonl(q_path)
        print(f"[stage11-data] cached: {len(docs)} docs / {len(questions)} questions")
    else:
        print(f"[stage11-data] streaming the NQ train split: skipping {skip} rows, "
              f"excluding {len(exclude)} Stage 8 titles, keeping {n_docs} documents",
              flush=True)
        started = time.perf_counter()
        stream = rf._load_train_stream()
        if skip:
            stream = stream.skip(skip)
        docs, questions, stats = collect_window(stream, exclude_titles=exclude,
                                                n_docs=n_docs)
        if len(docs) < n_docs:
            raise SystemExit(f"[stage11-data] stream ended early: {len(docs)}/"
                             f"{n_docs} documents")
        _write_jsonl(docs_path, docs)
        _write_jsonl(q_path, questions)
        if groups_path.exists():
            groups_path.unlink()               # mined from a different window
        stats["minutes"] = round((time.perf_counter() - started) / 60, 1)
        meta = {"window": window, "stream": stats, "n_docs": len(docs),
                "n_questions": len(questions)}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[stage11-data] kept {len(docs)} docs / {len(questions)} questions "
              f"after {stats['rows_scanned']} rows ({stats['excluded_rows']} "
              f"excluded as Stage 8 titles) in {stats['minutes']} min", flush=True)

    if not args.smoke:
        leaked = {d["title"] for d in docs} & stage8_titles
        if leaked:
            raise SystemExit(f"[stage11-data] {len(leaked)} Stage 8 title(s) leaked "
                             "into the Stage 11 window - refusing to continue")
        print(f"[stage11-data] disjoint from all {len(stage8_titles)} Stage 8 "
              "train and dev titles")

    if args.stream_only:
        print("[stage11-data] --stream-only: done. Mine on a GPU runtime with "
              "`python scripts/26_build_stage11_data.py`.")
        return

    if groups_path.exists() and meta.get("mining") and not args.force:
        print(f"[stage11-data] groups already mined: {meta['mining']['kept']} "
              f"({groups_path})")
    else:
        print("[stage11-data] mining groups with the Stage 8 rule (fixed "
              f"{C.STAGE8_TRAIN_CHUNK_SIZE}/{C.STAGE8_TRAIN_CHUNK_OVERLAP}, BGE top-"
              f"{C.STAGE8_MINE_DEPTH}, 1 + {C.STAGE8_NUM_NEGATIVES})", flush=True)
        groups, mining = rf.mine_training_groups(docs, questions)
        _write_jsonl(groups_path, groups)
        meta["mining"] = mining
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[stage11-data] wrote {groups_path} ({len(groups)} groups)")
        for key in ("questions", "kept", "dropped_no_positive",
                    "dropped_too_few_negatives", f"train_pool_recall@{C.STAGE8_MINE_DEPTH}",
                    "n_chunks"):
            print(f"  {key}: {mining.get(key)}")
    if args.smoke:
        print("[stage11-data] SMOKE MODE: plumbing only; the smoke_ folder is never "
              "used for training.")
    else:
        print("[stage11-data] next: python scripts/27_train_reranker_rl.py")


if __name__ == "__main__":
    main()
