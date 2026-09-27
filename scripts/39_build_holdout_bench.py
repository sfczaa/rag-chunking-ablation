"""Stage 16 (step 1) - build an NQ validation bench disjoint from the Stage 6 bench.

Streams the NQ validation split with the rules of nq_data.prepare_nq and skips every
row whose title is in the Stage 6 bench or whose question text is a Stage 6
question. It keeps the first STAGE16_N_DOCS new titles, or all of them if the split
ends first, and the questions met before the last one is added.

Writes docs.jsonl, questions.jsonl and holdout_meta.json under
NQ_DIR/STAGE16_BENCH_DIRNAME. An existing bench is kept unless --force is given.

Usage:
    python scripts/39_build_holdout_bench.py --smoke            # 20 docs, temp dir
    python scripts/39_build_holdout_bench.py --account A
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402


def stage6_dir() -> pathlib.Path:
    return C.NQ_DIR / f"large_n{int(C.N_NQ_DOCS_LARGE)}"


def bench_dir() -> pathlib.Path:
    return C.NQ_DIR / C.STAGE16_BENCH_DIRNAME


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def stage6_bench() -> tuple[list[dict], list[dict], str]:
    d = stage6_dir()
    if not (d / "docs.jsonl").exists() or not (d / "questions.jsonl").exists():
        raise SystemExit(f"[stage16] no Stage 6 bench under {d}")
    docs, questions = _read_jsonl(d / "docs.jsonl"), _read_jsonl(d / "questions.jsonl")
    if len(docs) != int(C.N_NQ_DOCS_LARGE) or not questions:
        raise SystemExit(f"[stage16] the Stage 6 bench under {d} is missing or incomplete")
    h = hashlib.sha1()
    for doc in docs:
        h.update(doc["title"].encode("utf-8") + b"\n")
    return docs, questions, h.hexdigest()


def build(n_docs: int, skip_titles: set, skip_questions: set):
    """prepare_nq's loop with the two skip rules; returns (docs, questions, stats)."""
    from rag_chunk import nq_data, wiki_data

    ds = nq_data._load_nq_stream()
    docs: dict[str, dict] = {}
    questions: list[dict] = []
    n_seen = n_err = n_skip_title = n_skip_question = 0
    for row in ds:
        n_seen += 1
        try:
            span = nq_data._first_short_answer(row)
            if span is None:
                continue
            start, end = span
            title = row["document"]["title"]
            if title in skip_titles:
                n_skip_title += 1
                continue
            qtext = row["question"]["text"]
            if qtext in skip_questions:
                n_skip_question += 1
                continue
            toks, is_html = nq_data._doc_tokens(row)
            answer = nq_data._join_tokens(toks, is_html, start, end).strip()
            if not answer:
                continue
            if title not in docs:
                if len(docs) >= n_docs:
                    continue
                sents = wiki_data.split_sentences(nq_data._join_tokens(toks, is_html))
                if len(sents) < 2:
                    continue
                docs[title] = {"id": f"doc_{len(docs)}", "title": title, "sentences": sents}
            questions.append({"question": qtext, "answer": answer, "doc_title": title})
        except (KeyError, TypeError, IndexError) as exc:
            n_err += 1
            if n_err <= 3:
                print(f"[stage16] WARN skipped a row (schema mismatch?): {exc!r}")
            continue
        if n_seen % 500 == 0:
            print(f"[stage16] {n_seen} rows scanned, {len(docs)} docs, "
                  f"{len(questions)} questions", flush=True)
        if len(docs) >= n_docs:
            break
    stats = {"rows_scanned": n_seen, "schema_errors": n_err,
             "rows_skipped_stage6_title": n_skip_title,
             "rows_skipped_stage6_question": n_skip_question}
    return list(docs.values()), questions, stats


def write(out: pathlib.Path, docs, questions, meta) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("docs.jsonl", docs), ("questions.jsonl", questions)):
        tmp = out / f"{name}.tmp"
        with tmp.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(out / name)
    (out / "holdout_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 16 step 1: the holdout NQ bench.")
    ap.add_argument("--smoke", action="store_true", help="20 docs into a temp dir, nothing kept")
    ap.add_argument("--force", action="store_true", help="rebuild an existing bench")
    ap.add_argument("--account", default="unlabelled", help="label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()

    from rag_chunk import run_guard

    if not args.smoke:
        run_guard.check_root_identity(C.DATA_ROOT, C.STAGE16_ROOT_ID)
    s6_docs, s6_questions, s6_sha1 = stage6_bench()
    skip_titles = {d["title"] for d in s6_docs}
    skip_questions = {q["question"] for q in s6_questions}

    if args.smoke:
        docs, questions, stats = build(20, skip_titles, skip_questions)
        out = pathlib.Path(tempfile.mkdtemp(prefix="stage16_smoke_"))
        write(out, docs, questions, {"smoke": True, **stats})
        ok = len(docs) == 20 and not skip_titles & {d["title"] for d in docs}
        shutil.rmtree(out, ignore_errors=True)
        print(f"[stage16] smoke: {len(docs)} docs, {len(questions)} questions, "
              f"{stats['rows_scanned']} rows; {'OK' if ok else 'FAILED'}. Nothing was kept.")
        if not ok:
            raise SystemExit(1)
        return

    out = bench_dir()
    if (out / "holdout_meta.json").exists() and not args.force:
        meta = json.loads((out / "holdout_meta.json").read_text(encoding="utf-8"))
        print(f"[stage16] bench already built: {meta['n_docs']} docs, "
              f"{meta['n_questions']} questions - nothing to do")
        return
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE16_LOCK_FILE,
                             run_version=C.STAGE16_RUN_VERSION, account=args.account,
                             task="build-bench", clear_stale=args.clear_stale_lock):
        run_guard.probe_directory(out)
        docs, questions, stats = build(int(C.STAGE16_N_DOCS), skip_titles, skip_questions)
        overlap = len(skip_titles & {d["title"] for d in docs})
        if overlap:
            raise SystemExit(f"[stage16] {overlap} titles overlap the Stage 6 bench")
        from rag_chunk import sentence_tokenizer
        meta = {"n_docs": len(docs), "n_questions": len(questions),
                "dataset": C.NQ_DATASET, "config": C.NQ_CONFIG, "split": C.NQ_SPLIT,
                "stage6_titles_sha1": s6_sha1, "sentence_splitter": sentence_tokenizer.__name__,
                **stats}
        write(out, docs, questions, meta)
    print(f"[stage16] built {len(docs)} docs / {len(questions)} questions after "
          f"{stats['rows_scanned']} rows; skipped {stats['rows_skipped_stage6_title']} rows "
          f"with Stage 6 titles and {stats['rows_skipped_stage6_question']} with Stage 6 "
          f"questions")
    if len(docs) < int(C.STAGE16_N_DOCS):
        print(f"[stage16] the split ran out after {len(docs)} new documents")
    print("[stage16] next: python scripts/40_eval_holdout.py")


if __name__ == "__main__":
    main()
