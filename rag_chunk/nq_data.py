"""Phase 4 (data side) — Natural Questions documents + answerable questions.

Sampling (be precise about what this does)
------------------------------------------
We stream the NQ ``validation`` split and scan rows until we have collected
``N_NQ_DOCS`` **distinct** documents that have a usable short answer, then stop.
Each scanned row that carries a short answer contributes a ``(question, answer,
gold doc)`` triple *for documents already in the set*. Because we stop as soon
as the N-th document appears, questions are those seen up to that point — this
keeps the streamed download small (Colab-friendly) but means later questions
for the same documents are not collected. If you want a fixed document set with
all of its questions instead, raise ``N_NQ_DOCS`` (more corpus + more queries)
rather than relying on a longer scan.

For each kept question we record its gold *short answer* as a string.

Key consistency trick for Recall@k matching
-------------------------------------------
Both the document text (what we chunk & index) and the answer string are
reconstructed from the **same** NQ token list with the **same** join rule
(non-HTML tokens joined by single spaces).  After identical normalisation
(lower-case + whitespace collapse) the gold answer is therefore guaranteed to
appear verbatim inside the document, so a chunk that covers the answer span
will substring-match it.  (An answer is only "missed" if chunking splits it or
the chunk isn't retrieved — exactly what we want Recall@k to measure.)

Everything is cached to ``NQ_DIR`` and reused on later runs.
"""

from __future__ import annotations

import json

import config as C
from rag_chunk import wiki_data


def _docs_path():
    return C.NQ_DIR / "docs.jsonl"


def _questions_path():
    return C.NQ_DIR / "questions.jsonl"


def _meta_path():
    return C.NQ_DIR / "nq_meta.json"


def _current_meta(n_docs: int) -> dict:
    """Fingerprint of the settings the cached NQ corpus was built with."""
    return {
        "n_docs": int(n_docs),
        "dataset": C.NQ_DATASET,
        "config": C.NQ_CONFIG,
        "split": C.NQ_SPLIT,
    }


def _load_meta() -> dict | None:
    p = _meta_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------- #
# Defensive accessors for NQ's nested schema
# --------------------------------------------------------------------------- #
def _spans(sa) -> tuple[list, list]:
    """Return (start_tokens, end_tokens) from one annotator's short_answers."""
    if isinstance(sa, dict):
        return list(sa.get("start_token") or []), list(sa.get("end_token") or [])
    if isinstance(sa, list):
        return ([d["start_token"] for d in sa], [d["end_token"] for d in sa])
    return [], []


def _first_short_answer(row) -> tuple[int, int] | None:
    ann = row.get("annotations")
    if not ann:
        return None
    sa_field = ann.get("short_answers")
    if sa_field is None:
        return None
    if isinstance(sa_field, dict):          # no annotator dimension -> wrap
        sa_field = [sa_field]
    for sa in sa_field:
        starts, ends = _spans(sa)
        for s, e in zip(starts, ends):
            if s is not None and s >= 0 and e > s:
                return int(s), int(e)
    return None


def _doc_tokens(row) -> tuple[list, list]:
    td = row["document"]["tokens"]
    return td["token"], td["is_html"]


def _join_tokens(tokens, is_html, start=None, end=None) -> str:
    if start is None:
        start, end = 0, len(tokens)
    return " ".join(
        tok for tok, h in zip(tokens[start:end], is_html[start:end]) if not h
    )


# --------------------------------------------------------------------------- #
# Build / cache the NQ corpus + queries
# --------------------------------------------------------------------------- #
def _load_nq_stream():
    """Stream the NQ split, retrying with trust_remote_code if needed."""
    from datasets import load_dataset

    try:
        return load_dataset(C.NQ_DATASET, C.NQ_CONFIG, split=C.NQ_SPLIT, streaming=True)
    except Exception as e1:  # older datasets need an explicit opt-in
        try:
            return load_dataset(
                C.NQ_DATASET, C.NQ_CONFIG, split=C.NQ_SPLIT,
                streaming=True, trust_remote_code=True,
            )
        except Exception as e2:
            raise RuntimeError(
                f"Could not load NQ ({C.NQ_DATASET}/{C.NQ_CONFIG}/{C.NQ_SPLIT}). "
                f"Tried with and without trust_remote_code.\n"
                f"  without: {e1!r}\n  with   : {e2!r}\n"
                f"Tip: try NQ_CONFIG='dev', or upgrade `datasets`."
            ) from e2


def prepare_nq(n_docs: int | None = None, force: bool = False):
    """Return ``(docs, questions)``; build & cache them on first run."""
    if n_docs is None:
        n_docs = C.N_NQ_DOCS
    C.ensure_dirs()
    meta_ok = _load_meta() == _current_meta(n_docs)
    if not force and meta_ok and _docs_path().exists() and _questions_path().exists():
        docs, questions = load_docs(), load_questions()
        if docs and questions:
            print(f"[nq] cached: {len(docs)} docs, {len(questions)} questions")
            return docs, questions
    if not force and _docs_path().exists() and not meta_ok:
        print("[nq] cached corpus was built with different settings — rebuilding")

    ds = _load_nq_stream()
    docs: dict[str, dict] = {}
    questions: list[dict] = []
    n_seen = n_err = 0
    for row in ds:
        n_seen += 1
        try:
            span = _first_short_answer(row)
            if span is None:
                continue
            start, end = span
            title = row["document"]["title"]
            toks, is_html = _doc_tokens(row)
            answer = _join_tokens(toks, is_html, start, end).strip()
            if not answer:
                continue
            if title not in docs:
                if len(docs) >= n_docs:
                    continue
                sents = wiki_data.split_sentences(_join_tokens(toks, is_html))
                if len(sents) < 2:
                    continue
                docs[title] = {"id": f"doc_{len(docs)}", "title": title, "sentences": sents}
            questions.append(
                {"question": row["question"]["text"], "answer": answer, "doc_title": title}
            )
        except (KeyError, TypeError, IndexError) as exc:
            n_err += 1
            if n_err <= 3:
                print(f"[nq] WARN skipped a row (schema mismatch?): {exc!r}")
            continue
        if len(docs) >= n_docs:
            break

    if not docs:
        raise RuntimeError(
            f"NQ yielded 0 usable documents after scanning {n_seen} rows "
            f"({n_err} schema errors). The NQ feature layout may differ from "
            f"what nq_data.py expects — inspect one row with "
            f"`next(iter(_load_nq_stream()))` and adjust the accessors."
        )
    docs_list = list(docs.values())
    with _docs_path().open("w", encoding="utf-8") as f:
        for d in docs_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with _questions_path().open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    _meta_path().write_text(json.dumps(_current_meta(n_docs)), encoding="utf-8")
    print(f"[nq] built {len(docs_list)} docs, {len(questions)} questions")
    return docs_list, questions


def load_docs() -> list[dict]:
    if not _docs_path().exists():
        return []
    with _docs_path().open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_questions() -> list[dict]:
    if not _questions_path().exists():
        return []
    with _questions_path().open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
