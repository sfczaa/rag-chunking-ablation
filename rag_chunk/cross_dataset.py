"""Stage 7 — TriviaQA ``rc.wikipedia`` corpus (cross-dataset robustness check).

Mirrors the ``nq_data`` interface: :func:`prepare_trivia` returns
``(docs, questions)`` where each doc is ``{"id", "title", "sentences"}`` and
each question is ``{"question", "answer", "doc_title", "doc_titles"}``. The
extra ``doc_titles`` lists *every* gold page (1-2 per question); ``doc_title``
stays the first of them, so code written against the NQ schema keeps working.

Gold-document definition (doc-constrained recall)
-------------------------------------------------
TriviaQA's ``entity_pages`` are full Wikipedia pages about entities in the
question — distant supervision, not human-annotated support (weaker than NQ's
gold docs; every report of Stage 7 numbers must say so). The loader:

1. splits each entity page into sentences (>= 2 sentences to be usable —
   same rule as the NQ loader);
2. normalizes the answer candidates — ``answer.value`` first, then
   ``answer.aliases`` in dataset order — with the *same* ``normalize_text``
   the metric uses;
3. keeps the first candidate that appears as a substring of some page's
   sentence-joined normalized text. That candidate becomes the question's
   single ``answer`` (so Recall@k stays literally identical to NQ's), and
   the gold set is every page containing it;
4. drops the question when no candidate appears anywhere (counted, reported).

The candidate is checked against the *sentence-joined* text — exactly what
chunks are built from — so the NQ honesty guarantee holds: a doc-constrained
miss means chunking split the answer span or retrieval missed the chunk.

Corpus = every usable entity page of every kept question, gold or not (a
non-gold page of a kept question is a natural distractor), deduplicated by
title.

Comparability guard: :func:`prepare_trivia` aborts if the median corpus
document is shorter than twice the largest grid chunk size — the
degenerate-sweep trap that disqualified raw HotpotQA paragraphs (see
``docs/stage7_cross_dataset.md``).

Everything caches under ``data/triviaqa/n<N>/`` (the NQ caches are never
touched) and is reused on later runs.
"""

from __future__ import annotations

import json

import config as C
from rag_chunk import wiki_data
from rag_chunk.metrics import normalize_text

DATASET_LABEL = "triviaqa-rc.wikipedia"


def _dataset_dir(n_questions: int):
    return C.DATA_DIR / "triviaqa" / f"n{int(n_questions)}"


def _docs_path(n_questions: int):
    return _dataset_dir(n_questions) / "docs.jsonl"


def _questions_path(n_questions: int):
    return _dataset_dir(n_questions) / "questions.jsonl"


def _meta_path(n_questions: int):
    return _dataset_dir(n_questions) / "trivia_meta.json"


def _current_meta(n_questions: int) -> dict:
    """Fingerprint of the settings the cached corpus was built with."""
    return {
        "n_questions": int(n_questions),
        "dataset": C.STAGE7_DATASET,
        "config": C.STAGE7_DATASET_CONFIG,
        "split": C.STAGE7_SPLIT,
    }


def load_meta(n_questions: int) -> dict | None:
    p = _meta_path(n_questions)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_stream():
    from datasets import load_dataset

    return load_dataset(C.STAGE7_DATASET, C.STAGE7_DATASET_CONFIG,
                        split=C.STAGE7_SPLIT, streaming=True)


# --------------------------------------------------------------------------- #
# Defensive accessors for TriviaQA's nested schema
# --------------------------------------------------------------------------- #
def _entity_pages(row) -> list[tuple[str, str]]:
    """``(title, wiki_context)`` pairs; robust to dict-of-lists (datasets)
    vs list-of-dicts layouts."""
    ep = row.get("entity_pages")
    if isinstance(ep, dict):
        titles = ep.get("title") or []
        contexts = ep.get("wiki_context") or []
        return [(t, c) for t, c in zip(titles, contexts) if t and c]
    if isinstance(ep, list):
        return [(d.get("title"), d.get("wiki_context")) for d in ep
                if d.get("title") and d.get("wiki_context")]
    return []


def _answer_candidates(row) -> list[str]:
    """``answer.value`` first, then aliases in dataset order, de-duplicated —
    a fixed order so the kept answer string is deterministic."""
    ans = row.get("answer") or {}
    cands: list[str] = []
    for s in [ans.get("value")] + list(ans.get("aliases") or []):
        s = (s or "").strip()
        if s and s not in cands:
            cands.append(s)
    return cands


# --------------------------------------------------------------------------- #
# Build / cache the TriviaQA corpus + queries
# --------------------------------------------------------------------------- #
def prepare_trivia(n_questions: int | None = None, force: bool = False):
    """Return ``(docs, questions)``; build & cache them on first run."""
    if n_questions is None:
        n_questions = C.STAGE7_N_QUESTIONS
    C.ensure_dirs()
    meta = load_meta(n_questions)
    meta_ok = (meta is not None
               and {k: meta.get(k) for k in _current_meta(n_questions)}
               == _current_meta(n_questions))
    if (not force and meta_ok and _docs_path(n_questions).exists()
            and _questions_path(n_questions).exists()):
        docs, questions = load_docs(n_questions), load_questions(n_questions)
        if docs and questions:
            print(f"[trivia] cached: {len(docs)} docs, "
                  f"{len(questions)} questions")
            return docs, questions
    if not force and _docs_path(n_questions).exists() and not meta_ok:
        print("[trivia] cached corpus was built with different settings — "
              "rebuilding")

    ds = _load_stream()
    docs: dict[str, dict] = {}
    sents_cache: dict[str, list[str] | None] = {}   # title -> sents (None=unusable)
    norm_cache: dict[str, str] = {}                 # title -> normalized joined text
    questions: list[dict] = []
    stats = {"scanned": 0, "kept": 0, "dropped_no_answer_in_pages": 0,
             "dropped_no_usable_page": 0, "pages_too_short": 0,
             "multi_gold_questions": 0}
    n_err = 0
    for row in ds:
        stats["scanned"] += 1
        try:
            pages: list[str] = []
            for title, context in _entity_pages(row):
                if title not in sents_cache:
                    sents = wiki_data.split_sentences(context)
                    if len(sents) < 2:
                        sents_cache[title] = None
                        stats["pages_too_short"] += 1
                    else:
                        sents_cache[title] = sents
                        norm_cache[title] = normalize_text(" ".join(sents))
                if sents_cache[title] is not None:
                    pages.append(title)
            if not pages:
                stats["dropped_no_usable_page"] += 1
                continue

            chosen, gold = None, []
            for cand in _answer_candidates(row):
                n_cand = normalize_text(cand)
                if not n_cand:
                    continue
                hits = [t for t in pages if n_cand in norm_cache[t]]
                if hits:
                    chosen, gold = cand, hits
                    break
            if chosen is None:
                stats["dropped_no_answer_in_pages"] += 1
                continue

            for t in pages:                       # only kept questions add docs
                if t not in docs:
                    docs[t] = {"id": f"doc_{len(docs)}", "title": t,
                               "sentences": sents_cache[t]}
            if len(gold) > 1:
                stats["multi_gold_questions"] += 1
            questions.append({"question": row["question"], "answer": chosen,
                              "doc_title": gold[0], "doc_titles": gold})
            stats["kept"] += 1
        except (KeyError, TypeError, IndexError) as exc:
            n_err += 1
            if n_err <= 3:
                print(f"[trivia] WARN skipped a row (schema mismatch?): {exc!r}")
            continue
        if stats["kept"] >= n_questions:
            break

    if not questions:
        raise RuntimeError(
            f"TriviaQA yielded 0 usable questions after scanning "
            f"{stats['scanned']} rows ({n_err} schema errors). The feature "
            "layout may differ from what cross_dataset.py expects — inspect "
            "one row with `next(iter(_load_stream()))` and adjust the "
            "accessors.")

    # Comparability guard: with documents shorter than ~2x the largest grid
    # size, every config chunks them nearly identically and the sweep can no
    # longer separate size from method (the trap that disqualified raw
    # HotpotQA paragraphs). Abort loudly instead of producing incomparable
    # numbers.
    lengths = sorted(len(d["sentences"]) for d in docs.values())
    median_len = lengths[len(lengths) // 2]
    guard = 2 * max(C.FIXED_SIZE_GRID)
    if median_len < guard:
        raise SystemExit(
            f"[trivia] median document length is {median_len} sentences "
            f"(< {guard} = 2x the largest grid chunk size). The chunk-size "
            "sweep would degenerate on this corpus — pick another dataset "
            "instead of running an incomparable sweep.")

    docs_list = list(docs.values())
    _dataset_dir(n_questions).mkdir(parents=True, exist_ok=True)
    with _docs_path(n_questions).open("w", encoding="utf-8") as f:
        for d in docs_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with _questions_path(n_questions).open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    meta_out = _current_meta(n_questions)
    meta_out.update({
        "n_docs": len(docs_list),
        "n_kept_questions": len(questions),
        "median_doc_sentences": median_len,
        "min_doc_sentences": lengths[0],
        "max_doc_sentences": lengths[-1],
        "mean_doc_sentences": round(sum(lengths) / len(lengths), 1),
        "stats": stats,
        "schema_errors": n_err,
    })
    _meta_path(n_questions).write_text(json.dumps(meta_out, indent=2),
                                       encoding="utf-8")
    print(f"[trivia] built {len(docs_list)} docs, {len(questions)} questions "
          f"(scanned {stats['scanned']}; dropped "
          f"{stats['dropped_no_answer_in_pages']} no-answer-in-pages, "
          f"{stats['dropped_no_usable_page']} no-usable-page; "
          f"median doc = {median_len} sentences)")
    return docs_list, questions


def load_docs(n_questions: int) -> list[dict]:
    if not _docs_path(n_questions).exists():
        return []
    with _docs_path(n_questions).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_questions(n_questions: int) -> list[dict]:
    if not _questions_path(n_questions).exists():
        return []
    with _questions_path(n_questions).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
