"""Stage 8 — training data for fine-tuning the cross-encoder reranker.

Everything here uses the NQ **train** split; all eval benches (Stages 1-7)
use the validation split (or TriviaQA), so train/eval stay disjoint by
construction. One streaming pass builds both:

* the **training corpus** — the first ``STAGE8_N_TRAIN_DOCS`` usable
  documents and their questions (same collection rule as ``nq_data``);
* the **dev bench** — the *next* ``STAGE8_N_DEV_DOCS`` documents and the
  questions whose gold doc lies in that window. Dev questions never point at
  a training document, so the go/no-go decision never touches the final
  (Stage 6) bench.

Mining is deployment-matched: the training corpus is chunked with the
deployment config (fixed ``STAGE8_TRAIN_CHUNK_SIZE`` / overlap
``STAGE8_TRAIN_CHUNK_OVERLAP``), BGE retrieves each question's
top-``STAGE8_MINE_DEPTH`` pool — exactly what the reranker sees at eval time
— and per question:

* **positive** = the highest-dense-ranked pool chunk that is answer-bearing
  *and* from the gold document (the metric's own hit rule);
* **hard negatives** = the first ``STAGE8_NUM_NEGATIVES`` remaining pool
  chunks by dense rank (the distractors BGE currently ranks high);
* questions with no positive in the pool are dropped and counted — the
  reranker cannot rescue them at eval time either.

Groups are fixed-width (1 positive + exactly ``STAGE8_NUM_NEGATIVES``
negatives) so the training loop can batch them; questions with too few
negatives are dropped and counted (practically never happens at depth 20).

Caches live under ``data/nq_train/`` and never touch the eval caches.
Heavy deps are imported lazily, matching the rest of the package.
"""

from __future__ import annotations

import json

import config as C
from rag_chunk import nq_data, wiki_data
from rag_chunk.metrics import normalize_text


def train_data_dir():
    return C.DATA_DIR / "nq_train"


def _docs_path(kind: str):
    return train_data_dir() / f"{kind}_docs.jsonl"


def _questions_path(kind: str):
    return train_data_dir() / f"{kind}_questions.jsonl"


def _meta_path():
    return train_data_dir() / "nq_train_meta.json"


def groups_path():
    return train_data_dir() / C.STAGE8_TRAIN_GROUPS_JSONL


def _current_meta() -> dict:
    return {
        "n_train_docs": int(C.STAGE8_N_TRAIN_DOCS),
        "n_dev_docs": int(C.STAGE8_N_DEV_DOCS),
        "dataset": C.NQ_DATASET,
        "config": C.NQ_CONFIG,
        "split": "train",
    }


def load_meta() -> dict | None:
    p = _meta_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_train_stream():
    """Stream the NQ *train* split (same retry pattern as nq_data)."""
    from datasets import load_dataset

    try:
        return load_dataset(C.NQ_DATASET, C.NQ_CONFIG, split="train",
                            streaming=True)
    except Exception as e1:
        try:
            return load_dataset(C.NQ_DATASET, C.NQ_CONFIG, split="train",
                                streaming=True, trust_remote_code=True)
        except Exception as e2:
            raise RuntimeError(
                f"Could not stream NQ train split ({C.NQ_DATASET}/"
                f"{C.NQ_CONFIG}).\n  without trust_remote_code: {e1!r}\n"
                f"  with: {e2!r}") from e2


def _write_jsonl(path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------- #
# Train corpus + dev bench (one streaming pass, cached)
# --------------------------------------------------------------------------- #
def prepare_train_and_dev(force: bool = False):
    """Return ``(train_docs, train_questions, dev_docs, dev_questions)``;
    build and cache them on first run."""
    C.ensure_dirs()
    train_data_dir().mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    meta_ok = (meta is not None
               and {k: meta.get(k) for k in _current_meta()} == _current_meta())
    if (not force and meta_ok
            and all(_docs_path(k).exists() and _questions_path(k).exists()
                    for k in ("train", "dev"))):
        out = (load_bench("train"), load_bench("dev"))
        if all(part for pair in out for part in pair):
            print(f"[stage8-data] cached: train {len(out[0][0])} docs / "
                  f"{len(out[0][1])} q; dev {len(out[1][0])} docs / "
                  f"{len(out[1][1])} q")
            return out[0][0], out[0][1], out[1][0], out[1][1]
    if not force and _docs_path("train").exists() and not meta_ok:
        print("[stage8-data] cache was built with different settings — "
              "rebuilding")

    n_train, n_dev = int(C.STAGE8_N_TRAIN_DOCS), int(C.STAGE8_N_DEV_DOCS)
    ds = _load_train_stream()
    train_docs: dict[str, dict] = {}
    dev_docs: dict[str, dict] = {}
    train_q: list[dict] = []
    dev_q: list[dict] = []
    n_seen = n_err = 0
    for row in ds:
        n_seen += 1
        try:
            span = nq_data._first_short_answer(row)
            if span is None:
                continue
            start, end = span
            title = row["document"]["title"]
            toks, is_html = nq_data._doc_tokens(row)
            answer = nq_data._join_tokens(toks, is_html, start, end).strip()
            if not answer:
                continue
            bucket = None
            if title in train_docs:
                bucket = "train"
            elif title in dev_docs:
                bucket = "dev"
            else:
                sents = wiki_data.split_sentences(
                    nq_data._join_tokens(toks, is_html))
                if len(sents) < 2:
                    continue
                if len(train_docs) < n_train:
                    train_docs[title] = {"id": f"doc_{len(train_docs)}",
                                         "title": title, "sentences": sents}
                    bucket = "train"
                elif len(dev_docs) < n_dev:
                    dev_docs[title] = {"id": f"dev_doc_{len(dev_docs)}",
                                       "title": title, "sentences": sents}
                    bucket = "dev"
                else:
                    break                      # both windows full -> stop
            q = {"question": row["question"]["text"], "answer": answer,
                 "doc_title": title}
            (train_q if bucket == "train" else dev_q).append(q)
        except (KeyError, TypeError, IndexError) as exc:
            n_err += 1
            if n_err <= 3:
                print(f"[stage8-data] WARN skipped a row: {exc!r}")
            continue

    if len(train_docs) < n_train or len(dev_docs) < n_dev:
        raise RuntimeError(
            f"[stage8-data] stream ended early: train {len(train_docs)}/"
            f"{n_train} docs, dev {len(dev_docs)}/{n_dev} docs "
            f"(scanned {n_seen} rows, {n_err} schema errors)")

    _write_jsonl(_docs_path("train"), train_docs.values())
    _write_jsonl(_questions_path("train"), train_q)
    _write_jsonl(_docs_path("dev"), dev_docs.values())
    _write_jsonl(_questions_path("dev"), dev_q)
    meta_out = _current_meta()
    meta_out.update({
        "rows_scanned": n_seen,
        "schema_errors": n_err,
        "n_train_questions": len(train_q),
        "n_dev_questions": len(dev_q),
    })
    _meta_path().write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"[stage8-data] built: train {len(train_docs)} docs / "
          f"{len(train_q)} questions; dev {len(dev_docs)} docs / "
          f"{len(dev_q)} questions (scanned {n_seen} rows)")
    return list(train_docs.values()), train_q, list(dev_docs.values()), dev_q


def load_bench(kind: str):
    """``(docs, questions)`` for ``kind`` in {"train", "dev"} (cached files)."""
    if kind not in ("train", "dev"):
        raise ValueError(f"kind must be 'train' or 'dev', got {kind!r}")
    return _read_jsonl(_docs_path(kind)), _read_jsonl(_questions_path(kind))


# --------------------------------------------------------------------------- #
# Hard-negative mining
# --------------------------------------------------------------------------- #
def groups_from_pool(questions: list[dict], pool: list[list[dict]],
                     num_negatives: int) -> tuple[list[dict], dict]:
    """Build fixed-width training groups from already-retrieved pools.

    Pure function (``pool[i]`` is the ranked ``{'text','doc_id'}`` list for
    ``questions[i]``) so the mining rule is locally testable without faiss.
    Positive = highest-ranked answer-bearing gold-doc chunk (the metric's hit
    rule); negatives = the first ``num_negatives`` other candidates by dense
    rank. Returns ``(groups, stats)``.
    """
    groups: list[dict] = []
    stats = {"questions": len(questions), "kept": 0,
             "dropped_no_positive": 0, "dropped_too_few_negatives": 0}
    for q, cands in zip(questions, pool):
        ans = normalize_text(q["answer"])
        gold = q.get("doc_title")
        pos_idx = None
        neg_idx: list[int] = []
        for i, c in enumerate(cands):
            is_pos = ans in normalize_text(c["text"]) and c["doc_id"] == gold
            if is_pos:
                if pos_idx is None:
                    pos_idx = i
            elif len(neg_idx) < num_negatives:
                neg_idx.append(i)
        if pos_idx is None:
            stats["dropped_no_positive"] += 1
            continue
        if len(neg_idx) < num_negatives:
            stats["dropped_too_few_negatives"] += 1
            continue
        groups.append({
            "question": q["question"],
            "answer": q["answer"],
            "doc_title": gold,
            "pos": cands[pos_idx]["text"],
            "pos_rank": pos_idx,
            "negs": [cands[i]["text"] for i in neg_idx],
            "neg_ranks": neg_idx,
        })
        stats["kept"] += 1
    return groups, stats


def mine_training_groups(docs: list[dict], questions: list[dict]):
    """Chunk the training corpus with the deployment config, retrieve each
    question's BGE pool, and mine the groups. Returns ``(groups, stats)``."""
    from rag_chunk import metrics, retrieval

    depth = int(C.STAGE8_MINE_DEPTH)
    index = retrieval.build_index_for_config(
        "fixed", docs,
        fixed_size=int(C.STAGE8_TRAIN_CHUNK_SIZE),
        fixed_overlap=int(C.STAGE8_TRAIN_CHUNK_OVERLAP))
    pool = index.search_chunks([q["question"] for q in questions], depth)
    pool_rec = metrics.recall_from_retrieved(pool, questions, (depth,))
    groups, stats = groups_from_pool(questions, pool,
                                     int(C.STAGE8_NUM_NEGATIVES))
    stats[f"train_pool_recall@{depth}"] = pool_rec["doc_constrained"][depth]
    stats["n_chunks"] = len(index.chunk_texts)
    stats["avg_chunk_size"] = index.avg_chunk_size()
    return groups, stats


def write_groups(groups: list[dict], stats: dict) -> None:
    train_data_dir().mkdir(parents=True, exist_ok=True)
    _write_jsonl(groups_path(), groups)
    stats_path = train_data_dir() / "stage8_mining_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[stage8-data] wrote {groups_path()} ({len(groups)} groups)")
    print(f"[stage8-data] wrote {stats_path}")


def load_groups() -> list[dict]:
    return _read_jsonl(groups_path())
