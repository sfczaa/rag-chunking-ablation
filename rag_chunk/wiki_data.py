"""Phase 1 — Wikipedia data preparation with reliable section pseudo-labels.

Why the MediaWiki API instead of the cleaned ``text`` field?
-----------------------------------------------------------
The cleaned ``text`` in ``wikimedia/wikipedia`` does **not** preserve the
``== Section ==`` markers, so section boundaries cannot be recovered reliably
from it.  We therefore:

  1. pick article *titles* deterministically by streaming the HF dump
     (no full download), then
  2. fetch the *raw wikitext* for those titles from the MediaWiki API and
     parse the real ``== Section ==`` structure with ``mwparserfromhell``.

Boundary label convention
--------------------------
``labels[i]`` is the label of the boundary *between* ``sentences[i]`` and
``sentences[i+1]`` (so ``len(labels) == len(sentences) - 1``).  It is ``1``
iff ``sentences[i+1]`` is the first sentence of a new section.

Everything is cached to Drive and resumable: raw wikitext lives in
``wiki_raw/<slug>.txt`` and is skipped if already present.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

import config as C

# Drop sentence fragments shorter than this (chars) — strip_code leftovers.
MIN_SENTENCE_CHARS = 20

_NLTK_READY = False


def ensure_nltk() -> None:
    """Make ``nltk.sent_tokenize`` usable (idempotent, Colab-friendly).

    Fails loud: if the required tokenizer can't be made available we raise here
    with a clear message, instead of letting a cryptic error surface later from
    deep inside ``sent_tokenize``.
    """
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    def _have(pkg: str) -> bool:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            return True
        except LookupError:
            return False

    download_errors: dict[str, str] = {}
    for pkg in ("punkt", "punkt_tab"):          # punkt_tab needed on nltk>=3.9
        if _have(pkg):
            continue
        try:
            nltk.download(pkg, quiet=True)
        except Exception as exc:                # network/SSL/etc. — record, verify below
            download_errors[pkg] = repr(exc)

    # `punkt_tab` only exists on newer nltk; success means at least one is present
    # AND `sent_tokenize` actually works on a probe string.
    if not (_have("punkt") or _have("punkt_tab")):
        raise RuntimeError(
            "NLTK sentence tokenizer is unavailable: could not find or download "
            f"'punkt'/'punkt_tab'. Download errors: {download_errors or 'none'}. "
            "On Colab run `import nltk; nltk.download('punkt'); "
            "nltk.download('punkt_tab')` in a cell with network access, then retry."
        )
    from nltk.tokenize import sent_tokenize
    try:
        sent_tokenize("This is a probe. It has two sentences.")
    except Exception as exc:
        raise RuntimeError(
            f"NLTK punkt is present but sent_tokenize failed: {exc!r}. "
            "The tokenizer data may be partially downloaded — delete the nltk_data "
            "tokenizers folder and re-download 'punkt' and 'punkt_tab'."
        ) from exc
    _NLTK_READY = True


def split_sentences(text: str) -> list[str]:
    ensure_nltk()
    from nltk.tokenize import sent_tokenize

    out = []
    for s in sent_tokenize(text):
        s = s.strip()
        if len(s) >= MIN_SENTENCE_CHARS:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Title selection (deterministic, streamed — no full dump download)
# --------------------------------------------------------------------------- #
def get_article_titles(n: int = C.N_WIKI_ARTICLES) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(
        C.WIKI_DUMP, C.WIKI_DUMP_CONFIG, split="train", streaming=True
    )
    titles: list[str] = []
    for row in islice(ds, n):
        t = row.get("title")
        if t:
            titles.append(t)
    return titles


# --------------------------------------------------------------------------- #
# Raw wikitext fetching (MediaWiki API, batched + cached + resumable)
# --------------------------------------------------------------------------- #
def _slug(title: str) -> str:
    h = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    base = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:60]
    return f"{base}_{h}" if base else h


def _batches(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_wikitext(titles: list[str], sleep: float = 0.1) -> int:
    """Fetch raw wikitext for ``titles`` into ``wiki_raw/``. Returns #new files."""
    import requests

    sess = requests.Session()
    sess.headers.update({"User-Agent": C.WIKI_API_USER_AGENT})
    n_new = 0
    for batch in _batches(titles, C.WIKI_API_BATCH):
        pending = [t for t in batch if not (C.WIKI_RAW_DIR / f"{_slug(t)}.txt").exists()]
        if not pending:
            continue
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2",
            "titles": "|".join(pending),
        }
        resp = sess.get(C.WIKI_API_ENDPOINT, params=params, timeout=60)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", [])
        for page in pages:
            if page.get("missing") or "revisions" not in page:
                continue
            try:
                content = page["revisions"][0]["slots"]["main"]["content"]
            except (KeyError, IndexError):
                continue
            (C.WIKI_RAW_DIR / f"{_slug(page['title'])}.txt").write_text(
                content, encoding="utf-8"
            )
            n_new += 1
        time.sleep(sleep)
    return n_new


# --------------------------------------------------------------------------- #
# wikitext -> (sentences, labels)
# --------------------------------------------------------------------------- #
def _clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", " ", text)              # leftover [12] ref marks
    lines = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if ln.count("|") >= 2:                        # table-row residue
            continue
        lines.append(ln)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_refs_and_comments(wikitext: str) -> str:
    wikitext = re.sub(r"<ref[^>]*>.*?</ref>", " ", wikitext, flags=re.DOTALL | re.IGNORECASE)
    wikitext = re.sub(r"<ref[^>]*/>", " ", wikitext, flags=re.IGNORECASE)
    wikitext = re.sub(r"<!--.*?-->", " ", wikitext, flags=re.DOTALL)
    return wikitext


def parse_article(wikitext: str):
    """Return ``(sentences, labels)`` or ``(None, None)`` if too short.

    ``labels[i]`` == 1 iff ``sentences[i+1]`` begins a new section.
    """
    import mwparserfromhell

    code = mwparserfromhell.parse(_strip_refs_and_comments(wikitext))

    section_sentences: list[list[str]] = []
    for section in code.get_sections(levels=[2], include_lead=True, flat=True):
        sec = mwparserfromhell.parse(str(section))
        headings = sec.filter_headings()
        name = (
            str(headings[0].title.strip_code()).strip().lower()
            if headings
            else "__lead__"
        )
        if name in C.WIKI_SKIP_SECTIONS:
            continue
        for h in list(sec.filter_headings()):         # drop heading from body text
            sec.remove(h)
        text = _clean_text(str(sec.strip_code()))
        sents = split_sentences(text)
        if len(sents) < C.MIN_SENTENCES_PER_SECTION:
            continue
        section_sentences.append(sents)

    sentences: list[str] = []
    labels: list[int] = []
    for sec_sents in section_sentences:
        for j, s in enumerate(sec_sents):
            if sentences:                             # a boundary precedes this sentence
                labels.append(1 if j == 0 else 0)
            sentences.append(s)

    if len(sentences) < C.MIN_SENTENCES_PER_ARTICLE:
        return None, None
    return sentences, labels


# --------------------------------------------------------------------------- #
# Orchestration: build & split the dataset
# --------------------------------------------------------------------------- #
def _split_path(split: str) -> Path:
    return C.WIKI_PROCESSED_DIR / f"{split}.jsonl"


def prepare_dataset(n_articles: int | None = None) -> dict:
    """Full Phase-1 pipeline. Writes train/val/test jsonl, returns stats."""
    if n_articles is None:
        n_articles = C.N_WIKI_ARTICLES
    C.ensure_dirs()
    titles = get_article_titles(n_articles)
    print(f"[wiki] selected {len(titles)} titles; fetching wikitext ...")
    n_new = fetch_wikitext(titles)
    print(f"[wiki] fetched {n_new} new articles (rest were cached)")

    records = []
    for txt_path in sorted(C.WIKI_RAW_DIR.glob("*.txt")):
        sents, labels = parse_article(txt_path.read_text(encoding="utf-8"))
        if sents is None:
            continue
        records.append({"id": txt_path.stem, "sentences": sents, "labels": labels})
        if len(records) >= n_articles:
            break
    print(f"[wiki] parsed {len(records)} usable articles")

    rng = random.Random(C.SEED)
    rng.shuffle(records)
    n = len(records)
    n_train = int(C.SPLIT_RATIOS[0] * n)
    n_val = int(C.SPLIT_RATIOS[1] * n)
    splits = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }

    pos = neg = 0
    for split, recs in splits.items():
        with _split_path(split).open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                pos += sum(r["labels"])
                neg += len(r["labels"]) - sum(r["labels"])

    stats = {
        "n_articles": n,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "pos_boundaries": pos,
        "neg_boundaries": neg,
        "pos_rate": pos / (pos + neg) if (pos + neg) else 0.0,
        "pos_weight": neg / pos if pos else float("nan"),
    }
    print(f"[wiki] stats: {stats}")
    return stats


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def load_split(split: str) -> list[dict]:
    path = _split_path(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run prepare_dataset() first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def iter_all_records() -> Iterable[dict]:
    for split in ("train", "val", "test"):
        if _split_path(split).exists():
            yield from load_split(split)
