"""Stage 17 reader prompts, answer scoring and verdict rules.

Normalization follows SQuAD: lower case, punctuation removed, the articles a, an and
the removed, whitespace collapsed. A prediction is correct when the normalized stored
answer appears as a contiguous token sequence in the normalized prediction. The
prompt text and the verdict rules are part of docs/stage17_answer_quality.md.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

ARM_CLOSED, ARM_GOLD = "closed_book", "gold"
# retrieval arm -> (config label, reranker label)
RETRIEVAL_ARMS = {
    "fixed15_ft": ("fixed15", "ft"),
    "fixed6_ft": ("fixed6", "ft"),
    "bilstm15_ft": ("bilstm15", "ft"),
    "fixed15_large": ("fixed15", "large"),
}
ARMS = (ARM_CLOSED, ARM_GOLD) + tuple(RETRIEVAL_ARMS)

INSTRUCTION_PASSAGES = ("Answer the question using the passages below. Reply with the "
                        "answer only, as a short phrase, without explanation.")
INSTRUCTION_CLOSED = ("Answer the question. Reply with the answer only, as a short phrase, "
                      "without explanation.")

_PUNCT = frozenset(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_THINK = re.compile(r"<think>.*?</think>", re.S)


def normalize_answer(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", s).split())


def prediction_from_output(text: str) -> str:
    """Drop any think block, cut at the first line break, strip."""
    text = _THINK.sub("", text).strip()
    return re.split(r"\r?\n", text, maxsplit=1)[0].strip()


def contains_answer(prediction: str, answer: str) -> bool:
    gold = normalize_answer(answer).split()
    pred = normalize_answer(prediction).split()
    n = len(gold)
    return n > 0 and any(pred[i:i + n] == gold for i in range(len(pred) - n + 1))


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(answer)


def token_f1(prediction: str, answer: str) -> float:
    pred, gold = normalize_answer(prediction).split(), normalize_answer(answer).split()
    same = sum((Counter(pred) & Counter(gold)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(gold)
    return 2 * precision * recall / (precision + recall)


def user_message(question: str, passages: list[tuple[str, str]] | None) -> str:
    """The pre-registered user message; ``passages`` is None for closed book."""
    if passages is None:
        return f"{INSTRUCTION_CLOSED}\n\nQuestion: {question}"
    blocks = [f"[{i}] {title}\n{text}" for i, (title, text) in enumerate(passages, 1)]
    return INSTRUCTION_PASSAGES + "\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def cut_passage(text: str, tokenizer, max_tokens: int) -> tuple[str, bool]:
    """Keep the first ``max_tokens`` reader tokens of ``text``."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, False
    return tokenizer.decode(ids[:max_tokens]), True


def mean_ci(diffs: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """Mean of paired differences with mean +/- z standard errors."""
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1) / n)
    return mean, mean - z * se, mean + z * se


def reader_check(diffs: list[float]) -> tuple[bool, float, float, float]:
    """Criterion 2: gold minus closed book must have a 95% lower bound above 0."""
    m, lo, hi = mean_ci(diffs)
    return lo > 0, m, lo, hi


def size_verdict(diffs: list[float], floor: float) -> tuple[str, float, float, float]:
    """Criterion 3 on unrounded values: fixed15_ft minus fixed6_ft."""
    m, lo, hi = mean_ci(diffs)
    if lo > 0:
        return ("SIZE-CARRIES" if m >= floor else "SIZE-SMALL"), m, lo, hi
    if hi < 0:
        return "SIZE-REVERSED", m, lo, hi
    return "SIZE-NOT-DETECTED", m, lo, hi


def placement_verdict(diffs: list[float], margin: float) -> tuple[str, float, float, float,
                                                                   float, float]:
    """Criterion 4: bilstm15_ft minus fixed15_ft. Returns the verdict, the mean, the
    95% bounds and the 90% bounds."""
    m, lo, hi = mean_ci(diffs)
    _, lo90, hi90 = mean_ci(diffs, z=1.645)
    if lo > 0:
        verdict = "BILSTM-BETTER"
    elif hi < 0:
        verdict = "BILSTM-WORSE"
    elif -margin <= lo90 and hi90 <= margin:
        verdict = "EQUIVALENT"
    else:
        verdict = "INCONCLUSIVE"
    return verdict, m, lo, hi, lo90, hi90
