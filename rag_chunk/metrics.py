"""Evaluation metrics: Recall@k (answer-string match), Boundary F1, balance."""

from __future__ import annotations

import re

import numpy as np

import config as C


def normalize_text(s: str) -> str:
    """Lower-case + collapse whitespace — used on both chunks and answers so
    the substring match is robust to tokenisation spacing."""
    return re.sub(r"\s+", " ", s).strip().lower()


def recall_at_k(index, questions: list[dict], ks=C.RECALL_KS) -> dict[str, dict]:
    """Recall@k under two definitions, returned as
    ``{"doc_constrained": {k: r}, "unconstrained": {k: r}}``.

    * ``unconstrained`` — the gold answer string appears in *any* top-k chunk.
      This over-counts: a short answer (a year, a common name, a place) can match
      a chunk from an unrelated document, inflating recall and doing so unevenly
      between the two chunking methods.
    * ``doc_constrained`` — the answer appears in a top-k chunk **that came from
      the question's own gold document**. This is the credible measure of whether
      better chunking actually surfaces the answer-bearing passage, so it is the
      headline number; ``unconstrained`` is kept only for reference.
    """
    if not questions:
        empty = {k: 0.0 for k in ks}
        return {"doc_constrained": dict(empty), "unconstrained": dict(empty)}
    maxk = max(ks)
    queries = [q["question"] for q in questions]
    retrieved = index.search_chunks(queries, maxk)
    return recall_from_retrieved(retrieved, questions, ks)


def recall_from_retrieved(retrieved: list[list[dict]], questions: list[dict],
                          ks=C.RECALL_KS) -> dict[str, dict]:
    """Recall@k from already-retrieved per-question chunk lists.

    ``retrieved[i]`` is the ranked list of ``{'text', 'doc_id'}`` dicts for
    ``questions[i]``. Same definitions as :func:`recall_at_k` — factored out so
    the Stage 4 BM25/RRF retrievers are scored by the identical code path.
    """
    empty = {k: 0.0 for k in ks}
    if not questions:
        return {"doc_constrained": dict(empty), "unconstrained": dict(empty)}
    hits_dc = {k: 0 for k in ks}
    hits_un = {k: 0 for k in ks}
    for q, chunks in zip(questions, retrieved):
        ans = normalize_text(q["answer"])
        # Gold is one document (NQ: "doc_title") or several ("doc_titles",
        # e.g. Stage 7 TriviaQA entity pages) — a hit must come from one of
        # them. The single-title fallback keeps the NQ behaviour unchanged.
        gold_docs = tuple(q.get("doc_titles") or ()) or (q.get("doc_title"),)
        norm = [(normalize_text(c["text"]), c["doc_id"]) for c in chunks]
        for k in ks:
            top = norm[:k]
            if any(ans in text for text, _ in top):
                hits_un[k] += 1
            if any(ans in text and doc in gold_docs for text, doc in top):
                hits_dc[k] += 1
    n = len(questions)
    return {
        "doc_constrained": {k: hits_dc[k] / n for k in ks},
        "unconstrained": {k: hits_un[k] / n for k in ks},
    }


def boundary_f1(model, threshold: float | None = None, split: str = "test") -> dict:
    """Precision/Recall/F1 of predicted vs pseudo-label boundaries (Wikipedia).

    ``threshold`` defaults to the *current* ``C.BOUNDARY_THRESHOLD`` (read at
    call time, so ``C.apply(...)`` overrides take effect)."""
    import torch

    from rag_chunk import embedding, wiki_data

    if threshold is None:
        threshold = C.BOUNDARY_THRESHOLD
    device = next(model.parameters()).device
    model.eval()
    tp = fp = fn = 0
    for rec in wiki_data.load_split(split):
        labels = np.asarray(rec["labels"], dtype=int)
        if labels.size == 0:
            continue
        emb = embedding.load_article_embeddings(rec["id"])
        with torch.no_grad():
            t = torch.as_tensor(emb, dtype=torch.float32, device=device)
            pred = model.predict_boundaries(t, threshold).cpu().numpy().astype(int)
        m = min(len(pred), len(labels))    # defensive: lengths should match
        pred, labels = pred[:m], labels[:m]
        tp += int(((pred == 1) & (labels == 1)).sum())
        fp += int(((pred == 1) & (labels == 0)).sum())
        fn += int(((pred == 0) & (labels == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------- #
# Boundary-threshold calibration (Stage 2.1)
#
# The shared BOUNDARY_THRESHOLD (0.8) was tuned for the BiLSTM; the Transformer's
# sigmoid outputs sit in a different range, so 0.8 can report Boundary F1 = 0 even
# when the boundaries are learned. These helpers find the F1-optimal threshold on
# validation and explain the probability distribution. They are pure (operate on
# already-collected scores) so a single forward pass feeds the sweep, the
# diagnostics and the held-out report. ``boundary_f1`` above is left untouched, so
# Stage 1 (BiLSTM) reporting is unchanged.
# --------------------------------------------------------------------------- #
def _prf(pred_pos, labels) -> tuple:
    """``(precision, recall, f1, tp, fp, fn)`` from a boolean prediction mask and
    0/1 labels — same definitions as :func:`boundary_f1`."""
    pred_pos = np.asarray(pred_pos, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    tp = int((pred_pos & (labels == 1)).sum())
    fp = int((pred_pos & (labels == 0)).sum())
    fn = int(((~pred_pos) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, fp, fn


def collect_boundary_scores(model, split: str = "val"):
    """Concatenate per-boundary ``(probs, labels)`` over a split.

    Runs the model **once per document** (one forward pass), so the threshold
    sweep and the probability diagnostics share a single pass instead of
    re-encoding per threshold. Returns two equal-length 1-D numpy arrays.
    """
    import torch

    from rag_chunk import embedding, wiki_data

    device = next(model.parameters()).device
    model.eval()
    probs_all, labels_all = [], []
    for rec in wiki_data.load_split(split):
        labels = np.asarray(rec["labels"], dtype=int)
        if labels.size == 0:
            continue
        emb = embedding.load_article_embeddings(rec["id"])
        with torch.no_grad():
            t = torch.as_tensor(emb, dtype=torch.float32, device=device)
            p = model.predict_proba(t).cpu().numpy()
        m = min(len(p), len(labels))       # defensive: lengths should match
        probs_all.append(np.asarray(p[:m], dtype="float32"))
        labels_all.append(labels[:m])
    if not probs_all:
        return np.zeros(0, dtype="float32"), np.zeros(0, dtype=int)
    return np.concatenate(probs_all), np.concatenate(labels_all)


def boundary_f1_from_scores(probs, labels, threshold: float) -> dict:
    """Boundary precision/recall/F1 at ``threshold`` over precomputed scores.

    Same definition as :func:`boundary_f1` (strict ``probs > threshold``, to match
    ``model.predict_boundaries``) but reuses already-collected scores instead of a
    second forward pass — so a threshold picked on val reproduces that exact F1.
    """
    probs = np.asarray(probs, dtype="float32")
    prec, rec, f1, tp, fp, fn = _prf(probs > float(threshold), labels)
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "threshold": float(threshold)}


def boundary_threshold_sweep(probs, labels, thresholds=None) -> list[dict]:
    """Boundary P/R/F1 at every threshold over precomputed ``(probs, labels)``.

    One row per threshold, so a miscalibrated fixed cut (e.g. 0.8) is visible
    against the best operating point. ``thresholds`` defaults to
    ``C.BOUNDARY_THRESHOLD_GRID``. Uses strict ``probs > t`` to match
    :func:`boundary_f1`.
    """
    if thresholds is None:
        thresholds = C.BOUNDARY_THRESHOLD_GRID
    probs = np.asarray(probs, dtype="float32")
    labels = np.asarray(labels, dtype=int)
    rows = []
    for t in thresholds:
        prec, rec, f1, tp, fp, fn = _prf(probs > float(t), labels)
        rows.append({"threshold": float(t), "precision": prec, "recall": rec,
                     "f1": f1, "n_pred_pos": tp + fp, "tp": tp, "fp": fp, "fn": fn})
    return rows


def best_threshold(sweep_rows: list[dict]) -> dict:
    """Sweep row with the highest F1; ties broken toward the **higher** threshold
    (fewer, more precise cuts). ``{}`` for an empty sweep."""
    if not sweep_rows:
        return {}
    return max(sweep_rows, key=lambda r: (r["f1"], r["threshold"]))


def boundary_prob_diagnostics(probs, labels) -> dict:
    """Distribution of predicted boundary probabilities, overall and by gold label.

    Surfaces *why* a fixed threshold mis-scores: e.g. a max probability below 0.8
    forces Boundary F1 to 0 at the BiLSTM's 0.8 cut. Reports percentiles, the mean
    probability on true-boundary vs non-boundary positions, and a coarse 10-bin
    histogram.
    """
    probs = np.asarray(probs, dtype="float32")
    labels = np.asarray(labels, dtype=int)

    def stats(a):
        if a.size == 0:
            return {"n": 0}
        qs = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
        return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
                "min": float(qs[0]), "p5": float(qs[1]), "p25": float(qs[2]),
                "median": float(qs[3]), "p75": float(qs[4]), "p95": float(qs[5]),
                "max": float(qs[6])}

    edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(probs, bins=edges)
    return {
        "n": int(probs.size),
        "pos_rate": float((labels == 1).mean()) if labels.size else 0.0,
        "overall": stats(probs),
        "positives": stats(probs[labels == 1]),
        "negatives": stats(probs[labels == 0]),
        "histogram_edges": [float(e) for e in edges],
        "histogram_counts": [int(c) for c in hist],
    }


def class_balance(split: str = "train") -> dict:
    """Positive-rate and neg/pos ratio of boundary labels in a split."""
    from rag_chunk import wiki_data

    pos = neg = 0
    for rec in wiki_data.load_split(split):
        labels = rec["labels"]
        pos += sum(labels)
        neg += len(labels) - sum(labels)
    return {
        "pos": pos,
        "neg": neg,
        "pos_rate": pos / (pos + neg) if (pos + neg) else 0.0,
        "pos_weight": neg / pos if pos else float("nan"),
    }
