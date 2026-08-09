"""Interactive demo: retrieval-aware RAG chunking (ZeroGPU Space).

Two chunking strategies side by side on the study's 1000-doc / 1032-question
Natural Questions bench — fixed 15 sentences vs a learned BiLSTM boundary model
at target size 15 — ranked by one of three arms that share the same BGE top-20
pool: dense only, the off-the-shelf cross-encoder, and the NQ-fine-tuned one.

This runs the study's own pipeline (`config.py` + `rag_chunk/`, copied in
unchanged); only the entry point differs. The FAISS indices and the bench corpus
are prebuilt and downloaded from a dataset repo, so nothing is embedded at boot.
"""

# `spaces` patches torch and must be imported before it. The fallback keeps the
# app runnable off-Space (local CPU smoke test), where the decorator is a no-op.
try:
    import spaces
except ImportError:  # pragma: no cover - only off-Space
    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda fn: fn
    spaces = _SpacesShim()

import html
import json
import os
import pathlib
import re
import time

from huggingface_hub import snapshot_download

ASSETS_REPO = os.environ.get("ASSETS_REPO", "sfczaa/rag-chunking-ablation-demo-assets")
FT_REPO = os.environ.get("FT_REPO", "sfczaa/bge-reranker-base-nq-ft")
GITHUB_URL = "https://github.com/sfczaa/rag-chunking-ablation"

# Local override lets the smoke test point at staged assets without downloading.
_local = os.environ.get("LOCAL_ASSETS")
ASSETS = _local or snapshot_download(repo_id=ASSETS_REPO, repo_type="dataset")
print(f"[demo] assets: {ASSETS}", flush=True)

# config.py resolves every path from RAG_DATA_ROOT, so the snapshot layout
# (data/nq/large_n1000/...) is picked up unchanged. Must precede the import.
os.environ["RAG_DATA_ROOT"] = str(ASSETS)

import config as C  # noqa: E402

DEPTH = 20                      # pool depth, matches the study's rerank20 arm
TOP_SHOW = 5                    # chunks displayed per side
DEMO_CONFIGS = (("fixed", 15, 0), ("bilstm", 15, 0))
ARM_BGE, ARM_OTS, ARM_FT = "bge", "rerank20", "rerank20_ft"
ARM_LABELS = {
    ARM_BGE: "BGE dense (no rerank)",
    ARM_OTS: "+ off-the-shelf rerank20",
    ARM_FT: "+ fine-tuned rerank20",
}

C.apply(RETRIEVAL_EMBED_MODEL="BAAI/bge-base-en-v1.5",
        RETRIEVAL_EMBED_NORMALIZE=True)
_N = int(C.N_NQ_DOCS_LARGE)
C.apply(N_NQ_DOCS=_N)
C.apply(NQ_DIR=C.NQ_DIR / f"large_n{_N}")


# --------------------------------------------------------------------------- #
# Corpus + indices (read-only; never built here)
# --------------------------------------------------------------------------- #
def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_bench():
    """Read the cached bench directly. Deliberately avoids the dataset-streaming
    loader so the Space can never try to re-download Natural Questions."""
    docs_p = C.NQ_DIR / "docs.jsonl"
    qs_p = C.NQ_DIR / "questions.jsonl"
    for p in (docs_p, qs_p):
        if not p.exists():
            raise SystemExit(f"[demo] missing bench file {p} — the assets repo "
                             "layout must mirror the study's data root")
    return _read_jsonl(docs_p), _read_jsonl(qs_p)


def _load_indices():
    """Load the prebuilt FAISS indices. No build fallback on purpose: building
    would silently embed ~40k chunks at boot and (worse) could disagree with the
    archived numbers. A missing index is a deployment error, so fail loud."""
    from rag_chunk.retrieval import ChunkIndex

    demo_dir = C.NQ_DIR / "indices" / "demo"
    out = {}
    for method, _, _ in DEMO_CONFIGS:
        prefix = demo_dir / f"index_{method}"
        if not ChunkIndex.exists(prefix):
            raise SystemExit(f"[demo] prebuilt index missing: {prefix}.faiss/.json")
        out[method] = ChunkIndex.load(prefix)
        print(f"[demo] loaded index {method}: {len(out[method].chunk_texts)} chunks",
              flush=True)
    return out


docs, questions = _load_bench()
indices = _load_indices()
print(f"[demo] bench: {len(docs)} docs / {len(questions)} questions", flush=True)


# --------------------------------------------------------------------------- #
# Models — placed on the device at module scope (ZeroGPU requirement); no
# inference happens here, only weight placement.
# --------------------------------------------------------------------------- #
import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MAXLEN = int(C.RERANK_MAX_LENGTH)

from rag_chunk import embedding  # noqa: E402

embedding.get_embedder(role="retrieval")          # placement only
_scorers = {
    ARM_OTS: CrossEncoder(C.RERANKER_MODEL, max_length=_MAXLEN, device=DEVICE),
    ARM_FT: CrossEncoder(FT_REPO, max_length=_MAXLEN, device=DEVICE),
}
print(f"[demo] rerankers on {DEVICE}: {C.RERANKER_MODEL} | {FT_REPO}", flush=True)


# --------------------------------------------------------------------------- #
# Pure helpers (same hit rule and rendering as the study's local demo)
# --------------------------------------------------------------------------- #
def is_hit(chunk: dict, answer: str, gold_docs: tuple) -> bool:
    """The metric's own hit rule: normalized answer substring AND gold doc."""
    from rag_chunk.metrics import normalize_text

    return (normalize_text(answer) in normalize_text(chunk["text"])
            and chunk["doc_id"] in gold_docs)


def highlight(text: str, answer: str | None) -> str:
    escaped = html.escape(text)
    if not answer:
        return escaped
    pattern = re.escape(html.escape(answer))
    pattern = re.sub(r"(\\\s|\s)+", r"\\s+", pattern)   # metric collapses whitespace
    try:
        return re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>",
                      escaped, flags=re.IGNORECASE)
    except re.error:
        return escaped


_CARD_CSS = """
<style>
.demo-card {border:1px solid #8884; border-radius:8px; padding:8px 10px;
            margin:8px 0; font-size:0.9em; line-height:1.45;}
.demo-card.hit {border-color:#2ca02c; box-shadow:0 0 0 1px #2ca02c66;}
.demo-head {display:flex; gap:8px; flex-wrap:wrap; align-items:baseline;
            margin-bottom:4px;}
.demo-rank {font-weight:bold;}
.demo-move {color:#888; font-size:0.85em;}
.demo-doc {color:#888; font-size:0.85em; font-style:italic;}
.demo-badge {font-size:0.75em; border-radius:4px; padding:1px 6px;
             color:white; background:#2ca02c;}
.demo-badge.gold {background:#b8860b;}
mark {background:#ffd54f; color:#000; padding:0 1px;}
</style>
"""


def _render_side(title, avg_size, ranked, answer, gold_docs, arm) -> str:
    parts = [f"<h3 style='margin:4px 0'>{html.escape(title)}</h3>",
             f"<div class='demo-doc'>avg chunk size {avg_size:.1f} sentences · "
             f"top-{len(ranked)} of the shared BGE top-{DEPTH} pool</div>"]
    for shown_rank, (dense_rank, chunk) in enumerate(ranked, 1):
        hit = bool(answer) and is_hit(chunk, answer, gold_docs)
        badges = []
        if chunk["doc_id"] in gold_docs:
            badges.append("<span class='demo-badge gold'>gold doc</span>")
        if hit:
            badges.append("<span class='demo-badge'>answer hit</span>")
        move = ""
        if arm != ARM_BGE:
            arrow = ("=" if dense_rank + 1 == shown_rank else
                     f"dense&nbsp;#{dense_rank + 1}&nbsp;&rarr;&nbsp;#{shown_rank}")
            move = f"<span class='demo-move'>{arrow}</span>"
        parts.append(
            f"<div class='demo-card{' hit' if hit else ''}'>"
            f"<div class='demo-head'><span class='demo-rank'>#{shown_rank}</span>"
            f"{move}<span class='demo-doc'>{html.escape(str(chunk['doc_id']))}</span>"
            f"{''.join(badges)}</div>{highlight(chunk['text'], answer)}</div>")
    return _CARD_CSS + "".join(parts)


# --------------------------------------------------------------------------- #
# Inference — the only place that touches the GPU
# --------------------------------------------------------------------------- #
@spaces.GPU(duration=30)
def _retrieve_and_rank(qtext: str, arm: str):
    """Returns [(avg_size, [(dense_rank, chunk), ...], seconds), ...] per config."""
    from rag_chunk.rerank import rerank_order

    out = []
    for method, _, _ in DEMO_CONFIGS:
        index = indices[method]
        t0 = time.perf_counter()
        pool = index.search_chunks([qtext], DEPTH)[0]
        if arm == ARM_BGE:
            order = list(range(len(pool)))
        else:
            scores = _scorers[arm].predict(
                [(qtext, c["text"]) for c in pool],
                batch_size=int(C.RERANK_BATCH_SIZE), show_progress_bar=False)
            order = rerank_order(scores)
        secs = time.perf_counter() - t0
        ranked = [(j, pool[j]) for j in order[:TOP_SHOW]]
        out.append((index.avg_chunk_size(), ranked, secs))
    return out


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_app():
    import gradio as gr

    bench_labels = [f"{i + 1}. {q['question']}" for i, q in enumerate(questions)]
    by_label = dict(zip(bench_labels, questions))
    arms = [ARM_BGE, ARM_OTS, ARM_FT]
    label_to_arm = {ARM_LABELS[a]: a for a in arms}

    def run(bench_label, free_text, arm_choice):
        arm = label_to_arm.get(arm_choice, ARM_BGE)
        free_text = (free_text or "").strip()
        if free_text:
            qtext, answer, gold_docs = free_text, None, ()
            meta = ("<i>Free-text question — no gold answer/document known, "
                    "so nothing is highlighted.</i>")
        elif bench_label in by_label:
            q = by_label[bench_label]
            qtext, answer = q["question"], q["answer"]
            gold_docs = (tuple(q.get("doc_titles") or ()) or (q.get("doc_title"),))
            meta = (f"<b>Answer:</b> {html.escape(answer)} &nbsp;·&nbsp; "
                    f"<b>Gold doc:</b> "
                    f"{html.escape(', '.join(map(str, gold_docs)))}")
        else:
            return "Pick a bench question or type your own.", "", ""

        results = _retrieve_and_rank(qtext, arm)
        sides, timing = [], []
        for (method, size, overlap), (avg, ranked, secs) in zip(DEMO_CONFIGS, results):
            timing.append(f"{method} {secs:.2f}s")
            title = (f"fixed size={size}, overlap={overlap}" if method == "fixed"
                     else f"{method} target={size}, overlap={overlap}")
            sides.append(_render_side(title, avg, ranked, answer, gold_docs, arm))
        meta += (f" &nbsp;·&nbsp; <span class='demo-doc'>{ARM_LABELS[arm]} · "
                 f"{' / '.join(timing)}</span>")
        return meta, sides[0], sides[1]

    with gr.Blocks(title="RAG chunking demo") as app:
        gr.Markdown(
            "# Retrieval-aware RAG chunking — interactive demo\n"
            f"Two chunking strategies side by side on a {len(docs)}-document "
            f"Wikipedia bench with {len(questions)} Natural Questions, three "
            "ranking arms sharing one BGE top-20 pool.\n\n"
            "**What the study found:** chunk **size** dominates recall "
            "(size effect ≈ 18× the largest chunking-method effect); fixed, "
            "BiLSTM and Transformer chunking **tie** at a matched size. The one "
            "intervention that moved the sweet spot was fine-tuning the "
            "cross-encoder reranker (+0.107 R@1 in-domain) — try the third arm "
            "and watch chunks climb the ranking.\n\n"
            f"[Code, data and full write-up]({GITHUB_URL})")
        with gr.Row():
            bench = gr.Dropdown(bench_labels, label="Bench question "
                                "(NQ validation split)", value=None, scale=3)
            free = gr.Textbox(label="…or your own question (overrides the "
                              "dropdown)", scale=2)
        arm = gr.Radio([ARM_LABELS[a] for a in arms],
                       value=ARM_LABELS[ARM_FT], label="Ranking arm")
        go = gr.Button("Retrieve", variant="primary")
        meta = gr.HTML()
        with gr.Row():
            left = gr.HTML()
            right = gr.HTML()
        go.click(run, [bench, free, arm], [meta, left, right])
    return app


if __name__ == "__main__":
    build_app().launch()
