"""Route D - interactive retrieval demo (Gradio).

One question in, two chunking strategies side by side — **fixed 15/0** vs
**BiLSTM t15/0** (the Stage 6/8 deployment-size configs) on the Stage 6 bench
corpus (1000 docs / 1032 questions) — ranked by one of three arms that share
the same BGE top-20 pool:

    bge          the dense order (no reranking);
    rerank20     + the off-the-shelf ``BAAI/bge-reranker-base``;
    rerank20_ft  + the Stage 8 fine-tuned reranker (shown only if
                 ``models/bge_reranker_ft/final`` exists).

Bench questions carry their gold answer + document, so the demo highlights
the answer string, badges chunks from the gold document, and shows each
chunk's dense-rank movement (e.g. "dense #7 -> #1") — the rerank
before/after in one glance. Free-text questions just show retrieval.

The demo runs no experiments and writes nothing under ``results/`` — the two
FAISS indices it needs are built once and cached under
``data/nq/large_n<N>/indices/demo/`` (first build embeds ~40k chunks, a few
minutes on a T4; later launches load instantly).

Usage:
    python scripts/19_demo.py --share       # Colab: prints a public link
    python scripts/19_demo.py               # local: http://127.0.0.1:7860
    python scripts/19_demo.py --rebuild-index
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

DEPTH = 20                      # pool depth, matches Stage 5/6/8 rerank20
TOP_SHOW = 5                    # chunks displayed per side
DEMO_CONFIGS = (("fixed", 15, 0), ("bilstm", 15, 0))
ARM_BGE, ARM_OTS, ARM_FT = "bge", "rerank20", "rerank20_ft"
ARM_LABELS = {
    ARM_BGE: "BGE dense (no rerank)",
    ARM_OTS: "+ off-the-shelf rerank20",
    ARM_FT: "+ fine-tuned rerank20 (Stage 8)",
}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable without torch/gradio)
# --------------------------------------------------------------------------- #
def is_hit(chunk: dict, answer: str, gold_docs: tuple) -> bool:
    """The metric's own hit rule: normalized answer substring AND gold doc."""
    from rag_chunk.metrics import normalize_text

    return (normalize_text(answer) in normalize_text(chunk["text"])
            and chunk["doc_id"] in gold_docs)


def highlight(text: str, answer: str | None) -> str:
    """HTML-escaped chunk text with the answer marked (case- and
    whitespace-insensitive, mirroring the metric's normalization)."""
    escaped = html.escape(text)
    if not answer:
        return escaped
    pattern = re.escape(html.escape(answer))
    # the metric collapses whitespace before matching; do the same here
    pattern = re.sub(r"(\\\s|\s)+", r"\\s+", pattern)
    try:
        return re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>",
                      escaped, flags=re.IGNORECASE)
    except re.error:
        return escaped


def top_with_dense_ranks(pool: list[dict], order: list[int],
                         k: int) -> list[tuple[int, dict]]:
    """First ``k`` of ``order`` as (dense_rank, chunk) pairs."""
    return [(j, pool[j]) for j in order[:k]]


# --------------------------------------------------------------------------- #
# Index cache
# --------------------------------------------------------------------------- #
def _demo_index_dir() -> pathlib.Path:
    return C.NQ_DIR / "indices" / "demo"


def _index_signature(n_docs: int) -> dict:
    return {"retrieval_model": C.RETRIEVAL_EMBED_MODEL,
            "n_docs": n_docs, "depth": DEPTH,
            "configs": [list(c) for c in DEMO_CONFIGS]}


def _load_or_build_indices(docs: list[dict], rebuild: bool) -> dict:
    from rag_chunk import retrieval, sweep, training
    from rag_chunk.retrieval import ChunkIndex

    demo_dir = _demo_index_dir()
    manifest = demo_dir / "demo_manifest.json"
    prefixes = {m: demo_dir / f"index_{m}" for m, _, _ in DEMO_CONFIGS}
    sig = _index_signature(len(docs))

    if not rebuild and manifest.exists():
        try:
            cached_sig = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached_sig = None
        if cached_sig == sig and all(ChunkIndex.exists(p)
                                     for p in prefixes.values()):
            out = {m: ChunkIndex.load(p) for m, p in prefixes.items()}
            print(f"[demo] loaded {len(out)} cached indices from {demo_dir}")
            return out

    print("[demo] building demo indices (one-time; embeds every chunk) ...")
    demo_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for method, size, overlap in DEMO_CONFIGS:
        print(f"[demo]   {method} size={size} overlap={overlap}")
        if method == "fixed":
            out[method] = retrieval.build_index_for_config(
                method, docs, fixed_size=size, fixed_overlap=overlap,
                save_prefix=prefixes[method])
        else:
            mn, mx = sweep._semantic_window(size)
            out[method] = retrieval.build_index_for_config(
                method, docs, model=training.load_model(method),
                semantic_policy="target", semantic_target_size=size,
                semantic_min_size=mn, semantic_max_size=mx,
                semantic_overlap=overlap, save_prefix=prefixes[method])
    manifest.write_text(json.dumps(sig), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_CARD_CSS = """
<style>
.demo-card {border:1px solid #8884; border-radius:8px; padding:8px 10px;
            margin:8px 0; font-size:0.9em; line-height:1.45;}
.demo-card.hit {border-color:#2ca02c; box-shadow:0 0 0 1px #2ca02c66;}
.demo-head {display:flex; gap:8px; flex-wrap:wrap; align-items:baseline;
            margin-bottom:4px;}
.demo-rank {font-weight:bold;}
.demo-move {color:#666; font-size:0.85em;}
.demo-doc {color:#666; font-size:0.85em; font-style:italic;}
.demo-badge {font-size:0.75em; border-radius:4px; padding:1px 6px;
             color:white; background:#2ca02c;}
.demo-badge.gold {background:#b8860b;}
mark {background:#ffd54f; color:inherit; padding:0 1px;}
</style>
"""


def _render_side(title: str, avg_size: float, ranked: list[tuple[int, dict]],
                 answer: str | None, gold_docs: tuple, arm: str) -> str:
    parts = [f"<h3 style='margin:4px 0'>{html.escape(title)}</h3>",
             f"<div class='demo-doc'>avg chunk size "
             f"{avg_size:.1f} sentences · top-{len(ranked)} of the shared "
             f"BGE top-{DEPTH} pool</div>"]
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
                     f"dense&nbsp;#{dense_rank + 1}&nbsp;→&nbsp;#{shown_rank}")
            move = f"<span class='demo-move'>{arrow}</span>"
        parts.append(
            f"<div class='demo-card{' hit' if hit else ''}'>"
            f"<div class='demo-head'><span class='demo-rank'>#{shown_rank}"
            f"</span>{move}<span class='demo-doc'>"
            f"{html.escape(str(chunk['doc_id']))}</span>{''.join(badges)}"
            f"</div>{highlight(chunk['text'], answer)}</div>")
    return _CARD_CSS + "".join(parts)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def build_app(docs, questions, indices, scorers):
    import gradio as gr

    from rag_chunk.rerank import rerank_order

    bench_labels = [f"{i + 1}. {q['question']}"
                    for i, q in enumerate(questions)]
    by_label = dict(zip(bench_labels, questions))
    arms = [a for a in (ARM_BGE, ARM_OTS, ARM_FT) if a == ARM_BGE
            or a in scorers]

    def run(bench_label, free_text, arm):
        free_text = (free_text or "").strip()
        if free_text:
            qtext, answer, gold_docs = free_text, None, ()
            meta = ("<i>Free-text question — no gold answer/document known, "
                    "so nothing is highlighted.</i>")
        elif bench_label in by_label:
            q = by_label[bench_label]
            qtext, answer = q["question"], q["answer"]
            gold_docs = (tuple(q.get("doc_titles") or ())
                         or (q.get("doc_title"),))
            meta = (f"<b>Answer:</b> {html.escape(answer)} &nbsp;·&nbsp; "
                    f"<b>Gold doc:</b> "
                    f"{html.escape(', '.join(map(str, gold_docs)))}")
        else:
            return ("Pick a bench question or type your own.", "", "")

        sides, timing = [], []
        for method, size, overlap in DEMO_CONFIGS:
            index = indices[method]
            t0 = time.perf_counter()
            pool = index.search_chunks([qtext], DEPTH)[0]
            if arm == ARM_BGE:
                order = list(range(len(pool)))
            else:
                scores = scorers[arm]([(qtext, c["text"]) for c in pool])
                order = rerank_order(scores)
            secs = time.perf_counter() - t0
            timing.append(f"{method} {secs:.2f}s")
            title = (f"fixed size={size}, overlap={overlap}"
                     if method == "fixed" else
                     f"{method} target={size}, overlap={overlap}")
            sides.append(_render_side(title, index.avg_chunk_size(),
                                      top_with_dense_ranks(pool, order,
                                                           TOP_SHOW),
                                      answer, gold_docs, arm))
        meta += (f" &nbsp;·&nbsp; <span class='demo-doc'>"
                 f"{ARM_LABELS[arm]} · {' / '.join(timing)}</span>")
        return meta, sides[0], sides[1]

    with gr.Blocks(title="RAG chunking demo") as app:
        gr.Markdown(
            "# Retrieval-aware RAG chunking — interactive demo\n"
            "Two chunking strategies side by side on the Stage 6 bench "
            f"({len(docs)} Wikipedia docs / {len(questions)} NQ questions), "
            "three ranking arms sharing one BGE top-20 pool. Project "
            "findings: chunk **size** dominates recall, chunking methods tie "
            "— and the Stage 8 **fine-tuned** reranker is the only "
            "intervention that lifts R@1 at the size-15 sweet spot "
            "(+0.107).")
        with gr.Row():
            bench = gr.Dropdown(bench_labels, label="Bench question "
                                "(NQ validation split)", value=None, scale=3)
            free = gr.Textbox(label="…or your own question (overrides the "
                              "dropdown)", scale=2)
        arm = gr.Radio([ARM_LABELS[a] for a in arms],
                       value=ARM_LABELS[arms[-1]], label="Ranking arm")
        go = gr.Button("Retrieve", variant="primary")
        meta = gr.HTML()
        with gr.Row():
            left = gr.HTML()
            right = gr.HTML()

        label_to_arm = {ARM_LABELS[a]: a for a in arms}

        def _run(bench_label, free_text, arm_choice):
            return run(bench_label, free_text,
                       label_to_arm.get(arm_choice, arm_choice))

        go.click(_run, [bench, free, arm], [meta, left, right])
    return app


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Route D: interactive Gradio demo of the retrieval "
                    "pipeline (no experiments, nothing written to results/).")
    ap.add_argument("--share", action="store_true",
                    help="create a public gradio.live link (use on Colab)")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="rebuild the cached demo indices")
    ap.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5",
                    help="dense retrieval model; must match the bench")
    args = ap.parse_args()

    try:
        import gradio  # noqa: F401
    except ImportError:
        raise SystemExit("[demo] gradio is not installed — "
                         "pip install gradio")

    # Same corpus redirect as Stage 6/8: reuse the archived bench cache.
    C.apply(RETRIEVAL_EMBED_MODEL=args.retrieval_model,
            RETRIEVAL_EMBED_NORMALIZE=True)
    n = int(C.N_NQ_DOCS_LARGE)
    C.apply(N_NQ_DOCS=n)
    C.apply(NQ_DIR=C.NQ_DIR / f"large_n{n}")
    C.ensure_dirs()

    from rag_chunk import nq_data

    docs, questions = nq_data.prepare_nq()
    indices = _load_or_build_indices(docs, args.rebuild_index)

    from sentence_transformers import CrossEncoder
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = None
    if device != "cuda":
        print(f"[demo] WARN: no GPU — reranking {2 * DEPTH} pairs/question "
              "on CPU takes a minute or more; the bge arm stays fast")

    max_len = int(C.RERANK_MAX_LENGTH)

    def _scorer(model):
        def score(pairs):
            return model.predict(pairs, batch_size=int(C.RERANK_BATCH_SIZE),
                                 show_progress_bar=False)
        return score

    scorers = {ARM_OTS: _scorer(CrossEncoder(C.RERANKER_MODEL,
                                             max_length=max_len,
                                             device=device))}
    ft_dir = C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"
    if (ft_dir / "config.json").exists():
        scorers[ARM_FT] = _scorer(CrossEncoder(str(ft_dir),
                                               max_length=max_len,
                                               device=device))
    else:
        print(f"[demo] WARN: no fine-tuned reranker at {ft_dir} — "
              "the rerank20_ft arm is hidden")

    app = build_app(docs, questions, indices, scorers)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
