"""Stage 17 (step 2) - reader outputs for every arm.

Reads results/latest/stage17_contexts.jsonl and has the pinned reader answer each
bench question once per arm: greedy, fp16, one question per forward pass. Every
STAGE17_SAVE_EVERY outputs are appended to stage17_generations/<arm>.jsonl with the
prompt hash and the GPU name, so a lost runtime loses at most that many and a restart
continues after the last saved row. The prompt, passage cap and decoding are fixed in
docs/stage17_answer_quality.md.

--smoke runs the reader on 16 NQ-train groups (closed book, the positive alone, and
the positive with four negatives), checks every generation step for non-finite
logits and every output for an empty prediction, and keeps nothing. Either failure is
the fallback condition of the pre-registration.

Usage:
    python scripts/42_stage17_generate.py --smoke
    python scripts/42_stage17_generate.py --account A
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402
from rag_chunk import answer_eval as AE  # noqa: E402


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S41 = _load_script("41_stage17_contexts.py", "stage17_contexts")
SMOKE_GROUPS = 16


def load_reader(device: str, name: str | None = None, revision: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if name is None:
        name, revision = S41.reader_spec()
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype)
    return model.to(device).eval(), tok


def build_prompt(tok, question: str, passages):
    """(prompt text, passages after the cap or None, number of cut passages)."""
    cut, n_cut = None, 0
    if passages is not None:
        cut = []
        for title, text in passages:
            kept, was_cut = AE.cut_passage(text, tok, int(C.STAGE17_PASSAGE_TOKENS))
            cut.append((title, kept))
            n_cut += was_cut
    message = AE.user_message(question, cut)
    prompt = tok.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                     add_generation_prompt=True)
    return prompt, cut, n_cut


def generate(model, tok, prompt: str, scores: bool = False):
    """(decoded new text, prompt tokens, all step logits finite or None)."""
    import torch

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=int(C.STAGE17_MAX_NEW_TOKENS),
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=pad, output_scores=scores,
                             return_dict_in_generate=True)
    n_in = int(enc["input_ids"].shape[1])
    text = tok.decode(out.sequences[0, n_in:], skip_special_tokens=True)
    finite = all(bool(torch.isfinite(s).all()) for s in out.scores) if scores else None
    return text, n_in, finite


def answer_in_prompt(question: dict, cut) -> bool | None:
    """Whether the answer string survives the cap in a passage from the gold document."""
    from rag_chunk.metrics import normalize_text

    if cut is None:
        return None
    ans = normalize_text(question["answer"])
    return any(title == question["doc_title"] and ans in normalize_text(text)
               for title, text in cut)


def kept_questions(questions: list[dict]) -> list[int]:
    """Questions whose stored answer is not empty after normalization."""
    return [i for i, q in enumerate(questions) if AE.normalize_answer(q["answer"])]


def work_items(arm: str, kept: list[int], ctx_rows: dict) -> list[tuple]:
    """(question index, passages or None) for every question this arm answers."""
    items = []
    for qi in kept:
        if arm == AE.ARM_CLOSED:
            items.append((qi, None))
            continue
        ctx = ctx_rows[arm]["contexts"][qi]
        if ctx:
            items.append((qi, [(c["doc_id"], c["text"]) for c in ctx]))
    return items


def _write_lines(path: pathlib.Path, rows: list[dict], mode: str) -> None:
    with path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_rows(path: pathlib.Path) -> list[dict]:
    """Saved rows. A torn last line left by a lost runtime is dropped and the file is
    rewritten without it."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise SystemExit(f"[stage17] {path} line {i + 1} is unreadable; stopping")
            print(f"[stage17] {path.name}: dropping a torn last line", flush=True)
    if raw and (not raw.endswith("\n") or len(rows) != len(lines)):
        tmp = path.with_name(path.name + ".tmp")
        _write_lines(tmp, rows, "w")
        os.replace(tmp, path)
    return rows


def prompt_sha1(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def generate_arm(arm, items, questions, model, tok, gpu, out_path, gen=generate):
    """Answer every item not yet saved for ``arm``; returns the number generated."""
    done = {r["qid"]: r for r in read_rows(out_path)}
    other = sorted({r["gpu"] for r in done.values()} - {gpu})
    if other:
        raise SystemExit(f"[stage17] {arm}: saved rows came from {other}, this runtime has "
                         f"{gpu}. The run uses one GPU type; stopping.")
    for qi, passages in items:
        if qi in done:
            prompt, _, _ = build_prompt(tok, questions[qi]["question"], passages)
            if prompt_sha1(prompt) != done[qi]["prompt_sha1"]:
                raise SystemExit(f"[stage17] {arm}: the prompt of saved question {qi} "
                                 "differs from the one rebuilt now; stopping.")
    todo = [(qi, p) for qi, p in items if qi not in done]
    print(f"[stage17] {arm}: {len(done)} saved, {len(todo)} to generate", flush=True)
    every = int(C.STAGE17_SAVE_EVERY)
    buf, t_arm = [], time.perf_counter()
    for n, (qi, passages) in enumerate(todo, 1):
        q = questions[qi]
        t0 = time.perf_counter()
        prompt, cut, n_cut = build_prompt(tok, q["question"], passages)
        raw, n_in, _ = gen(model, tok, prompt)
        buf.append({"qid": qi, "arm": arm, "raw": raw,
                    "prediction": AE.prediction_from_output(raw), "prompt_tokens": n_in,
                    "n_cut": n_cut, "answer_in_prompt": answer_in_prompt(q, cut),
                    "prompt_sha1": prompt_sha1(prompt),
                    "seconds": round(time.perf_counter() - t0, 3), "gpu": gpu})
        if len(buf) >= every or n == len(todo):
            _write_lines(out_path, buf, "a")
            buf = []
            rate = (time.perf_counter() - t_arm) / n
            print(f"[stage17] {arm}: {len(done) + n}/{len(items)} saved, "
                  f"{rate:.2f} s per question", flush=True)
    return len(todo)


def smoke() -> None:
    import torch

    from rag_chunk import rerank_finetune as rf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name, revision = S41.reader_spec()
    print(f"[stage17] smoke: reader {name} at {revision} on {device}", flush=True)
    model, tok = load_reader(device)
    bad_logits = empty = correct = n = 0
    for g in rf.load_groups()[:SMOKE_GROUPS]:
        # negatives carry no title in the groups file, so the group's title labels all five
        sets = {"closed": None, "positive": [(g["doc_title"], g["pos"])],
                "five": [(g["doc_title"], t) for t in [g["pos"]] + list(g["negs"])[:4]]}
        for kind, passages in sets.items():
            prompt, _, n_cut = build_prompt(tok, g["question"], passages)
            raw, n_in, finite = generate(model, tok, prompt, scores=True)
            pred = AE.prediction_from_output(raw)
            ok = AE.contains_answer(pred, g["answer"])
            n += 1
            correct += ok
            bad_logits += not finite
            empty += not pred
            print(f"[stage17]   {kind:8s} {n_in:5d} tok  cut {n_cut}  finite {finite}  "
                  f"{'hit ' if ok else 'miss'} {pred[:60]!r} | gold {g['answer']!r}",
                  flush=True)
    print(f"[stage17] smoke: {n} generations, non-finite steps in {bad_logits}, empty "
          f"predictions {empty}, answer contained in {correct}", flush=True)
    if bad_logits or empty:
        raise SystemExit("[stage17] FALLBACK CONDITION MET: record the switch to the fallback "
                         "reader in docs/stage17_answer_quality.md and set "
                         "STAGE17_USE_FALLBACK before any bench question is generated.")
    print("[stage17] smoke: the reader passed. Nothing was kept.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 17 step 2: reader outputs per arm.")
    ap.add_argument("--smoke", action="store_true",
                    help=f"reader check on {SMOKE_GROUPS} NQ-train groups, nothing kept")
    ap.add_argument("--account", default="unlabelled", help="label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return

    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE17_ROOT_ID)
    gen_dir = C.RESULTS_LATEST_DIR / C.STAGE17_GENERATIONS_DIRNAME
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE17_LOCK_FILE,
                             run_version=C.STAGE17_RUN_VERSION, account=args.account,
                             task="generate", clear_stale=args.clear_stale_lock):
        run(gen_dir)


def run(gen_dir: pathlib.Path) -> None:
    import torch

    from rag_chunk import run_guard

    latest = C.RESULTS_LATEST_DIR
    ident = latest / C.STAGE17_IDENTITY_JSON
    if not ident.exists():
        raise SystemExit("[stage17] no run identity: run scripts/41_stage17_contexts.py "
                         "--mode fresh first")
    _, questions, titles_sha1 = S41.S16.B16.stage6_bench()
    run_guard.check_run_identity(ident, S41.run_identity(titles_sha1))
    ctx_path = latest / C.STAGE17_CONTEXTS_JSONL
    if not ctx_path.exists():
        raise SystemExit("[stage17] no contexts: run scripts/41_stage17_contexts.py first")
    meta, ctx_rows = S41.read_contexts(ctx_path)
    if meta["n_questions"] != len(questions):
        raise SystemExit(f"[stage17] the contexts hold {meta['n_questions']} questions, the "
                         f"bench {len(questions)}")
    if not meta["valid"]:
        raise SystemExit(f"[stage17] the contexts failed criterion 1 ({meta['why_invalid']}); "
                         "the verdict is INVALID and nothing is generated.")
    if not torch.cuda.is_available():
        raise SystemExit("[stage17] no CUDA device on this runtime")
    gpu = torch.cuda.get_device_name(0)
    run_guard.probe_directory(gen_dir)
    seen = sorted({r["gpu"] for arm in AE.ARMS
                   for r in read_rows(gen_dir / f"{arm}.jsonl")} - {gpu})
    if seen:
        raise SystemExit(f"[stage17] saved rows came from {seen}, this runtime has {gpu}. "
                         "The run uses one GPU type; stopping.")
    kept = kept_questions(questions)
    print(f"[stage17] {len(kept)} of {len(questions)} questions scored; GPU {gpu}", flush=True)
    t0 = time.perf_counter()
    model, tok = load_reader("cuda")
    print(f"[stage17] reader loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    for arm in AE.ARMS:
        generate_arm(arm, work_items(arm, kept, ctx_rows), questions, model, tok, gpu,
                     gen_dir / f"{arm}.jsonl")
    print("[stage17] every arm is complete; run scripts/43_stage17_score.py", flush=True)


if __name__ == "__main__":
    main()
