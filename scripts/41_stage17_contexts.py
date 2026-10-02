"""Stage 17 (step 1) - the reader inputs for every arm.

Builds the fixed 15/0, fixed 6/0 and bilstm 15/0 indices over the Stage 6 bench,
reorders each BGE top-20 pool with the Stage 8 reranker (at fixed 15/0 also with the
Stage 15 reranker), and writes the top five chunks per question and arm to
results/latest/stage17_contexts.jsonl, with one gold chunk per question for the
`gold` arm. The retrieval rows are checked against stage8/final and stage15/final,
criterion 1 of docs/stage17_answer_quality.md.

At fixed 15/0 the Stage 15 score cache is copied in and reused when its scored pairs
match.

Usage:
    python scripts/41_stage17_contexts.py --smoke                  # 40 questions, temp dir
    python scripts/41_stage17_contexts.py --mode fresh --account A
    python scripts/41_stage17_contexts.py --mode resume --account B
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S11 = _load_script("28_eval_reranker_rl.py", "stage11_eval")
S15 = _load_script("37_eval_large_reranker.py", "stage15_eval")
S16 = _load_script("40_eval_holdout.py", "stage16_eval")
CONFIGS = {"fixed15": ("fixed", 15, 0), "fixed6": ("fixed", 6, 0),
           "bilstm15": ("bilstm", 15, 0)}
STAGE15_CACHE = {"rerank20_ft": "fixed15_ft", "rerank20_s15_large": "fixed15_large"}
SMOKE_QUESTIONS = 40


def weight_paths() -> dict:
    return {"ft": C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final" / "model.safetensors",
            "large": (C.MODELS_DIR / C.STAGE15_MODEL_DIRNAME / "large" / "final"
                      / "model.safetensors"),
            "bilstm": C.model_path("bilstm")}


def expected_hashes() -> dict:
    return {"ft": C.STAGE16_STAGE8_SHA256, "large": C.STAGE16_LARGE_SHA256,
            "bilstm": C.STAGE17_BILSTM_SHA256}


def reader_spec() -> tuple[str, str]:
    if C.STAGE17_USE_FALLBACK:
        return C.STAGE17_FALLBACK_MODEL, C.STAGE17_FALLBACK_REVISION
    return C.STAGE17_READER_MODEL, C.STAGE17_READER_REVISION


def run_identity(titles_sha1: str) -> dict:
    """Everything a resumed run must share with the first one."""
    from rag_chunk import answer_eval as AE

    model, revision = reader_spec()
    prompt = AE.user_message("q", [("t", "x")]) + "\x00" + AE.user_message("q", None)
    return {"run_version": C.STAGE17_RUN_VERSION, "reader_model": model,
            "reader_revision": revision, "top_k": int(C.STAGE17_TOP_K),
            "passage_tokens": int(C.STAGE17_PASSAGE_TOKENS),
            "max_new_tokens": int(C.STAGE17_MAX_NEW_TOKENS),
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            "bench_titles_sha1": titles_sha1, "weights_sha256": expected_hashes()}


def read_contexts(path: pathlib.Path) -> tuple[dict, dict]:
    """``(meta, {arm: row})`` from a contexts file."""
    meta, rows = None, {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if "meta" in obj:
                meta = obj["meta"]
            else:
                rows[obj["arm"]] = obj
    if meta is None:
        raise SystemExit(f"[stage17] {path} has no meta line")
    return meta, rows


def gold_contexts(docs, questions) -> list[list[dict]]:
    """The first fixed 15/0 chunk of each question's document that holds the answer."""
    from rag_chunk.chunking import chunk_text, fixed_chunks
    from rag_chunk.metrics import normalize_text

    by_title = {d["title"]: d for d in docs}
    out = []
    for q in questions:
        ans, found = normalize_text(q["answer"]), []
        for chunk in fixed_chunks(by_title[q["doc_title"]]["sentences"], 15, 0):
            text = chunk_text(chunk)
            if ans in normalize_text(text):
                found = [{"doc_id": q["doc_title"], "text": text}]
                break
        out.append(found)
    return out


def config_arms(label, docs, questions, scorers, cache_dir, probs=None) -> dict:
    """One dense pool per config, reordered by every scorer; ``{arm: row}``."""
    from rag_chunk import metrics, retrieval, sweep
    from rag_chunk.rerank import rerank_order

    method, size, overlap = CONFIGS[label]
    if method == "fixed":
        dense = retrieval.build_index_for_config("fixed", docs, fixed_size=size,
                                                 fixed_overlap=overlap)
    else:
        lo, hi = sweep._semantic_window(size)
        dense = retrieval.build_index_for_config(
            method, docs, boundary_probs_by_id=probs, semantic_policy="target",
            semantic_target_size=size, semantic_min_size=lo, semantic_max_size=hi,
            semantic_overlap=overlap)
    k = int(C.STAGE17_TOP_K)
    queries = [q["question"] for q in questions]
    pool = dense.search_chunks(queries, S11.DEPTH)
    pool_rec = metrics.recall_from_retrieved(pool, questions, (S11.DEPTH,))
    pairs, counts = [], []
    for qtext, cands in zip(queries, pool):
        pairs.extend((qtext, c["text"]) for c in cands)
        counts.append(len(cands))

    ranked = {f"{label}_bge": ([cands[:k] for cands in pool], None)}
    for arm, scorer in scorers.items():
        scores = scorer(pairs)
        out, off = [], 0
        for cands, n in zip(pool, counts):
            order = rerank_order(scores[off:off + n])
            out.append([cands[j] for j in order[:k]])
            off += n
        meta = cache_dir / f"{arm}.json"
        seconds = (json.loads(meta.read_text(encoding="utf-8"))["seconds"]
                   if meta.exists() else None)
        ranked[arm] = (out, seconds)

    rows = {}
    for arm, (retrieved, seconds) in ranked.items():
        rec = metrics.recall_from_retrieved(retrieved, questions, C.RECALL_KS)
        row = {"arm": arm, "config": label, "method": method,
               "fixed_size": size if method == "fixed" else None,
               "fixed_overlap": overlap if method == "fixed" else None,
               "n_chunks": len(dense.chunk_texts), "avg_chunk_size": dense.avg_chunk_size(),
               f"pool_recall@{S11.DEPTH}": pool_rec["doc_constrained"][S11.DEPTH],
               "n_questions": len(questions), "rerank_seconds": seconds}
        for kk in C.RECALL_KS:
            row[f"recall@{kk}"] = rec["doc_constrained"][kk]
        row["hits1"], row["hits5"] = S11._hits(retrieved, questions, 1), S11._hits(
            retrieved, questions, 5)
        if not arm.endswith("_bge"):
            row["contexts"] = [[{"doc_id": c["doc_id"], "text": c["text"]} for c in r]
                               for r in retrieved]
        rows[arm] = row
    return rows


def archive_checks(rows: dict) -> tuple[bool, list[dict]]:
    """Criterion 1: dense rows against stage8/final, reranked fixed 15/0 rows against
    stage15/final, within S11.CHECK_TOLERANCE with the same chunk and question counts."""
    def read(path):
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    s8 = read(C.RESULTS_DIR / "stage8" / "final" / C.STAGE8_RESULTS_CSV)
    s15 = {r["arm"]: r for r in read(C.RESULTS_DIR / "stage15" / "final"
                                     / C.STAGE15_RESULTS_CSV)}
    targets = []
    for label, (method, size, overlap) in CONFIGS.items():
        for r in s8:
            if r["arm"] != "bge" or r["method"] != method:
                continue
            if method == "fixed" and (int(r["fixed_size"]), int(r["fixed_overlap"])) != (size,
                                                                                        overlap):
                continue
            targets.append((f"{label}_bge", "stage8/final", r))
    targets += [("fixed15_ft", "stage15/final", s15["rerank20_ft"]),
                ("fixed15_large", "stage15/final", s15["rerank20_s15_large"])]
    out, ok = [], True
    for arm, source, ref in targets:
        now = rows[arm]
        for metric in ("recall@1", "recall@3", "recall@5"):
            diff = now[metric] - float(ref[metric])
            passed = abs(diff) <= S11.CHECK_TOLERANCE
            ok &= passed
            out.append({"arm": arm, "archive": source, "metric": metric,
                        "archived": float(ref[metric]), "now": round(now[metric], 4),
                        "diff": round(diff, 4), "pass": passed})
        for metric in ("n_chunks", "n_questions"):
            passed = int(float(ref[metric])) == int(now[metric])
            ok &= passed
            out.append({"arm": arm, "archive": source, "metric": metric,
                        "archived": int(float(ref[metric])), "now": int(now[metric]),
                        "diff": int(now[metric]) - int(float(ref[metric])), "pass": passed})
    return ok, out


def write_contexts(path: pathlib.Path, meta: dict, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 17 step 1: reader inputs per arm.")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_QUESTIONS} questions, temp dir, no checks kept")
    ap.add_argument("--mode", choices=("fresh", "resume"),
                    help="fresh starts the run; resume continues it")
    ap.add_argument("--account", default="unlabelled", help="label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()
    if args.smoke:
        build(args)
        return
    if args.mode is None:
        raise SystemExit("[stage17] choose --mode fresh or --mode resume")

    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE17_ROOT_ID)
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE17_LOCK_FILE,
                             run_version=C.STAGE17_RUN_VERSION, account=args.account,
                             task="contexts", clear_stale=args.clear_stale_lock):
        run_guard.probe_directory(C.RESULTS_LATEST_DIR)
        build(args)


def build(args) -> None:
    from rag_chunk import run_guard

    docs, questions, titles_sha1 = S16.B16.stage6_bench()
    latest = (pathlib.Path(tempfile.mkdtemp(prefix="stage17_contexts_smoke_")) if args.smoke
              else C.RESULTS_LATEST_DIR)
    out_path = latest / C.STAGE17_CONTEXTS_JSONL
    if not args.smoke:
        ident = latest / C.STAGE17_IDENTITY_JSON
        if args.mode == "fresh" and ident.exists():
            raise SystemExit(f"[stage17] {ident} exists: a Stage 17 run has started. Use "
                             "--mode resume. Nothing was changed.")
        if args.mode == "resume" and not ident.exists():
            raise SystemExit(f"[stage17] no {ident.name}: there is no run to resume. Use "
                             "--mode fresh.")
        status = run_guard.check_run_identity(ident, run_identity(titles_sha1))
        print(f"[stage17] run identity {status}", flush=True)
        if out_path.exists():
            meta, _ = read_contexts(out_path)
            print(f"[stage17] contexts already written ({meta['n_questions']} questions, "
                  f"valid {meta['valid']}); nothing to do", flush=True)
            return

    hashes = {k: (S16.sha256(p) if p.exists() else None) for k, p in weight_paths().items()}
    bad_hash = sorted(k for k, h in hashes.items() if h != expected_hashes()[k])
    print(f"[stage17] weights: {'match' if not bad_hash else f'MISMATCH {bad_hash}'}",
          flush=True)
    n_questions_all = len(questions)
    if args.smoke:
        questions = questions[:SMOKE_QUESTIONS]
    print(f"[stage17] Stage 6 bench: {len(docs)} docs / {len(questions)} questions"
          f"{' (smoke)' if args.smoke else ''}", flush=True)

    C.apply(RETRIEVAL_EMBED_MODEL="BAAI/bge-base-en-v1.5", RETRIEVAL_EMBED_NORMALIZE=True)
    import torch
    from sentence_transformers import CrossEncoder

    from rag_chunk import sweep, training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = latest / C.STAGE17_SCORES_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    src = C.RESULTS_LATEST_DIR / "stage15_scores"
    for s15_arm, arm in STAGE15_CACHE.items():
        if (src / f"{s15_arm}.npy").exists() and not (cache_dir / f"{arm}.npy").exists():
            for ext in (".npy", ".json"):
                shutil.copy2(src / f"{s15_arm}{ext}", cache_dir / f"{arm}{ext}")
            print(f"[stage17] copied the Stage 15 cache {s15_arm} as {arm}", flush=True)

    paths = {"ft": str(weight_paths()["ft"].parent), "large": str(weight_paths()["large"].parent)}

    def scorer(arm, which):
        def load():
            return CrossEncoder(paths[which], max_length=int(C.RERANK_MAX_LENGTH), device=device)
        return S15.cached_scorer(arm, load, cache_dir)

    rows = {}
    rows.update(config_arms("fixed15", docs, questions,
                            {"fixed15_ft": scorer("fixed15_ft", "ft"),
                             "fixed15_large": scorer("fixed15_large", "large")}, cache_dir))
    rows.update(config_arms("fixed6", docs, questions, {"fixed6_ft": scorer("fixed6_ft", "ft")},
                            cache_dir))
    probs = sweep._precompute_boundary_probs(docs, {"bilstm": training.load_model("bilstm")})
    rows.update(config_arms("bilstm15", docs, questions,
                            {"bilstm15_ft": scorer("bilstm15_ft", "ft")}, cache_dir,
                            probs=probs["bilstm"]))
    gold = gold_contexts(docs, questions)

    for arm, r in rows.items():
        sec = r["rerank_seconds"]
        print(f"[stage17] {arm:15s} R@1 {r['recall@1']:.4f}  R@5 {r['recall@5']:.4f}  "
              f"chunks {r['n_chunks']}"
              + (f"  rerank {sec / 60:.1f} min" if sec is not None else ""), flush=True)
    print(f"[stage17] gold chunk found for {sum(bool(g) for g in gold)} of {len(gold)} "
          "questions", flush=True)

    if args.smoke:
        shutil.rmtree(latest, ignore_errors=True)
        print("[stage17] smoke: every arm ranked and the gold chunks found. The numbers are "
              "not results and nothing was kept.", flush=True)
        return

    check_ok, check_rows = archive_checks(rows)
    counts_ok = len(docs) == int(C.N_NQ_DOCS_LARGE) and n_questions_all == 1032
    valid = check_ok and counts_ok and not bad_hash
    why = ("" if valid else
           f"weights differ from the pre-registered hashes: {bad_hash}" if bad_hash else
           "the bench does not hold 1000 documents and 1032 questions" if not counts_ok else
           "a retrieval row did not reproduce its archive")
    S11._write_rows(latest / C.STAGE17_CHECK_CSV, check_rows)
    meta = {"stage": "stage17", "n_docs": len(docs), "n_questions": len(questions),
            "bench_titles_sha1": titles_sha1, "weights_sha256": hashes, "valid": valid,
            "why_invalid": why, "arms": list(rows) + ["gold"]}
    write_contexts(out_path, meta, list(rows.values()) + [{"arm": "gold", "contexts": gold}])
    print(f"[stage17] archive check {'PASS' if check_ok else 'FAIL'}; valid {valid}"
          + (f" ({why})" if why else ""), flush=True)
    print(f"[stage17] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
