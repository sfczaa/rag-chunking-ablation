"""Stage 18 (step 1) - reader inputs on the holdout bench.

Each completed chunking config is saved under results/latest before the final
contexts file is written. Retrieval uses the Stage 17 functions and settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile

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
S42 = _load_script("42_stage17_generate.py", "stage17_generate")
S39 = S41.S16.B16
S11 = S41.S11
S15 = S41.S15
ARMS = (AE.ARM_CLOSED, AE.ARM_GOLD, "fixed15_ft", "fixed6_ft", "bilstm15_ft")
CONFIG_ARMS = {"fixed15": ("fixed15_bge", "fixed15_ft"),
               "fixed6": ("fixed6_bge", "fixed6_ft"),
               "bilstm15": ("bilstm15_bge", "bilstm15_ft")}


def bench() -> tuple[list[dict], list[dict], str, dict]:
    folder = S39.bench_dir()
    docs = S39._read_jsonl(folder / "docs.jsonl")
    questions = S39._read_jsonl(folder / "questions.jsonl")
    meta = json.loads((folder / "holdout_meta.json").read_text(encoding="utf-8"))
    sha = hashlib.sha1()
    for doc in docs:
        sha.update(doc["title"].encode("utf-8") + b"\n")
    return docs, questions, sha.hexdigest(), meta


def weight_paths() -> dict:
    paths = S41.weight_paths()
    return {name: paths[name] for name in ("ft", "bilstm")}


def expected_hashes() -> dict:
    return {"ft": C.STAGE16_STAGE8_SHA256, "bilstm": C.STAGE17_BILSTM_SHA256}


def run_identity(titles_sha1: str, questions: list[dict]) -> dict:
    identity = S41.run_identity(titles_sha1, questions)
    identity["run_version"] = C.STAGE18_RUN_VERSION
    identity["weights_sha256"] = expected_hashes()
    return identity


def partial_path(latest: pathlib.Path, label: str) -> pathlib.Path:
    return latest / f"stage18_contexts_{label}.jsonl"


def precheck(docs, questions, bench_meta, hashes, gold) -> list[str]:
    why = []
    if hashes != expected_hashes():
        why.append("weight hashes differ from the pre-registered values")
    if (len(docs), len(questions)) != (C.STAGE18_N_DOCS, C.STAGE18_N_QUESTIONS):
        why.append("holdout bench counts differ")
    if (bench_meta.get("n_docs"), bench_meta.get("n_questions")) != (
            len(docs), len(questions)):
        why.append("holdout metadata counts differ")
    if gold is not None:
        if len(S42.kept_questions(questions)) != C.STAGE18_N_SCORED:
            why.append("scored question count differs")
        if sum(bool(row) for row in gold) != C.STAGE18_N_GOLD:
            why.append("gold chunk count differs")
    return why


def archive_checks(rows: dict) -> tuple[bool, list[dict]]:
    path = C.RESULTS_DIR / "stage16" / "final" / C.STAGE16_RESULTS_CSV
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            archive = list(csv.DictReader(fh))
    except FileNotFoundError:
        archive = []
    checks, valid = [], True
    for arm, ref_arm in (("fixed15_bge", "bge"), ("fixed15_ft", "rerank20_ft")):
        refs = [r for r in archive if r["arm"] == ref_arm and r["method"] == "fixed"
                and int(r["fixed_size"]) == 15 and int(r["fixed_overlap"]) == 0]
        if len(refs) != 1:
            valid = False
            checks.append({"arm": arm, "metric": "reference rows", "archived": len(refs),
                           "now": None, "diff": None, "pass": False})
            continue
        ref, now = refs[0], rows[arm]
        for metric in ("recall@1", "recall@3", "recall@5"):
            diff = now[metric] - float(ref[metric])
            passed = abs(diff) <= S11.CHECK_TOLERANCE
            valid &= passed
            checks.append({"arm": arm, "metric": metric, "archived": float(ref[metric]),
                           "now": round(now[metric], 4), "diff": round(diff, 4),
                           "pass": passed})
        for metric, expected in (("n_chunks", C.STAGE18_N_FIXED15_CHUNKS),
                                 ("n_questions", C.STAGE18_N_QUESTIONS)):
            archived = int(float(ref[metric]))
            passed = archived == expected == int(now[metric])
            valid &= passed
            checks.append({"arm": arm, "metric": metric, "archived": archived,
                           "now": int(now[metric]), "diff": int(now[metric]) - archived,
                           "pass": passed})
    return valid, checks


def load_partial(latest, label, n_questions, identity):
    path = partial_path(latest, label)
    if not path.exists():
        return None
    meta, rows = S41.read_contexts(path)
    expected = {"stage": "stage18", "config": label, "n_questions": n_questions,
                "bench_titles_sha1": identity["bench_titles_sha1"],
                "bench_questions_sha1": identity["bench_questions_sha1"]}
    if meta != expected:
        raise SystemExit(f"[stage18] {path.name} metadata differs; stopping")
    if set(rows) != set(CONFIG_ARMS[label]):
        raise SystemExit(f"[stage18] {path.name} arms differ; stopping")
    for arm, row in rows.items():
        if row["n_questions"] != n_questions or len(row["hits5"]) != n_questions:
            raise SystemExit(f"[stage18] {arm} partial counts differ; stopping")
        if "contexts" in row and len(row["contexts"]) != n_questions:
            raise SystemExit(f"[stage18] {arm} partial contexts differ; stopping")
    print(f"[stage18] reused {path.name}", flush=True)
    return rows


def build(args) -> None:
    from rag_chunk import chunking, run_guard

    docs, questions, titles_sha1, bench_meta = bench()
    latest = (pathlib.Path(tempfile.mkdtemp(prefix="stage18_contexts_smoke_"))
              if args.smoke else C.RESULTS_LATEST_DIR)
    out_path = latest / C.STAGE18_CONTEXTS_JSONL
    if not args.smoke:
        ident = latest / C.STAGE18_IDENTITY_JSON
        if args.mode == "fresh" and ident.exists():
            raise SystemExit("[stage18] run identity exists; use --mode resume")
        if args.mode == "resume" and not ident.exists():
            raise SystemExit("[stage18] no run identity; use --mode fresh")
        status = run_guard.check_run_identity(ident, run_identity(titles_sha1, questions))
        print(f"[stage18] run identity {status}", flush=True)
        if out_path.exists():
            meta, _ = S41.read_contexts(out_path)
            print(f"[stage18] contexts already written; valid {meta['valid']}", flush=True)
            return

    hashes = {name: S41.S16.sha256(path) if path.exists() else None
              for name, path in weight_paths().items()}
    early = precheck(docs, questions, bench_meta, hashes, None)
    if early:
        invalid(out_path, docs, questions, titles_sha1, hashes, early)
        if args.smoke:
            shutil.rmtree(latest, ignore_errors=True)
        return
    gold = S41.gold_contexts(docs, questions)
    kept = set(S42.kept_questions(questions))
    gold = [row if i in kept else [] for i, row in enumerate(gold)]
    early = precheck(docs, questions, bench_meta, hashes, gold)
    if early and args.smoke:
        invalid(out_path, docs, questions, titles_sha1, hashes, early)
        shutil.rmtree(latest, ignore_errors=True)
        return
    if not args.smoke:
        s6_docs, s6_questions, _ = S39.stage6_bench()
        title_overlap = len({d["title"] for d in docs} & {d["title"] for d in s6_docs})
        question_overlap = len({q["question"] for q in questions}
                               & {q["question"] for q in s6_questions})
        if title_overlap or question_overlap:
            early.append(f"Stage 6 overlap: {title_overlap} titles, "
                         f"{question_overlap} questions")
        n_fixed6 = sum(len(chunking.fixed_chunks(d["sentences"], 6, 0)) for d in docs)
        if n_fixed6 != C.STAGE18_N_FIXED6_CHUNKS:
            early.append(f"fixed 6/0 has {n_fixed6} chunks")
        if early:
            invalid(out_path, docs, questions, titles_sha1, hashes, early)
            return
    else:
        questions = questions[:S41.SMOKE_QUESTIONS]
        gold = gold[:S41.SMOKE_QUESTIONS]
    print(f"[stage18] holdout bench: {len(docs)} docs, {len(questions)} questions",
          flush=True)

    C.apply(RETRIEVAL_EMBED_MODEL="BAAI/bge-base-en-v1.5", RETRIEVAL_EMBED_NORMALIZE=True)
    import torch
    from sentence_transformers import CrossEncoder
    from rag_chunk import sweep, training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = latest / C.STAGE18_SCORES_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    src = C.RESULTS_LATEST_DIR / "stage16_scores"
    dst = cache_dir / "fixed15_ft"
    if (all((src / f"rerank20_ft{ext}").exists() for ext in (".npy", ".json"))
            and not any(dst.with_suffix(ext).exists() for ext in (".npy", ".json"))):
        for ext in (".npy", ".json"):
            shutil.copy2(src / f"rerank20_ft{ext}", dst.with_suffix(ext))
        print("[stage18] seeded fixed15_ft score cache", flush=True)

    def scorer(arm):
        def load():
            return CrossEncoder(str(weight_paths()["ft"].parent),
                                max_length=int(C.RERANK_MAX_LENGTH), device=device)
        return S15.cached_scorer(arm, load, cache_dir)

    rows = {}
    for label in CONFIG_ARMS:
        identity = run_identity(titles_sha1, questions)
        part = None if args.smoke else load_partial(latest, label, len(questions), identity)
        if part is None:
            probs = None
            if label == "bilstm15":
                probs = sweep._precompute_boundary_probs(
                    docs, {"bilstm": training.load_model("bilstm")})["bilstm"]
            arm = CONFIG_ARMS[label][1]
            part = S41.config_arms(label, docs, questions, {arm: scorer(arm)}, cache_dir,
                                   probs=probs)
            if not args.smoke:
                S41.write_contexts(partial_path(latest, label),
                                   {"stage": "stage18", "config": label,
                                    "n_questions": len(questions),
                                    "bench_titles_sha1": titles_sha1,
                                    "bench_questions_sha1": identity["bench_questions_sha1"]},
                                   list(part.values()))
                print(f"[stage18] saved {label} contexts", flush=True)
        rows.update(part)
    if args.smoke:
        shutil.rmtree(latest, ignore_errors=True)
        print("[stage18] smoke complete; nothing kept", flush=True)
        return

    valid, checks = archive_checks(rows)
    S11._write_rows(latest / C.STAGE18_CHECK_CSV, checks)
    why = [] if valid else ["fixed 15/0 rows differ from the Stage 16 archive"]
    if rows["fixed6_ft"]["n_chunks"] != C.STAGE18_N_FIXED6_CHUNKS:
        why.append("fixed 6/0 chunk count differs")
    for arm in AE.RETRIEVAL_ARMS:
        if arm == "fixed15_large":
            continue
        contexts = rows[arm]["contexts"]
        if len(contexts) != len(questions) or any(len(c) != C.STAGE17_TOP_K for c in contexts):
            why.append(f"{arm} lacks a top-five list for every question")
    meta = {"stage": "stage18", "n_docs": len(docs), "n_questions": len(questions),
            "n_scored": len(S42.kept_questions(questions)),
            "n_gold": sum(bool(g) for g in gold),
            "n_bilstm_chunks": rows["bilstm15_ft"]["n_chunks"],
            "bench_titles_sha1": titles_sha1, "weights_sha256": hashes,
            "valid": not why, "why_invalid": "; ".join(why), "arms": list(ARMS)}
    S41.write_contexts(out_path, meta,
                       [rows[arm] for arm in rows] + [{"arm": "gold", "contexts": gold}])
    print(f"[stage18] contexts written; valid {meta['valid']}; "
          f"bilstm chunks {meta['n_bilstm_chunks']}", flush=True)


def invalid(out_path, docs, questions, titles_sha1, hashes, reasons):
    meta = {"stage": "stage18", "n_docs": len(docs), "n_questions": len(questions),
            "bench_titles_sha1": titles_sha1, "weights_sha256": hashes,
            "valid": False, "why_invalid": "; ".join(reasons), "arms": []}
    S41.write_contexts(out_path, meta, [])
    print(f"[stage18] INVALID ({meta['why_invalid']}); retrieval skipped", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 18 step 1: holdout reader inputs.")
    ap.add_argument("--smoke", action="store_true", help="40 questions, nothing kept")
    ap.add_argument("--mode", choices=("fresh", "resume"))
    ap.add_argument("--account", default="unlabelled", help="label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()
    if args.smoke:
        build(args)
        return
    if args.mode is None:
        raise SystemExit("[stage18] choose --mode fresh or --mode resume")
    from rag_chunk import run_guard
    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE14_ROOT_ID)
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE18_LOCK_FILE,
                             run_version=C.STAGE18_RUN_VERSION, account=args.account,
                             task="contexts", clear_stale=args.clear_stale_lock):
        run_guard.probe_directory(C.RESULTS_LATEST_DIR)
        build(args)


if __name__ == "__main__":
    main()
