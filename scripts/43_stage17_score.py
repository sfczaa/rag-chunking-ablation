"""Stage 17 (step 3) - scores, paired comparisons and verdicts.

Reads the contexts and the generations, scores every prediction and applies the
pre-registered rules of docs/stage17_answer_quality.md. Runs on CPU.

Usage:
    python scripts/43_stage17_score.py --account A
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402
from rag_chunk import answer_eval as AE  # noqa: E402


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S42 = _load_script("42_stage17_generate.py", "stage17_generate")
S41 = S42.S41
S11 = S41.S11


def _mean(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def arm_scores(arm, items, questions, ctx_rows, gen_dir) -> dict:
    """Per-question scores of one arm; stops if any question is not generated yet."""
    saved = {r["qid"]: r for r in S42.read_rows(gen_dir / f"{arm}.jsonl")}
    missing = [qi for qi, _ in items if qi not in saved]
    if missing:
        raise SystemExit(f"[stage17] {arm}: {len(missing)} of {len(items)} questions are not "
                         "generated yet; finish scripts/42_stage17_generate.py first")
    hits5 = ctx_rows[arm]["hits5"] if arm in AE.RETRIEVAL_ARMS else None
    out = {}
    for qi, _ in items:
        r, ans = saved[qi], questions[qi]["answer"]
        out[qi] = {"acc": int(AE.contains_answer(r["prediction"], ans)),
                   "em": int(AE.exact_match(r["prediction"], ans)),
                   "f1": AE.token_f1(r["prediction"], ans),
                   "words": len(r["prediction"].split()), "prompt_tokens": r["prompt_tokens"],
                   "n_cut": r["n_cut"], "seconds": r["seconds"], "gpu": r["gpu"],
                   "hit5": hits5[qi] if hits5 is not None else None,
                   "lost": int(hits5 is not None and hits5[qi] == 1
                               and r["answer_in_prompt"] is False),
                   "short": len(ans.split()) <= 5}
    return out


def arm_row(arm, scores, ctx_rows) -> dict:
    s = list(scores.values())
    row = {"arm": arm, "n_questions": len(s), "accuracy": _mean(x["acc"] for x in s),
           "exact_match": _mean(x["em"] for x in s), "f1": _mean(x["f1"] for x in s)}
    row.update({"config": None, "recall@1": None, "recall@5": None, "n_chunks": None,
                "accuracy_hit5": None, "n_hit5": None, "accuracy_miss5": None,
                "n_miss5": None})
    if arm in AE.RETRIEVAL_ARMS:
        c = ctx_rows[arm]
        row.update({"config": c["config"], "recall@1": c["recall@1"], "recall@5": c["recall@5"],
                    "n_chunks": c["n_chunks"]})
        for name, flag in (("hit5", 1), ("miss5", 0)):
            part = [x["acc"] for x in s if x["hit5"] == flag]
            row[f"accuracy_{name}"], row[f"n_{name}"] = _mean(part), len(part)
    for name, flag in (("short", True), ("long", False)):
        part = [x["acc"] for x in s if x["short"] == flag]
        row[f"accuracy_{name}_answer"], row[f"n_{name}_answer"] = _mean(part), len(part)
    row.update({"mean_prediction_words": _mean(x["words"] for x in s),
                "mean_prompt_tokens": _mean(x["prompt_tokens"] for x in s),
                "passages_cut": sum(x["n_cut"] for x in s),
                "answers_lost_to_cut": sum(x["lost"] for x in s),
                "s_per_question": _mean(x["seconds"] for x in s)})
    return row


def paired(a: dict, b: dict, qids) -> list[float]:
    return [a[q]["acc"] - b[q]["acc"] for q in qids]


def _fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 17 step 3: scores and verdicts.")
    ap.add_argument("--account", default="unlabelled", help="label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()

    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE17_ROOT_ID)
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE17_LOCK_FILE,
                             run_version=C.STAGE17_RUN_VERSION, account=args.account,
                             task="score", clear_stale=args.clear_stale_lock):
        run_guard.probe_directory(C.RESULTS_LATEST_DIR)
        score(C.RESULTS_LATEST_DIR)


def score(latest: pathlib.Path) -> dict:
    from rag_chunk import run_guard

    _, questions, titles_sha1 = S41.S16.B16.stage6_bench()
    run_guard.check_run_identity(latest / C.STAGE17_IDENTITY_JSON, S41.run_identity(titles_sha1))
    meta, ctx_rows = S41.read_contexts(latest / C.STAGE17_CONTEXTS_JSONL)
    if not meta["valid"]:
        verdict = {"claim1": "INVALID", "claim2": "INVALID", "valid": False,
                   "why_invalid": meta["why_invalid"]}
        (latest / C.STAGE17_VERDICT_JSON).write_text(json.dumps(verdict, indent=2),
                                                      encoding="utf-8")
        (latest / C.STAGE17_SUMMARY_MD).write_text(
            "# Stage 17 - answer accuracy with a reader on top of retrieval\n\n"
            f"Claim 1: INVALID. Claim 2: INVALID. Reason: {meta['why_invalid']}.\n",
            encoding="utf-8")
        print(f"[stage17] INVALID: {meta['why_invalid']}", flush=True)
        return verdict
    kept = S42.kept_questions(questions)
    gen_dir = latest / C.STAGE17_GENERATIONS_DIRNAME
    items = {arm: S42.work_items(arm, kept, ctx_rows) for arm in AE.ARMS}
    scores = {arm: arm_scores(arm, items[arm], questions, ctx_rows, gen_dir)
              for arm in AE.ARMS}
    rows = [arm_row(arm, scores[arm], ctx_rows) for arm in AE.ARMS]
    gpus = sorted({x["gpu"] for s in scores.values() for x in s.values()})

    gold_q = [qi for qi, _ in items[AE.ARM_GOLD]]
    reader_ok, rm, rlo, rhi = AE.reader_check(
        paired(scores[AE.ARM_GOLD], scores[AE.ARM_CLOSED], gold_q))
    c1, m1, lo1, hi1 = AE.size_verdict(
        paired(scores["fixed15_ft"], scores["fixed6_ft"], kept), float(C.STAGE17_PRACTICAL_FLOOR))
    c2, m2, lo2, hi2, lo2_90, hi2_90 = AE.placement_verdict(
        paired(scores["bilstm15_ft"], scores["fixed15_ft"], kept), float(C.STAGE17_EQUIV_MARGIN))
    m3, lo3, hi3 = AE.mean_ci(paired(scores["fixed15_large"], scores["fixed15_ft"], kept))
    r5 = [ctx_rows["fixed15_ft"]["hits5"][q] - ctx_rows["fixed6_ft"]["hits5"][q] for q in kept]
    r5_gap = sum(r5) / len(r5)

    overall = None if reader_ok else "INVALID-READER"
    claim1 = overall or c1
    claim2 = overall or c2
    comparisons = [
        ("reader check", "gold - closed_book", len(gold_q), rm, rlo, rhi, None, None),
        ("claim 1", "fixed15_ft - fixed6_ft", len(kept), m1, lo1, hi1, None, None),
        ("claim 2", "bilstm15_ft - fixed15_ft", len(kept), m2, lo2, hi2, lo2_90, hi2_90),
        ("reported", "fixed15_large - fixed15_ft", len(kept), m3, lo3, hi3, None, None),
    ]
    paired_rows = [{"role": role, "comparison": comp, "metric": "accuracy", "n_questions": n,
                    "mean": round(m, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                    "ci90_low": None if l90 is None else round(l90, 4),
                    "ci90_high": None if h90 is None else round(h90, 4)}
                   for role, comp, n, m, lo, hi, l90, h90 in comparisons]
    S11._write_rows(latest / C.STAGE17_RESULTS_CSV, rows)
    S11._write_rows(latest / C.STAGE17_PAIRED_CSV, paired_rows)

    name, revision = S41.reader_spec()
    lines = ["# Stage 17 - answer accuracy with a reader on top of retrieval", "",
             f"Claim 1 (chunk size): {claim1}. Claim 2 (boundary placement): {claim2}.", "",
             f"- reader: `{name}` at `{revision}`, greedy, fp16, GPU {', '.join(gpus)}",
             f"- Stage 6 bench: {meta['n_docs']} docs / {meta['n_questions']} questions, "
             f"{len(kept)} scored; gold chunk for {len(gold_q)}",
             "- criterion 1 (weights and archive check): PASS",
             f"- criterion 2 (gold - closed book): {rm:+.4f}, 95% CI [{rlo:+.4f}, {rhi:+.4f}], "
             f"{'PASS' if reader_ok else 'FAIL'}",
             f"- claim 1 R@5 gap over the scored questions: {r5_gap:+.4f}; accuracy gap / R@5 "
             f"gap = {m1 / r5_gap:.2f}" if r5_gap else "- claim 1 R@5 gap: 0",
             "", "| arm | accuracy | EM | F1 | R@1 | R@5 | acc, answer in top 5 | "
             "acc, answer outside top 5 | words | prompt tokens | cut | lost to cut | "
             "s per question |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['arm']} | {_fmt(r['accuracy'])} | {_fmt(r['exact_match'])} | "
                     f"{_fmt(r['f1'])} | {_fmt(r.get('recall@1'))} | {_fmt(r.get('recall@5'))} | "
                     f"{_fmt(r.get('accuracy_hit5'))} | {_fmt(r.get('accuracy_miss5'))} | "
                     f"{_fmt(r['mean_prediction_words'], 2)} | "
                     f"{_fmt(r['mean_prompt_tokens'], 0)} | "
                     f"{r['passages_cut']} | {r['answers_lost_to_cut']} | "
                     f"{_fmt(r['s_per_question'], 2)} |")
    lines += ["", "| role | comparison | n | mean | 95% CI | 90% CI |",
              "| --- | --- | --- | --- | --- | --- |"]
    for p in paired_rows:
        ci90 = ("-" if p["ci90_low"] is None
                else f"[{p['ci90_low']:+.4f}, {p['ci90_high']:+.4f}]")
        lines.append(f"| {p['role']} | {p['comparison']} | {p['n_questions']} | "
                     f"{p['mean']:+.4f} | [{p['ci95_low']:+.4f}, "
                     f"{p['ci95_high']:+.4f}] | {ci90} |")
    lines += ["", "| arm | acc, answer <= 5 NQ tokens | n | acc, longer answer | n |",
              "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['arm']} | {_fmt(r['accuracy_short_answer'])} | "
                     f"{r['n_short_answer']} | {_fmt(r['accuracy_long_answer'])} | "
                     f"{r['n_long_answer']} |")
    (latest / C.STAGE17_SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    verdict = {"claim1": claim1, "claim2": claim2, "valid": meta["valid"],
               "reader_check": {"pass": reader_ok, "mean": rm, "ci95": [rlo, rhi],
                                "n_questions": len(gold_q)},
               "claim1_detail": {"comparison": "fixed15_ft - fixed6_ft", "mean": m1,
                                 "ci95": [lo1, hi1], "floor": float(C.STAGE17_PRACTICAL_FLOOR),
                                 "r5_gap": r5_gap},
               "claim2_detail": {"comparison": "bilstm15_ft - fixed15_ft", "mean": m2,
                                 "ci95": [lo2, hi2], "ci90": [lo2_90, hi2_90],
                                 "margin": float(C.STAGE17_EQUIV_MARGIN)},
               "reported": {"comparison": "fixed15_large - fixed15_ft", "mean": m3,
                            "ci95": [lo3, hi3]},
               "n_questions": len(kept), "reader_model": name, "reader_revision": revision,
               "gpus": gpus}
    (latest / C.STAGE17_VERDICT_JSON).write_text(json.dumps(verdict, indent=2),
                                                  encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\n[stage17] claim 1: {claim1}; claim 2: {claim2}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
