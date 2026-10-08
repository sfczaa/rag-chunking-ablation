"""Stage 18 (step 3) - holdout scores and pre-registered verdicts.

Stage 17 scoring functions apply to the five holdout arms. Stage 6 comparisons
are reported when their saved generations are available.
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


S45 = _load_script("45_stage18_contexts.py", "stage18_contexts")
S42 = _load_script("42_stage17_generate.py", "stage17_generate")
S43 = _load_script("43_stage17_score.py", "stage17_score")
S11 = S45.S11


def pooled_scores(questions) -> dict | None:
    arms = ("fixed15_ft", "fixed6_ft", "bilstm15_ft")
    folders = (C.RESULTS_DIR / "stage17" / "final" / C.STAGE17_GENERATIONS_DIRNAME,
               C.RESULTS_LATEST_DIR / C.STAGE17_GENERATIONS_DIRNAME)
    folder = next((d for d in folders if all((d / f"{a}.jsonl").exists() for a in arms)),
                  None)
    if folder is None:
        print("[stage18] Stage 17 generations absent; pooled rows skipped", flush=True)
        return None
    kept = S42.kept_questions(questions)
    out = {}
    for arm in arms:
        rows = S42.read_rows(folder / f"{arm}.jsonl")
        if len(rows) != len(kept) or {r["qid"] for r in rows} != set(kept):
            raise SystemExit(f"[stage18] Stage 17 {arm} generations are incomplete")
        out[arm] = {r["qid"]: int(AE.contains_answer(r["prediction"],
                                                    questions[r["qid"]]["answer"]))
                    for r in rows}
    source = "stage17/final" if folder == folders[0] else "latest"
    print(f"[stage18] Stage 17 generations read from {source}", flush=True)
    return out


def paired_row(role, comparison, diffs, ci90=None):
    mean, lo, hi = AE.mean_ci(diffs)
    low90, high90 = (None, None) if ci90 is None else ci90
    return {"role": role, "comparison": comparison, "metric": "accuracy",
            "n_questions": len(diffs), "mean": round(mean, 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "ci90_low": None if low90 is None else round(low90, 4),
            "ci90_high": None if high90 is None else round(high90, 4)}


def score(latest: pathlib.Path) -> dict:
    from rag_chunk import run_guard

    _, questions, titles_sha1, _ = S45.bench()
    ident = latest / C.STAGE18_IDENTITY_JSON
    if not ident.exists():
        raise SystemExit("[stage18] no run identity; run scripts/45_stage18_contexts.py first")
    run_guard.check_run_identity(ident, S45.run_identity(titles_sha1, questions))
    meta, ctx_rows = S45.S41.read_contexts(latest / C.STAGE18_CONTEXTS_JSONL)
    if meta["n_questions"] != len(questions):
        raise SystemExit("[stage18] contexts and bench question counts differ")
    if not meta["valid"]:
        verdict = {"claim1": "INVALID", "claim2": "INVALID", "valid": False,
                   "why_invalid": meta["why_invalid"]}
        (latest / C.STAGE18_VERDICT_JSON).write_text(json.dumps(verdict, indent=2),
                                                      encoding="utf-8")
        (latest / C.STAGE18_SUMMARY_MD).write_text(
            "# Stage 18 - answer accuracy on the holdout bench\n\n"
            f"Claim 1: INVALID. Claim 2: INVALID. Reason: {meta['why_invalid']}.\n",
            encoding="utf-8")
        print(f"[stage18] INVALID: {meta['why_invalid']}", flush=True)
        return verdict

    kept = S42.kept_questions(questions)
    gen_dir = latest / C.STAGE18_GENERATIONS_DIRNAME
    items = {arm: S42.work_items(arm, kept, ctx_rows) for arm in S45.ARMS}
    scores = {arm: S43.arm_scores(arm, items[arm], questions, ctx_rows, gen_dir)
              for arm in S45.ARMS}
    rows = [S43.arm_row(arm, scores[arm], ctx_rows) for arm in S45.ARMS]
    gpus = sorted({x["gpu"] for arm in scores for x in scores[arm].values()})
    versions = sorted({x["versions"] for arm in scores for x in scores[arm].values()})
    if len(gpus) != 1:
        raise SystemExit(f"[stage18] generations use multiple GPUs: {gpus}")

    gold_q = [qi for qi, _ in items[AE.ARM_GOLD]]
    reader_ok, rm, rlo, rhi = AE.reader_check(
        S43.paired(scores[AE.ARM_GOLD], scores[AE.ARM_CLOSED], gold_q))
    d1 = S43.paired(scores["fixed15_ft"], scores["fixed6_ft"], kept)
    d2 = S43.paired(scores["bilstm15_ft"], scores["fixed15_ft"], kept)
    c1, m1, lo1, hi1 = AE.size_verdict(d1, float(C.STAGE17_PRACTICAL_FLOOR))
    c2, m2, lo2, hi2, lo2_90, hi2_90 = AE.placement_verdict(
        d2, float(C.STAGE17_EQUIV_MARGIN))
    claim1 = c1 if reader_ok else "INVALID-READER"
    claim2 = c2 if reader_ok else "INVALID-READER"
    r5 = [ctx_rows["fixed15_ft"]["hits5"][q]
          - ctx_rows["fixed6_ft"]["hits5"][q] for q in kept]
    r5_gap = sum(r5) / len(r5)

    paired_rows = [
        paired_row("reader check", "gold - closed_book",
                   S43.paired(scores[AE.ARM_GOLD], scores[AE.ARM_CLOSED], gold_q)),
        paired_row("claim 1", "fixed15_ft - fixed6_ft", d1),
        paired_row("claim 2", "bilstm15_ft - fixed15_ft", d2, (lo2_90, hi2_90)),
    ]
    _, s6_questions, _ = S45.S39.stage6_bench()
    stage6 = pooled_scores(s6_questions)
    if stage6 is not None:
        old_kept = S42.kept_questions(s6_questions)
        for role, a, b in (("pooled claim 1", "fixed15_ft", "fixed6_ft"),
                           ("pooled claim 2", "bilstm15_ft", "fixed15_ft")):
            old = [stage6[a][q] - stage6[b][q] for q in old_kept]
            new = S43.paired(scores[a], scores[b], kept)
            paired_rows.append(paired_row(role, f"{a} - {b}", old + new))

    S11._write_rows(latest / C.STAGE18_RESULTS_CSV, rows)
    S11._write_rows(latest / C.STAGE18_PAIRED_CSV, paired_rows)
    name, revision = S45.S41.reader_spec()
    lines = ["# Stage 18 - answer accuracy on the holdout bench", "",
             f"Claim 1 (chunk size): {claim1}. Claim 2 (boundary placement): {claim2}.",
             "", f"- reader: `{name}` at `{revision}`, greedy, fp16, GPU {gpus[0]}",
             f"- libraries: {'; '.join(versions)}",
             f"- holdout bench: {meta['n_docs']} docs / {meta['n_questions']} questions, "
             f"{len(kept)} scored; gold chunk for {len(gold_q)}",
             f"- bilstm 15/0 chunks: {meta['n_bilstm_chunks']}",
             (f"- pooled comparisons: Stage 6 and holdout, "
              f"{len(old_kept) + len(kept)} questions" if stage6 is not None
              else "- pooled comparisons: Stage 17 generations unavailable"),
             "- criterion 1 (weights, counts, overlap and archive): PASS",
             f"- criterion 2 (gold - closed book): {rm:+.4f}, "
             f"95% CI [{rlo:+.4f}, {rhi:+.4f}], {'PASS' if reader_ok else 'FAIL'}",
             (f"- claim 1 R@5 gap: {r5_gap:+.4f}; accuracy gap / R@5 gap = "
              f"{m1 / r5_gap:.2f}" if r5_gap else "- claim 1 R@5 gap: 0"),
             "", "| arm | accuracy | EM | F1 | R@1 | R@5 | acc, hit5 | acc, miss5 | "
             "words | prompt tokens | cut | lost to cut | s per question |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
             "--- | --- |"]
    for row in rows:
        fmt = S43._fmt
        lines.append(f"| {row['arm']} | {fmt(row['accuracy'])} | "
                     f"{fmt(row['exact_match'])} | {fmt(row['f1'])} | "
                     f"{fmt(row['recall@1'])} | {fmt(row['recall@5'])} | "
                     f"{fmt(row['accuracy_hit5'])} | {fmt(row['accuracy_miss5'])} | "
                     f"{fmt(row['mean_prediction_words'], 2)} | "
                     f"{fmt(row['mean_prompt_tokens'], 0)} | {row['passages_cut']} | "
                     f"{row['answers_lost_to_cut']} | {fmt(row['s_per_question'], 2)} |")
    lines += ["", "| role | comparison | n | mean | 95% CI | 90% CI |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in paired_rows:
        ci90 = ("-" if row["ci90_low"] is None else
                f"[{row['ci90_low']:+.4f}, {row['ci90_high']:+.4f}]")
        lines.append(f"| {row['role']} | {row['comparison']} | {row['n_questions']} | "
                     f"{row['mean']:+.4f} | [{row['ci95_low']:+.4f}, "
                     f"{row['ci95_high']:+.4f}] | {ci90} |")
    lines += ["", "| arm | acc, answer <= 5 NQ tokens | n | acc, longer answer | n |",
              "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['arm']} | {S43._fmt(row['accuracy_short_answer'])} | "
                     f"{row['n_short_answer']} | "
                     f"{S43._fmt(row['accuracy_long_answer'])} | "
                     f"{row['n_long_answer']} |")
    (latest / C.STAGE18_SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    verdict = {"claim1": claim1, "claim2": claim2, "valid": True,
               "reader_check": {"pass": reader_ok, "mean": rm, "ci95": [rlo, rhi],
                                "n_questions": len(gold_q)},
               "claim1_detail": {"comparison": "fixed15_ft - fixed6_ft", "mean": m1,
                                 "ci95": [lo1, hi1],
                                 "floor": float(C.STAGE17_PRACTICAL_FLOOR),
                                 "r5_gap": r5_gap},
               "claim2_detail": {"comparison": "bilstm15_ft - fixed15_ft", "mean": m2,
                                 "ci95": [lo2, hi2], "ci90": [lo2_90, hi2_90],
                                 "margin": float(C.STAGE17_EQUIV_MARGIN)},
               "n_questions": len(kept), "reader_model": name,
               "reader_revision": revision, "gpus": gpus, "versions": versions,
               "pooled_n_questions": (len(old_kept) + len(kept)
                                      if stage6 is not None else None)}
    (latest / C.STAGE18_VERDICT_JSON).write_text(json.dumps(verdict, indent=2),
                                                  encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"[stage18] claim 1: {claim1}; claim 2: {claim2}", flush=True)
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 18 step 3: scores and verdicts.")
    ap.add_argument("--account", default="unlabelled", help="label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()
    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE14_ROOT_ID)
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE18_LOCK_FILE,
                             run_version=C.STAGE18_RUN_VERSION, account=args.account,
                             task="score", clear_stale=args.clear_stale_lock):
        run_guard.probe_directory(C.RESULTS_LATEST_DIR)
        score(C.RESULTS_LATEST_DIR)


if __name__ == "__main__":
    main()
