"""Stage 12: score one extra listwise CE epoch against the Stage 8 reranker.

In the Stage 11 dev gate the CE control arm was ahead of the Stage 8 weights it
started from (R@1 +0.0222, 95% CI [+0.0001, +0.0442], n = 406). This script tests
that on the Stage 6 bench. For each config all arms rerank the same BGE top-20
pool:

    bge              dense order, used for the pipeline check
    rerank20         off-the-shelf cross-encoder, used for the pipeline check
    rerank20_ft      Stage 8 weights
    rerank20_s11_ce  Stage 8 weights plus one CE epoch on the Stage 11 groups

Primary comparison: fixed 15/0, R@1, CE minus Stage 8, paired over questions.
The criteria are in docs/stage12_ce_epoch.md.

The ``bge`` and ``rerank20`` rows must match ``stage8/final`` within 0.005, with
the same chunk and question counts, or the verdict is INVALID. The
``rerank20_ft`` row is compared with its archived value but does not affect
validity, because the Drive weights were retrained after the originals were lost.

The scoring and statistics helpers come from scripts/28_eval_reranker_rl.py and
are loaded unchanged, so some log lines start with ``[stage11]``. The RL arm is
not loaded and no Stage 11 output is written.

Usage:
    python scripts/29_eval_ce_epoch.py                       # primary, fixed 15/0
    python scripts/29_eval_ce_epoch.py --configs 15:0,6:0    # + secondary
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

import config as C  # noqa: E402


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S11 = _load_script("28_eval_reranker_rl.py", "stage11_eval")
S11_TRAIN = _load_script("27_train_reranker_rl.py", "stage11_train")

ARM_BGE, ARM_OTS, ARM_FT, ARM_CE = S11.ARM_BGE, S11.ARM_OTS, S11.ARM_FT, S11.ARM_CE
RETRIEVAL_MODEL = "BAAI/bge-base-en-v1.5"
CHECKPOINT = "stage12_checkpoint_final.jsonl"
VERDICT_JSON = "stage12_verdict.json"


def _verdict(diffs: list[int], check_ok: bool, floor: float):
    """Pre-registered rule on unrounded values: (verdict, why, mean, lo, hi)."""
    m, lo, hi = S11._mean_ci(diffs)
    ci = f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    if not check_ok:
        return ("INVALID", "the dense or off-the-shelf rows did not reproduce "
                "stage8/final", m, lo, hi)
    if lo > 0 and m >= floor:
        return ("CE-BETTER", f"CE beats Stage 8 by {m:+.4f} R@1, {ci}, at or above "
                f"the {floor} floor", m, lo, hi)
    if hi < 0:
        return "STAGE8-BETTER", f"CE trails Stage 8 by {m:+.4f} R@1, {ci}", m, lo, hi
    if lo > 0:
        return ("TIE", f"CE - Stage 8 = {m:+.4f} R@1, {ci}: excludes 0 but below the "
                f"{floor} floor", m, lo, hi)
    return ("TIE", f"CE - Stage 8 = {m:+.4f} R@1, {ci}: not a detectable difference",
            m, lo, hi)


def _questions_match(archived: list[dict], n_questions: int) -> bool:
    return {int(float(r["n_questions"])) for r in archived} == {n_questions}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 12: one more CE epoch vs the Stage 8 reranker, "
                    "on the Stage 6 bench.")
    ap.add_argument("--configs", default=None,
                    help="size:overlap list from 15:0,6:0 (default 15:0, the claim)")
    args = ap.parse_args()
    configs = S11._parse_configs(args.configs)
    stale = [name for name in ("STAGE12_PRACTICAL_FLOOR", "STAGE12_RESULTS_CSV",
                               "STAGE12_PAIRED_CSV", "STAGE12_CHECK_CSV",
                               "STAGE12_SUMMARY_MD") if not hasattr(C, name)]
    if stale:
        raise SystemExit(f"[stage12] config.py has no {stale} - it predates Stage 12")

    from rag_chunk import rerank_rl as rr

    ft_dir = C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"
    ce_dir = C.MODELS_DIR / C.STAGE11_MODEL_DIRNAME / "ce"
    if not (ft_dir / "model.safetensors").exists():
        raise SystemExit(f"[stage12] no Stage 8 weights at {ft_dir}")
    ce_meta = rr.arm_meta(ce_dir)
    if ce_meta is None:
        raise SystemExit(f"[stage12] no completed CE arm at {ce_dir}")
    started_from = ce_meta["fingerprint"]["init_identity"]
    if started_from != S11_TRAIN._init_identity(ft_dir):
        raise SystemExit(f"[stage12] the CE arm did not start from {ft_dir}: "
                         f"recorded {started_from}, found "
                         f"{S11_TRAIN._init_identity(ft_dir)}")

    s8 = C.RESULTS_DIR / "stage8" / "final" / C.STAGE8_RESULTS_CSV
    if not s8.exists():
        raise SystemExit(f"[stage12] missing {s8} - the run checks itself against "
                         "the Stage 8 archive")
    with open(s8, newline="", encoding="utf-8") as fh:
        archived = list(csv.DictReader(fh))

    C.apply(RETRIEVAL_EMBED_MODEL=RETRIEVAL_MODEL, RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()
    n = int(C.N_NQ_DOCS_LARGE)
    C.apply(N_NQ_DOCS=n)
    C.apply(NQ_DIR=C.NQ_DIR / f"large_n{n}")

    from rag_chunk import nq_data
    from rag_chunk.large_eval import append_checkpoint, load_checkpoint

    docs, questions = nq_data.prepare_nq()
    print(f"[stage12] Stage 6 bench: {len(docs)} docs / {len(questions)} questions")

    from sentence_transformers import CrossEncoder
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_paths = {ARM_OTS: C.RERANKER_MODEL, ARM_FT: str(ft_dir), ARM_CE: str(ce_dir)}
    models = {arm: CrossEncoder(path, max_length=int(C.RERANK_MAX_LENGTH),
                                device=device)
              for arm, path in model_paths.items()}

    def scorer(model):
        def run(pairs):
            if not pairs:
                return np.zeros(0, dtype="float32")
            return np.asarray(model.predict(pairs, batch_size=int(C.RERANK_BATCH_SIZE),
                                            show_progress_bar=False), dtype="float32")
        return run

    scorers = {arm: scorer(m) for arm, m in models.items()}

    latest = C.RESULTS_LATEST_DIR
    checkpoint = latest / CHECKPOINT
    meta = {"stage": "stage12", "n_docs": len(docs), "n_questions": len(questions),
            "retrieval_model": C.RETRIEVAL_EMBED_MODEL, "models": model_paths,
            "ce_fingerprint": ce_meta["fingerprint"]}
    done = load_checkpoint(checkpoint, meta)
    if not done and not checkpoint.exists():
        append_checkpoint(checkpoint, {"meta": meta})
    rows = list(done)
    done_configs = {(r["fixed_size"], r["fixed_overlap"]) for r in done}
    for size, overlap in configs:
        if (size, overlap) in done_configs:
            print(f"[stage12] fixed {size}/{overlap} already in the checkpoint")
            continue
        print(f"[stage12] fixed {size}/{overlap}", flush=True)
        new_rows = S11._eval_config(docs, questions, size, overlap, scorers)
        append_checkpoint(checkpoint, {"rows": new_rows})
        rows += new_rows

    paired = S11._paired(rows)
    check_ok, check_rows = S11._check(rows, archived)
    questions_ok = _questions_match(archived, len(questions))
    valid = check_ok and questions_ok
    S11._write_rows(latest / C.STAGE12_RESULTS_CSV, rows)
    S11._write_rows(latest / C.STAGE12_PAIRED_CSV, paired)
    S11._write_rows(latest / C.STAGE12_CHECK_CSV, check_rows)

    by = {(r["fixed_size"], r["fixed_overlap"], r["arm"]): r for r in rows}
    ce, ft = by[(15, 0, ARM_CE)], by[(15, 0, ARM_FT)]
    diffs = [a - b for a, b in zip(ce["hits1"], ft["hits1"])]
    floor = float(C.STAGE12_PRACTICAL_FLOOR)
    verdict, why, m, lo, hi = _verdict(diffs, valid, floor)
    if not questions_ok:
        why = (f"the bench has {len(questions)} questions, not the archived count"
               if check_ok else why + "; the question count differs too")

    for p in paired:
        print(f"[stage12] {p['config']} {p['metric']:9s} {p['comparison']:32s} "
              f"{p['mean']:+.4f} [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]")

    ft_check = next((c for c in check_rows if c["arm"] == ARM_FT
                     and c["config"] == "fixed 15/0"), None)
    lines = ["# Stage 12 - one extra CE epoch vs the Stage 8 reranker", "",
             f"Verdict: {verdict}. {why}.", "",
             f"- Stage 6 bench: {len(docs)} docs / {len(questions)} questions; all "
             f"arms rerank the same BGE top-{S11.DEPTH} pool per config",
             f"- pipeline check (bge, off-the-shelf vs stage8/final, within "
             f"{S11.CHECK_TOLERANCE}, same chunk and question counts): "
             f"{'PASS' if valid else 'FAIL'}",
             f"- primary comparison: fixed 15/0, R@1, CE - Stage 8, paired over "
             f"{len(diffs)} questions; threshold {floor}",
             f"- CE arm: {ce_meta['steps']} steps on {ce_meta['fingerprint']['n_groups']}"
             f" groups, started from {started_from}"]
    if ft_check is not None and "now_recall@1" in ft_check:
        lines.append(f"- Stage 8 weights now vs archived at fixed 15/0: R@1 "
                     f"{ft_check['now_recall@1']} vs {ft_check['archived_recall@1']} "
                     "(retrained weights, not used for validity)")
    lines += ["", "| config | arm | R@1 | R@3 | R@5 | pool@20 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| fixed {r['fixed_size']}/{r['fixed_overlap']} | {r['arm']} | "
                     f"{r['recall@1']:.4f} | {r['recall@3']:.4f} | "
                     f"{r['recall@5']:.4f} | {r[f'pool_recall@{S11.DEPTH}']:.4f} |")
    lines += ["", "| config | metric | comparison | mean | 95% CI |",
              "| --- | --- | --- | --- | --- |"]
    for p in paired:
        lines.append(f"| {p['config']} | {p['metric']} | {p['comparison']} | "
                     f"{p['mean']:+.4f} | [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}] |")
    (latest / C.STAGE12_SUMMARY_MD).write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    (latest / VERDICT_JSON).write_text(json.dumps(
        {"verdict": verdict, "why": why, "check_ok": check_ok,
         "questions_ok": questions_ok, "ce_minus_stage8_r1": m, "ci95": [lo, hi],
         "floor": floor, "n_questions": len(diffs),
         "configs": [f"fixed {s}/{o}" for s, o in sorted(
             {(r["fixed_size"], r["fixed_overlap"]) for r in rows}, reverse=True)],
         "ce_init_identity": started_from}, indent=2), encoding="utf-8")
    print(f"[stage12] wrote {latest / C.STAGE12_SUMMARY_MD}")
    print(f"\n[stage12] VERDICT: {verdict} - {why}.")


if __name__ == "__main__":
    main()
