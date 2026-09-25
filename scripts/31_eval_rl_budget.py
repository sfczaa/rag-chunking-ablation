"""Stage 13 (step 2) - does RL catch up once its training budget matches CE?

Every arm reorders one identical BGE top-20 pool per config:

    bge                dense order, used for the pipeline check
    rerank20           off-the-shelf cross-encoder, used for the pipeline check
    rerank20_ft        Stage 8 weights, the common starting point
    rerank20_s13_ce4   4 cross-entropy epochs
    rerank20_s13_rl4   4 policy-gradient epochs, the primary arm
    rerank20_s13_ce8   ce4 continued for 4 more epochs
    rerank20_s13_rl8   rl4 continued for 4 more epochs

Primary comparison: fixed 15/0, R@1, rl4 minus ce4, paired over questions. The
criteria, including the budget condition on the RL arm's live groups, are in
docs/stage13_rl_budget.md.

The dev mode scores the Stage 8 dev bench for provenance and to catch a broken
arm. It decides nothing: the claim bench runs in either direction.

Scoring and statistics helpers come from scripts/28_eval_reranker_rl.py and are
loaded unchanged, so some log lines start with [stage11]. No Stage 11 or Stage 12
output is written.

Usage:
    python scripts/31_eval_rl_budget.py --dev
    python scripts/31_eval_rl_budget.py                       # the claim, fixed 15/0
    python scripts/31_eval_rl_budget.py --configs 15:0,6:0    # optional secondary
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
ARM_BGE, ARM_OTS, ARM_FT = S11.ARM_BGE, S11.ARM_OTS, S11.ARM_FT
ARMS = {"ce4": "rerank20_s13_ce4", "rl4": "rerank20_s13_rl4",
        "ce8": "rerank20_s13_ce8", "rl8": "rerank20_s13_rl8"}
PAIRS = ((ARMS["rl4"], ARMS["ce4"]), (ARMS["rl8"], ARMS["ce8"]),
         (ARMS["ce4"], ARM_FT), (ARMS["rl4"], ARM_FT),
         (ARMS["ce8"], ARM_FT), (ARMS["rl8"], ARM_FT))
CHECKPOINT = "stage13_checkpoint_{mode}.jsonl"
VERDICT_JSON = "stage13_verdict.json"


def _paired(rows) -> list[dict]:
    """Stage 13 pairs; the same estimator scripts/28 uses."""
    by = {(r["fixed_size"], r["fixed_overlap"], r["arm"]): r for r in rows}
    configs = sorted({(r["fixed_size"], r["fixed_overlap"]) for r in rows}, reverse=True)
    out = []
    for size, overlap in configs:
        for a, b in PAIRS:
            ra, rb = by.get((size, overlap, a)), by.get((size, overlap, b))
            if ra is None or rb is None:
                continue
            for metric, key in (("recall@1", "hits1"), ("recall@5", "hits5")):
                diffs = [x - y for x, y in zip(ra[key], rb[key])]
                mean, lo, hi = S11._mean_ci(diffs)
                out.append({"config": f"fixed {size}/{overlap}", "metric": metric,
                            "comparison": f"{a} - {b}", "mean": round(mean, 4),
                            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                            "n_questions": len(diffs)})
    return out


def _verdict(diffs, valid, live_fraction, live_groups, floor):
    """Pre-registered rule on unrounded values: (verdict, why, mean, lo, hi)."""
    m, lo, hi = S11._mean_ci(diffs)
    ci = f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    min_live = float(C.STAGE11_MIN_LIVE_FRACTION)
    budget = int(C.STAGE13_MIN_LIVE_GROUPS)
    if not valid:
        return ("INVALID", "the dense or off-the-shelf rows did not reproduce "
                "stage8/final", m, lo, hi)
    if live_fraction is not None and live_fraction < min_live:
        return ("FAILED-OPTIMISATION", f"only {live_fraction:.1%} of RL groups "
                "produced a gradient", m, lo, hi)
    if live_groups is not None and live_groups < budget:
        return ("INCONCLUSIVE-BUDGET", f"the RL arm accumulated {live_groups} live "
                f"groups, short of the {budget} the budget condition asks for",
                m, lo, hi)
    if lo > 0 and m >= floor:
        return ("RL-BETTER", f"RL beats CE by {m:+.4f} R@1 at a matched budget, "
                f"{ci}, at or above the {floor} floor", m, lo, hi)
    if hi < 0:
        return "CE-BETTER", f"RL trails CE by {m:+.4f} R@1, {ci}", m, lo, hi
    if lo > 0:
        return ("TIE", f"RL - CE = {m:+.4f} R@1, {ci}: excludes 0 but below the "
                f"{floor} floor", m, lo, hi)
    return "TIE", f"RL - CE = {m:+.4f} R@1, {ci}: not a detectable difference", m, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 13 step 2: the RL arm at a matched budget against its "
                    "cross-entropy control.")
    ap.add_argument("--dev", action="store_true",
                    help="Stage 8 dev bench, provenance only, no verdict")
    ap.add_argument("--configs", default=None,
                    help="final mode: size:overlap list from 15:0,6:0 (default 15:0)")
    args = ap.parse_args()
    mode = "dev" if args.dev else "final"
    configs = S11.DEV_CONFIGS if args.dev else S11._parse_configs(args.configs)
    stale = [n for n in ("STAGE13_MIN_LIVE_GROUPS", "STAGE13_PRACTICAL_FLOOR",
                         "STAGE13_RESULTS_CSV", "STAGE13_PAIRED_CSV",
                         "STAGE13_CHECK_CSV", "STAGE13_SUMMARY_MD",
                         "STAGE13_MODEL_DIRNAME") if not hasattr(C, n)]
    if stale:
        raise SystemExit(f"[stage13] config.py has no {stale} - it predates Stage 13")

    from rag_chunk import rerank_rl as rr

    ft_dir = C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"
    if not (ft_dir / "model.safetensors").exists():
        raise SystemExit(f"[stage13] no Stage 8 weights at {ft_dir}")
    arm_dirs = {name: C.MODELS_DIR / C.STAGE13_MODEL_DIRNAME / key
                for key, name in ARMS.items()}
    metas = {name: rr.arm_meta(d) for name, d in arm_dirs.items()}
    missing = [name for name, meta in metas.items() if meta is None]
    if missing:
        raise SystemExit(f"[stage13] no completed training for {missing} - run "
                         "scripts/30_train_rl_budget.py first")
    rl4_meta = metas[ARMS["rl4"]]
    live_fraction = rl4_meta.get("live_fraction")
    live_groups = rl4_meta.get("live_groups_cumulative", rl4_meta.get("live_groups"))

    archived = None
    if mode == "final":
        s8 = C.RESULTS_DIR / "stage8" / "final" / C.STAGE8_RESULTS_CSV
        if not s8.exists():
            raise SystemExit(f"[stage13] missing {s8} - the final mode checks itself "
                             "against the Stage 8 archive")
        with open(s8, newline="", encoding="utf-8") as fh:
            archived = list(csv.DictReader(fh))

    C.apply(RETRIEVAL_EMBED_MODEL="BAAI/bge-base-en-v1.5",
            RETRIEVAL_EMBED_NORMALIZE=True)
    C.ensure_dirs()

    from rag_chunk.large_eval import append_checkpoint, load_checkpoint

    if mode == "final":
        n = int(C.N_NQ_DOCS_LARGE)
        C.apply(N_NQ_DOCS=n)
        C.apply(NQ_DIR=C.NQ_DIR / f"large_n{n}")
        from rag_chunk import nq_data
        docs, questions = nq_data.prepare_nq()
    else:
        from rag_chunk import rerank_finetune as rf
        docs, questions = rf.load_bench("dev")
    print(f"[stage13] mode={mode}: {len(docs)} docs / {len(questions)} questions")

    from sentence_transformers import CrossEncoder
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_paths = {ARM_FT: str(ft_dir)}
    model_paths.update({name: str(d) for name, d in arm_dirs.items()})
    if mode == "final":
        model_paths = {ARM_OTS: C.RERANKER_MODEL, **model_paths}
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
    checkpoint = latest / CHECKPOINT.format(mode=mode)
    meta = {"stage": "stage13", "mode": mode, "n_docs": len(docs),
            "n_questions": len(questions), "retrieval_model": C.RETRIEVAL_EMBED_MODEL,
            "models": model_paths,
            "arm_fingerprints": {arm: metas[arm]["fingerprint"] for arm in metas}}
    done = load_checkpoint(checkpoint, meta)
    if not done and not checkpoint.exists():
        append_checkpoint(checkpoint, {"meta": meta})
    rows = list(done)
    done_configs = {(r["fixed_size"], r["fixed_overlap"]) for r in done}
    for size, overlap in configs:
        if (size, overlap) in done_configs:
            print(f"[stage13] fixed {size}/{overlap} already in the checkpoint")
            continue
        print(f"[stage13] fixed {size}/{overlap}", flush=True)
        new_rows = S11._eval_config(docs, questions, size, overlap, scorers)
        append_checkpoint(checkpoint, {"rows": new_rows})
        rows += new_rows

    paired = _paired(rows)
    for p in paired:
        print(f"[stage13] {p['config']} {p['metric']:9s} {p['comparison']:40s} "
              f"{p['mean']:+.4f} [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]")
    S11._write_rows(latest / (C.STAGE13_DEV_CSV if mode == "dev"
                              else C.STAGE13_RESULTS_CSV), rows)
    S11._write_rows(latest / C.STAGE13_PAIRED_CSV.replace(".csv", f"_{mode}.csv"), paired)

    if mode == "dev":
        print("[stage13] dev mode is provenance only; the claim bench runs either way.")
        return

    check_ok, check_rows = S11._check(rows, archived)
    questions_ok = {int(float(r["n_questions"])) for r in archived} == {len(questions)}
    valid = check_ok and questions_ok
    S11._write_rows(latest / C.STAGE13_CHECK_CSV, check_rows)

    by = {(r["fixed_size"], r["fixed_overlap"], r["arm"]): r for r in rows}
    rl4, ce4 = by[(15, 0, ARMS["rl4"])], by[(15, 0, ARMS["ce4"])]
    diffs = [a - b for a, b in zip(rl4["hits1"], ce4["hits1"])]
    floor = float(C.STAGE13_PRACTICAL_FLOOR)
    verdict, why, m, lo, hi = _verdict(diffs, valid, live_fraction, live_groups, floor)
    if not questions_ok and check_ok:
        why = f"the bench has {len(questions)} questions, not the archived count"

    ft_check = next((c for c in check_rows if c["arm"] == ARM_FT
                     and c["config"] == "fixed 15/0"), None)
    lines = ["# Stage 13 - the RL reranker at a matched training budget", "",
             f"Verdict: {verdict}. {why}.", "",
             f"- Stage 6 bench: {len(docs)} docs / {len(questions)} questions; all "
             f"arms rerank the same BGE top-{S11.DEPTH} pool per config",
             f"- pipeline check (bge, off-the-shelf vs stage8/final, within "
             f"{S11.CHECK_TOLERANCE}, same chunk and question counts): "
             f"{'PASS' if valid else 'FAIL'}",
             f"- primary comparison: fixed 15/0, R@1, rl4 - ce4, paired over "
             f"{len(diffs)} questions; threshold {floor}",
             f"- RL budget: {live_groups} live groups against the "
             f"{int(C.STAGE13_MIN_LIVE_GROUPS)} the condition asks for; live fraction "
             f"{live_fraction:.3f}" if live_fraction is not None else
             f"- RL budget: {live_groups} live groups"]
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
    (latest / C.STAGE13_SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (latest / VERDICT_JSON).write_text(json.dumps(
        {"verdict": verdict, "why": why, "check_ok": check_ok,
         "questions_ok": questions_ok, "rl4_minus_ce4_r1": m, "ci95": [lo, hi],
         "floor": floor, "n_questions": len(diffs), "live_fraction": live_fraction,
         "live_groups": live_groups,
         "budget_required": int(C.STAGE13_MIN_LIVE_GROUPS)}, indent=2),
        encoding="utf-8")
    print(f"[stage13] wrote {latest / C.STAGE13_SUMMARY_MD}")
    print(f"\n[stage13] VERDICT: {verdict} - {why}.")


if __name__ == "__main__":
    main()
