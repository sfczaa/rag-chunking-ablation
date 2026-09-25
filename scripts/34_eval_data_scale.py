"""Stage 14 (step 2) - does the reranker trained on more data rank better?

Every arm reorders one identical BGE top-20 pool per config:

    bge               dense order, used for the pipeline check
    rerank20          off-the-shelf cross-encoder, used for the pipeline check
    rerank20_ft       Stage 8 weights, 2034 groups
    rerank20_s14_4k   the same recipe on 4011 groups
    rerank20_s14_10k  the same recipe on the 4k set plus three new shards

Primary comparison: fixed 15/0, R@1, 10k minus Stage 8, paired over questions. The
criteria, including the size condition on the 10k set, are in
docs/stage14_data_scale.md.

The dev mode scores the Stage 8 dev bench for provenance and to catch a broken arm.
It decides nothing: the claim bench runs in either direction.

Scoring and statistics helpers come from scripts/28_eval_reranker_rl.py and are
loaded unchanged, so some log lines start with [stage11].

Usage:
    python scripts/34_eval_data_scale.py --dev
    python scripts/34_eval_data_scale.py                       # the claim, fixed 15/0
    python scripts/34_eval_data_scale.py --configs 15:0,6:0    # optional secondary
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
ARMS = {"4k": "rerank20_s14_4k", "10k": "rerank20_s14_10k"}
PAIRS = ((ARMS["10k"], ARM_FT), (ARMS["4k"], ARM_FT), (ARMS["10k"], ARMS["4k"]))
CHECKPOINT = "stage14_checkpoint_{mode}.jsonl"
VERDICT_JSON = "stage14_verdict.json"


def _paired(rows) -> list[dict]:
    """Stage 14 pairs; the same estimator scripts/28 uses."""
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


def _verdict(diffs, valid, n_groups_10k, floor):
    """Pre-registered rule on unrounded values: (verdict, why, mean, lo, hi)."""
    m, lo, hi = S11._mean_ci(diffs)
    ci = f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    need = int(C.STAGE14_MIN_GROUPS_10K)
    if not valid:
        return ("INVALID", "the dense or off-the-shelf rows did not reproduce "
                "stage8/final", m, lo, hi)
    if n_groups_10k < need:
        return ("INCONCLUSIVE-SIZE", f"the 10k arm trained on {n_groups_10k} groups, "
                f"short of the {need} the size condition asks for", m, lo, hi)
    if lo > 0 and m >= floor:
        return ("MORE-DATA-BETTER", f"10k beats Stage 8 by {m:+.4f} R@1, {ci}, at or "
                f"above the {floor} floor", m, lo, hi)
    if hi < 0:
        return "MORE-DATA-WORSE", f"10k trails Stage 8 by {m:+.4f} R@1, {ci}", m, lo, hi
    if lo > 0:
        return ("TIE", f"10k - Stage 8 = {m:+.4f} R@1, {ci}: excludes 0 but below the "
                f"{floor} floor", m, lo, hi)
    return ("TIE", f"10k - Stage 8 = {m:+.4f} R@1, {ci}: not a detectable difference",
            m, lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 14 step 2: the Stage 8 recipe on more data, against Stage 8.")
    ap.add_argument("--dev", action="store_true",
                    help="Stage 8 dev bench, provenance only, no verdict")
    ap.add_argument("--configs", default=None,
                    help="final mode: size:overlap list from 15:0,6:0 (default 15:0)")
    ap.add_argument("--account", default="unlabelled",
                    help="operational label recorded in the lock")
    ap.add_argument("--clear-stale-lock", action="store_true",
                    help="only after confirming the runtime that held the lock is stopped")
    args = ap.parse_args()
    if not args.dev:
        S11._parse_configs(args.configs)
    stale = [n for n in ("STAGE14_MIN_GROUPS_10K", "STAGE14_PRACTICAL_FLOOR",
                         "STAGE14_RESULTS_CSV", "STAGE14_PAIRED_CSV", "STAGE14_CHECK_CSV",
                         "STAGE14_SUMMARY_MD", "STAGE14_MODEL_DIRNAME", "STAGE14_ROOT_ID",
                         "STAGE14_RUN_VERSION", "STAGE14_LOCK_FILE") if not hasattr(C, n)]
    if stale:
        raise SystemExit(f"[stage14] config.py has no {stale} - it predates Stage 14")

    from rag_chunk import run_guard

    run_guard.check_root_identity(C.DATA_ROOT, C.STAGE14_ROOT_ID)
    with run_guard.held_lock(C.DATA_ROOT / C.STAGE14_LOCK_FILE,
                             run_version=C.STAGE14_RUN_VERSION, account=args.account,
                             task="eval-dev" if args.dev else "eval",
                             clear_stale=args.clear_stale_lock):
        run_guard.probe_directory(C.RESULTS_LATEST_DIR)
        evaluate(args)


def evaluate(args) -> None:
    mode = "dev" if args.dev else "final"
    configs = S11.DEV_CONFIGS if args.dev else S11._parse_configs(args.configs)

    from rag_chunk import rerank_rl as rr

    ft_dir = C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"
    if not (ft_dir / "model.safetensors").exists():
        raise SystemExit(f"[stage14] no Stage 8 weights at {ft_dir}")
    arm_dirs = {name: C.MODELS_DIR / C.STAGE14_MODEL_DIRNAME / key / "final"
                for key, name in ARMS.items()}
    metas = {name: rr.arm_meta(d) for name, d in arm_dirs.items()}
    missing = [name for name, meta in metas.items() if meta is None]
    if missing:
        raise SystemExit(f"[stage14] no completed training for {missing} - run "
                         "scripts/33_train_data_scale.py first")
    n_groups = {name: int(meta["fingerprint"]["n_groups"]) for name, meta in metas.items()}

    archived = None
    if mode == "final":
        s8 = C.RESULTS_DIR / "stage8" / "final" / C.STAGE8_RESULTS_CSV
        if not s8.exists():
            raise SystemExit(f"[stage14] missing {s8} - the final mode checks itself "
                             "against the Stage 8 archive")
        with open(s8, newline="", encoding="utf-8") as fh:
            archived = list(csv.DictReader(fh))

    C.apply(RETRIEVAL_EMBED_MODEL="BAAI/bge-base-en-v1.5", RETRIEVAL_EMBED_NORMALIZE=True)
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
    print(f"[stage14] mode={mode}: {len(docs)} docs / {len(questions)} questions")

    from sentence_transformers import CrossEncoder
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_paths = {ARM_FT: str(ft_dir)}
    model_paths.update({name: str(d) for name, d in arm_dirs.items()})
    if mode == "final":
        model_paths = {ARM_OTS: C.RERANKER_MODEL, **model_paths}
    models = {arm: CrossEncoder(path, max_length=int(C.RERANK_MAX_LENGTH), device=device)
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
    meta = {"stage": "stage14", "mode": mode, "n_docs": len(docs),
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
            print(f"[stage14] fixed {size}/{overlap} already in the checkpoint")
            continue
        print(f"[stage14] fixed {size}/{overlap}", flush=True)
        new_rows = S11._eval_config(docs, questions, size, overlap, scorers)
        append_checkpoint(checkpoint, {"rows": new_rows})
        rows += new_rows

    paired = _paired(rows)
    for p in paired:
        print(f"[stage14] {p['config']} {p['metric']:9s} {p['comparison']:36s} "
              f"{p['mean']:+.4f} [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]")
    S11._write_rows(latest / (C.STAGE14_DEV_CSV if mode == "dev"
                              else C.STAGE14_RESULTS_CSV), rows)
    S11._write_rows(latest / C.STAGE14_PAIRED_CSV.replace(".csv", f"_{mode}.csv"), paired)

    if mode == "dev":
        print("[stage14] dev mode is provenance only; the claim bench runs either way.")
        return

    check_ok, check_rows = S11._check(rows, archived)
    questions_ok = {int(float(r["n_questions"])) for r in archived} == {len(questions)}
    valid = check_ok and questions_ok
    S11._write_rows(latest / C.STAGE14_CHECK_CSV, check_rows)

    by = {(r["fixed_size"], r["fixed_overlap"], r["arm"]): r for r in rows}
    ten, ft = by[(15, 0, ARMS["10k"])], by[(15, 0, ARM_FT)]
    diffs = [a - b for a, b in zip(ten["hits1"], ft["hits1"])]
    floor = float(C.STAGE14_PRACTICAL_FLOOR)
    verdict, why, m, lo, hi = _verdict(diffs, valid, n_groups[ARMS["10k"]], floor)
    if not questions_ok and check_ok:
        why = f"the bench has {len(questions)} questions, not the archived count"

    r1 = {arm: by[(15, 0, arm)]["recall@1"] for arm in (ARM_FT, ARMS["4k"], ARMS["10k"])}
    monotone = r1[ARM_FT] <= r1[ARMS["4k"]] <= r1[ARMS["10k"]]
    ft_check = next((c for c in check_rows if c["arm"] == ARM_FT
                     and c["config"] == "fixed 15/0"), None)
    lines = ["# Stage 14 - more training data for the fine-tuned reranker", "",
             f"Verdict: {verdict}. {why}.", "",
             f"- Stage 6 bench: {len(docs)} docs / {len(questions)} questions; all arms "
             f"rerank the same BGE top-{S11.DEPTH} pool per config",
             f"- pipeline check (bge, off-the-shelf vs stage8/final, within "
             f"{S11.CHECK_TOLERANCE}, same chunk and question counts): "
             f"{'PASS' if valid else 'FAIL'}",
             f"- primary comparison: fixed 15/0, R@1, 10k - Stage 8, paired over "
             f"{len(diffs)} questions; threshold {floor}",
             f"- training groups: Stage 8 2034, 4k {n_groups[ARMS['4k']]}, "
             f"10k {n_groups[ARMS['10k']]} (size condition "
             f"{int(C.STAGE14_MIN_GROUPS_10K)})",
             f"- R@1 rises monotonically from 2k to 4k to 10k: {'yes' if monotone else 'no'}"]
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
    (latest / C.STAGE14_SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (latest / VERDICT_JSON).write_text(json.dumps(
        {"verdict": verdict, "why": why, "check_ok": check_ok,
         "questions_ok": questions_ok, "10k_minus_stage8_r1": m, "ci95": [lo, hi],
         "floor": floor, "n_questions": len(diffs), "n_groups": n_groups,
         "size_required": int(C.STAGE14_MIN_GROUPS_10K), "r1_monotone": monotone},
        indent=2), encoding="utf-8")
    print(f"[stage14] wrote {latest / C.STAGE14_SUMMARY_MD}")
    print(f"\n[stage14] VERDICT: {verdict} - {why}.")


if __name__ == "__main__":
    main()
