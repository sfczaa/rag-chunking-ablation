"""Stage 13 (step 1) - train both objectives at a matched budget, then double it.

Stage 11 matched the step count, which gave the RL estimator less usable signal
than cross-entropy: it is silent on a group whose sampled rankings agree, and that
was 68% of groups. This trains four arms from the same Stage 8 weights over the
same Stage 11 groups:

    ce4  4 epochs of listwise cross-entropy
    rl4  4 epochs of the policy gradient (the primary arm)
    ce8  ce4 continued for 4 more epochs
    rl8  rl4 continued for 4 more epochs

The 8-epoch arms continue from the 4-epoch checkpoints, so their schedule restarts
halfway. Both objectives get the same treatment.

Each arm writes a training_meta.json after its checkpoint is saved and verified,
so a rerun skips finished arms. The RL arms also record cumulative live groups,
which is what the budget condition in docs/stage13_rl_budget.md is checked against.

Usage:
    python scripts/30_train_rl_budget.py --smoke    # 2 steps per arm, nothing kept
    python scripts/30_train_rl_budget.py            # all four arms in order
    python scripts/30_train_rl_budget.py --arm rl4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

SCRIPTS = pathlib.Path(__file__).resolve().parent
ARMS = ("ce4", "rl4", "ce8", "rl8")
CONTINUES = {"ce8": "ce4", "rl8": "rl4"}


def _load_stage11_train():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stage11_train", SCRIPTS / "27_train_reranker_rl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage13_dir() -> pathlib.Path:
    return C.MODELS_DIR / C.STAGE13_MODEL_DIRNAME


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 13 step 1: four training arms at two budgets.")
    ap.add_argument("--arm", choices=ARMS + ("all",), default="all")
    ap.add_argument("--epochs", type=int, default=None,
                    help="epochs per stage (default STAGE13_EPOCHS)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 groups, 2 steps per arm, written to a temp dir and deleted")
    ap.add_argument("--fresh", action="store_true",
                    help="retrain an arm even if its completion marker matches")
    args = ap.parse_args()
    C.ensure_dirs()

    s11 = _load_stage11_train()
    from rag_chunk import rerank_rl as rr

    groups_path = C.DATA_DIR / C.STAGE11_DATA_DIRNAME / C.STAGE11_GROUPS_JSONL
    if not groups_path.exists():
        raise SystemExit(f"[stage13] no training groups at {groups_path} - run "
                         "scripts/26_build_stage11_data.py first")
    with open(groups_path, encoding="utf-8") as fh:
        groups = [json.loads(line) for line in fh]
    if not groups:
        raise SystemExit(f"[stage13] {groups_path} holds no groups")

    stage8 = C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"
    if not (stage8 / "config.json").exists():
        raise SystemExit(f"[stage13] no Stage 8 weights at {stage8}")

    epochs = args.epochs or int(C.STAGE13_EPOCHS)
    lr = float(C.STAGE11_LR)
    samples = int(C.STAGE11_RL_SAMPLES)
    temperature = float(C.STAGE11_RL_TEMPERATURE)
    seed = int(C.STAGE11_SEED)
    gps = 1 if args.smoke else int(C.STAGE8_GROUPS_PER_STEP)
    max_groups = 2 if args.smoke else None
    log_every = 1 if args.smoke else int(C.STAGE11_LOG_EVERY)
    if args.smoke:
        epochs = 1

    wanted = ARMS if args.arm == "all" else (args.arm,)
    summaries: dict[str, dict] = {}
    for arm in wanted:
        objective = "ce" if arm.startswith("ce") else "rl"
        parent = CONTINUES.get(arm)
        if parent:
            init = stage13_dir() / parent
            if not args.smoke and rr.arm_meta(init) is None:
                raise SystemExit(f"[stage13] {arm} continues from {parent}, which is "
                                 f"not trained yet ({init})")
            if args.smoke:
                init = stage8
        else:
            init = stage8

        fingerprint = {
            "arm": arm, "objective": objective, "init": str(init),
            "init_identity": s11._init_identity(init), "epochs": epochs,
            "continues": parent, "lr": lr,
            "warmup_frac": float(C.STAGE11_WARMUP_FRAC),
            "weight_decay": float(C.STAGE8_WEIGHT_DECAY), "groups_per_step": gps,
            "seed": seed, "n_groups": max_groups or len(groups),
            "groups": str(groups_path), "groups_sha1": s11._file_sha1(groups_path),
            "max_length": int(C.RERANK_MAX_LENGTH),
            "samples": samples if objective == "rl" else None,
            "temperature": temperature if objective == "rl" else None,
        }

        if args.smoke:
            out_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"stage13_smoke_{arm}_"))
            log_path = None
        else:
            out_dir = stage13_dir() / arm
            log_path = C.RESULTS_LATEST_DIR / C.STAGE13_TRAIN_LOG_CSV.replace(
                ".csv", f"_{arm}.csv")
            done = rr.arm_meta(out_dir)
            if done and done.get("fingerprint") == fingerprint and not args.fresh:
                print(f"[stage13] arm {arm} already trained with these settings "
                      f"({out_dir}) - skipping", flush=True)
                summaries[arm] = done
                continue

        stats = rr.train_arm(
            groups, arm=objective, init_model=str(init), out_dir=out_dir,
            epochs=epochs, lr=lr, warmup_frac=float(C.STAGE11_WARMUP_FRAC),
            weight_decay=float(C.STAGE8_WEIGHT_DECAY), groups_per_step=gps,
            seed=seed, samples=samples, temperature=temperature,
            log_every=log_every, log_path=log_path, max_groups=max_groups)
        meta = {"fingerprint": fingerprint, **stats}
        if objective == "rl":
            seen = (max_groups or len(groups)) * epochs
            meta["groups_seen"] = seen
            meta["live_groups"] = round((stats.get("live_fraction") or 0.0) * seen)
            if parent:
                parent_meta = summaries.get(parent) or rr.arm_meta(stage13_dir() / parent)
                if parent_meta:
                    meta["live_groups_cumulative"] = (
                        meta["live_groups"] + int(parent_meta.get("live_groups_cumulative")
                                                  or parent_meta.get("live_groups") or 0))
            else:
                meta["live_groups_cumulative"] = meta["live_groups"]

        if args.smoke:
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            (out_dir / "training_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")
        summaries[arm] = meta

    print("\n[stage13] summary")
    floor = int(C.STAGE13_MIN_LIVE_GROUPS)
    for arm, meta in summaries.items():
        line = (f"  {arm}: {meta['steps']} steps, {meta['minutes']} min, greedy RR "
                f"{meta['greedy_rr_first_tenth']:.4f} -> {meta['greedy_rr_last_tenth']:.4f}")
        if meta.get("live_fraction") is not None:
            line += (f"; live fraction {meta['live_fraction']:.3f}, live groups "
                     f"{meta.get('live_groups_cumulative', meta.get('live_groups'))}")
        print(line)
    rl4 = summaries.get("rl4")
    if rl4 and rl4.get("live_groups_cumulative") is not None:
        seen = rl4["live_groups_cumulative"]
        verdict = "met" if seen >= floor else "NOT MET"
        print(f"[stage13] budget condition ({floor} live groups): {seen} - {verdict}")
    if args.smoke:
        print("[stage13] SMOKE MODE: nothing was kept.")
    else:
        print("[stage13] next: python scripts/31_eval_rl_budget.py --dev")


if __name__ == "__main__":
    main()
