"""Stage 11 (step 1) - continue the Stage 8 reranker under two objectives.

Both arms start from the Stage 8 fine-tuned weights and see the same fresh groups
(built by scripts/26_build_stage11_data.py) in the same order for the same
number of steps:

    ce  listwise softmax cross-entropy, exactly Stage 8's objective (control)
    rl  policy gradient over sampled Plackett-Luce rankings, rewarded by the
        positive's reciprocal rank, leave-one-out baseline

See ``rag_chunk/rerank_rl.py`` for the formulation and
``docs/stage11_reranker_rl.md`` for the pre-registered criteria.

Checkpointing: each arm is saved to Drive when it finishes, and a
``training_meta.json`` written after the save is its completion marker. A
rerun skips any arm whose marker matches this run's settings, so a lost runtime
costs at most the arm that was in progress.

Usage:
    python scripts/27_train_reranker_rl.py --smoke    # 2 steps per arm, nothing kept
    python scripts/27_train_reranker_rl.py            # both arms
    python scripts/27_train_reranker_rl.py --arm rl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402


def stage11_dir() -> pathlib.Path:
    return C.MODELS_DIR / C.STAGE11_MODEL_DIRNAME


def default_init() -> pathlib.Path:
    return C.MODELS_DIR / C.STAGE8_FT_MODEL_DIRNAME / "final"


def _file_sha1(path: pathlib.Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _init_identity(init: pathlib.Path) -> str:
    """Cheap identity of the starting weights: config hash plus weight size."""
    cfg = hashlib.sha1((init / "config.json").read_bytes()).hexdigest()[:12]
    weights = init / "model.safetensors"
    size = weights.stat().st_size if weights.exists() else 0
    return f"{cfg}:{size}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 11 step 1: continue the Stage 8 reranker under the "
                    "listwise (ce) and policy-gradient (rl) objectives.")
    ap.add_argument("--arm", choices=("ce", "rl", "both"), default="both")
    ap.add_argument("--init-model", default=None,
                    help="starting checkpoint (default: Stage 8 final)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="8 groups, 2 steps per arm, weights written to a temp "
                         "dir and deleted; checks the loop, the save and a reload")
    ap.add_argument("--fresh", action="store_true",
                    help="retrain an arm even if its completion marker matches")
    args = ap.parse_args()
    C.ensure_dirs()

    from rag_chunk import rerank_finetune as rf
    from rag_chunk import rerank_rl as rr

    groups_path = C.DATA_DIR / C.STAGE11_DATA_DIRNAME / C.STAGE11_GROUPS_JSONL
    if groups_path.exists():
        with open(groups_path, encoding="utf-8") as fh:
            groups = [json.loads(line) for line in fh]
    elif args.smoke:
        groups_path = rf.groups_path()     # plumbing only, before the fresh data exists
        groups = rf.load_groups()
        print("[stage11] smoke: no Stage 11 groups yet - using Stage 8's for plumbing")
    else:
        raise SystemExit(f"[stage11] no Stage 11 training groups at {groups_path} - "
                         "run scripts/26_build_stage11_data.py first")
    if not groups:
        raise SystemExit(f"[stage11] {groups_path} holds no groups")
    init = pathlib.Path(args.init_model) if args.init_model else default_init()
    if not (init / "config.json").exists():
        raise SystemExit(f"[stage11] no starting checkpoint at {init}")

    epochs = args.epochs or int(C.STAGE11_EPOCHS)
    lr = args.lr or float(C.STAGE11_LR)
    samples = args.samples or int(C.STAGE11_RL_SAMPLES)
    temperature = args.temperature or float(C.STAGE11_RL_TEMPERATURE)
    seed = args.seed if args.seed is not None else int(C.STAGE11_SEED)
    # Smoke keeps one group per step: two 8-pair steps at full 512-token length
    # still exercise the loop, the save and the reload, and fit in laptop RAM.
    gps = 1 if args.smoke else int(C.STAGE8_GROUPS_PER_STEP)
    max_groups = 2 if args.smoke else None
    log_every = 1 if args.smoke else int(C.STAGE11_LOG_EVERY)
    arms = ["ce", "rl"] if args.arm == "both" else [args.arm]

    summaries = {}
    for arm in arms:
        fingerprint = {
            "arm": arm, "init": str(init), "init_identity": _init_identity(init),
            "epochs": epochs, "lr": lr, "warmup_frac": float(C.STAGE11_WARMUP_FRAC),
            "weight_decay": float(C.STAGE8_WEIGHT_DECAY), "groups_per_step": gps,
            "seed": seed, "n_groups": len(groups) if not max_groups else max_groups,
            "groups": str(groups_path), "groups_sha1": _file_sha1(groups_path),
            "max_length": int(C.RERANK_MAX_LENGTH),
            "samples": samples if arm == "rl" else None,
            "temperature": temperature if arm == "rl" else None,
        }
        if args.smoke:
            out_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"stage11_smoke_{arm}_"))
            log_path = None
        else:
            out_dir = stage11_dir() / arm
            log_path = C.RESULTS_LATEST_DIR / C.STAGE11_TRAIN_LOG_CSV.replace(
                ".csv", f"_{arm}.csv")
            done = rr.arm_meta(out_dir)
            if done and done.get("fingerprint") == fingerprint and not args.fresh:
                print(f"[stage11] arm {arm} already trained with these settings "
                      f"({out_dir}) - skipping", flush=True)
                summaries[arm] = done
                continue

        stats = rr.train_arm(
            groups, arm=arm, init_model=str(init), out_dir=out_dir,
            epochs=epochs, lr=lr, warmup_frac=float(C.STAGE11_WARMUP_FRAC),
            weight_decay=float(C.STAGE8_WEIGHT_DECAY), groups_per_step=gps,
            seed=seed, samples=samples, temperature=temperature,
            log_every=log_every, log_path=log_path, max_groups=max_groups)
        meta = {"fingerprint": fingerprint, **stats}

        if args.smoke:
            from sentence_transformers import CrossEncoder

            probe = CrossEncoder(str(out_dir), max_length=int(C.RERANK_MAX_LENGTH))
            scores = probe.predict([(groups[0]["question"], groups[0]["pos"]),
                                    (groups[0]["question"], groups[0]["negs"][0])],
                                   show_progress_bar=False)
            print(f"[stage11] smoke: reloaded the saved {arm} arm and scored 2 "
                  f"pairs {list(map(float, scores))}", flush=True)
            del probe
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            # written after the verified save: this file is the completion marker
            (out_dir / "training_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")
        summaries[arm] = meta

    print("\n[stage11] summary")
    for arm, meta in summaries.items():
        line = (f"  {arm}: greedy RR on training groups "
                f"{meta['greedy_rr_first_tenth']:.4f} -> "
                f"{meta['greedy_rr_last_tenth']:.4f} over {meta['steps']} steps")
        if arm == "rl":
            line += (f"; live fraction {meta['live_fraction']:.3f}, "
                     f"mean |advantage| {meta['mean_abs_advantage']:.4f}")
        print(line)
    rl_meta = summaries.get("rl")
    if rl_meta and rl_meta["live_fraction"] < float(C.STAGE11_MIN_LIVE_FRACTION):
        print(f"[stage11] WARN: only {rl_meta['live_fraction']:.1%} of RL groups "
              "produced a gradient - the sampled rankings almost never "
              "disagreed. This is a failed optimisation.")
    if args.smoke:
        print("[stage11] SMOKE MODE: nothing was kept.")
    else:
        print("[stage11] next: python scripts/28_eval_reranker_rl.py --dev")


if __name__ == "__main__":
    main()
