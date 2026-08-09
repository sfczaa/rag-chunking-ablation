"""Create the Hugging Face ZeroGPU Space and push the demo.

Assembles the Space payload itself — `deploy/space/*` plus a fresh copy of
`config.py` and `rag_chunk/` from the repo root — so the deployed app can never
drift from the study's code.

**Eligibility.** ZeroGPU is free only for accounts in good standing: verified
email AND account older than 30 days. Before that, `create_repo` returns
``402 Payment Required`` for a Gradio Space on any hardware. That is an account
gate, not a configuration problem; do not "fix" it by subscribing.

Usage:
    python deploy/create_space.py                 # create + push
    python deploy/create_space.py --public        # world-readable Space
    python deploy/create_space.py --dry-run       # assemble only, no network
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import tempfile

REPO_ID = "sfczaa/rag-chunking-ablation-demo"
HARDWARE = "zero-a10g"          # the ZeroGPU tier; NOT a paid GPU
ROOT = pathlib.Path(__file__).resolve().parent.parent


def assemble(dest: pathlib.Path) -> pathlib.Path:
    """deploy/space/* + config.py + rag_chunk/ -> a self-contained Space dir."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(ROOT / "deploy" / "space" / name, dest / name)
    shutil.copy2(ROOT / "config.py", dest / "config.py")
    shutil.copytree(ROOT / "rag_chunk", dest / "rag_chunk",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    dirs_exist_ok=True)
    files = sorted(p.relative_to(dest).as_posix()
                   for p in dest.rglob("*") if p.is_file())
    print(f"[space] assembled {len(files)} files in {dest}")
    for f in files:
        print(f"[space]   {f}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Create/push the ZeroGPU Space.")
    ap.add_argument("--public", action="store_true",
                    help="create the Space world-readable (default: private)")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the payload and stop (no Hub calls)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="space_") as tmp:
        payload = assemble(pathlib.Path(tmp) / "space")
        if args.dry_run:
            print("[space] --dry-run: nothing pushed")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        try:
            api.create_repo(REPO_ID, repo_type="space", space_sdk="gradio",
                            space_hardware=HARDWARE, private=not args.public,
                            exist_ok=True)
        except Exception as exc:
            if "402" in str(exc) or "Payment" in str(exc):
                raise SystemExit(
                    "[space] 402 Payment Required — this account cannot host a "
                    "Gradio Space yet.\n"
                    "[space] ZeroGPU hosting is free only with a verified email "
                    "AND an account older than 30 days.\n"
                    "[space] Wait until the account qualifies and re-run; do not "
                    "subscribe to work around it.") from exc
            raise
        print(f"[space] repo ready ({'PUBLIC' if args.public else 'PRIVATE'}, "
              f"gradio, {HARDWARE}): {REPO_ID}")

        api.upload_folder(folder_path=str(payload), repo_id=REPO_ID,
                          repo_type="space",
                          commit_message="Add interactive retrieval demo "
                                         "(prebuilt indices, ZeroGPU)")
        print("[space] pushed")
        print(f"[space] https://huggingface.co/spaces/{REPO_ID}")
        print("[space] NOTE: if the model/dataset repos are private, add an "
              "HF_TOKEN secret in Space settings, or make them public.")


if __name__ == "__main__":
    main()
