"""Assemble and upload the retrieval demo to a Hugging Face Space.

The payload contains ``deploy/space/*``, ``config.py``, and ``rag_chunk/``.
Space creation requires account access to the requested hardware.

Usage:
    python deploy/create_space.py
    python deploy/create_space.py --public
    python deploy/create_space.py --dry-run
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import tempfile

REPO_ID = "sfczaa/rag-chunking-ablation-demo"
HARDWARE = "zero-a10g"
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
                    "[space] Hosting access denied (HTTP 402). Check the account's "
                    "current Space hardware entitlements.") from exc
            raise
        print(f"[space] repo ready ({'PUBLIC' if args.public else 'PRIVATE'}, "
              f"gradio, {HARDWARE}): {REPO_ID}")

        api.upload_folder(folder_path=str(payload), repo_id=REPO_ID,
                          repo_type="space",
                          commit_message="Add interactive retrieval demo "
                                         "(prebuilt indices, ZeroGPU)")
        print("[space] pushed")
        print(f"[space] https://huggingface.co/spaces/{REPO_ID}")
        print("[space] Private assets require a read-only HF_TOKEN Space secret.")


if __name__ == "__main__":
    main()
