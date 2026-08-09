"""Upload the fine-tuned reranker to a Hugging Face model repo.

Already run (2026-08-09) — kept so the artifact can be reproduced or refreshed.
The source directory is the Stage 8 checkpoint: config.json, tokenizer*.json and
model.safetensors, plus `deploy/model_card.md` uploaded as the repo README.

Usage:
    python deploy/upload_model.py --source "<data root>/models/bge_reranker_ft/final"
    python deploy/upload_model.py --source <dir> --public
"""
from __future__ import annotations
import argparse, pathlib, shutil, tempfile

REPO_ID = "sfczaa/bge-reranker-base-nq-ft"
CARD = pathlib.Path(__file__).resolve().parent / "model_card.md"


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload the fine-tuned reranker.")
    ap.add_argument("--source", required=True,
                    help="dir holding config.json / tokenizer*.json / model.safetensors")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.source)
    need = ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"]
    missing = [n for n in need if not (src / n).exists()]
    if missing:
        raise SystemExit(f"[model] missing in {src}: {', '.join(missing)}")

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="ftmodel_") as tmp:
        stage = pathlib.Path(tmp)
        for n in need:
            shutil.copy2(src / n, stage / n)
        shutil.copy2(CARD, stage / "README.md")          # card ships as the README
        api = HfApi()
        api.create_repo(REPO_ID, repo_type="model", private=not args.public, exist_ok=True)
        print(f"[model] repo ready ({'PUBLIC' if args.public else 'PRIVATE'}): {REPO_ID}")
        api.upload_folder(folder_path=str(stage), repo_id=REPO_ID, repo_type="model",
                          commit_message="Add NQ-train fine-tuned bge-reranker-base with model card")
        print("[model] upload complete")
        print("[model] files:", sorted(api.list_repo_files(REPO_ID, repo_type="model")))


if __name__ == "__main__":
    main()
