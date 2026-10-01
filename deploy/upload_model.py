"""Upload the fine-tuned reranker to a Hugging Face model repo.

The source directory is the Stage 8 checkpoint: config.json, tokenizer*.json and
model.safetensors, plus `deploy/model_card.md` uploaded as the repo README.

Usage:
    python deploy/upload_model.py --source "<data root>/models/bge_reranker_ft/final"
    python deploy/upload_model.py --source <dir> --public
    python deploy/upload_model.py --source "<data root>/models/bge_reranker_stage15/large/final"         --repo-id sfczaa/bge-reranker-large-nq-ft --card deploy/model_card_large.md --public
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
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--card", default=str(CARD), help="model card uploaded as README.md")
    ap.add_argument("--message", default="Add NQ-train fine-tuned reranker with model card")
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
        shutil.copy2(args.card, stage / "README.md")     # card ships as the README
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
        print(f"[model] repo ready ({'PUBLIC' if args.public else 'PRIVATE'}): {args.repo_id}")
        info = api.upload_folder(folder_path=str(stage), repo_id=args.repo_id,
                                 repo_type="model", commit_message=args.message)
        print(f"[model] upload complete, commit {info.oid}")
        print("[model] files:", sorted(api.list_repo_files(args.repo_id, repo_type="model")))


if __name__ == "__main__":
    main()
