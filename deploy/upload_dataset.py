"""Upload the demo assets (bench corpus + prebuilt FAISS indices) to a dataset repo.

Already run (2026-08-09) — kept for reproducibility. The uploaded layout MIRRORS
the study's data root, so the Space can point RAG_DATA_ROOT at a snapshot and let
config.py resolve every path unchanged. Do not flatten it.

Usage:
    python deploy/upload_dataset.py --data-root "<data root>"
    python deploy/upload_dataset.py --data-root <dir> --public
"""
from __future__ import annotations
import argparse, pathlib, shutil, tempfile

REPO_ID = "sfczaa/rag-chunking-ablation-demo-assets"
CARD = pathlib.Path(__file__).resolve().parent / "dataset_card.md"
SUBDIR = pathlib.Path("data/nq/large_n1000")
FILES = ["docs.jsonl", "questions.jsonl", "nq_meta.json",
         "indices/demo/demo_manifest.json",
         "indices/demo/index_fixed.faiss", "indices/demo/index_fixed.json",
         "indices/demo/index_bilstm.faiss", "indices/demo/index_bilstm.json"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload the demo assets.")
    ap.add_argument("--data-root", required=True,
                    help="the study's data root (the dir that contains data/)")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    base = pathlib.Path(args.data_root) / SUBDIR
    missing = [f for f in FILES if not (base / f).exists()]
    if missing:
        raise SystemExit(f"[dataset] missing under {base}: {', '.join(missing)}")

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="dsassets_") as tmp:
        stage = pathlib.Path(tmp)
        for f in FILES:
            dst = stage / SUBDIR / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / f, dst)
        shutil.copy2(CARD, stage / "README.md")
        api = HfApi()
        api.create_repo(REPO_ID, repo_type="dataset", private=not args.public, exist_ok=True)
        print(f"[dataset] repo ready ({'PUBLIC' if args.public else 'PRIVATE'}): {REPO_ID}")
        api.upload_folder(folder_path=str(stage), repo_id=REPO_ID, repo_type="dataset",
                          commit_message="Add NQ bench corpus and prebuilt demo FAISS indices")
        print("[dataset] upload complete")
        print("[dataset] files:", sorted(api.list_repo_files(REPO_ID, repo_type="dataset")))


if __name__ == "__main__":
    main()
