"""Archive the current ``results/latest/`` into a stage snapshot folder.

``results/latest/`` is the *working* output — every new sweep overwrites it. This
utility copies its contents into ``results/<stage>/final/`` so a finished run is
preserved while ``latest/`` stays free to be overwritten. It only ever **copies**;
nothing is moved or deleted.

Usage:
    python scripts/save_stage_results.py --stage stage1
"""

import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402

VALID_STAGES = ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6",
                "stage7", "stage8", "stage9")


def stage_final_dir(stage: str) -> pathlib.Path:
    """``artifacts/results/<stage>/final`` for a validated stage name."""
    if stage not in VALID_STAGES:
        raise ValueError(f"--stage must be one of {VALID_STAGES}, got {stage!r}")
    return C.RESULTS_DIR / stage / "final"


def save_stage_results(stage: str, src: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Copy every file in ``results/latest/`` into ``results/<stage>/final/``.

    Returns the list of destination paths written. Never deletes anything; the
    destination is created if missing and existing files there are overwritten by
    the fresh copy (a re-archive of the same stage).
    """
    dest = stage_final_dir(stage)
    src = pathlib.Path(src) if src is not None else C.RESULTS_LATEST_DIR

    if not src.exists():
        raise SystemExit(
            f"[save-stage] nothing to copy: {src} does not exist.\n"
            f"             run `python scripts/6_sweep_chunking.py` first.")

    dest.mkdir(parents=True, exist_ok=True)

    copied: list[pathlib.Path] = []
    skipped_dirs: list[str] = []
    for item in sorted(src.iterdir()):
        if item.is_file():
            out = dest / item.name
            shutil.copy2(item, out)
            copied.append(out)
        elif item.is_dir():
            skipped_dirs.append(item.name)

    if not copied:
        print(f"[save-stage] {src} has no files to copy.")
    else:
        print(f"[save-stage] copied {len(copied)} file(s): {src} -> {dest}")
        for p in copied:
            print(f"  {p}")
    if skipped_dirs:
        print(f"[save-stage] skipped sub-folders (files only): {skipped_dirs}")
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Archive results/latest/ into results/<stage>/final/ (copy only).")
    ap.add_argument("--stage", required=True, choices=VALID_STAGES,
                    help="stage folder to snapshot into (stage1 | ... | stage9)")
    args = ap.parse_args()
    save_stage_results(args.stage)


if __name__ == "__main__":
    main()
