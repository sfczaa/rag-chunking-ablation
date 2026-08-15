"""Local CPU smoke test for the Space app — run this BEFORE every push.

Assembles the exact payload `create_space.py` would deploy, points it at local
assets instead of the Hub, and exercises the whole boot path: config resolution
from RAG_DATA_ROOT, prebuilt-index loading, model placement, both ranking arms,
and HTML rendering. Runs on CPU (slow but correct); `@spaces.GPU` is a no-op off
Space.

It is the cheapest check that the shipped indices really are the study's: the
printed chunk counts and mean chunk sizes must match the archived Stage 6 row
(fixed 15/0 -> 19507 chunks / 14.649 sentences; bilstm t15/0 -> 19306 / 14.801).

Usage:
    python deploy/smoke_space.py --data-root "<data root>" \
                                 --ft-model "<data root>/models/bge_reranker_ft/final"
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from create_space import assemble  # noqa: E402

EXPECTED = {"fixed": (19507, 14.649), "bilstm": (19306, 14.801)}
TOL_SIZE = 0.01
BENCH_SUBDIR = pathlib.Path("data/nq/large_n1000")


def _ascii_safe_data_root(data_root: pathlib.Path,
                          tmp: pathlib.Path) -> pathlib.Path:
    """FAISS reads indices through a C++ ``const char*``, which on Windows
    cannot open a path containing non-ASCII characters — and a localised Drive
    mount produces exactly that. Python-side loads (JSONL, safetensors) handle
    Unicode fine, so only the bench subtree needs relocating; staging it under an
    ASCII temp dir keeps the smoke test usable without remounting anything.
    Spaces run on Linux under ASCII cache paths, so this never applies in
    production."""
    if str(data_root).isascii():
        return data_root
    src = data_root / BENCH_SUBDIR
    if not src.is_dir():
        raise SystemExit(f"[smoke] bench dir not found: {src}")
    dest_root = tmp / "assets"
    dest = dest_root / BENCH_SUBDIR
    print(f"[smoke] non-ASCII data root detected — staging the bench to an "
          f"ASCII path so FAISS can read it\n[smoke]   {src}\n[smoke]   -> {dest}")
    shutil.copytree(src, dest)
    mb = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"[smoke] staged {mb:.0f} MB")
    return dest_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the Space app locally.")
    ap.add_argument("--data-root", required=True,
                    help="dir containing data/nq/large_n1000/... (the assets)")
    ap.add_argument("--ft-model", required=True,
                    help="local fine-tuned reranker dir")
    ap.add_argument("--question-index", type=int, default=0)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="smoke_") as tmp:
        payload = assemble(pathlib.Path(tmp) / "space")
        data_root = _ascii_safe_data_root(
            pathlib.Path(args.data_root).resolve(), pathlib.Path(tmp))
        os.environ["LOCAL_ASSETS"] = str(data_root)
        os.environ["FT_REPO"] = str(pathlib.Path(args.ft_model).resolve())
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        sys.path.insert(0, str(payload))

        t0 = time.perf_counter()
        app = importlib.import_module("app")
        print(f"\n[smoke] boot OK in {time.perf_counter() - t0:.1f}s "
              f"(device={app.DEVICE})")
        print(f"[smoke] bench: {len(app.docs)} docs / {len(app.questions)} questions")

        # the indices must be the archived ones, not a rebuild
        for method, (n_exp, avg_exp) in EXPECTED.items():
            idx = app.indices[method]
            n, avg = len(idx.chunk_texts), idx.avg_chunk_size()
            ok = (n == n_exp) and abs(avg - avg_exp) < TOL_SIZE
            print(f"[smoke] index {method:<7} {n} chunks, avg {avg:.3f} "
                  f"(archive: {n_exp} / {avg_exp}) {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit(f"[smoke] {method} index does not match the "
                                 "Stage 6 archive — wrong or rebuilt assets")

        q = app.questions[args.question_index]
        gold = tuple(q.get("doc_titles") or ()) or (q.get("doc_title"),)
        print(f"\n[smoke] question: {q['question']!r}")
        print(f"[smoke] answer:   {q['answer']!r}  gold={gold}")

        last = None
        for arm in (app.ARM_BGE, app.ARM_OTS, app.ARM_FT):
            t = time.perf_counter()
            last = app._retrieve_and_rank(q["question"], arm)
            print(f"\n[smoke] arm={arm} ({time.perf_counter() - t:.1f}s)")
            for (method, _, _), (_, ranked, secs) in zip(app.DEMO_CONFIGS, last):
                if len(ranked) != app.TOP_SHOW:
                    raise SystemExit(f"[smoke] expected {app.TOP_SHOW} chunks, "
                                     f"got {len(ranked)}")
                hits = [i for i, (_, c) in enumerate(ranked, 1)
                        if app.is_hit(c, q["answer"], gold)]
                print(f"[smoke]   {method:<8} {secs:6.2f}s  hit ranks: "
                      f"{hits or 'none'}")

        rendered = app._render_side("t", 15.0, last[0][1], q["answer"], gold,
                                    app.ARM_FT)
        if "<script" in rendered.lower():
            raise SystemExit("[smoke] unescaped script tag in rendered HTML")
        print(f"\n[smoke] render OK ({len(rendered)} chars)")
        print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
