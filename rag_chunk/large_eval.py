"""Stage 6 — larger-scale robustness evaluation.

Re-runs the Stage 3/5 protocol at a larger corpus scale (``N_NQ_DOCS_LARGE``,
~1000 docs/questions) to test whether the Stage 1-5 *directions* replicate.
Nothing else changes: same dataset source, same chunking logic and grids, same
boundary models/weights, same BGE dense retriever, same off-the-shelf reranker
(no fine-tuning), and no rerank50 arm.

Per chunking config:

* ``bge``      — dense-only, on every config of the full 30-config grid;
* ``rerank20`` — BGE top-20 reordered by the cross-encoder, only on the
                 ``C.STAGE6_RERANK_CONFIGS`` subset (small-chunk + size-15
                 configs — enough to test the Stage 5 direction claims).

Because a large run takes hours on the Colab free tier, :func:`run_stage6` is
**resume-safe**: every finished config is appended to a JSONL checkpoint, and
a restarted run skips finished configs (boundary probabilities are recomputed
only for the model types that still have pending configs).

The eval set differs from Stages 3/5 (more documents *and* more questions), so
exact deltas against the archived baselines are not comparable. Instead
:func:`direction_checks` re-tests the four Stage 1-5 direction claims on the
large-scale rows and reports whether each replicates.

Heavy deps are imported lazily inside functions, matching the rest of the
package, so the pure-Python helpers work without them installed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import config as C
from rag_chunk.hybrid import config_key, config_label

BASELINE_ARM = "bge"


def rerank_depth() -> int:
    """The single Stage 6 rerank depth. Stage 6 deliberately runs one rerank
    arm only (top-20); rerank50 is out of scope per the Stage 5 finding."""
    from rag_chunk.rerank import rerank_depths

    depths = rerank_depths()
    if len(depths) != 1:
        raise ValueError(
            f"Stage 6 expects exactly one rerank depth, got {depths} — "
            "set RERANK_DEPTHS=(20,) (rerank50 is out of scope)")
    return depths[0]


def rerank_arm() -> str:
    return f"rerank{rerank_depth()}"


# --------------------------------------------------------------------------- #
# Config plan (full grid) + the selected rerank subset
# --------------------------------------------------------------------------- #
def _key_for(method: str, size: int, overlap: int) -> tuple:
    """The ``hybrid.config_key`` identity for a (method, size, overlap) spec."""
    from rag_chunk import sweep

    if method == "fixed":
        return ("fixed", int(size), int(overlap))
    mn, mx = sweep._semantic_window(int(size))
    return (method, "target", int(size), mn, mx, int(overlap))


def config_plan() -> list[tuple[str, int, int]]:
    """Full-grid configs in the canonical sweep order (same as Stages 3-5)."""
    from rag_chunk import sweep

    fixed_sizes, target_sizes, overlaps = sweep._grids(False)
    plan: list[tuple[str, int, int]] = []
    for overlap in overlaps:
        for size in fixed_sizes:
            plan.append(("fixed", size, overlap))
        for mtype in ("bilstm", "transformer"):
            for target in target_sizes:
                plan.append((mtype, target, overlap))
    return plan


def selected_rerank_keys() -> set[tuple]:
    """Config keys of the subset that also gets the rerank arm; every entry of
    ``C.STAGE6_RERANK_CONFIGS`` must be on the full grid."""
    plan_keys = {_key_for(*cfg) for cfg in config_plan()}
    keys = set()
    for method, size, overlap in C.STAGE6_RERANK_CONFIGS:
        key = _key_for(method, size, overlap)
        if key not in plan_keys:
            raise ValueError(
                f"STAGE6_RERANK_CONFIGS entry {(method, size, overlap)} is not "
                "on the sweep grid")
        keys.add(key)
    return keys


# --------------------------------------------------------------------------- #
# Checkpointing (restart/resume-safe runs)
# --------------------------------------------------------------------------- #
def checkpoint_meta(docs: list[dict], questions: list[dict], mode: str) -> dict:
    """Fingerprint of the run a checkpoint belongs to. A resumed run must match
    exactly, or its rows would silently mix different corpora/models."""
    return {
        "mode": mode,
        "n_docs": len(docs),
        "n_questions": len(questions),
        "retrieval_model": C.RETRIEVAL_EMBED_MODEL,
        "boundary_model": C.BOUNDARY_EMBED_MODEL,
        "reranker_model": C.RERANKER_MODEL,
        "rerank_depth": rerank_depth(),
    }


def load_checkpoint(path, expect_meta: dict) -> list[dict]:
    """Rows already finished by an earlier (interrupted) run; ``[]`` if the
    checkpoint does not exist. A truncated final line (killed mid-write) is
    skipped; a meta mismatch aborts rather than mixing incompatible rows."""
    p = Path(path)
    if not p.exists():
        return []
    meta = None
    rows: list[dict] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("[stage6] WARN: skipping a truncated checkpoint line "
                      "(config not counted as done)")
                continue
            if "meta" in obj:
                meta = obj["meta"]
            elif "rows" in obj:
                rows += obj["rows"]
    if meta != expect_meta:
        raise SystemExit(
            f"[stage6] checkpoint {p.name} belongs to a different run.\n"
            f"[stage6]   checkpoint meta: {meta}\n"
            f"[stage6]   this run       : {expect_meta}\n"
            "[stage6] Re-run with --fresh to discard it and start over."
        )
    return rows


def append_checkpoint(path, obj: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


# --------------------------------------------------------------------------- #
# Stage 6 driver
# --------------------------------------------------------------------------- #
def run_stage6(
    docs: list[dict],
    questions: list[dict],
    *,
    bilstm=None,
    transformer_model=None,
    mode: str = "large",
    checkpoint_path=None,
    scorer=None,
) -> list[dict]:
    """Score the full grid with the bge arm and the selected subset with the
    rerank arm. Resume-safe via ``checkpoint_path``; returns all rows in the
    canonical sweep order."""
    from rag_chunk import rerank, sweep

    depth = rerank_depth()
    sel = selected_rerank_keys()
    plan = config_plan()

    done_rows: list[dict] = []
    if checkpoint_path is not None:
        done_rows = load_checkpoint(checkpoint_path,
                                    checkpoint_meta(docs, questions, mode))
        if not done_rows and not Path(checkpoint_path).exists():
            append_checkpoint(checkpoint_path,
                              {"meta": checkpoint_meta(docs, questions, mode)})
    done_keys = {config_key(r) for r in done_rows}

    remaining = [cfg for cfg in plan if _key_for(*cfg) not in done_keys]
    if done_rows:
        print(f"[stage6] resuming: {len(plan) - len(remaining)}/{len(plan)} "
              "configs already in the checkpoint")

    # Boundary probabilities only for model types that still have pending
    # configs — a resumed run with only fixed configs left skips this entirely.
    needed = {m for m, _, _ in remaining if m in ("bilstm", "transformer")}
    models = {}
    for mtype, model in (("bilstm", bilstm), ("transformer", transformer_model)):
        if mtype in needed:
            if model is None:
                raise ValueError(f"{mtype} configs remain but no {mtype} model given")
            models[mtype] = model
    probs_by_type = sweep._precompute_boundary_probs(docs, models) if models else {}

    rows = list(done_rows)
    done_ct = len(plan) - len(remaining)
    for method, size, overlap in plan:
        key = _key_for(method, size, overlap)
        if key in done_keys:
            continue
        done_ct += 1
        tag = " (+rerank)" if key in sel else ""
        if method == "fixed":
            print(f"[stage6] ({done_ct}/{len(plan)}) fixed        "
                  f"size={size} overlap={overlap}{tag}")
            if key in sel:
                new_rows = rerank._eval_config_rerank(
                    "fixed", docs, questions, scorer=scorer,
                    fixed_size=size, fixed_overlap=overlap)
            else:
                new_rows = [sweep._eval_config(
                    "fixed", docs, questions,
                    fixed_size=size, fixed_overlap=overlap)]
        else:
            mn, mx = sweep._semantic_window(size)
            print(f"[stage6] ({done_ct}/{len(plan)}) {method:<12} target={size} "
                  f"(min={mn},max={mx}) overlap={overlap}{tag}")
            kwargs = dict(
                boundary_probs_by_id=probs_by_type[method],
                semantic_policy="target", semantic_target_size=size,
                semantic_min_size=mn, semantic_max_size=mx,
                semantic_overlap=overlap)
            if key in sel:
                new_rows = rerank._eval_config_rerank(
                    method, docs, questions, scorer=scorer, **kwargs)
            else:
                new_rows = [sweep._eval_config(method, docs, questions, **kwargs)]

        for r in new_rows:
            r.setdefault("arm", BASELINE_ARM)
            r.setdefault("rerank_depth", None)
            r.setdefault("reranker_model", "none")
            r.setdefault("rerank_seconds", None)
            r.setdefault("n_pairs", None)
            r.setdefault(f"pool_recall@{depth}", None)
            r["n_docs"] = len(docs)
            r["n_questions"] = len(questions)
        if checkpoint_path is not None:
            append_checkpoint(checkpoint_path, {"rows": new_rows})
        rows += new_rows
    return rows


# --------------------------------------------------------------------------- #
# Matched summary (the selected rerank configs, bge vs rerank side by side)
# --------------------------------------------------------------------------- #
def matched_summary(rows: list[dict]) -> list[dict]:
    depth = rerank_depth()
    arm = rerank_arm()
    ks = sorted(C.RECALL_KS)
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault(config_key(r), {})[r["arm"]] = r
    out: list[dict] = []
    seen: set[tuple] = set()
    for r in rows:                              # preserve sweep order
        key = config_key(r)
        if key in seen:
            continue
        group = by_key[key]
        if arm not in group or BASELINE_ARM not in group:
            continue
        seen.add(key)
        base, rr = group[BASELINE_ARM], group[arm]
        row = {
            "method": rr["method"],
            "chunk_config": config_label(rr),
            "avg_chunk_size": rr["avg_chunk_size"],
            "n_chunks": rr["n_chunks"],
            f"pool_recall@{depth}": rr.get(f"pool_recall@{depth}"),
        }
        for k in ks:
            row[f"bge_recall@{k}"] = base[f"recall@{k}"]
            row[f"{arm}_recall@{k}"] = rr[f"recall@{k}"]
            row[f"{arm}_minus_bge@{k}"] = rr[f"recall@{k}"] - base[f"recall@{k}"]
        row["rerank_seconds"] = rr.get("rerank_seconds")
        row["n_pairs"] = rr.get("n_pairs")
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Direction checks — do the Stage 1-5 conclusions replicate at large N?
# --------------------------------------------------------------------------- #
def _nominal_size(row: dict) -> int:
    return (row["fixed_size"] if row["method"] == "fixed"
            else row["semantic_target_size"])


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def direction_checks(rows: list[dict], n_questions: int) -> list[dict]:
    """Re-test the four Stage 1-5 direction claims on the Stage 6 rows.

    Each check has an explicit, documented rule (see docs/stage6_large_eval.md)
    and reports the observed value next to the small-eval reference, so a human
    can judge the borderline cases. ``2 SE`` uses the actual question count.
    """
    topk = max(C.RECALL_KS)
    k1 = min(C.RECALL_KS)
    depth = rerank_depth()
    arm = rerank_arm()
    sel = selected_rerank_keys()

    bge = [r for r in rows if r["arm"] == BASELINE_ARM]
    pbar = sum(r[f"recall@{k1}"] for r in bge) / len(bge) if bge else 0.5
    se = (pbar * (1 - pbar) / n_questions) ** 0.5

    checks: list[dict] = []

    def add(claim, metric, value, rule, reference, replicates):
        checks.append({
            "claim": claim, "metric": metric,
            "value": None if value is None else round(value, 4),
            "rule": rule, "reference_small_eval": reference,
            "replicates": replicates,
        })

    # 1a. size > method: recall tracks chunk size.
    r = _pearson([r["avg_chunk_size"] for r in bge],
                 [r[f"recall@{topk}"] for r in bge]) if len(bge) >= 2 else None
    add("1. chunk size matters more than chunking method",
        f"pearson_r(avg_chunk_size, recall@{topk}) over bge configs",
        r, "r >= 0.5",
        "Stage 2/3 (n=203): r ~ +0.77",
        "n/a" if r is None else ("yes" if r >= 0.5 else "no"))

    # 1b. ... and the size effect exceeds the method spread.
    by_size: dict[int, list[float]] = {}
    cells: dict[tuple, dict[str, float]] = {}
    for row in bge:
        size = _nominal_size(row)
        ov = row["fixed_overlap"] if row["method"] == "fixed" else row["semantic_overlap"]
        by_size.setdefault(size, []).append(row[f"recall@{topk}"])
        cells.setdefault((size, ov), {})[row["method"]] = row[f"recall@{topk}"]
    size_effect = method_spread = None
    if by_size:
        lo, hi = min(by_size), max(by_size)
        if lo != hi:
            size_effect = (sum(by_size[hi]) / len(by_size[hi])
                           - sum(by_size[lo]) / len(by_size[lo]))
        spreads = [max(v.values()) - min(v.values())
                   for v in cells.values() if len(v) >= 2]
        method_spread = sum(spreads) / len(spreads) if spreads else None
    ok = (size_effect is not None and method_spread is not None
          and size_effect > method_spread)
    add("1. chunk size matters more than chunking method",
        f"recall@{topk} size effect (largest vs smallest size) minus mean "
        "method spread at matched (size, overlap)",
        None if size_effect is None or method_spread is None
        else size_effect - method_spread,
        "size effect > method spread",
        "Stage 2 (n=203): size ~ +0.07 vs method <= 0.014",
        "yes" if ok else "no")

    # 2. BGE-only remains strong.
    best5 = max((r[f"recall@{topk}"] for r in bge), default=None)
    add("2. BGE-only remains a strong baseline",
        f"best bge recall@{topk} over the grid",
        best5, ">= 0.80 (loose heuristic; the corpus is ~5x larger, so some "
        "absolute drop vs the 200-doc eval is expected)",
        "Stage 3 (200 docs): 0.921",
        "n/a" if best5 is None else ("yes" if best5 >= 0.80 else "no"))

    # 3 & 4 need the rerank deltas on the selected configs.
    deltas: dict[tuple, float] = {}
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        by_key.setdefault(config_key(row), {})[row["arm"]] = row
    for key in sel:
        group = by_key.get(key, {})
        if arm in group and BASELINE_ARM in group:
            deltas[key] = (group[arm][f"recall@{k1}"]
                           - group[BASELINE_ARM][f"recall@{k1}"])
    small_key = _key_for("fixed", 6, 0)
    d_small = deltas.get(small_key)
    d_large = [d for key, d in deltas.items() if key != small_key]
    d_large_mean = sum(d_large) / len(d_large) if d_large else None

    # 3. rerank20 helps mainly at small chunks.
    ok = (d_small is not None and d_large_mean is not None
          and d_small >= 2 * se and d_small > d_large_mean)
    add(f"3. {arm} helps mainly at small chunks",
        f"recall@{k1} delta at fixed size 6 (vs mean delta at the size-15 "
        "configs)",
        d_small,
        f"delta(size 6) >= 2 SE ({2 * se:.4f}) and > mean size-15 delta"
        + (f" ({d_large_mean:+.4f})" if d_large_mean is not None else ""),
        "Stage 5 (n=203): +0.113 at size 6 vs ~+0.002 mean at size 15",
        "n/a" if d_small is None or d_large_mean is None
        else ("yes" if ok else "no"))

    # 4. rerank20 does not clearly improve the best size-15 setting.
    d_max15 = max(d_large, default=None)
    ok = d_max15 is not None and d_max15 < 2 * se
    add(f"4. {arm} does not clearly improve the size-15 sweet spot",
        f"max recall@{k1} delta over the size-15 configs",
        d_max15,
        f"max size-15 delta < 2 SE ({2 * se:.4f})",
        "Stage 5 (n=203): size-15 deltas within 1 SE ~ 0.034",
        "n/a" if d_max15 is None else ("yes" if ok else "no"))

    return checks


# --------------------------------------------------------------------------- #
# Writers (reuse the sweep formatting)
# --------------------------------------------------------------------------- #
def _stage6_columns() -> list[str]:
    from rag_chunk.sweep import _sweep_columns

    cols = ["arm"] + _sweep_columns()
    cols += ["rerank_depth", "reranker_model", "rerank_seconds", "n_pairs",
             f"pool_recall@{rerank_depth()}", "n_docs", "n_questions"]
    return cols


def write_results_csv(rows: list[dict], path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _stage6_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[stage6] wrote {path}  ({len(rows)} rows)")


def _matched_columns() -> list[str]:
    arm = rerank_arm()
    cols = ["method", "chunk_config", "avg_chunk_size", "n_chunks",
            f"pool_recall@{rerank_depth()}"]
    for k in sorted(C.RECALL_KS):
        cols += [f"bge_recall@{k}", f"{arm}_recall@{k}", f"{arm}_minus_bge@{k}"]
    cols += ["rerank_seconds", "n_pairs"]
    return cols


def write_matched_csv(table: list[dict], path) -> None:
    from rag_chunk.sweep import _fmt

    cols = _matched_columns()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow({c: _fmt(c, r.get(c)) for c in cols})
    print(f"[stage6] wrote {path}  ({len(table)} rows)")


def write_direction_csv(checks: list[dict], path) -> None:
    cols = ["claim", "metric", "value", "rule", "reference_small_eval",
            "replicates"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in checks:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    print(f"[stage6] wrote {path}  ({len(checks)} rows)")
