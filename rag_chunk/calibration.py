"""Stage 2.1 — boundary-threshold calibration + probability diagnostics.

The retrieval sweep cuts with the target-size (argmax) policy and never reads a
probability threshold, so a boundary model can drive perfectly good chunks yet
still report Boundary F1 = 0 when the *diagnostic* threshold is miscalibrated.
That is exactly what happened to the Transformer at the BiLSTM-tuned
``BOUNDARY_THRESHOLD = 0.8``.

This module calibrates the threshold on the validation split (max boundary F1),
re-reports the held-out test F1 at that threshold, and writes three diagnostics so
the choice is auditable instead of a magic number:

    results/latest/transformer_threshold_f1.csv          full val threshold sweep
    results/latest/transformer_boundary_diagnostics.json prob distribution + summary
    results/latest/transformer_boundary_threshold_f1.png F1 / precision / recall

Nothing here touches retrieval, the sweep or the BiLSTM — it only changes which
threshold the Boundary F1 *number* is read at.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import config as C


def calibrate_boundary_threshold(
    model,
    model_type: str = "transformer",
    val_split: str = "val",
    test_split: str = "test",
    out_dir=None,
    plot: bool = True,
) -> dict:
    """Calibrate the boundary threshold on ``val_split`` and write diagnostics.

    Returns a summary dict: the chosen ``threshold``, the val/test Boundary F1 at
    it, a reference at the old fixed ``BOUNDARY_THRESHOLD`` (the bug being fixed),
    the probability ``diagnostics`` and the written ``artifacts`` paths.
    """
    from rag_chunk import metrics

    out_dir = Path(out_dir) if out_dir is not None else C.RESULTS_LATEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # One forward pass per split, shared by the sweep, the diagnostics and the
    # held-out report (instead of re-running the model per threshold).
    val_probs, val_labels = metrics.collect_boundary_scores(model, val_split)
    test_probs, test_labels = metrics.collect_boundary_scores(model, test_split)

    sweep_rows = metrics.boundary_threshold_sweep(val_probs, val_labels)
    diagnostics = metrics.boundary_prob_diagnostics(val_probs, val_labels)

    best = metrics.best_threshold(sweep_rows)
    if not best or int(val_labels.sum()) == 0:
        # No positive val boundaries to fit — keep the configured default.
        threshold = float(C.TRANSFORMER_BOUNDARY_THRESHOLD)
        print(f"[calib] WARN no positive val boundaries to calibrate on; "
              f"falling back to TRANSFORMER_BOUNDARY_THRESHOLD={threshold}")
    else:
        threshold = float(best["threshold"])

    val_f1 = metrics.boundary_f1_from_scores(val_probs, val_labels, threshold)
    test_f1 = metrics.boundary_f1_from_scores(test_probs, test_labels, threshold)
    ref_t = float(C.BOUNDARY_THRESHOLD)
    fixed_ref = {
        "threshold": ref_t,
        "val": metrics.boundary_f1_from_scores(val_probs, val_labels, ref_t),
        "test": metrics.boundary_f1_from_scores(test_probs, test_labels, ref_t),
    }

    csv_path = out_dir / C.TRANSFORMER_THRESHOLD_CSV
    json_path = out_dir / C.TRANSFORMER_DIAGNOSTICS_JSON
    png_path = out_dir / C.TRANSFORMER_THRESHOLD_PNG

    _write_threshold_csv(sweep_rows, csv_path)

    summary = {
        "model_type": model_type,
        "threshold": threshold,
        "selection": "max boundary F1 on val (ties -> higher threshold)",
        "val_boundary_f1": val_f1,
        "test_boundary_f1": test_f1,
        "fixed_threshold_reference": fixed_ref,
        "diagnostics": diagnostics,
        "artifacts": {"threshold_csv": str(csv_path),
                      "diagnostics_json": str(json_path)},
    }
    if plot:
        try:
            _plot_threshold_f1(sweep_rows, threshold, ref_t, png_path)
            summary["artifacts"]["threshold_png"] = str(png_path)
        except Exception as e:                  # plot is a bonus; never fail training on it
            print(f"[calib] WARN could not write {png_path.name}: {e}")

    _write_diagnostics_json(summary, json_path)
    _print_summary(summary)
    return summary


# --------------------------------------------------------------------------- #
# Writers / plot / print
# --------------------------------------------------------------------------- #
def _write_threshold_csv(rows: list[dict], path) -> None:
    cols = ["threshold", "precision", "recall", "f1", "n_pred_pos", "tp", "fp", "fn"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = {c: r[c] for c in cols}
            for c in ("threshold", "precision", "recall", "f1"):
                row[c] = f"{r[c]:.4f}"
            w.writerow(row)
    print(f"[calib] wrote {path}  ({len(rows)} thresholds)")


def _json_default(o):
    # robustness if a numpy scalar ever sneaks through
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _write_diagnostics_json(summary: dict, path) -> None:
    Path(path).write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"[calib] wrote {path}")


def _plot_threshold_f1(rows: list[dict], chosen: float, fixed_ref: float, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ts, [r["f1"] for r in rows], label="F1", linewidth=2)
    ax.plot(ts, [r["precision"] for r in rows], label="Precision",
            linestyle="--", alpha=0.8)
    ax.plot(ts, [r["recall"] for r in rows], label="Recall",
            linestyle=":", alpha=0.8)
    ax.axvline(chosen, color="green", alpha=0.7, label=f"calibrated = {chosen:.2f}")
    ax.axvline(fixed_ref, color="red", alpha=0.5, label=f"old fixed = {fixed_ref:.2f}")
    ax.set_xlabel("Boundary probability threshold")
    ax.set_ylabel("Score (Wikipedia validation boundaries)")
    ax.set_title("Transformer boundary calibration: F1 / precision / recall vs threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[calib] wrote {path}")


def _print_summary(s: dict) -> None:
    d = s["diagnostics"]["overall"]
    pos = s["diagnostics"]["positives"]
    neg = s["diagnostics"]["negatives"]
    ref = s["fixed_threshold_reference"]
    nan = float("nan")
    print("\n[calib] transformer boundary-threshold calibration")
    print(f"  val prob: min={d.get('min', 0):.3f} median={d.get('median', 0):.3f} "
          f"max={d.get('max', 0):.3f}  "
          f"(mean on true boundaries={pos.get('mean', nan):.3f} "
          f"vs non-boundaries={neg.get('mean', nan):.3f})")
    print(f"  old fixed threshold {ref['threshold']:.2f}: "
          f"val F1={ref['val']['f1']:.4f}  test F1={ref['test']['f1']:.4f}")
    print(f"  calibrated threshold {s['threshold']:.2f}: "
          f"val F1={s['val_boundary_f1']['f1']:.4f}  "
          f"test F1={s['test_boundary_f1']['f1']:.4f}")
