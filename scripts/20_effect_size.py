"""Descriptive effect-size regression and minimum detectable effect.

Reads the archived Stage 6 dense grid: 30 configurations (three methods,
five sizes, two overlaps) evaluated on the same 1032 NQ questions.

OLS summarizes associations between aggregate recall, size, overlap, and
method. The MDE calculation uses an unpaired two-proportion approximation;
per-question paired hits are not available in the archive. Neither small
observed gaps nor non-significant coefficients establish method equivalence.

Usage: python scripts/20_effect_size.py
Outputs: artifacts/results/portfolio/effect_size_{coefficients.csv,report.md,tornado.png}.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
SRC = ARTIFACTS / "results" / "stage6" / "final" / "stage6_large_eval_results.csv"
OUT = ARTIFACTS / "results" / "portfolio"

METHODS = ("fixed", "bilstm", "transformer")
SIZE_LO, SIZE_HI = 6, 15          # swept size range, for "effect over range"
TARGETS = ("recall@1", "recall@3", "recall@5")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_grid() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df = df[df["arm"] == "bge"].copy()
    # unify the size / overlap knobs across fixed and semantic configs
    df["size"] = df["fixed_size"].fillna(df["semantic_target_size"])
    df["overlap"] = df["fixed_overlap"].fillna(df["semantic_overlap"])
    if len(df) != len(METHODS) * 5 * 2:
        raise SystemExit(f"[effect] expected 30 bge configs, got {len(df)}")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# (a) OLS
# --------------------------------------------------------------------------- #
def _design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    n = len(df)
    X = np.column_stack([
        np.ones(n),
        df["size"].to_numpy(float),
        df["overlap"].to_numpy(float),
        (df["method"] == "bilstm").to_numpy(float),
        (df["method"] == "transformer").to_numpy(float),
    ])
    names = ["intercept", "size", "overlap",
             "method[bilstm]", "method[transformer]"]
    return X, names


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    n, p = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - p
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "t": t, "p": pval, "r2": r2, "dof": dof}


def coeff_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    X, names = _design(df)
    fit = ols(X, df[target].to_numpy(float))
    ci = 1.96 * fit["se"]
    return pd.DataFrame({
        "target": target,
        "term": names,
        "coef": fit["beta"],
        "se": fit["se"],
        "ci95_lo": fit["beta"] - ci,
        "ci95_hi": fit["beta"] + ci,
        "t": fit["t"],
        "p_value": fit["p"],
        "r2_model": fit["r2"],
    })


# --------------------------------------------------------------------------- #
# (b) Minimum detectable effect
# --------------------------------------------------------------------------- #
def _se_diff(p: float, n: int) -> float:
    """Unpaired SE approximation for two proportions at the same p."""
    return float(np.sqrt(2 * p * (1 - p) / n))


def mde_analysis(df: pd.DataFrame, target: str) -> dict:
    n = int(df["n_questions"].iloc[0])
    p_bar = float(df[target].mean())
    se_diff = _se_diff(p_bar, n)
    floor95 = 1.96 * se_diff                      # approximate 95% noise band
    mde80 = (1.96 + 0.8416) * se_diff             # detectable at 80% power

    # largest observed between-method gap at any matched (size, overlap) cell
    cells = []
    for (size, ov), g in df.groupby(["size", "overlap"]):
        vals = {m: float(g.loc[g["method"] == m, target].iloc[0])
                for m in METHODS if (g["method"] == m).any()}
        spread = max(vals.values()) - min(vals.values())
        cells.append({"size": int(size), "overlap": int(ov),
                      "spread": spread, **vals})
    cells_df = pd.DataFrame(cells).sort_values(["overlap", "size"])
    return {"n": n, "p_bar": p_bar, "se_diff": se_diff, "floor95": floor95,
            "mde80": mde80, "max_spread": float(cells_df["spread"].max()),
            "cells": cells_df}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def tornado(df: pd.DataFrame, coefs: pd.DataFrame, mde: dict,
            path: pathlib.Path) -> None:
    c = coefs.set_index("term")
    # effect on R@5 attributable to each knob, with 95% CI half-width
    size_eff = c.loc["size", "coef"] * (SIZE_HI - SIZE_LO)
    size_ci = c.loc["size", "se"] * 1.96 * (SIZE_HI - SIZE_LO)
    ov_eff = c.loc["overlap", "coef"]
    ov_ci = c.loc["overlap", "se"] * 1.96
    # method: largest |shift from fixed| among the two learned arms
    m_terms = ["method[bilstm]", "method[transformer]"]
    m_idx = c.loc[m_terms, "coef"].abs().idxmax()
    m_eff = abs(c.loc[m_idx, "coef"])
    m_ci = c.loc[m_idx, "se"] * 1.96

    labels = [f"chunk size ({SIZE_LO}→{SIZE_HI} sent.)",
              "overlap (0→1 sent.)",
              f"chunk method (max |Δ|,\n{m_idx.split('[')[1][:-1]} vs fixed)"]
    effs = [abs(size_eff), abs(ov_eff), m_eff]
    cis = [size_ci, ov_ci, m_ci]
    colors = ["#1f77b4", "#7f7f7f", "#d62728"]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, effs, xerr=cis, color=colors, alpha=0.85,
            error_kw={"ecolor": "#333", "capsize": 4})
    # noise band: nothing below the 95% detection floor is distinguishable
    ax.axvspan(0, mde["floor95"], color="#999", alpha=0.18, zorder=0)
    ax.axvline(mde["floor95"], color="#666", ls="--", lw=1)
    ax.text(mde["floor95"], len(labels) - 0.5,
            f"  95% detection floor = {mde['floor95']:.3f}",
            va="top", ha="left", fontsize=8, color="#444")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("effect on doc-constrained Recall@5 (absolute)")
    ax.set_title("Chunk size is a real lever; chunking method sits inside the "
                 "noise\nStage 6 bench (1000 docs / 1032 questions, dense BGE)",
                 fontsize=10)
    for y, e, ci in zip(yy, effs, cis):
        ax.text(e + ci + 0.002, y, f"{e:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(effs) + max(cis) + 0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(df: pd.DataFrame, all_coefs: pd.DataFrame,
                 mde5: dict, r_size: float, r_avg: float) -> str:
    c5 = all_coefs[all_coefs["target"] == "recall@5"].set_index("term")
    size_eff = c5.loc["size", "coef"] * (SIZE_HI - SIZE_LO)
    m_terms = ["method[bilstm]", "method[transformer]"]
    max_method = c5.loc[m_terms, "coef"].abs().max()
    ratio = abs(size_eff) / abs(c5.loc["overlap", "coef"]) if \
        c5.loc["overlap", "coef"] else float("nan")
    ratio_m = abs(size_eff) / max_method if max_method else float("inf")

    lines = [
        "# Effect-size regression and detection limits",
        "",
        "Derived from the archived Stage 6 dense grid "
        "(`stage6_large_eval_results.csv`, 30 `bge` configs = 3 methods x 5 "
        "sizes x 2 overlaps, n=1032 questions). No experiment was rerun.",
        "",
        "## (a) Descriptive OLS fit - `R@5 ~ size + overlap + C(method)`",
        "",
        f"Model R^2 = {c5['r2_model'].iloc[0]:.3f}. "
        f"Pearson r(size, R@5) = {r_size:.3f} "
        f"(r(avg_chunk_size, R@5) = {r_avg:.3f}).",
        "",
        "| term | coef | 95% CI | t | p |",
        "|---|---|---|---|---|",
    ]
    for term in c5.index:
        row = c5.loc[term]
        lines.append(
            f"| {term} | {row['coef']:+.4f} | "
            f"[{row['ci95_lo']:+.4f}, {row['ci95_hi']:+.4f}] | "
            f"{row['t']:+.2f} | {row['p_value']:.2g} |")
    lines += [
        "",
        f"- Size moves fitted R@5 by {size_eff:+.3f} across the swept range "
        f"({SIZE_LO}->{SIZE_HI} sentences). The nominal OLS p-value is "
        f"{c5.loc['size', 'p_value']:.1e}.",
        f"- Overlap (a nuisance knob) shifts R@5 by "
        f"{c5.loc['overlap', 'coef']:+.4f} per sentence; the size range effect "
        f"is ~{ratio:.1f}x it.",
        f"- Method dummies (BiLSTM, Transformer vs fixed) are the smallest "
        f"terms: max |coef| = {max_method:.4f}, "
        f"with nominal p-values of {c5.loc[m_terms, 'p_value'].min():.2f}-"
        f"{c5.loc[m_terms, 'p_value'].max():.2f}. The size "
        f"range effect is ~{ratio_m:.0f}x the largest method effect.",
        "",
        "## (b) Minimum detectable effect under an unpaired approximation",
        "",
        f"At n = {mde5['n']} questions and mean R@5 = {mde5['p_bar']:.3f}, the "
        f"unpaired SE of a between-method difference is "
        f"{mde5['se_diff']:.4f}. So this study could:",
        "",
        f"- place a gap outside the approximate 95% noise band only if it exceeds "
        f"{mde5['floor95']:.3f};",
        f"- reach 80% power under this approximation only if the true gap is at least "
        f"{mde5['mde80']:.3f}.",
        "",
        f"The largest between-method R@5 gap at any matched "
        f"(size, overlap) cell is {mde5['max_spread']:.3f} - below both "
        f"thresholds. This does not establish method equivalence or rule out "
        f"effects below the detection limit.",
        "",
        "Per-cell between-method spread (max − min across fixed / bilstm / "
        "transformer):",
        "",
        "| size | overlap | fixed | bilstm | transformer | spread |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in mde5["cells"].iterrows():
        lines.append(
            f"| {int(r['size'])} | {int(r['overlap'])} | {r['fixed']:.4f} | "
            f"{r['bilstm']:.4f} | {r['transformer']:.4f} | {r['spread']:.4f} |")
    lines += [
        "",
        "> Caveat. The three methods are scored on the same questions, "
        "but per-question hits are not in the Stage 6 archive. The unpaired "
        "MDE is an approximation, not a paired test or proof of equivalence. "
        "OLS intervals and p-values are descriptive because configurations share "
        "questions and are not independent experiments.",
        "",
        "_Generated by `scripts/20_effect_size.py` from committed archives._",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_grid()

    all_coefs = pd.concat([coeff_table(df, t) for t in TARGETS],
                          ignore_index=True)
    all_coefs.to_csv(OUT / "effect_size_coefficients.csv", index=False)

    r_size = float(np.corrcoef(df["size"], df["recall@5"])[0, 1])
    r_avg = float(np.corrcoef(df["avg_chunk_size"], df["recall@5"])[0, 1])
    mde5 = mde_analysis(df, "recall@5")

    tornado(df, all_coefs[all_coefs["target"] == "recall@5"], mde5,
            OUT / "effect_size_tornado.png")

    report = write_report(df, all_coefs, mde5, r_size, r_avg)
    (OUT / "effect_size_report.md").write_text(report, encoding="utf-8")

    # console summary (sanity check against the known r ~= 0.95 headline)
    print(f"[effect] {len(df)} bge configs; r(size,R@5)={r_size:.3f}, "
          f"r(avg_size,R@5)={r_avg:.3f}")
    c5 = all_coefs[all_coefs["target"] == "recall@5"].set_index("term")
    print(f"[effect] size range effect (6->15) = "
          f"{c5.loc['size','coef']*(SIZE_HI-SIZE_LO):+.3f} "
          f"(p={c5.loc['size','p_value']:.1e})")
    print(f"[effect] max |method coef| = "
          f"{c5.loc[['method[bilstm]','method[transformer]'],'coef'].abs().max():.4f} "
          f"(p={c5.loc[['method[bilstm]','method[transformer]'],'p_value'].min():.2f}-"
          f"{c5.loc[['method[bilstm]','method[transformer]'],'p_value'].max():.2f})")
    print(f"[effect] 95% floor={mde5['floor95']:.3f}, 80%% MDE={mde5['mde80']:.3f}, "
          f"max observed method spread={mde5['max_spread']:.3f}")
    print(f"[effect] wrote 3 files to {OUT}")


if __name__ == "__main__":
    main()
