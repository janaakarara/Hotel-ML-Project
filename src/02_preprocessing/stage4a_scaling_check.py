"""Stage 4a: scaling check. Only MEASURES and DRAWS, no column is changed.

Compares the 4 numeric columns before and after three scalers.
HasChildren is a 0/1 flag and is left out of this comparison. It is standardized in stage 4b (see the stage 5a balance check).

Usage (from the repo root):
  python src/02_preprocessing/stage4a_scaling_check.py
Reads:  data/processed/h2_clustering_stage3.csv
Writes: outputs/04_scaling/stage4a_scaling_distributions.png   (big: every column x every scaler)
        outputs/04_scaling/stage4a_scaling_spread.png          (spread of each column before and after)
        outputs/04_scaling/stage4a_scaling_summary.csv
        outputs/04_scaling/stage4a_scaling_ranking.csv       (which scaler is best, and the score)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

NUMERIC = ["Adults", "LeadTime_yj", "StaysInWeekNights_yj", "StaysInWeekendNights_sqrt"]


def versions(x):
    """Every option we compare. Returns {name: values}."""
    a = x.values.reshape(-1, 1)
    return {
        "Original": x.values,
        "StandardScaler": StandardScaler().fit_transform(a).ravel(),
        "MinMax (0 to 1)": MinMaxScaler().fit_transform(a).ravel(),
        "RobustScaler": RobustScaler().fit_transform(a).ravel(),
    }


def ranking(df):
    """Equal-influence score per scaler: max std / min std across the columns (1.0 = perfectly equal).
    Distance-based clustering works best when no column is wider than the others."""
    rows = []
    for m in ["Original", "StandardScaler", "MinMax (0 to 1)", "RobustScaler"]:
        stds = [float(np.std(versions(df[c])[m])) for c in NUMERIC]
        rows.append({"version": m, "widest_over_narrowest": round(max(stds) / min(stds), 2)})
    r = pd.DataFrame(rows)
    best = r[r.version != "Original"].sort_values("widest_over_narrowest").iloc[0]["version"]
    return r, best


def distributions_figure(df, path, best):
    n_methods = 4
    fig, axes = plt.subplots(len(NUMERIC), n_methods, figsize=(7 * n_methods, 5.5 * len(NUMERIC)))
    rows = []
    for i, c in enumerate(NUMERIC):
        for j, (m, v) in enumerate(versions(df[c]).items()):
            rows.append({"column": c, "version": m, "std": round(float(np.std(v)), 3),
                         "min": round(float(np.min(v)), 3), "max": round(float(np.max(v)), 3),
                         "range": round(float(np.max(v) - np.min(v)), 3)})
            ax = axes[i, j]
            is_best = (m == best)
            ax.hist(v, bins=int(min(40, max(5, df[c].nunique()))), color="tab:red" if is_best else "tab:blue",
                    edgecolor="white", linewidth=0.3)
            tag = "  <- BEST" if is_best else ""
            ax.set_title(f"{c} | {m}{tag}\nspread (std) = {np.std(v):.2f}   range = {np.min(v):.1f} to {np.max(v):.1f}",
                         fontsize=17, fontweight="bold" if is_best else "normal",
                         color="tab:red" if is_best else "black")
            ax.tick_params(labelsize=13)
            if j == 0:
                ax.set_ylabel("rows", fontsize=14)
    fig.suptitle("Scaling check: every column (rows) x every scaler (columns).  "
                 "Red = best scaler (most equal spread across columns)", fontsize=24, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def spread_figure(summary, path, best):
    methods = ["Original", "StandardScaler", "MinMax (0 to 1)", "RobustScaler"]
    base = {"Original": "tab:gray", "StandardScaler": "tab:blue", "MinMax (0 to 1)": "tab:green", "RobustScaler": "tab:orange"}
    colors = ["tab:red" if m == best else base[m] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    for ax, key, title in zip(axes, ["std", "range"], ["Spread (std) of each column", "Range (max - min) of each column"]):
        w = 0.2
        for k, (m, col) in enumerate(zip(methods, colors)):
            sub = summary[summary.version == m].set_index("column").loc[NUMERIC]
            ax.bar(np.arange(len(NUMERIC)) + (k - 1.5) * w, sub[key], w, label=m + (" (BEST)" if m == best else ""), color=col)
        ax.set_xticks(range(len(NUMERIC)))
        ax.set_xticklabels(NUMERIC, rotation=20, ha="right", fontsize=15)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_title(title, fontsize=20)
        ax.legend(fontsize=15)
    fig.suptitle(f"Before scaling (grey) LeadTime_yj is far wider than the others. Red = best scaler: {best}",
                 fontsize=22, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage3.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "04_scaling").mkdir(parents=True, exist_ok=True)
    (out / "04_scaling").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.input)
    print(f"rows={len(df)} columns={df.shape[1]}")

    rank, best = ranking(df)
    rank.to_csv(out / "04_scaling/stage4a_scaling_ranking.csv", index=False)
    summary = distributions_figure(df, out / "04_scaling/stage4a_scaling_distributions.png", best)
    summary.to_csv(out / "04_scaling/stage4a_scaling_summary.csv", index=False)
    spread_figure(summary, out / "04_scaling/stage4a_scaling_spread.png", best)
    print("\n=== WHICH SCALER GIVES THE MOST EQUAL SPREAD? (1.0 = perfectly equal, lower is better) ===")
    print(rank.to_string(index=False))
    print("BEST:", best)
    print("\n=== SPREAD (std) BEFORE AND AFTER SCALING ===")
    print(summary.pivot(index="column", columns="version", values="std")[
        ["Original", "StandardScaler", "MinMax (0 to 1)", "RobustScaler"]].to_string())
    print("\nFigures and tables saved in", out)


if __name__ == "__main__":
    main()
