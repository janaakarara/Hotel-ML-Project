"""Stage 3a: look at skew (numeric columns) and rare categories (category columns).

Only MEASURES and DRAWS. No column is changed. You decide what to do by reading the figures.

Usage (from the repo root):
  python src/02_preprocessing/stage3a_distributions_check.py
Reads:  data/processed/h2_clustering_stage2.csv
Writes: outputs/figures/stage3a_numeric_skew.png        (big: every column x every method, best option in red)
        outputs/figures/stage3a_categorical_shares.png
        outputs/tables/stage3a_numeric_skew.csv
        outputs/tables/stage3a_rare_categories.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson

NUMERIC = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Children"]
CATEGORICAL = ["Country", "MarketSegment", "Meal", "CustomerType", "ReservedRoomType"]
RARE = 0.01          # a category under 1% of rows counts as rare


def transforms(x):
    """Every option we want to compare. Returns {name: transformed values}."""
    x = x.astype(float).values
    return {
        "Original": x,
        "log(1+x)": np.log1p(x),
        "Square root": np.sqrt(x),
        "Yeo-Johnson": yeojohnson(x)[0],
        "Cap at 99th pct": np.minimum(x, np.percentile(x, 99)),
    }


def skew_of(v):
    return float(pd.Series(v).skew())


def skew_label(s):
    """Plain-words reading of a skew value."""
    a = abs(s)
    return "balanced" if a < 0.5 else ("moderate" if a <= 1 else "HIGH")


def numeric_figure(df, path):
    n_methods = 5
    fig, axes = plt.subplots(len(NUMERIC), n_methods, figsize=(7 * n_methods, 5.5 * len(NUMERIC)))
    rows = []
    for i, c in enumerate(NUMERIC):
        t = transforms(df[c])
        skews = {m: skew_of(v) for m, v in t.items()}
        # if the original is already balanced (|skew| < 0.5) no transform is needed
        best = "Original" if abs(skews["Original"]) < 0.5 else min(skews, key=lambda m: abs(skews[m]))
        rows.append({"column": c, **{f"skew_{m}": round(s, 2) for m, s in skews.items()},
                     "best_option": best, "best_skew_label": skew_label(skews[best]),
                     "share_zero": round((df[c] == 0).mean(), 3)})
        for j, (m, v) in enumerate(t.items()):
            ax = axes[i, j]
            color = "tab:red" if m == best else "tab:blue"
            ax.hist(v, bins=int(min(40, max(5, df[c].nunique()))), color=color, edgecolor="white", linewidth=0.3)
            tag = ("  <- BEST" if m == best else "")
            ax.set_title(f"{c} | {m}{tag}\nskew = {skews[m]:.2f} ({skew_label(skews[m])})", fontsize=17,
                         fontweight="bold" if m == best else "normal", color="tab:red" if m == best else "black")
            ax.tick_params(labelsize=13)
            if j == 0:
                ax.set_ylabel("rows", fontsize=14)
    fig.suptitle("Skew check: every column (rows) x every method (columns).  "
                 "Red = best option.  Skew: under 0.5 balanced, 0.5 to 1 moderate, over 1 HIGH", fontsize=24, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def categorical_figure(df, path, top_country=25):
    cats = [c for c in CATEGORICAL if c in df.columns]   # skip columns dropped in stage 2b
    fig, axes = plt.subplots(len(cats), 1, figsize=(20, 6 * len(cats)), squeeze=False)
    axes = axes.ravel()
    rows = []
    for ax, c in zip(axes, cats):
        vc = df[c].value_counts(normalize=True)
        rare = vc[vc < RARE]
        rows.append({"column": c, "n_categories": len(vc), "n_rare_under_1pct": len(rare),
                     "rows_covered_by_rare": round(rare.sum(), 4),
                     "rare_values": ", ".join(rare.index.astype(str)[:40])})
        show = vc.head(top_country)
        ax.bar(show.index.astype(str), show.values * 100,
               color=np.where(show.values < RARE, "tab:red", "tab:blue"))
        ax.axhline(RARE * 100, ls="--", c="k")
        extra = len(vc) - len(show)
        note = f" (showing top {len(show)}, {extra} more not drawn)" if extra > 0 else ""
        ax.set_title(f"{c}: {len(vc)} categories, {len(rare)} under 1% (red){note}", fontsize=19)
        ax.set_ylabel("% of rows", fontsize=15)
        ax.tick_params(axis="x", rotation=60, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
    fig.suptitle("Rare-category check: dashed line = 1%, red bars = candidates for 'Other'", fontsize=24, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage2.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.input)
    print(f"rows={len(df)} columns={df.shape[1]}")

    sk = numeric_figure(df, out / "figures/stage3a_numeric_skew.png")
    sk.to_csv(out / "tables/stage3a_numeric_skew.csv", index=False)
    print("\n=== SKEW OF EVERY METHOD (under 0.5 balanced, over 1 HIGH) ===")
    print(sk.to_string(index=False))

    cat = categorical_figure(df, out / "figures/stage3a_categorical_shares.png")
    cat.to_csv(out / "tables/stage3a_rare_categories.csv", index=False)
    print("\n=== RARE CATEGORIES (under 1% of rows) ===")
    print(cat.drop(columns=["rare_values"]).to_string(index=False))

    print("\nFigures and tables saved in", out)


if __name__ == "__main__":
    main()

