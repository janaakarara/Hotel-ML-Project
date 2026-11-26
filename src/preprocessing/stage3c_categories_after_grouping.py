"""Stage 3c: draw the categorical columns AFTER grouping (the 'after' version of the stage 3a figure).

Only MEASURES and DRAWS. Nothing is changed.

Usage (from the repo root):
  python src/preprocessing/stage3c_categories_after_grouping.py
Reads:  data/processed/h2_clustering_stage3.csv
        data/processed/stage3b_country_group_profile.csv   (written by stage 3b)
Writes: outputs/figures/stage3c_categories_after_grouping.png   (share of every category, all 5 columns)
        outputs/figures/stage3c_country_group_profile.png       (do the 4 Country groups differ?)
        outputs/tables/stage3c_category_shares.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORICAL = ["Country", "MarketSegment", "Meal", "CustomerType", "ReservedRoomType"]
RARE = 0.01
COUNTRY_ORDER = ["Portugal", "Latin Europe", "Rest of Europe", "Outside Europe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage3.csv")
    ap.add_argument("--profile", default="data/processed/stage3b_country_group_profile.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.input)

    # ---- figure 1: every category column after grouping ----
    cats = [c for c in CATEGORICAL if c in df.columns]   # skip columns dropped in stage 2b
    ncols = 2 if len(cats) <= 3 else 3                   # one spare panel holds the note
    fig, axes = plt.subplots(2, ncols, figsize=(20 if ncols == 3 else 14, 11))
    axes = axes.ravel()
    rows = []
    for ax, c in zip(axes, cats):
        vc = df[c].value_counts(normalize=True) * 100
        if c == "Country":
            vc = vc.reindex(COUNTRY_ORDER)
        bars = ax.bar(vc.index.astype(str), vc.values,
                      color=np.where(vc.values < RARE * 100, "tab:red", "tab:blue"))
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in vc.values], fontsize=12)
        ax.axhline(RARE * 100, ls="--", c="k", lw=1)
        ax.set_ylim(0, vc.max() * 1.15)
        ax.set_title(f"{c}: {len(vc)} categories, smallest {vc.min():.1f}%", fontsize=15)
        ax.set_ylabel("% of rows", fontsize=12)
        ax.tick_params(axis="x", rotation=25, labelsize=12)
        for k, v in vc.items():
            rows.append({"column": c, "category": k, "pct_of_rows": round(v, 2)})
    axes[-1].axis("off")
    axes[-1].text(0.0, 0.6, "Dashed line = 1% rare threshold.\nRed bar would be a rare category.\n"
                  "Country went from 167 raw codes to 4 groups,\nall above 13%.",
                  fontsize=14, va="center")
    fig.suptitle("Categorical columns after grouping (clustering input, 52,817 rows)", fontsize=20)
    fig.tight_layout()
    fig.savefig(out / "figures/stage3c_categories_after_grouping.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "tables/stage3c_category_shares.csv", index=False)

    # ---- figure 2: do the Country groups differ? ----
    prof = pd.read_csv(a.profile, index_col=0).reindex(COUNTRY_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title, scale in [(axes[0], "lead", "Median lead time (days)", 1),
                                  (axes[1], "week", "Median week nights", 1),
                                  (axes[2], "kids", "Share of bookings with children (%)", 100)]:
        v = prof[col].values * scale
        bars = ax.bar(prof.index, v, color="tab:blue")
        ax.bar_label(bars, labels=[f"{x:.0f}" if col != "kids" else f"{x:.1f}" for x in v], fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, max(v.max() * 1.2, 1))
        ax.tick_params(axis="x", rotation=25, labelsize=12)
    fig.suptitle("Country groups: lead time separates them, stay length and children barely do", fontsize=16)
    fig.tight_layout()
    fig.savefig(out / "figures/stage3c_country_group_profile.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    print("\n=== CATEGORY SHARES AFTER GROUPING (% of rows) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSaved figures in", out / "figures")


if __name__ == "__main__":
    main()
