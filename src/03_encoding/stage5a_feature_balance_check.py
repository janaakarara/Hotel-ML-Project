"""Stage 5a: feature-balance check on the encoded matrix.

A column contributes to squared distances in proportion to its variance, so
  share(group) = sum of Var(column) over the group's columns / sum of Var over all columns
is the fraction of the distance that the group controls. Fair share = 1 / number of groups.
Numeric-type groups far below the fair share are flagged. One-hot groups are shown with the share of
their most common category, because their variance is capped by 1 - sum(p_k^2).

Usage (from the repo root):
  python src/03_encoding/stage5a_feature_balance_check.py
Reads:  data/processed/h2_clustering_stage5.csv
Writes: outputs/05_encoding_balance/stage5a_balance_<tag>.png, outputs/05_encoding_balance/stage5a_balance_<tag>.csv
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUMERIC = ["Adults", "LeadTime_yj", "StaysInWeekNights_yj", "StaysInWeekendNights_sqrt", "HasChildren"]
CATEGORICAL = ["MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType"]
BLUE, GREEN, GREY, RED = "#2F5DA8", "#2E8B57", "#8A8A8A", "#C8433B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage5.csv")
    ap.add_argument("--tag", default="current")
    ap.add_argument("--unscaled-flag", action="store_true")
    a = ap.parse_args()
    fig_dir, tab_dir = Path("outputs/05_encoding_balance"), Path("outputs/05_encoding_balance")
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input)
    if a.unscaled_flag:
        df["HasChildren"] = (df["HasChildren"] > df["HasChildren"].min()).astype(int)
    groups = {c: [c] for c in NUMERIC}
    groups["ArrivalMonth (sin+cos)"] = ["ArrivalMonth_sin", "ArrivalMonth_cos"]
    for c in CATEGORICAL:
        groups[c + " (one-hot)"] = [x for x in df.columns if x.startswith(c + "_")]

    rows = []
    for g, cols in groups.items():
        kind = "one-hot" if g.endswith("(one-hot)") else "numeric"
        top = float(df[cols].mean().max() * 100) if kind == "one-hot" else np.nan
        rows.append((g, kind, len(cols), float(df[cols].var(ddof=0).sum()), top))
    tab = pd.DataFrame(rows, columns=["group", "kind", "n_columns", "variance", "top_category_pct"])
    tab["share_pct"] = tab.variance / tab.variance.sum() * 100
    fair = 100 / len(tab)
    tab["vs_fair_share"] = tab.share_pct / fair
    tab["flag"] = np.where((tab.kind == "numeric") & (tab.share_pct < fair / 2), "LOW", "")
    tab = tab.sort_values("share_pct", ascending=False).reset_index(drop=True)
    tab.round(3).to_csv(tab_dir / f"stage5a_balance_{a.tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(tab))[::-1]
    colors = [GREY if k == "one-hot" else GREEN if g.startswith("ArrivalMonth") else BLUE
              for g, k in zip(tab.group, tab.kind)]
    ax.barh(y, tab.share_pct, color=colors)
    ax.axvline(fair, color="black", ls="--", lw=1.2)
    ax.text(fair + .15, len(tab) - .35, f"fair share = {fair:.1f}%", fontsize=9)
    for yi, r in zip(y, tab.itertuples()):
        note = f"   (top category {r.top_category_pct:.0f}%)" if r.kind == "one-hot" else ""
        low = "   <- low" if r.flag else ""
        ax.text(r.share_pct + .2, yi, f"{r.share_pct:.1f}%{note}{low}", va="center", fontsize=9,
                color=RED if r.flag else "black", fontweight="bold" if r.flag else "normal")
    ax.set_yticks(y); ax.set_yticklabels(tab.group)
    ax.set_xlim(0, tab.share_pct.max() * 1.9)
    ax.set_xlabel("% of the total distance this feature group controls")
    ax.set_title("Feature balance: share of distance per feature group", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"stage5a_balance_{a.tag}.png", bbox_inches="tight", dpi=130); plt.close(fig)

    print(tab[["group", "n_columns", "share_pct", "vs_fair_share", "top_category_pct", "flag"]]
          .round(2).to_string(index=False))
    print(f"\nfair share {fair:.1f}%")
    print("Saved", fig_dir / f"stage5a_balance_{a.tag}.png")


if __name__ == "__main__":
    main()
