"""Stage 3b: group rare categories (Country -> 4 groups) and fix skew in the numeric columns.

Applies the decisions made from the stage 3a figures. Writes a readable explanation file.

Usage (from the repo root):
  python src/preprocessing/stage3b_group_and_fix_skew.py
Reads:  data/processed/h2_clustering_stage2.csv
Writes: data/processed/h2_clustering_stage3.csv        (same rows, same order)
        data/processed/stage3_decisions.md             (why each thing was done)
        data/processed/stage3_transform_params.json    (fitted Yeo-Johnson lambdas)
        data/processed/stage3b_country_group_profile.csv (lead/stay/children by Country group)
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `src.` imports work
from src.features.country_groups import country_group, GROUP_ORDER  # noqa: E402

# column: ({old value: new value}, why)
MERGES = {
    "CustomerType": ({"Group": "Transient-Party/Group", "Transient-Party": "Transient-Party/Group"},
                     "Group is 0.5% of rows. Both are bookings tied to other bookings, so they are merged."),
    "MarketSegment": ({"Complementary": "Other", "Aviation": "Other"},
                      "Both are under 1% of rows and are special cases with no close relative."),
}

SKEW_FIXES = {
    "LeadTime": ("Yeo-Johnson", "LeadTime_yj"),
    "StaysInWeekNights": ("Yeo-Johnson", "StaysInWeekNights_yj"),
    "StaysInWeekendNights": ("Square root", "StaysInWeekendNights_sqrt"),
}


def share_table(series):
    vc = series.value_counts(normalize=True) * 100
    return vc


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage2.csv")
    ap.add_argument("--out-dir", default="data/processed")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input)
    out = df.copy()
    n = len(df)
    md = ["# Stage 3b: grouping and skew fixes", "",
          "Applies the decisions made from the stage 3a figures. No row is removed and the row order is unchanged.", "",
          f"- **Input:** `{a.input}` ({n} rows, {df.shape[1]} columns)",
          "- **Output:** `h2_clustering_stage3.csv`", ""]

    # ---------- 1) Country: 4 balanced groups ----------
    country_before = df["Country"].nunique()
    raw_code = df["Country"].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})
    out["Country"] = raw_code.map(country_group)
    sh = share_table(out["Country"]).reindex(GROUP_ORDER)
    prof = df.assign(G=out["Country"].values, HasKids=(df["Children"] > 0)).groupby("G").agg(
        lead=("LeadTime", "median"), week=("StaysInWeekNights", "median"),
        weekend=("StaysInWeekendNights", "median"), kids=("HasKids", "mean")).reindex(GROUP_ORDER)
    md += ["## 1. Country: 4 groups", "",
           f"Country had **{country_before} values**, which would create {country_before} encoded columns and "
           f"swamp every other feature. It now has **{out['Country'].nunique()} groups**, each at least 14% of rows.", "",
           "| Group | Members |", "|---|---|",
           "| Portugal | PRT (the hotel is in Lisbon: domestic guests) |",
           "| Latin Europe | FRA, ESP, ITA (Portugal's nearest Latin-language neighbours) |",
           "| Rest of Europe | all other European codes (GBR, DEU, BEL, NLD, CHE, IRL, AUT, Nordics, Eastern Europe ...) |",
           "| Outside Europe | Asia, the Americas, Africa, Oceania, and unknown / junk codes (`Unknown`, `TMP`, `ATA` ...) |", "",
           "**Why this grouping?** The earlier version (top 5 countries + continents) gave 11 columns and let Country "
           "outvote the numeric features. Four groups of 14% to 36% remove every rare category and need only 4 columns. "
           "Groups were chosen for balance and interpretability, not strict geography: 'Latin Europe' is only the three "
           "Latin-language neighbours, and 'Rest of Europe' is everyone else. Germany and Britain are merged, which "
           "averages two different booking habits; lead time is already a clustering feature, so most of that "
           "difference is still captured.", ""]
    md += md_table(["Group", "% of rows"], [(g, f"{v:.1f}") for g, v in sh.items()])
    md += ["", "**Do the groups differ?** (medians from the untransformed stage 2 columns)", ""]
    md += md_table(["Group", "Median lead time (days)", "Median week nights", "Median weekend nights", "Share with children"],
                   [(g, f"{r.lead:.0f}", f"{r.week:.0f}", f"{r.weekend:.0f}", f"{100 * r.kids:.1f}%") for g, r in prof.iterrows()])
    md += ["", "Lead time is the clear separator; stay length and the children share differ little. "
           "Note: Russia counts as Europe, Turkey and Cyprus as Asia (Outside Europe), `CN` is China.", ""]
    prof.round(3).to_csv(out_dir / "stage3b_country_group_profile.csv")

    # ---------- 2) other categorical merges ----------
    md += ["## 2. Rare categories in the other columns", "",
           "Rule: a value under 1% of rows is merged. Where a value has an obvious close relative it is merged "
           "with it, otherwise it goes into `Other`.", ""]
    rows = []
    for col, (mapping, why) in MERGES.items():
        before = share_table(df[col])
        out[col] = df[col].replace(mapping)
        after = share_table(out[col])
        rows.append((col, ", ".join(f"{k} -> {v}" for k, v in mapping.items() if k in before.index and
                                    before[k] < 1 or k in before.index and v == "Other"), why,
                     f"{df[col].nunique()} -> {out[col].nunique()}"))
    md += md_table(["Column", "Merge", "Why", "Categories"], rows)
    md += ["", "The CustomerType merge is a judgment call. If you prefer, change them to `Other` in `MERGES`.", ""]

    # ---------- 3) skew ----------
    params = {}
    md += ["## 3. Skew fixes", "",
           "Skew under 0.5 is balanced, 0.5 to 1 moderate, over 1 high. Long tails would stretch the distance "
           "between points and let a few extreme bookings dominate the clusters. For each column we used the "
           "method with the lowest skew in the stage 3a figure.", ""]
    srows = []
    for col, (method, new) in SKEW_FIXES.items():
        x = df[col].astype(float).values
        if method == "Yeo-Johnson":
            y, lam = yeojohnson(x)
            params[new] = {"method": "yeo-johnson", "source": col, "lambda": float(lam)}
        else:
            y = np.sqrt(x)
            params[new] = {"method": "sqrt", "source": col}
        out[new] = y
        out = out.drop(columns=[col])
        srows.append((col, method, new, f"{pd.Series(x).skew():.2f}", f"{pd.Series(y).skew():.2f}"))
    md += md_table(["Column", "Method", "New column", "Skew before", "Skew after"], srows)
    out["HasChildren"] = (df["Children"] > 0).astype(int)
    out = out.drop(columns=["Children"])
    params["HasChildren"] = {"method": "binary", "source": "Children", "rule": "Children > 0"}
    md += ["", f"**Children** became a 0/1 flag (`HasChildren`). {100 * (df['Children'] == 0).mean():.0f}% of values are 0, "
           "so no transform can fix the skew (every method stayed above 2.8 in the stage 3a figure). "
           "The question that matters for segments is whether children are present.", "",
           "**Adults** was checked and left unchanged (skew -0.24, already balanced).", "",
           "**ArrivalDateMonth** is unchanged here. It is a cyclical category and is handled in the encoding stage.", ""]

    # ---------- checks + output ----------
    assert len(out) == n, "row count changed"
    assert out.isna().sum().sum() == 0, "missing values appeared"
    out.to_csv(out_dir / "h2_clustering_stage3.csv", index=False)
    (out_dir / "stage3_transform_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    md += ["## 4. Result", "",
           f"{df.shape[1]} columns in, {out.shape[1]} columns out, {n} rows kept.", "",
           "Final columns: " + ", ".join(f"`{c}`" for c in out.columns), "",
           "The skew fixes learn nothing from any target. Clustering is unsupervised, so fitting the Yeo-Johnson "
           "setting on all rows leaks nothing. The fitted settings are saved in `stage3_transform_params.json` "
           "so the same transform can be repeated on new bookings.", "",
           "The untransformed columns stay in `h2_clustering_stage2.csv` (same row order) for describing the "
           "clusters afterwards.", ""]
    (out_dir / "stage3_decisions.md").write_text("\n".join(md), encoding="utf-8")

    print("\n========== STAGE 3b COMPLETED ==========")
    print(f"rows {n}, columns {df.shape[1]} -> {out.shape[1]}")
    print("columns:", ", ".join(out.columns))
    print("\nCountry groups (% of rows):")
    print(sh.round(1).to_string())
    print("\nSaved in", out_dir)
    print("=======================================")


if __name__ == "__main__":
    main()
