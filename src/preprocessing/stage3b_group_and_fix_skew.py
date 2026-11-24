"""Stage 5: group rare categories (incl. Country hybrid) and fix skew in the numeric columns.

Applies the decisions made from the stage 3a figures. Writes a readable explanation file.

Usage (from the repo root):
  python src/preprocessing/stage3b_group_and_fix_skew.py
Reads:  data/processed/h2_clustering_stage2.csv
Writes: data/processed/h2_clustering_stage3.csv        (same rows, same order)
        data/processed/stage3_decisions.md             (why each thing was done)
        data/processed/stage3_transform_params.json    (fitted Yeo-Johnson lambdas)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson

TOP_COUNTRIES = ["PRT", "FRA", "GBR", "DEU", "ESP"]

EUROPE = [
    "ALB", "AND", "AUT", "BEL", "BGR", "BIH", "BLR", "CHE", "CZE", "DEU", "DNK", "ESP", "EST",
    "FIN", "FRA", "FRO", "GBR", "GGY", "GIB", "GRC", "HRV", "HUN", "IMN", "IRL", "ISL", "ITA",
    "JEY", "LIE", "LTU", "LUX", "LVA", "MCO", "MKD", "MLT", "MNE", "NLD", "NOR", "POL", "PRT",
    "ROU", "RUS", "SRB", "SVK", "SVN", "SWE", "UKR",
]
ASIA = [
    "ARE", "ARM", "AZE", "BGD", "BHR", "CHN", "CN", "CYP", "GEO", "HKG", "IDN", "IND", "IRN", "IRQ",
    "ISR", "JOR", "JPN", "KAZ", "KHM", "KOR", "KWT", "LAO", "LBN", "LKA", "MAC", "MDV", "MMR",
    "MYS", "OMN", "PAK", "PHL", "QAT", "SAU", "SGP", "SYR", "THA", "TJK", "TUR", "TWN", "UZB", "VNM",
]
SOUTH_AMERICA = [
    "ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PER", "PRY", "SUR", "URY", "VEN",
]
NORTH_AMERICA = [
    "ABW", "AIA", "BRB", "CRI", "CUB", "DMA", "DOM", "GLP", "GTM", "HND", "JAM", "KNA", "LCA",
    "MEX", "NIC", "PAN", "PRI", "SLV", "USA", "VGB",
]
AFRICA = [
    "AGO", "BEN", "BFA", "CAF", "CIV", "CMR", "COM", "CPV", "DZA", "EGY", "ETH", "GAB", "GHA",
    "GNB", "KEN", "LBY", "MAR", "MLI", "MOZ", "MRT", "MUS", "MYT", "NAM", "NGA", "RWA", "SDN",
    "SEN", "SLE", "STP", "SYC", "TGO", "TUN", "TZA", "UGA", "ZAF", "ZMB", "ZWE",
]
OCEANIA = [
    "ASM", "AUS", "KIR", "NCL", "NZL", "PYF",
]

CONTINENTS = {"Rest of Europe": EUROPE, "Asia": ASIA, "South America": SOUTH_AMERICA,
              "North America": NORTH_AMERICA, "Africa": AFRICA}
CODE_TO_GROUP = {code: name for name, codes in CONTINENTS.items() for code in codes}
# anything not listed (Oceania, Unknown, TMP, ATA ...) becomes "Other"

# column: ({old value: new value}, why)
MERGES = {
    "Meal": ({"FB": "HB/FB", "HB": "HB/FB"},
             "FB is 0.02% of rows. Both FB and HB are meal plans beyond breakfast, so they are merged."),
    "CustomerType": ({"Group": "Transient-Party/Group", "Transient-Party": "Transient-Party/Group"},
                     "Group is 0.5% of rows. Both are bookings tied to other bookings, so they are merged."),
    "MarketSegment": ({"Complementary": "Other", "Aviation": "Other"},
                      "Both are under 1% of rows and are special cases with no close relative."),
    "ReservedRoomType": ({"G": "Other", "C": "Other"},
                         "Both are under 1% of rows. Room codes are anonymous, so there is no meaning to merge by."),
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
    md = ["# Stage 5: grouping and skew fixes", "",
          "Applies the decisions made from the stage 3a figures. No row is removed and the row order is unchanged.", "",
          f"- **Input:** `{a.input}` ({n} rows, {df.shape[1]} columns)",
          "- **Output:** `h2_clustering_stage3.csv`", ""]

    # ---------- 1) Country: top 5 countries + continents ----------
    country_before = df["Country"].nunique()
    out["Country"] = df["Country"].where(df["Country"].isin(TOP_COUNTRIES),
                                         df["Country"].map(CODE_TO_GROUP).fillna("Other"))
    sh = share_table(out["Country"])
    md += ["## 1. Country: top 5 countries plus continents", "",
           f"Country had **{country_before} values**, which would create {country_before} encoded columns and "
           f"swamp every other feature. It now has **{out['Country'].nunique()} groups**.", "",
           "**Why not pure continents?** About 86% of bookings are European, so one huge Europe group would hide big "
           "differences. For example the median booking is made 26 days ahead for Portugal but 83 days ahead for "
           "Germany. So the 5 biggest countries stay separate and only the rest are grouped by continent.", ""]
    md += md_table(["Group", "% of rows"], [(g, f"{v:.1f}") for g, v in sh.items()])
    md += ["", "Notes: `CN` is China in this dataset and is mapped to Asia by hand. Oceania and junk codes "
           "(`Unknown`, `TMP`, `ATA` ...) fall into `Other`. Russia counts as Europe and Turkey and Cyprus as Asia.", ""]

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
    md += ["", "The merges of Meal and CustomerType are judgment calls. If you prefer, change them to `Other` in `MERGES`.", ""]

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

    print("\n========== STAGE 5 COMPLETED ==========")
    print(f"rows {n}, columns {df.shape[1]} -> {out.shape[1]}")
    print("columns:", ", ".join(out.columns))
    print("\nCountry groups (% of rows):")
    print(sh.round(1).to_string())
    print("\nSaved in", out_dir)
    print("=======================================")


if __name__ == "__main__":
    main()
