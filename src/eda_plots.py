
"""Phase 1 (part 2): plots that show what needs deciding in cleaning and feature choice. Read-only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/H2.csv")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, dtype=str)
df = df.apply(lambda s: s.str.strip()).replace({"NULL": np.nan, "": np.nan})
for c in ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["TotalNights"] = df.StaysInWeekendNights + df.StaysInWeekNights

# 1. Skew: raw versus log1p for the long-tailed numeric columns
cols = ["LeadTime", "TotalNights", "StaysInWeekNights"]
fig, ax = plt.subplots(2, 3, figsize=(11, 5.5))
for j, c in enumerate(cols):
    ax[0, j].hist(df[c].dropna(), bins=50, color="#2F5DA8")
    ax[0, j].set_title(f"{c} (raw), skew {df[c].skew():.2f}")
    ax[1, j].hist(np.log1p(df[c].dropna()), bins=50, color="#0E8A8A")
    ax[1, j].set_title(f"log1p({c}), skew {np.log1p(df[c]).skew():.2f}")
plt.tight_layout(); plt.savefig(FIG / "p1_skew_raw_vs_log.png", dpi=140); plt.close()

# 2. How dominant is the most common value in each column? (near-constant columns separate nothing)
share = {c: df[c].value_counts(normalize=True, dropna=True).iloc[0] for c in df.columns if c != "ADR"}
s = pd.Series(share).sort_values()
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(s.index, s.values, color=["#C8433B" if v > 0.9 else "#2F5DA8" for v in s.values])
ax.axvline(0.9, color="#C8433B", ls="--", lw=1)
ax.set_title("Share of rows holding the most common value (red: over 90%)")
plt.tight_layout(); plt.savefig(FIG / "p1_dominant_value_share.png", dpi=140); plt.close()

# 3. The categorical columns we may cluster on
cats = ["MarketSegment", "CustomerType", "DistributionChannel", "Meal"]
fig, ax = plt.subplots(2, 2, figsize=(11, 6))
for a, c in zip(ax.ravel(), cats):
    df[c].value_counts().plot.bar(ax=a, color="#6B4FB3"); a.set_title(c); a.set_xlabel("")
    a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(FIG / "p1_categories.png", dpi=140); plt.close()

# 4. Duplicates: how many distinct rows are there really?
full = len(df); dedup = len(df.drop_duplicates())
cand = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Adults", "IsRepeatedGuest", "MarketSegment", "CustomerType"]
distinct = len(df[cand].drop_duplicates())
print(f"rows: {full}; after dropping exact duplicates: {dedup}; distinct rows on the 7 candidate columns only: {distinct}")
print("figures saved in outputs/figures")
