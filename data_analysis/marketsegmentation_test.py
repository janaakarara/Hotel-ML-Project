import pandas as pd
import numpy as np

df = pd.read_csv("Hotel_Data.csv")

# 1. Cancellation rate per market segment
summary = (
    df.groupby("MarketSegment")["IsCanceled"]
    .agg(["mean", "count"])
    .sort_values("mean")
)
summary["share_of_bookings"] = summary["count"] / summary["count"].sum()
print("Cancellation rate by market segment:")
print(summary)
summary.to_csv("cancellation_by_segment.csv")

# 2. Chi-square statistic, computed manually
table = pd.crosstab(df["MarketSegment"], df["IsCanceled"])
observed = table.values
n = observed.sum()
row_totals = observed.sum(axis=1, keepdims=True)
col_totals = observed.sum(axis=0, keepdims=True)
expected = row_totals @ col_totals / n

chi2 = ((observed - expected) ** 2 / expected).sum()
dof = (table.shape[0] - 1) * (table.shape[1] - 1)
print(f"\nChi-square = {chi2:.2f}, dof = {dof}")

# 3. Share of cells with expected count < 5 (chi-square is unreliable if high)
small = (expected < 5).mean()
print(f"Share of cells with expected count < 5: {small:.1%}")

# 4. Cramér's V: effect size (0 = none, 1 = perfect)
k = min(table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * k))
print(f"Cramér's V = {cramers_v:.3f}")