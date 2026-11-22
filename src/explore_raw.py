"""Phase 1: understand the raw H2 data. Read-only: never changes the CSV."""
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/H2.csv")
OUT = Path("outputs/tables")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, dtype=str)
df = df.apply(lambda s: s.str.strip()).replace({"NULL": np.nan, "": np.nan})
NUM = ["IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
       "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies", "IsRepeatedGuest",
       "PreviousCancellations", "PreviousBookingsNotCanceled", "BookingChanges", "DaysInWaitingList",
       "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests"]
for c in NUM:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("rows x columns:", df.shape)
dups = int(df.duplicated().sum())
print(f"exact duplicate rows: {dups} ({dups / len(df):.1%}); rows left after dropping them: {len(df) - dups}")

miss = df.isna().sum()
print("\nmissing values (only columns with any):")
print(miss[miss > 0].to_string())

rows = []
for c in df.columns:
    vc = df[c].value_counts(dropna=True)
    rows.append({"column": c, "kind": "numeric" if c in NUM else "text", "missing": int(df[c].isna().sum()),
                 "n_unique": int(df[c].nunique()), "top_value": vc.index[0], "top_share": round(vc.iloc[0] / len(df), 3)})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "phase1_column_summary.csv", index=False)
print("\ncolumn summary (saved to outputs/tables/phase1_column_summary.csv):")
print(summary.to_string(index=False))

print("\nnumeric ranges:")
d = df[NUM].describe().T[["min", "25%", "50%", "75%", "max"]]
d["skew"] = df[NUM].skew()
print(d.round(2).to_string())

guests = df.Adults.fillna(0) + df.Children.fillna(0) + df.Babies.fillna(0)
nights = df.StaysInWeekendNights + df.StaysInWeekNights
checks = {"zero guests": (guests == 0).sum(), "zero nights": (nights == 0).sum(), "Babies > 2": (df.Babies > 2).sum(),
          "ADR == 0": (df.ADR == 0).sum(), "ADR < 0": (df.ADR < 0).sum(), "ADR > 1000": (df.ADR > 1000).sum(),
          "MarketSegment Undefined": (df.MarketSegment == "Undefined").sum(),
          "DistributionChannel Undefined": (df.DistributionChannel == "Undefined").sum()}
print("\nodd values (raw rows, before any deduplication):")
for k, v in checks.items():
    print(f"  {k}: {int(v)}")
