"""Stage 5: encoding. One-hot for the categorical columns, sin/cos for the month.

- MarketSegment, Country, CustomerType -> one-hot (0/1), all levels kept
  (distance-based clustering needs no dropped level).
- ArrivalDateMonth (1-12) -> ArrivalMonth_sin, ArrivalMonth_cos, so December sits next to January.
- Numeric columns and HasChildren (all already standardized in stage 4b) are left as they are.
- Optional --cat-weight multiplies the one-hot columns (default 1.0 = no change). Values below 1
  (for example 0.7071) stop the many one-hot columns from dominating the distances.

Usage (from the repo root):
  python src/preprocessing/stage5_encode.py
Reads:  data/processed/h2_clustering_stage4.csv
Writes: data/processed/h2_clustering_stage5.csv
        data/processed/stage5_encoding_columns.json   (which columns came from which source)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORICAL = ["MarketSegment", "Country", "CustomerType"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage4.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--cat-weight", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input)
    n_rows = len(df)

    # month -> sin/cos
    angle = 2 * np.pi * (df["ArrivalDateMonth"] - 1) / 12
    df["ArrivalMonth_sin"] = np.sin(angle)
    df["ArrivalMonth_cos"] = np.cos(angle)
    df = df.drop(columns="ArrivalDateMonth")

    # one-hot
    dummies = pd.get_dummies(df[CATEGORICAL], dtype=int) * a.cat_weight
    mapping = {c: [d for d in dummies.columns if d.startswith(c + "_")] for c in CATEGORICAL}
    df = pd.concat([df.drop(columns=CATEGORICAL), dummies], axis=1)

    assert len(df) == n_rows and not df.isna().any().any()
    df.to_csv(out / "h2_clustering_stage5.csv", index=False)
    (out / "stage5_encoding_columns.json").write_text(json.dumps(
        {"one_hot": mapping, "month": ["ArrivalMonth_sin", "ArrivalMonth_cos"], "cat_weight": a.cat_weight}, indent=2))

    print(f"rows={len(df)} columns={df.shape[1]}")
    for c, cols in mapping.items():
        print(f"  {c}: {len(cols)} columns")
    print("Saved", out / "h2_clustering_stage5.csv")


if __name__ == "__main__":
    main()
