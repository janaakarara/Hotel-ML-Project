"""Stage 5: encoding. Month as sin/cos, categorical columns as one-hot (default) or frequency.

--encoding onehot (default, unchanged behaviour)
    MarketSegment, Meal, Country, CustomerType, ReservedRoomType -> one-hot (0/1), all levels kept.
    Optional --cat-weight multiplies the one-hot columns (1.0 = no change).
    Writes data/processed/h2_clustering_stage5.csv
--encoding frequency
    Each categorical column becomes ONE column: how common its category is (share of rows), then z-scored
    like the numeric columns. Winner of experiments/comparing_experiments for Agglomerative (and a tie for DBSCAN).
    Writes data/processed/h2_clustering_stage5_frequency.csv   (the one-hot file is left untouched)

ArrivalDateMonth (1-12) -> ArrivalMonth_sin, ArrivalMonth_cos in both modes.
Numeric columns and HasChildren (already standardized in stage 4b) are left as they are.

Usage (from the repo root):
  python src/03_encoding/stage5_encode.py                       # one-hot, as before
  python src/03_encoding/stage5_encode.py --encoding frequency  # frequency
Reads:  data/processed/h2_clustering_stage4.csv
Writes: see above, plus data/processed/stage5_encoding_columns.json (one-hot) or
        data/processed/stage5_frequency_maps.json (frequency: category -> share, mean, std)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORICAL = ["MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage4.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--encoding", choices=["onehot", "frequency"], default="onehot")
    ap.add_argument("--cat-weight", type=float, default=0.5, help="one-hot only (0.5 = the weight used for the current stage5.csv)")
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

    if a.encoding == "onehot":
        dummies = pd.get_dummies(df[CATEGORICAL], dtype=int) * a.cat_weight
        mapping = {c: [d for d in dummies.columns if d.startswith(c + "_")] for c in CATEGORICAL}
        df = pd.concat([df.drop(columns=CATEGORICAL), dummies], axis=1)
        target = out / "h2_clustering_stage5.csv"
        (out / "stage5_encoding_columns.json").write_text(json.dumps(
            {"one_hot": mapping, "month": ["ArrivalMonth_sin", "ArrivalMonth_cos"], "cat_weight": a.cat_weight}, indent=2))
        for c, cols in mapping.items():
            print(f"  {c}: {len(cols)} columns")
    else:
        maps = {}
        for c in CATEGORICAL:
            share = df[c].map(df[c].value_counts(normalize=True))
            maps[c] = {"share": df[c].value_counts(normalize=True).to_dict(),
                       "mean": float(share.mean()), "std": float(share.std())}
            df[c] = (share - share.mean()) / share.std()
        target = out / "h2_clustering_stage5_frequency.csv"
        (out / "stage5_frequency_maps.json").write_text(json.dumps(maps, indent=2))

    assert len(df) == n_rows and not df.isna().any().any()
    df.to_csv(target, index=False)
    print(f"encoding={a.encoding} rows={len(df)} columns={df.shape[1]}")
    print("Saved", target)


if __name__ == "__main__":
    main()

