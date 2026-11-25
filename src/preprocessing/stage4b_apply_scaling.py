"""Stage 4b: apply StandardScaler to the 4 numeric columns (the winner of stage 4a).

HasChildren (0/1) and ArrivalDateMonth (encoded later as sin/cos) are NOT scaled.
Clustering is unsupervised, so fitting on all rows leaks nothing.

Usage (from the repo root):
  python src/preprocessing/stage4b_apply_scaling.py
Reads:  data/processed/h2_clustering_stage3.csv
Writes: data/processed/h2_clustering_stage4.csv
        data/processed/stage4_scaler_params.json   (mean and std, to repeat the scaling on new bookings)
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC = ["Adults", "LeadTime_yj", "StaysInWeekNights_yj", "StaysInWeekendNights_sqrt", "HasChildren"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage3.csv")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input)
    sc = StandardScaler().fit(df[NUMERIC])
    df[NUMERIC] = sc.transform(df[NUMERIC])  # same column names, values now scaled

    params = {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(NUMERIC, sc.mean_, sc.scale_)}
    (out / "stage4_scaler_params.json").write_text(json.dumps(params, indent=2))
    df.to_csv(out / "h2_clustering_stage4.csv", index=False)

    print(f"rows={len(df)} columns={df.shape[1]}")
    print(df[NUMERIC].agg(["mean", "std"]).round(3).to_string())
    print("Saved", out / "h2_clustering_stage4.csv")


if __name__ == "__main__":
    main()
