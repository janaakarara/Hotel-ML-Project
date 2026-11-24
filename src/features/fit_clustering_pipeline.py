"""Fit the clustering preprocessor on the training data, save it, and prove it matches the stage files.

Usage (from the repo root):
  python -m src.features.fit_clustering_pipeline
Reads:  data/processed/h2_clustering_stage2.csv   (the 11 source columns, before any transform)
Checks: data/processed/h2_clustering_stage5.csv   (the encoded file made by the stage scripts)
Writes: artifacts/clustering_preprocessor.joblib        (the fitted object; ignored by git)
        artifacts/clustering_preprocessor_params.json   (readable copy: lambdas, mean/std, categories)
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features.clustering_pipeline import ClusteringPreprocessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage2.csv")
    ap.add_argument("--check", default="data/processed/h2_clustering_stage5.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--cat-weight", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input)
    prep = ClusteringPreprocessor(cat_weight=a.cat_weight).fit(df)
    Z = prep.transform(df)
    joblib.dump(prep, out / "clustering_preprocessor.joblib")
    (out / "clustering_preprocessor_params.json").write_text(json.dumps(prep.to_dict(), indent=2), encoding="utf-8")
    print(f"fitted on {len(df)} rows -> {Z.shape[1]} features; saved in {out}/")
    print("Yeo-Johnson lambdas:", {k: round(v, 4) for k, v in prep.lambdas_.items()})

    # reload and compare, to be sure the saved file works
    again = joblib.load(out / "clustering_preprocessor.joblib").transform(df.head(100))
    assert np.allclose(again.values, Z.head(100).values), "reloaded object gives different numbers"
    print("reload check: OK")

    if a.cat_weight == 1.0 and Path(a.check).exists():
        ref = pd.read_csv(a.check)
        assert list(Z.columns) == list(ref.columns), "column names or order differ from the stage file"
        diff = float(np.abs(Z.values - ref.values).max())
        print(f"stage-file check: max difference = {diff:.2e}")
        assert diff < 1e-5, "pipeline output differs from the stage files"
        print("stage-file check: OK (the pipeline reproduces the stage scripts)")


if __name__ == "__main__":
    main()
