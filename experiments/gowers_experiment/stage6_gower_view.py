"""Stage 6: the Gower view (no models). Prepares the data for distance-based algorithms that use a
precomputed distance matrix (Agglomerative, DBSCAN) without one-hot encoding.

What it does
  1. Draws ONE fixed random sample of rows (seeded) and saves the row indices.
  2. Numeric columns: clip at the 1st / 99th percentile, then min-max to 0..1. The percentiles are
     learned on ALL rows (unsupervised, no target) and saved, so new bookings can be transformed the same way.
  3. Month: circular distance (December is next to January).
  4. Categories: stay as text; distance is 0 if equal, 1 if different.
  5. Gower distance = weighted mean of the per-variable distances (11 variables).
  6. Model-free checks: symmetric, zero diagonal, range 0..1, each variable's share of the distance,
     and how many pairs are exact copies (distance 0).

Usage (from the repo root):
  python gowers_experiment/stage6_gower_view.py
Reads:  data/processed/h2_clustering_stage3.csv   (grouped + skew-fixed, categories as text)
Writes: gowers_experiment/data/gower_sample_idx.csv         (row positions of the sample in the stage 3 file)
        gowers_experiment/data/gower_view_sample.csv        (sample: numerics scaled 0..1, month, categories as text)
        gowers_experiment/data/stage6_gower_params.json     (clip percentiles per numeric column, sample size, seed)
        gowers_experiment/tables/stage6_gower_checks.csv      (per-variable share of the mean distance)
        gowers_experiment/figures/stage6_gower_checks.png
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM = ["Adults", "LeadTime_yj", "StaysInWeekNights_yj", "StaysInWeekendNights_sqrt", "HasChildren"]
CATS = ["MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType"]


def scale01(v, lo, hi):
    """Clip to [lo, hi] (the learned 1st/99th percentiles) and min-max to 0..1."""
    return (np.clip(np.asarray(v, float), lo, hi) - lo) / ((hi - lo) or 1.0)


def pair_dist_num(v):
    return np.abs(v[:, None] - v[None, :]).astype(np.float32)


def pair_dist_cat(codes):
    return (codes[:, None] != codes[None, :]).astype(np.float32)


def pair_dist_month(m):
    d = np.abs(m[:, None] - m[None, :])
    return (np.minimum(d, 12 - d) / 6).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage3.csv")
    ap.add_argument("--out", default="gowers_experiment/data")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cat-weight", type=float, default=1.0,
                    help="weight of each categorical variable in the distance (1.0 = equal vote)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path("gowers_experiment/tables").mkdir(parents=True, exist_ok=True)
    Path("gowers_experiment/figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input)
    n_all = len(df)

    # 1) fixed sample
    idx = np.sort(np.random.RandomState(a.seed).choice(n_all, a.sample, replace=False))
    pd.Series(idx, name="row_in_stage3").to_csv(out / "gower_sample_idx.csv", index=False)
    S = df.iloc[idx].reset_index(drop=True)

    # 2) clip percentiles learned on ALL rows, applied to the sample
    params = {"sample_size": a.sample, "seed": a.seed, "cat_weight": a.cat_weight, "numeric": {}}
    view = pd.DataFrame(index=S.index)
    for c in NUM:
        lo, hi = float(df[c].quantile(0.01)), float(df[c].quantile(0.99))
        params["numeric"][c] = {"p01": lo, "p99": hi}
        view[c] = scale01(S[c].values, lo, hi)
    view["ArrivalDateMonth"] = S["ArrivalDateMonth"].values       # kept as 1..12, circular distance
    for c in CATS:
        view[c] = S[c].values
    view.to_csv(out / "gower_view_sample.csv", index=False)
    (out / "stage6_gower_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    # 3-5) Gower distance + checks (one variable at a time, so memory stays low)
    n = len(S)
    parts = {}
    for c in NUM:
        parts[c] = pair_dist_num(view[c].values)
    parts["ArrivalDateMonth (circular)"] = pair_dist_month(view["ArrivalDateMonth"].values.astype(float))
    for c in CATS:
        parts[c] = pair_dist_cat(pd.factorize(view[c])[0])
    weights = {k: (a.cat_weight if k in CATS else 1.0) for k in parts}
    total_w = sum(weights.values())
    D = np.zeros((n, n), np.float32)
    mean_part = {}
    for k, M in parts.items():
        D += weights[k] * M
        mean_part[k] = float(M.mean()) * weights[k]
    D /= total_w
    del parts

    sym = float(np.abs(D - D.T).max())
    diag = float(np.abs(np.diag(D)).max())
    zero_pairs = float(((D == 0).sum() - n) / (n * (n - 1)))
    rows_with_copy = float(((D == 0).sum(axis=1) > 1).mean())
    print(f"sample {n} of {n_all} rows (seed {a.seed}), category weight {a.cat_weight}")
    print(f"symmetric: max |D - D.T| = {sym:.1e} | zero diagonal: {diag:.1e} | range {D.min():.2f} to {D.max():.2f}")
    print(f"mean distance {D.mean():.3f} | pairs at distance exactly 0: {100 * zero_pairs:.2f}% | "
          f"rows that have an identical twin in the sample: {100 * rows_with_copy:.1f}%")
    assert sym < 1e-6 and diag < 1e-6 and D.min() >= 0 and D.max() <= 1 + 1e-6

    tot = sum(mean_part.values())
    share = pd.DataFrame({"variable": list(mean_part), "share_of_mean_distance_pct":
                          [100 * v / tot for v in mean_part.values()]}).sort_values("share_of_mean_distance_pct", ascending=False)
    fair = 100 / len(mean_part)
    share["vs_fair_share"] = (share.share_of_mean_distance_pct / fair).round(2)
    share.round(2).to_csv("gowers_experiment/tables/stage6_gower_checks.csv", index=False)
    print(f"\nShare of the mean distance by variable (fair share {fair:.1f}%):")
    print(share.round(2).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    ax[0].barh(share.variable[::-1], share.share_of_mean_distance_pct[::-1], color="tab:blue")
    ax[0].axvline(fair, ls="--", c="k")
    ax[0].set_title(f"Share of mean Gower distance by variable (dashed = fair {fair:.1f}%)")
    ax[0].set_xlabel("% of mean distance")
    tri = D[np.triu_indices(n, 1)]
    ax[1].hist(tri[np.random.RandomState(0).choice(len(tri), 500000, replace=False)], bins=60, color="tab:blue")
    ax[1].set_title(f"Pairwise Gower distances (500k random pairs), {100 * zero_pairs:.1f}% are exactly 0")
    ax[1].set_xlabel("Gower distance")
    fig.tight_layout()
    fig.savefig("gowers_experiment/figures/stage6_gower_checks.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:", out / "gower_sample_idx.csv", "|", out / "gower_view_sample.csv", "|",
          out / "stage6_gower_params.json", "| gowers_experiment/tables/stage6_gower_checks.csv | gowers_experiment/figures/stage6_gower_checks.png")


if __name__ == "__main__":
    main()

