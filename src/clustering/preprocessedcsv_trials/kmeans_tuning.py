"""
Task C - K-Means tuning on the preprocessed table (h2_clustering_preprocessed_v0.csv).

    python -m src.clustering.preprocessedcsv_trials.kmeans_tuning
    python -m src.clustering.preprocessedcsv_trials.kmeans_tuning --k 5   # force a k

Every k-specific output is named preprocessed_kmeans_k<k>_...  (so k=5 and k=6 never overwrite each other).

What it does
  1. K SCAN      k = 2..10, each k fitted with 5 different seeds (10 restarts each) and averaged,
                 so the choice does not depend on one lucky seed.
  2. CHOOSE k    inside k = 3..8, drop any k whose smallest cluster is under 3% or whose result
                 changes with the seed (stability below 0.90). Rank the rest by silhouette
                 (higher is better) and Davies-Bouldin (lower is better). Best average rank wins;
                 a tie goes to the smaller k (easier to describe).
  3. SEEDING     for the chosen k, try 2 initialisations (k-means++ and random) x 20 seeds, one
                 start each, and keep the (init, seed) with the lowest inertia.
  4. TRAIN/TEST  fit on 80% of the rows, then assign the unseen 20% to the nearest centre and
                 compare the scores. Similar scores = the clusters are not specific to the training rows.
  5. FINAL       refit on all rows with the best settings, describe the clusters with the raw
                 columns, and save the model.

No outcome column (IsCanceled, ADR ...) is read anywhere. Only distances between bookings are used.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[3]            # project folder (src/clustering/preprocessedcsv_trials/ -> 3 levels up)
PREP_CSV = ROOT / "data" / "processed" / "h2_clustering_preprocessed_v0.csv"
RAW_CSV = ROOT / "data" / "processed" / "h2_clustering_raw_features.csv"   # only to describe clusters
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
ARTIFACTS = ROOT / "artifacts"

SEED = 42                       # fixed seed for the split, the silhouette sample and the plots
K_SCAN = list(range(2, 11))     # k values that are tried
K_MIN, K_MAX = 3, 8             # k = 2 only halves the hotel; more than 8 is too many to describe
MIN_CLUSTER_SHARE = 0.03        # a cluster under 3% of the bookings is too small to act on
MIN_STABILITY = 0.90            # ARI between runs with different seeds must be at least this
SCAN_SEEDS = [0, 1, 2, 3, 4]    # seeds averaged for every k in the scan
SEED_TRIALS = list(range(20))   # seeds tried for the final model
INITS = ["k-means++", "random"]
SILHOUETTE_SAMPLE = 10_000      # silhouette compares every pair of points: score a fixed sample
TEST_SIZE = 0.20
PLOT_SAMPLE = 5_000


# ----------------------------------------------------------------------------- scoring
def score(X, labels, sample_idx=None):
    """Internal scores. They only look at the prepared data and the cluster numbers."""
    if sample_idx is None:
        sample_idx = np.random.RandomState(SEED).choice(
            len(X), min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    return {
        "silhouette": silhouette_score(X[sample_idx], labels[sample_idx]),   # higher = better
        "davies_bouldin": davies_bouldin_score(X, labels),                   # lower = better
        "calinski_harabasz": calinski_harabasz_score(X, labels),             # higher = better
    }


def smallest_share(labels):
    return np.bincount(labels).min() / len(labels)


# ----------------------------------------------------------------------------- 1. k scan
def scan_k(X, sample_idx):
    rows = []
    for k in K_SCAN:
        start = time.perf_counter()
        runs = []
        for seed in SCAN_SEEDS:
            model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
            runs.append((model.inertia_, model.labels_))
        # stability: does a different seed give the same clusters? (1.0 = identical)
        ref = runs[0][1]
        stability = float(np.mean([adjusted_rand_score(ref, lab) for _, lab in runs[1:]]))
        per_seed = [score(X, lab, sample_idx) for _, lab in runs]
        rows.append({
            "k": k,
            "inertia": np.mean([r[0] for r in runs]),
            "silhouette": np.mean([s["silhouette"] for s in per_seed]),
            "davies_bouldin": np.mean([s["davies_bouldin"] for s in per_seed]),
            "calinski_harabasz": np.mean([s["calinski_harabasz"] for s in per_seed]),
            "smallest_cluster_pct": 100 * np.mean([smallest_share(lab) for _, lab in runs]),
            "stability_ari": stability,
            "seconds": round(time.perf_counter() - start, 1),
        })
        print(f"  k={k:2d}  silhouette={rows[-1]['silhouette']:.3f}  "
              f"DB={rows[-1]['davies_bouldin']:.3f}  smallest={rows[-1]['smallest_cluster_pct']:.1f}%  "
              f"stability={stability:.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. choose k
def choose_k(scan):
    in_range = scan[(scan.k >= K_MIN) & (scan.k <= K_MAX)].copy()
    ok = in_range[(in_range.smallest_cluster_pct >= 100 * MIN_CLUSTER_SHARE) &
                  (in_range.stability_ari >= MIN_STABILITY)].copy()
    note = ""
    if ok.empty:                                    # never leave the user without an answer
        ok = in_range.copy()
        note = "WARNING: no k passed the size/stability guards; ranking all k in the range."
    ok["rank_silhouette"] = ok.silhouette.rank(ascending=False)
    ok["rank_davies_bouldin"] = ok.davies_bouldin.rank(ascending=True)
    ok["mean_rank"] = (ok.rank_silhouette + ok.rank_davies_bouldin) / 2
    best = ok.sort_values(["mean_rank", "k"]).iloc[0]          # tie -> smaller k
    return int(best.k), ok.sort_values(["mean_rank", "k"]), note


# ----------------------------------------------------------------------------- 3. seeding
def search_seed(X, k, sample_idx):
    rows, labels = [], {}
    for init in INITS:
        for seed in SEED_TRIALS:
            model = KMeans(n_clusters=k, init=init, n_init=1, random_state=seed).fit(X)
            labels[(init, seed)] = model.labels_
            rows.append({"init": init, "seed": seed, "inertia": model.inertia_,
                         "silhouette": score(X, model.labels_, sample_idx)["silhouette"],
                         "iterations": model.n_iter_})
    table = pd.DataFrame(rows).sort_values(["inertia", "silhouette"], ascending=[True, False])
    best = table.iloc[0]
    best_labels = labels[(best.init, int(best.seed))]
    # how typical is the winner? ARI against every other run (1.0 = every start finds it)
    typical = float(np.mean([adjusted_rand_score(best_labels, lab)
                             for key, lab in labels.items() if key != (best.init, int(best.seed))]))
    return table, best.init, int(best.seed), typical


# ----------------------------------------------------------------------------- 4. train / test
def train_test(X, k, init, seed):
    X_tr, X_te = train_test_split(X, test_size=TEST_SIZE, random_state=SEED)
    model = KMeans(n_clusters=k, init=init, n_init=1, random_state=seed).fit(X_tr)
    lab_tr, lab_te = model.labels_, model.predict(X_te)          # test rows are only ASSIGNED
    rows = []
    for name, data, lab in [("train", X_tr, lab_tr), ("test", X_te, lab_te)]:
        rows.append({"split": name, "rows": len(data), **score(data, lab),
                     "smallest_cluster_pct": 100 * smallest_share(lab)})
    shares = pd.DataFrame({
        "cluster": range(k),
        "train_share_pct": 100 * np.bincount(lab_tr, minlength=k) / len(lab_tr),
        "test_share_pct": 100 * np.bincount(lab_te, minlength=k) / len(lab_te)}).round(2)
    return pd.DataFrame(rows), shares


# ----------------------------------------------------------------------------- profiles
def profile_clusters(raw, labels):
    """Describe the clusters with the ORIGINAL (unscaled) columns."""
    d = raw.copy()
    d["cluster"] = labels
    out = pd.DataFrame({"bookings": d.groupby("cluster").size()})
    out["share_pct"] = (100 * out.bookings / out.bookings.sum()).round(1)
    for col in d.select_dtypes("number").columns.drop("cluster"):
        out[f"{col}_mean"] = d.groupby("cluster")[col].mean().round(2)
    if {"StaysInWeekendNights", "StaysInWeekNights"} <= set(d.columns):
        out["nights_mean"] = (d.StaysInWeekendNights + d.StaysInWeekNights).groupby(d.cluster).mean().round(2)
    if "LeadTime" in d:
        out["LeadTime_median"] = d.groupby("cluster").LeadTime.median()
    for col in d.select_dtypes(exclude="number").columns:
        top = d.groupby("cluster")[col].agg(lambda s: f"{s.mode().iloc[0]} ({100 * (s == s.mode().iloc[0]).mean():.0f}%)")
        out[f"top_{col}"] = top
    return out


def raw_is_aligned(raw, prep):
    """The raw table must be the same bookings in the same order as the preprocessed table."""
    if len(raw) != len(prep) or "LeadTime" not in raw or "LeadTime" not in prep:
        return False
    return raw["LeadTime"].corr(prep["LeadTime"], method="spearman") > 0.99


# ----------------------------------------------------------------------------- plots
def plot_scan(scan, ok, k, tag):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, title in zip(ax, ["inertia", "silhouette", "davies_bouldin"],
                             ["Inertia (look for an elbow)", "Silhouette (higher is better)",
                              "Davies-Bouldin (lower is better)"]):
        a.plot(scan.k, scan[col], "o-")
        a.axvspan(K_MIN - .5, K_MAX + .5, color="grey", alpha=.12)
        a.axvline(k, color="red", ls="--", lw=1.2, label=f"chosen k = {k}")
        a.set_title(title); a.set_xlabel("k")
    ax[0].legend()
    fig.suptitle("K-Means k scan (mean of 5 seeds; grey band = allowed range)")
    fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_k_scan.png", dpi=130); plt.close(fig)


def plot_pca(X, labels, centers, tag):
    idx = np.random.RandomState(SEED).choice(len(X), min(PLOT_SAMPLE, len(X)), replace=False)
    pca = PCA(2, random_state=SEED).fit(X)
    p, c = pca.transform(X[idx]), pca.transform(centers)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(p[:, 0], p[:, 1], c=labels[idx], s=4, cmap="tab10", alpha=.5)
    ax.scatter(c[:, 0], c[:, 1], c="black", marker="X", s=120, label="centres")
    ax.set_title(f"Clusters in PCA 2-D ({100 * pca.explained_variance_ratio_.sum():.0f}% of variance shown)")
    ax.legend(*sc.legend_elements(), title="cluster", loc="best")
    fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_pca_2d.png", dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(force_k=None):
    for folder in (TABLES, FIGURES, ARTIFACTS):
        folder.mkdir(parents=True, exist_ok=True)
    prep = pd.read_csv(PREP_CSV)
    assert not prep.isna().any().any(), "the preprocessed table must have no missing values"
    X = prep.to_numpy(dtype=float)
    sample_idx = np.random.RandomState(SEED).choice(len(X), min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    print(f"Loaded {X.shape[0]} bookings x {X.shape[1]} columns from {PREP_CSV.name}\n")

    print("1) k scan (5 seeds per k, 10 restarts each):")
    scan = scan_k(X, sample_idx)
    scan.round(4).to_csv(TABLES / "preprocessed_kmeans_k_scan.csv", index=False)

    chosen, ranked, note = choose_k(scan)
    k = force_k or chosen
    tag = f"preprocessed_kmeans_k{k}"          # every k-specific file starts with this
    print(f"\n2) Choice of k (range {K_MIN}-{K_MAX}, smallest cluster >= {MIN_CLUSTER_SHARE:.0%}, "
          f"stability >= {MIN_STABILITY}):")
    print(ranked[["k", "silhouette", "davies_bouldin", "smallest_cluster_pct", "stability_ari",
                  "mean_rank"]].round(3).to_string(index=False))
    if note:
        print(note)
    print(f"   -> best k = {chosen}" + (f"   (forced to k = {force_k})" if force_k else ""))
    ranked.round(4).to_csv(TABLES / "preprocessed_kmeans_k_ranking.csv", index=False)

    print(f"\n3) Seeding for k = {k}: {len(INITS)} inits x {len(SEED_TRIALS)} seeds, one start each")
    seeds, init, seed, typical = search_seed(X, k, sample_idx)
    seeds.round(4).to_csv(TABLES / f"{tag}_seed_search.csv", index=False)
    summary = seeds.groupby("init").inertia.agg(["min", "mean", "std"]).round(1)
    print(summary.to_string())
    print(f"   -> best init = {init}, best seed = {seed}  (lowest inertia; "
          f"agreement with the other runs = {typical:.3f})")

    print("\n4) Train / test (fit on 80%, assign the unseen 20%):")
    tt, shares = train_test(X, k, init, seed)
    print(tt.round(3).to_string(index=False))
    gap = abs(tt.iloc[0].silhouette - tt.iloc[1].silhouette)
    print(f"   silhouette gap train vs test = {gap:.3f}  "
          f"({'clusters carry over to unseen bookings' if gap < 0.02 else 'check: clusters may be specific to the training rows'})")
    print(shares.to_string(index=False))
    tt.round(4).to_csv(TABLES / f"{tag}_train_test.csv", index=False)
    shares.to_csv(TABLES / f"{tag}_train_test_shares.csv", index=False)

    print("\n5) Final model on all rows")
    model = KMeans(n_clusters=k, init=init, n_init=1, random_state=seed).fit(X)
    labels = model.labels_
    final = {"algorithm": "kmeans", "k": k, "init": init, "seed": seed, "rows": len(X),
             "columns": X.shape[1], **score(X, labels, sample_idx), "inertia": float(model.inertia_),
             "smallest_cluster_pct": float(100 * smallest_share(labels))}
    pd.DataFrame([final]).round(4).to_csv(TABLES / f"{tag}_final_metrics.csv", index=False)
    print({key: (round(v, 4) if isinstance(v, float) else v) for key, v in final.items()})

    if RAW_CSV.exists() and raw_is_aligned(pd.read_csv(RAW_CSV), prep):
        profile = profile_clusters(pd.read_csv(RAW_CSV), labels)
        profile.to_csv(TABLES / f"{tag}_cluster_profiles.csv")
        print("\nCluster profiles (original, unscaled columns):")
        print(profile.T.to_string())
    else:
        print("\n(Profiles skipped: raw feature file missing or not in the same row order.)")

    plot_scan(scan, ranked, k, tag)
    plot_pca(X, labels, model.cluster_centers_, tag)
    joblib.dump({"model": model, "columns": list(prep.columns), "k": k, "init": init, "seed": seed},
                ARTIFACTS / f"{tag}.joblib")
    print(f"\nSaved tables to {TABLES}, figures to {FIGURES}, model to {ARTIFACTS / (tag + '.joblib')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tune and train K-Means on the preprocessed table")
    ap.add_argument("--k", type=int, default=None, help="force a number of clusters")
    main(ap.parse_args().k)