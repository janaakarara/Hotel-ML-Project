"""Untuned (baseline) K-Means for Task C (clustering).

Baseline = KMeans fitted directly on the preprocessed 29-D clustering features
(h2_clustering_stage5.csv), no PCA and no hyper-parameter search:
  n_clusters=3 (fixed, same k as the GMM baseline), init='k-means++', n_init=10, max_iter=300, random_state=42.

Data loading, the fixed 10,000-row silhouette sample, the internal metrics and the cluster-profile table are
imported from clustering/gmm/baseline/run.py, so both baselines are computed by the same code and are directly
comparable. Metrics are always computed in the original 29-D space. ADR, IsCanceled and the other held-out
columns are used only afterwards, to describe the clusters.

Usage (from anywhere):
  python clustering/kmeans/baseline/run.py
  python -m clustering.kmeans.baseline.run
Writes to --output-dir (default clustering/kmeans/baseline/outputs):
  kmeans_baseline_model.joblib            fitted pipeline (passthrough + KMeans)
  kmeans_baseline_metadata.json           hyper-parameters, shape, runtime, seed ARI, cluster distribution
  clustering_kmeans_baseline_metrics.csv  one row, same column order as clustering_gmm_untuned_metrics.csv
                                          (GMM-only columns left empty, K-Means extras appended at the end)
  kmeans_baseline_pca2d.png               clusters + centroids on PC1-PC2
  results.txt, labels.npy                 README.md convention (summary text; one label per CSV row)
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline

from clustering.gmm.baseline.run import (  # shared utilities: identical loading / metrics / profiling
    PALETTE, RANDOM_STATE, build_cluster_profiles, cluster_metrics, data_zero_eigenvalues,
    load_data, make_dirs, silhouette_sample_idx)

BASELINE_PARAMS = dict(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=RANDOM_STATE)
STABILITY_SEEDS = [0, 1, 2, 3, 4]

# Column order of clustering_gmm_untuned_metrics.csv. Columns that only exist for a GMM stay empty (NaN).
GMM_COLUMNS = [
    "configuration", "feature_space", "model_dims", "pca_variance_kept", "n_components", "covariance_type",
    "reg_covar", "init_params", "n_init", "max_iter", "converged", "n_iter", "n_free_parameters",
    "n_clusters_found", "min_cluster_share_pct", "max_cluster_share_pct", "silhouette", "davies_bouldin",
    "calinski_harabasz", "bic_model_space", "aic_model_space", "mean_log_likelihood", "mean_max_posterior",
    "mean_normalised_entropy", "cluster_shares_pct", "training_time_sec", "data_zero_eigenvalues",
    "max_condition_number", "min_eig_at_floor_per_component"]
KMEANS_EXTRA_COLUMNS = ["inertia", "seed_ari_mean", "seed_ari_min", "cluster_counts"]


def resolve(path):
    """Relative paths are taken from the repo root, so the script works from any folder."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def seed_stability(X, ref_labels, seeds=STABILITY_SEEDS):
    """ARI of the partition from each seed against the random_state=42 partition."""
    aris = {}
    for s in seeds:
        km = KMeans(**{**BASELINE_PARAMS, "random_state": s}).fit(X)
        aris[int(s)] = round(float(adjusted_rand_score(ref_labels, km.labels_)), 4)
    return aris


def summarize_model(pipe, X, sil_idx, fit_time, aris, configuration="untuned_baseline"):
    km = pipe.named_steps["kmeans"]
    labels = km.labels_
    counts = pd.Series(labels).value_counts().sort_index()
    row = {c: np.nan for c in GMM_COLUMNS}
    row.update({
        "configuration": configuration,
        "feature_space": "raw",
        "model_dims": int(X.shape[1]),
        "pca_variance_kept": 1.0,
        "n_components": int(km.n_clusters),
        "init_params": km.init,
        "n_init": int(km.n_init),
        "max_iter": int(km.max_iter),
        "converged": bool(km.n_iter_ < km.max_iter),   # KMeans has no converged_; stopped before max_iter
        "n_iter": int(km.n_iter_),
        "n_free_parameters": int(km.n_clusters * X.shape[1]),   # centroid coordinates
    })
    row.update(cluster_metrics(X, labels, sil_idx))
    row["cluster_shares_pct"] = json.dumps({int(k): float(round(v, 2)) for k, v in (counts / len(labels) * 100).items()})
    row["training_time_sec"] = round(float(fit_time), 3)
    row["data_zero_eigenvalues"] = data_zero_eigenvalues(X)[0]
    row["inertia"] = round(float(km.inertia_), 2)
    vals = list(aris.values())
    row["seed_ari_mean"] = round(float(np.mean(vals)), 4)
    row["seed_ari_min"] = round(float(np.min(vals)), 4)
    row["cluster_counts"] = json.dumps({int(k): int(v) for k, v in counts.items()})
    return row, labels


def plot_pca_2d(X, labels, centers, out_path, title):
    """2-D PCA view of the clusters; centroids are projected with the same PCA."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    coords, cen2d = pca.transform(X), pca.transform(centers)
    var1, var2 = pca.explained_variance_ratio_ * 100
    show = np.random.RandomState(RANDOM_STATE).choice(len(X), size=min(15000, len(X)), replace=False)
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    for cid in np.unique(labels):
        color = PALETTE[int(cid) % len(PALETTE)]
        m = labels[show] == cid
        ax.scatter(coords[show][m, 0], coords[show][m, 1], s=10, alpha=0.35, color=color, edgecolors="none",
                   label=f"Cluster {cid} ({(labels == cid).mean() * 100:.1f}%)", zorder=2)
        ax.scatter(*cen2d[int(cid)], s=220, color=color, edgecolors="black", linewidths=1.8, marker="X", zorder=5)
    ax.set_xlabel(f"PC1 ({var1:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% of variance)")
    ax.set_title(f"{title}\nX = centroids. This view keeps only {var1 + var2:.1f}% of the variance: clusters that "
                 f"overlap here may still separate in the other dimensions.", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="best")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Untuned baseline K-Means (k=3, scikit-learn defaults otherwise).")
    ap.add_argument("--features", default="data/processed/h2_clustering_stage5_frequency.csv")
    ap.add_argument("--raw", default="data/processed/h2_clustering_stage2.csv")
    ap.add_argument("--profile-cols", default="data/processed/h2_clustering_profile_cols.csv")
    ap.add_argument("--output-dir", default="clustering/kmeans/baseline/outputs")
    args = ap.parse_args()
    (out_dir,) = make_dirs(resolve(args.output_dir))
    X_df, X, raw_df, prof_df = load_data(resolve(args.features), resolve(args.raw), resolve(args.profile_cols))
    sil_idx = silhouette_sample_idx(len(X))
    print(f"Untuned K-Means {BASELINE_PARAMS} on {X.shape[0]} bookings x {X.shape[1]} features")

    pipe = Pipeline([("reduce", "passthrough"), ("kmeans", KMeans(**BASELINE_PARAMS))])
    t0 = time.perf_counter()
    pipe.fit(X)
    fit_time = time.perf_counter() - t0

    km = pipe.named_steps["kmeans"]
    aris = seed_stability(X, km.labels_)
    metrics, labels = summarize_model(pipe, X, sil_idx, fit_time, aris)
    metrics_df = pd.DataFrame([metrics])[GMM_COLUMNS + KMEANS_EXTRA_COLUMNS]

    # GMM profile builder needs posteriors; K-Means has none, so pass a dummy and drop that column.
    profiles = (build_cluster_profiles(raw_df, prof_df, labels, np.ones((len(labels), 1)))
                .drop(columns="Mean_Max_Posterior"))

    counts = pd.Series(labels).value_counts().sort_index()
    meta = {"params": BASELINE_PARAMS, "feature_names": list(X_df.columns),
            "dataset_shape": [int(X.shape[0]), int(X.shape[1])], "training_time_sec": metrics["training_time_sec"],
            "inertia": metrics["inertia"], "n_iter": metrics["n_iter"],
            "stability_seeds": STABILITY_SEEDS, "seed_ari_vs_seed42": aris,
            "seed_ari_mean": metrics["seed_ari_mean"], "seed_ari_min": metrics["seed_ari_min"],
            "cluster_counts": {int(k): int(v) for k, v in counts.items()},
            "cluster_pcts": {int(k): round(float(v / len(labels) * 100), 2) for k, v in counts.items()},
            "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()}}

    joblib.dump(pipe, out_dir / "kmeans_baseline_model.joblib")
    (out_dir / "kmeans_baseline_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    metrics_df.to_csv(out_dir / "clustering_kmeans_baseline_metrics.csv", index=False)
    np.save(out_dir / "labels.npy", labels)
    plot_pca_2d(X, labels, km.cluster_centers_, out_dir / "kmeans_baseline_pca2d.png",
                f"K-Means untuned baseline (raw {X.shape[1]}-D, k=3, k-means++, n_init=10)")

    summary = [
        f"K-Means untuned baseline  params={BASELINE_PARAMS}",
        f"rows x features: {X.shape[0]} x {X.shape[1]}",
        f"cluster sizes: {meta['cluster_counts']}  pct: {meta['cluster_pcts']}",
        f"silhouette (10k sample): {metrics['silhouette']}   davies_bouldin: {metrics['davies_bouldin']}   "
        f"calinski_harabasz: {metrics['calinski_harabasz']}",
        f"inertia: {metrics['inertia']}   n_iter: {metrics['n_iter']}",
        f"seed ARI vs seed 42 ({STABILITY_SEEDS}): {aris}  mean={metrics['seed_ari_mean']}",
        "noise: n/a (K-Means assigns every row)"]
    (out_dir / "results.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n========== UNTUNED BASELINE K-MEANS ==========")
    for k, v in metrics_df.iloc[0].items():
        print(f"  {k:32s}: {v}")
    print("\nCluster profiles:")
    print(profiles[["Cluster", "Count", "Share_Pct", "LeadTime_Mean", "TotalNights_Mean", "HasChildren_Pct",
                    "Top_MarketSegment", "PostHoc_ADR_Mean", "PostHoc_CancelRate_Pct"]].to_string(index=False))
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
