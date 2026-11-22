"""Task C, Milestone 1: an untuned K-Means baseline on booking attributes (no outcome columns).

Untuned means sklearn's default settings. Only k is chosen. To avoid reporting one lucky
seed, every k is fitted with several seeds and the scores are averaged (this is a more
honest measurement, not tuning: the model settings are not changed).
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import project_path
from src.evaluation import clustering_report as report
from src.evaluation.clustering_metrics import score_clustering
from src.features.clustering_features import add_clustering_features
from src.preprocessing.clustering_preprocessor import build_clustering_preprocessor


def build_kmeans_pipeline(k, seed):
    """Raw booking rows in, cluster number out. Saved as one object, so a new booking
    goes through exactly the same steps as the data we fitted on."""
    return Pipeline([
        ("features", FunctionTransformer(add_clustering_features)),
        ("preprocess", build_clustering_preprocessor()),
        # Untuned baseline: only k is chosen (from the scan); every other setting is sklearn's default.
        ("kmeans", KMeans(n_clusters=k, random_state=seed)),
    ])


def scan_k(X, k_values, seed, silhouette_sample, n_seeds):
    """Fit K-Means for every k with n_seeds different seeds on the already-prepared matrix X.

    The scores are averaged over the seeds. silhouette_std shows how much they move, and
    stability_ari is the average agreement (1 = identical clusters) between the first
    seed's clusters and the other seeds' clusters for the same k.
    """
    rows = []
    for k in k_values:
        fits = []
        for s in range(n_seeds):
            start = time.perf_counter()
            model = KMeans(n_clusters=k, random_state=seed + s).fit(X)
            fits.append((model, time.perf_counter() - start))
        reference = fits[0][0].labels_
        scores = [score_clustering(X, m.labels_, silhouette_sample, seed) for m, _ in fits]
        aris = [adjusted_rand_score(reference, m.labels_) for m, _ in fits[1:]]
        rows.append({
            "k": k,
            "inertia": float(np.mean([m.inertia_ for m, _ in fits])),
            "silhouette": float(np.mean([sc["silhouette"] for sc in scores])),
            "davies_bouldin": float(np.mean([sc["davies_bouldin"] for sc in scores])),
            "calinski_harabasz": float(np.mean([sc["calinski_harabasz"] for sc in scores])),
            "fit_seconds": round(float(np.mean([t for _, t in fits])), 2),
            "silhouette_std": float(np.std([sc["silhouette"] for sc in scores])),
            "stability_ari": float(np.mean(aris)) if aris else 1.0,
        })
    return pd.DataFrame(rows)


def choose_k(scan, min_k, max_k):
    """Highest average silhouette inside the allowed range. No outcome column is involved."""
    allowed = scan[(scan["k"] >= min_k) & (scan["k"] <= max_k)]
    return int(allowed.loc[allowed["silhouette"].idxmax(), "k"])


def run(df, config, k=None):
    seed = config["seed"]
    settings = config["clustering"]
    paths = config["paths"]
    n_seeds = settings.get("stability_seeds", 5)

    # 1. Prepare the data once, then try many k on the same matrix.
    prepare = Pipeline([("features", FunctionTransformer(add_clustering_features)),
                        ("preprocess", build_clustering_preprocessor())])
    X = prepare.fit_transform(df)
    scan = scan_k(X, settings["k_values"], seed, settings["silhouette_sample"], n_seeds)
    if k is None:
        k = choose_k(scan, settings["min_k_to_choose"], settings["max_k_to_choose"])
    chosen = scan[scan["k"] == k].iloc[0]

    # 2. Fit the full pipeline with the chosen k on all cleaned rows (this is the saved model).
    start = time.perf_counter()
    pipeline = build_kmeans_pipeline(k, seed).fit(df)
    fit_seconds = time.perf_counter() - start
    labels = pipeline.named_steps["kmeans"].labels_
    X = pipeline[:-1].transform(df)  # the same prepared matrix K-Means saw
    metrics = {"algorithm": "kmeans", "config": "baseline", "k": k, "rows": len(df), "features_after_encoding": X.shape[1],
               **score_clustering(X, labels, settings["silhouette_sample"], seed),
               "inertia": pipeline.named_steps["kmeans"].inertia_, "fit_seconds": round(fit_seconds, 2),
               "seeds_averaged": n_seeds,
               "silhouette_mean_over_seeds": float(chosen["silhouette"]),
               "silhouette_std_over_seeds": float(chosen["silhouette_std"]),
               "stability_ari": float(chosen["stability_ari"])}

    # 3. Describe the clusters and save everything.
    profile = report.profile_clusters(df, labels)
    report.plot_k_scan(scan, k, project_path(f"{paths['figures']}/clustering_kmeans_k_scan.png"))
    pca_share = report.plot_pca_2d(X, labels, pipeline.named_steps["kmeans"].cluster_centers_,
                                   settings["plot_sample"], seed, project_path(f"{paths['figures']}/clustering_kmeans_pca_2d.png"))
    report.plot_profiles(profile, project_path(f"{paths['figures']}/clustering_kmeans_profiles.png"))
    metrics["pca_2d_variance_shown"] = round(float(pca_share.sum()), 3)

    scan.to_csv(project_path(f"{paths['tables']}/clustering_kmeans_k_scan.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(project_path(f"{paths['tables']}/clustering_kmeans_baseline_metrics.csv"), index=False)
    profile.to_csv(project_path(f"{paths['tables']}/clustering_kmeans_profiles.csv"))
    joblib.dump(pipeline, project_path(f"{paths['artifacts']}/clustering_kmeans_baseline.joblib"))
    return scan, metrics, profile