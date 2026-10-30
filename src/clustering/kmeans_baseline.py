"""Task C, Milestone 1: an untuned K-Means baseline on booking attributes (no outcome columns)."""
import time

import joblib
import pandas as pd
from sklearn.cluster import KMeans
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


def scan_k(X, k_values, seed, silhouette_sample):
    """Fit K-Means once for every k on the already-prepared matrix X and score each result."""
    rows = []
    for k in k_values:
        start = time.perf_counter()
        model = KMeans(n_clusters=k, random_state=seed).fit(X)
        seconds = time.perf_counter() - start
        scores = score_clustering(X, model.labels_, silhouette_sample, seed)
        rows.append({"k": k, "inertia": model.inertia_, **scores, "fit_seconds": round(seconds, 2)})
    return pd.DataFrame(rows)


def choose_k(scan, min_k, max_k):
    """Highest silhouette inside the allowed range. No outcome column is involved."""
    allowed = scan[(scan["k"] >= min_k) & (scan["k"] <= max_k)]
    return int(allowed.loc[allowed["silhouette"].idxmax(), "k"])


def run(df, config, k=None):
    seed = config["seed"]
    settings = config["clustering"]
    paths = config["paths"]

    # 1. Prepare the data once, then try many k on the same matrix.
    prepare = Pipeline([("features", FunctionTransformer(add_clustering_features)),
                        ("preprocess", build_clustering_preprocessor())])
    X = prepare.fit_transform(df)
    scan = scan_k(X, settings["k_values"], seed, settings["silhouette_sample"])
    if k is None:
        k = choose_k(scan, settings["min_k_to_choose"], settings["max_k_to_choose"])

    # 2. Fit the full pipeline with the chosen k on all cleaned rows.
    start = time.perf_counter()
    pipeline = build_kmeans_pipeline(k, seed).fit(df)
    fit_seconds = time.perf_counter() - start
    labels = pipeline.named_steps["kmeans"].labels_
    X = pipeline[:-1].transform(df)  # the same prepared matrix K-Means saw
    metrics = {"algorithm": "kmeans", "config": "baseline", "k": k, "rows": len(df), "features_after_encoding": X.shape[1],
               **score_clustering(X, labels, settings["silhouette_sample"], seed),
               "inertia": pipeline.named_steps["kmeans"].inertia_, "fit_seconds": round(fit_seconds, 2)}

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
