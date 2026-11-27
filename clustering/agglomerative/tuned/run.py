"""Tuned Agglomerative Clustering comparing raw 12-D frequency features vs. preprocessed PCA-90 (8-D).

Search space
  feature_space   : raw_12d (stage5 frequency-encoded) vs pca90_8d (preprocessed PCA-90)
  n_components k  : 2 .. 10
  linkage/metric  : (ward, euclidean), (average, euclidean/manhattan/cosine), (complete, euclidean/manhattan/cosine)
  propagation     : KNeighborsClassifier(n_neighbors=5, weights="distance") from 10,000-row subsample to full N

Selection
  1. Filter valid fits: all k clusters found and smallest cluster >= 2.0% of bookings (blocks outlier singletons).
  2. Shortlist top 6 valid candidates by Silhouette score.
  3. Test subsample stability across 5 different random 10,000-row draws (mean ARI >= 0.80).
  4. Select the stable candidate with the highest Silhouette (ties broken by lower Davies-Bouldin).
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier

SEED = 42
SAMPLE = 10_000
K_RANGE = list(range(2, 11))
MIN_SHARE_PCT = 2.0
SHORTLIST = 6
STABILITY_SEEDS = [10, 20, 30, 40, 50]
STABILITY_MIN_ARI = 0.80

LINKAGE_METRICS = [
    ("ward", "euclidean"),
    ("average", "euclidean"),
    ("average", "manhattan"),
    ("average", "cosine"),
    ("complete", "euclidean"),
    ("complete", "manhattan"),
    ("complete", "cosine"),
]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_spaces():
    raw_path = ROOT / "data/processed/h2_clustering_stage5_freq.csv"
    if not raw_path.exists():
        raw_path = ROOT / "data/processed/h2_clustering_stage5_frequency.csv"
    pca_path = ROOT / "data/processed/h2_clustering_pca90.csv"

    spaces = {
        "raw_12d": pd.read_csv(raw_path).to_numpy(dtype=float),
        "pca90_8d": pd.read_csv(pca_path).to_numpy(dtype=float),
    }
    return spaces


def fit_and_propagate(X, idx, k, link, metric):
    Xs = X[idx]
    model = AgglomerativeClustering(n_clusters=k, linkage=link, metric=metric)
    labels_s = model.fit_predict(Xs)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric=metric)
    labels = knn.fit(Xs, labels_s).predict(X)
    labels[idx] = labels_s
    return labels


def eval_config(space_name, X, idx, sil_idx, k, link, metric):
    t0 = time.perf_counter()
    labels = fit_and_propagate(X, idx, k, link, metric)
    elapsed = time.perf_counter() - t0

    counts = np.bincount(labels, minlength=k)
    shares = counts / len(X) * 100.0
    n_found = int((counts > 0).sum())
    min_share = float(shares.min())
    max_share = float(shares.max())

    if n_found < 2:
        sil, db, ch = -1.0, 999.0, 0.0
    else:
        sil = float(silhouette_score(X[sil_idx], labels[sil_idx], metric="euclidean"))
        db = float(davies_bouldin_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))

    valid = bool(n_found == k and min_share >= MIN_SHARE_PCT)
    return {
        "feature_space": space_name,
        "dims": int(X.shape[1]),
        "k": k,
        "linkage": link,
        "metric": metric,
        "n_clusters_found": n_found,
        "min_cluster_share_pct": round(min_share, 2),
        "max_cluster_share_pct": round(max_share, 2),
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 1),
        "fit_time_sec": round(elapsed, 2),
        "valid": valid,
    }


def check_stability(X, ref_labels, k, link, metric):
    aris = []
    for s in STABILITY_SEEDS:
        rng = np.random.default_rng(s)
        idx_s = np.sort(rng.choice(len(X), SAMPLE, replace=False))
        lab_s = fit_and_propagate(X, idx_s, k, link, metric)
        aris.append(adjusted_rand_score(ref_labels, lab_s))
    return round(float(np.mean(aris)), 4), round(float(np.min(aris)), 4)


def plot_comparison(grid_df, chosen, out_path):
    valid = grid_df[grid_df["valid"]].copy()
    spaces = ["raw_12d", "pca90_8d"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    for ax, sp in zip(axes, spaces):
        sub = valid[valid["feature_space"] == sp]
        for (link, metric), grp in sub.groupby(["linkage", "metric"]):
            grp = grp.sort_values("k")
            ax.plot(grp["k"], grp["silhouette"], marker="o", lw=1.7, label=f"{link} ({metric})")
        if chosen["feature_space"] == sp:
            ax.axvline(chosen["k"], color="black", ls="--", lw=1.3, label=f"selected k={chosen['k']}")
        ax.set_title(f"{sp} (valid fits: min share >= {MIN_SHARE_PCT}%)", fontweight="bold")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(K_RANGE)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Silhouette Score (higher = better)")
    axes[1].legend(fontsize=8, loc="best", frameon=True)
    fig.suptitle("Agglomerative Tuning: Raw 12-D vs. Preprocessed PCA-90 (8-D)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    spaces = load_spaces()
    n_rows = len(spaces["raw_12d"])
    rng = np.random.default_rng(SEED)
    base_idx = np.sort(rng.choice(n_rows, SAMPLE, replace=False))
    sil_idx = np.sort(rng.choice(n_rows, SAMPLE, replace=False))

    for sp, X in spaces.items():
        print(f"Loaded {sp:10s}: {X.shape[0]} rows x {X.shape[1]} dims")

    tasks = [(sp, k, link, metric) for sp in spaces for k in K_RANGE for (link, metric) in LINKAGE_METRICS]
    print(f"\nStep 1: Running {len(tasks)} Agglomerative fits across raw_12d and pca90_8d ...")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=-1, verbose=5)(
        delayed(eval_config)(sp, spaces[sp], base_idx, sil_idx, k, link, metric)
        for sp, k, link, metric in tasks
    )
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(OUT / "tuning_grid.csv", index=False)
    print(f"Grid finished in {time.perf_counter() - t0:.1f}s | Valid fits: {int(grid_df['valid'].sum())}/{len(grid_df)}")

    # Best per feature space summary
    print("\n========== BEST VALID CONFIG PER FEATURE SPACE ==========")
    for sp in spaces:
        sub = grid_df[(grid_df["feature_space"] == sp) & (grid_df["valid"])].sort_values(
            ["silhouette", "davies_bouldin"], ascending=[False, True]
        )
        if not sub.empty:
            print(sub.head(3).to_string(index=False))

    # Step 2: Shortlist top 6 overall and test subsample stability across 5 seeds
    top = (
        grid_df[grid_df["valid"]]
        .sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
        .head(SHORTLIST)
        .copy()
    )
    print(f"\nStep 2: Checking 5-subsample stability (ARI) on top {len(top)} candidates ...")
    ari_means, ari_mins = [], []
    for r in top.itertuples():
        X_sp = spaces[r.feature_space]
        ref_labels = fit_and_propagate(X_sp, base_idx, int(r.k), r.linkage, r.metric)
        m_ari, min_ari = check_stability(X_sp, ref_labels, int(r.k), r.linkage, r.metric)
        ari_means.append(m_ari)
        ari_mins.append(min_ari)

    top["stability_ari_mean"] = ari_means
    top["stability_ari_min"] = ari_mins

    stable = top[top["stability_ari_mean"] >= STABILITY_MIN_ARI]
    pool = stable if not stable.empty else top.sort_values("stability_ari_mean", ascending=False).head(1)
    chosen = pool.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0].to_dict()

    top["selected"] = [
        (r.feature_space, r.k, r.linkage, r.metric)
        == (chosen["feature_space"], chosen["k"], chosen["linkage"], chosen["metric"])
        for r in top.itertuples()
    ]
    top.to_csv(OUT / "shortlist_stability.csv", index=False)

    print("\n========== SHORTLIST & SUBSAMPLE STABILITY ==========")
    print(top[["feature_space", "dims", "k", "linkage", "metric", "silhouette", "davies_bouldin",
               "calinski_harabasz", "min_cluster_share_pct", "stability_ari_mean", "selected"]].to_string(index=False))

    # Final fit with winning configuration
    sp_win = chosen["feature_space"]
    k_win = int(chosen["k"])
    link_win = chosen["linkage"]
    metric_win = chosen["metric"]
    X_win = spaces[sp_win]
    labels_win = fit_and_propagate(X_win, base_idx, k_win, link_win, metric_win)
    counts_win = np.bincount(labels_win, minlength=k_win).tolist()

    lines = [
        "AGGLOMERATIVE TUNED RESULTS",
        f"selected_feature_space : {sp_win} ({X_win.shape[1]}-D)",
        f"settings               : linkage={link_win}, metric={metric_win}, k={k_win}, sample={SAMPLE}, knn=5(distance)",
        f"rows                   : {n_rows}",
        f"cluster_sizes          : {counts_win}",
        f"min_cluster_share_pct  : {chosen['min_cluster_share_pct']}",
        f"max_cluster_share_pct  : {chosen['max_cluster_share_pct']}",
        f"silhouette             : {chosen['silhouette']:.4f}",
        f"davies_bouldin         : {chosen['davies_bouldin']:.4f}",
        f"calinski_harabasz      : {chosen['calinski_harabasz']:.1f}",
        f"stability_ari_mean     : {chosen['stability_ari_mean']:.4f}",
        f"stability_ari_min      : {chosen['stability_ari_min']:.4f}",
    ]
    print("\n" + "\n".join(lines))
    (OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")
    np.save(OUT / "labels.npy", labels_win)

    # Plots
    plot_comparison(grid_df, chosen, OUT / "tuning_curves.png")

    plt.figure(figsize=(10, 5))
    Z_link = linkage(X_win[base_idx], method=link_win, metric=metric_win)
    dendrogram(Z_link, truncate_mode="lastp", p=30, no_labels=True)
    plt.title(f"Tuned Agglomerative dendrogram ({sp_win}, {link_win}/{metric_win}, last 30 merges)")
    plt.tight_layout()
    plt.savefig(OUT / "dendrogram_tuned.png", dpi=150)
    plt.close()

    p2 = PCA(n_components=2, random_state=SEED).fit_transform(X_win[sil_idx])
    plt.figure(figsize=(7, 6))
    plt.scatter(p2[:, 0], p2[:, 1], s=3, c=labels_win[sil_idx], cmap="tab10")
    plt.title(f"Tuned Agglomerative ({sp_win}, k={k_win}, {link_win}/{metric_win}) in 2D PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT / "pca_tuned.png", dpi=150)
    plt.close()
    print("\nSaved all tuned outputs to", OUT)


if __name__ == "__main__":
    main()