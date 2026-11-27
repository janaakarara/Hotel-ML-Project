from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import KNeighborsClassifier

SEED = 42
SAMPLE = 10_000
K_RANGE = range(2, 11)
CONFIGS = [("ward", "euclidean"), ("average", "euclidean"), ("average", "cosine"), ("complete", "euclidean"), ("complete", "cosine")]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

X = pd.read_csv(ROOT / "data/processed/h2_clustering_pca90.csv").to_numpy(dtype=float)
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(X), SAMPLE, replace=False))
sil_idx = np.sort(rng.choice(len(X), SAMPLE, replace=False))


def fit_predict_full(sub_idx, k, link, metric):
    Xs = X[sub_idx]
    ls = AgglomerativeClustering(n_clusters=k, linkage=link, metric=metric).fit_predict(Xs)
    labels = KNeighborsClassifier(n_neighbors=5, weights="distance", metric=metric).fit(Xs, ls).predict(X)
    labels[sub_idx] = ls
    return labels


def evaluate(k, link, metric):
    labels = fit_predict_full(idx, k, link, metric)
    shares = np.bincount(labels, minlength=k) / len(X) * 100.0
    if (shares == 0).any() or shares.min() < 2.0:
        return None
    sil = silhouette_score(X[sil_idx], labels[sil_idx])
    return {"k": k, "linkage": link, "metric": metric, "sil": sil, "labels": labels}


results = [r for r in Parallel(n_jobs=-1)(delayed(evaluate)(k, l, m) for k in K_RANGE for l, m in CONFIGS) if r]
results.sort(key=lambda r: r["sil"], reverse=True)

# Pick highest-silhouette candidate that is also stable across subsamples (mean ARI >= 0.60)
best = results[0]
for cand in results[:5]:
    aris = [adjusted_rand_score(cand["labels"], fit_predict_full(np.sort(np.random.default_rng(s).choice(len(X), SAMPLE, replace=False)), cand["k"], cand["linkage"], cand["metric"])) for s in [10, 20, 30]]
    cand["ari"] = float(np.mean(aris))
    if cand["ari"] >= 0.60:
        best = cand
        break

labels = best["labels"]
k, link, metric = best["k"], best["linkage"], best["metric"]
sil = silhouette_score(X, labels, sample_size=SAMPLE, random_state=SEED)
db = davies_bouldin_score(X, labels)
ch = calinski_harabasz_score(X, labels)

lines = [
    "AGGLOMERATIVE TUNED (PCA-90)",
    f"settings: linkage={link}, metric={metric}, k={k}, sample={SAMPLE}, knn=5(distance)",
    f"rows: {len(X)} | cluster sizes: {np.bincount(labels).tolist()}",
    f"silhouette: {sil:.4f} | davies-bouldin: {db:.4f} | calinski-harabasz: {ch:.1f} | stability_ari: {best.get('ari', 0):.4f}",
]
print("\n".join(lines))
(OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")
np.save(OUT / "labels.npy", labels)

plt.figure(figsize=(10, 5))
dendrogram(linkage(X[idx], method=link, metric=metric), truncate_mode="lastp", p=30, no_labels=True)
plt.title(f"Agglomerative tuned: dendrogram ({link}/{metric}, last 30 merges)")
plt.tight_layout(); plt.savefig(OUT / "dendrogram.png", dpi=150); plt.close()

p = PCA(n_components=2, random_state=SEED).fit_transform(X[sil_idx])
plt.figure(figsize=(7, 6))
plt.scatter(p[:, 0], p[:, 1], s=3, c=labels[sil_idx], cmap="tab10")
plt.title(f"Agglomerative tuned (k={k}, {link}/{metric}) in PCA space"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig(OUT / "pca.png", dpi=150); plt.close()