from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import KNeighborsClassifier

SEED = 42
SAMPLE = 10_000      # agglomerative needs an n x n distance matrix, so it runs on a sample
N_CLUSTERS = 4       # fixed untuned guess (sklearn's default of 2 would flatter silhouette)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

X = pd.read_csv(ROOT / "data/processed/h2_clustering_stage5_frequency.csv").to_numpy(dtype=float)
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(X), SAMPLE, replace=False))
Xs = X[idx]

# default settings otherwise: linkage="ward", metric="euclidean"
labels_s = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit_predict(Xs)

# give every other row a cluster via its 5 nearest sampled neighbours
labels = KNeighborsClassifier(n_neighbors=5).fit(Xs, labels_s).predict(X)
labels[idx] = labels_s

sil = silhouette_score(X, labels, sample_size=10_000, random_state=SEED)
db = davies_bouldin_score(X, labels)
ch = calinski_harabasz_score(X, labels)

lines = ["AGGLOMERATIVE BASELINE (untuned)",
         f"settings: linkage=ward, metric=euclidean, k={N_CLUSTERS} (sklearn defaults except k), clustered sample={SAMPLE}",
         f"rows: {len(X)}",
         f"cluster sizes: {np.bincount(labels).tolist()}",
         f"silhouette: {sil:.4f}",
         f"davies-bouldin: {db:.4f}",
         f"calinski-harabasz: {ch:.1f}"]
print("\n".join(lines))
(OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")
np.save(OUT / "labels.npy", labels)

# dendrogram of the same ward tree (scipy ward = sklearn ward), last 30 merges
plt.figure(figsize=(10, 5))
dendrogram(linkage(Xs, method="ward"), truncate_mode="lastp", p=30, no_labels=True)
plt.title("Agglomerative baseline: dendrogram (ward, last 30 merges)")
plt.tight_layout(); plt.savefig(OUT / "dendrogram.png", dpi=150); plt.close()

pi = np.sort(rng.choice(len(X), 10_000, replace=False))
p = PCA(n_components=2, random_state=SEED).fit_transform(X[pi])
plt.figure(figsize=(7, 6))
plt.scatter(p[:, 0], p[:, 1], s=3, c=labels[pi], cmap="tab10")
plt.title(f"Agglomerative baseline (k={N_CLUSTERS}) in PCA space"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig(OUT / "pca.png", dpi=150); plt.close()
print("saved to", OUT)
