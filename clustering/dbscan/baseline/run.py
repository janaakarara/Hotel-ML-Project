from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

SEED = 42

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

X = pd.read_csv(ROOT / "data/processed/h2_clustering_stage5.csv").to_numpy(dtype=float)

labels = DBSCAN(n_jobs=-1).fit_predict(X)     # sklearn defaults: eps=0.5, min_samples=5, all rows

keep = labels != -1
sizes = np.unique(labels[keep], return_counts=True)[1]
k = len(sizes)
lines = ["DBSCAN BASELINE (untuned)",
         "settings: eps=0.5, min_samples=5, metric=euclidean (sklearn defaults), all rows",
         f"rows: {len(X)} | clusters: {k} | noise %: {100 * (~keep).mean():.2f}"]
if k >= 2:
    Xk, lk = X[keep], labels[keep]
    lines += [f"largest cluster: {sizes.max()} | smallest: {sizes.min()} | clusters under 100 points: {int((sizes < 100).sum())}",
              f"silhouette (non-noise only): {silhouette_score(Xk, lk, sample_size=min(10_000, len(lk)), random_state=SEED):.4f}",
              f"davies-bouldin: {davies_bouldin_score(Xk, lk):.4f}",
              f"calinski-harabasz: {calinski_harabasz_score(Xk, lk):.1f}",
              "note: scores ignore noise points and are not comparable with a method that labels every row"]
else:
    lines.append("fewer than 2 clusters found, so the scores are undefined")
print("\n".join(lines))
(OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")
np.save(OUT / "labels.npy", labels)

# k-distance curve for min_samples=5: shows where the default eps sits
kd = np.sort(NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(X).kneighbors(X)[0][:, -1])
plt.figure(figsize=(8, 5))
plt.plot(kd, label="distance to 5th neighbour (sorted)")
plt.axhline(0.5, color="red", linestyle="--", label="default eps = 0.5")
plt.title("DBSCAN baseline: default eps vs the k-distance curve")
plt.xlabel("points (sorted)"); plt.ylabel("distance"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "k_distance.png", dpi=150); plt.close()

rng = np.random.default_rng(SEED)
pi = np.sort(rng.choice(len(X), 10_000, replace=False))
p = PCA(n_components=2, random_state=SEED).fit_transform(X[pi])
l = labels[pi]; noise = l == -1
plt.figure(figsize=(7, 6))
plt.scatter(p[noise, 0], p[noise, 1], s=3, c="lightgray", label="noise")
plt.scatter(p[~noise, 0], p[~noise, 1], s=3, c=l[~noise] % 10, cmap="tab10")
plt.title("DBSCAN baseline in PCA space (colors repeat: many tiny clusters)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(markerscale=4)
plt.tight_layout(); plt.savefig(OUT / "pca.png", dpi=150); plt.close()
print("saved to", OUT)
