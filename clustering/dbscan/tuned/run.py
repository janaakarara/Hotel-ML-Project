from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

SEED = 42
EPS_GRID = [0.60, 0.75, 0.90, 1.05, 1.20, 1.35, 1.50, 1.75, 2.00]
MIN_SAMPLES_GRID = [15, 25, 50, 100, 150]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

X = pd.read_csv(ROOT / "data/processed/h2_clustering_pca90.csv").to_numpy(dtype=float)


def eval_dbscan(eps, ms):
    labels = DBSCAN(eps=eps, min_samples=ms, metric="euclidean", n_jobs=1).fit_predict(X)
    keep = labels != -1
    noise_pct = 100.0 * (~keep).mean()
    if keep.sum() < 500 or noise_pct > 25.0:
        return None
    _, sizes = np.unique(labels[keep], return_counts=True)
    if not (2 <= len(sizes) <= 10) or (sizes.min() / len(X) * 100.0) < 2.0:
        return None
    sil = silhouette_score(X[keep], labels[keep], sample_size=min(10_000, int(keep.sum())), random_state=SEED)
    return {"eps": eps, "ms": ms, "sil": sil, "noise_pct": noise_pct, "labels": labels}


candidates = [r for r in Parallel(n_jobs=-1)(delayed(eval_dbscan)(e, m) for e in EPS_GRID for m in MIN_SAMPLES_GRID) if r]
if not candidates:
    raise RuntimeError("No valid DBSCAN config found within constraints; widen EPS_GRID or noise cap.")

best = max(candidates, key=lambda r: (r["sil"], -r["noise_pct"]))
labels, eps, ms = best["labels"], best["eps"], best["ms"]
keep = labels != -1
sizes = np.unique(labels[keep], return_counts=True)[1]
Xk, lk = X[keep], labels[keep]

sil = silhouette_score(Xk, lk, sample_size=min(10_000, len(lk)), random_state=SEED)
db = davies_bouldin_score(Xk, lk)
ch = calinski_harabasz_score(Xk, lk)

lines = [
    "DBSCAN TUNED (PCA-90)",
    f"settings: eps={eps}, min_samples={ms}, metric=euclidean, all rows",
    f"rows: {len(X)} | clusters: {len(sizes)} | noise %: {best['noise_pct']:.2f}",
    f"cluster sizes: {sizes.tolist()}",
    f"silhouette (non-noise): {sil:.4f} | davies-bouldin: {db:.4f} | calinski-harabasz: {ch:.1f}",
]
print("\n".join(lines))
(OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")
np.save(OUT / "labels.npy", labels)

kd = np.sort(NearestNeighbors(n_neighbors=ms, n_jobs=-1).fit(X).kneighbors(X)[0][:, -1])
plt.figure(figsize=(8, 5))
plt.plot(kd, label=f"distance to {ms}th neighbour (sorted)")
plt.axhline(eps, color="green", linestyle="--", label=f"tuned eps = {eps}")
plt.title("DBSCAN tuned: tuned eps vs k-distance curve")
plt.xlabel("points (sorted)"); plt.ylabel("distance"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "k_distance.png", dpi=150); plt.close()

pi = np.sort(np.random.default_rng(SEED).choice(len(X), 10_000, replace=False))
p = PCA(n_components=2, random_state=SEED).fit_transform(X[pi])
l = labels[pi]; noise = l == -1
plt.figure(figsize=(7, 6))
plt.scatter(p[noise, 0], p[noise, 1], s=3, c="lightgray", label=f"noise ({best['noise_pct']:.1f}%)")
plt.scatter(p[~noise, 0], p[~noise, 1], s=3, c=l[~noise] % 10, cmap="tab10")
plt.title(f"DBSCAN tuned (k={len(sizes)}, eps={eps}, ms={ms}) in PCA space")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(markerscale=4)
plt.tight_layout(); plt.savefig(OUT / "pca.png", dpi=150); plt.close()