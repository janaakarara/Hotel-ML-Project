from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from sklearn.neighbors import KNeighborsClassifier

SEED = 42
SAMPLE = 10_000       # same sample as the agglomerative baseline
N_CLUSTERS = 4        # same k as the agglomerative baseline
VAR_KEPT = 0.80       # keep enough FAMD components to explain this much variance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data/processed/h2_clustering_stage5.csv")
Xb = df.to_numpy(dtype=float)                       # the main (baseline) space

# one-hot columns are the ones with exactly two values {0, w}; everything else is numeric
cat_cols = [c for c in df.columns if df[c].min() == 0 and df[c].nunique() == 2]
num_cols = [c for c in df.columns if c not in cat_cols]

# ---- FAMD ----
num = df[num_cols].to_numpy(dtype=float)
num = (num - num.mean(axis=0)) / num.std(axis=0)                 # standardize numeric
dum = (df[cat_cols].to_numpy(dtype=float) > 0).astype(float)     # undo the 0.5 weight -> plain 0/1
p = dum.mean(axis=0)                                             # share of each category
cat = (dum - p) / np.sqrt(p)                                     # FAMD weighting
M = np.hstack([num, cat])

U, S, Vt = np.linalg.svd(M, full_matrices=False)                 # PCA via SVD (M is already centred)
var_ratio = S ** 2 / np.sum(S ** 2)
n_comp = int(np.searchsorted(np.cumsum(var_ratio), VAR_KEPT) + 1)
F = (U * S)[:, :n_comp]                                          # FAMD coordinates
print(f"numeric cols: {len(num_cols)} | one-hot cols: {len(cat_cols)} | FAMD components kept: {n_comp} (of {M.shape[1]})")

# ---- cluster exactly like the agglomerative baseline ----
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(F), SAMPLE, replace=False))
labels_s = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit_predict(F[idx])
labels = KNeighborsClassifier(n_neighbors=5).fit(F[idx], labels_s).predict(F)
labels[idx] = labels_s


def scores(A, lab):
    return (silhouette_score(A, lab, sample_size=10_000, random_state=SEED),
            davies_bouldin_score(A, lab), calinski_harabasz_score(A, lab))


s_own = scores(F, labels)
s_main = scores(Xb, labels)
lines = ["FAMD EXPERIMENT (ward, k=4, 10k sample, same as the agglomerative baseline)",
         f"numeric columns: {len(num_cols)} | one-hot columns: {len(cat_cols)} | components kept: {n_comp} (explain {np.cumsum(var_ratio)[n_comp - 1] * 100:.1f}% of variance)",
         f"FAMD cluster sizes: {np.bincount(labels).tolist()}",
         "",
         "1) FAMD labels scored in the FAMD space:   silhouette %.4f | davies-bouldin %.4f | calinski-harabasz %.1f" % s_own,
         "2) FAMD labels scored in the main space:   silhouette %.4f | davies-bouldin %.4f | calinski-harabasz %.1f" % s_main]

base_file = ROOT / "clustering/agglomerative/baseline/outputs/labels.npy"
if base_file.exists():
    base = np.load(base_file)
    s_base = scores(Xb, base)
    lines += ["3) baseline labels scored in the main space: silhouette %.4f | davies-bouldin %.4f | calinski-harabasz %.1f" % s_base,
              f"4) adjusted Rand index FAMD vs baseline labels: {adjusted_rand_score(base, labels):.4f}  (1 = same clusters, 0 = unrelated)",
              "",
              "Fair comparison = rows 2 and 3 (same space). Row 1 is in a different space and is not comparable with them."]
else:
    lines += ["", "Baseline labels not found, so rows 3 and 4 were skipped. Run the agglomerative baseline first."]

print("\n".join(lines))
(OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")
np.save(OUT / "labels.npy", labels)

# ---- plots ----
plt.figure(figsize=(8, 4.5))
plt.bar(range(1, len(var_ratio) + 1), var_ratio * 100, label="per component")
plt.plot(range(1, len(var_ratio) + 1), np.cumsum(var_ratio) * 100, color="red", marker="o", label="cumulative")
plt.axvline(n_comp, color="gray", linestyle="--", label=f"kept: {n_comp}")
plt.xlabel("FAMD component"); plt.ylabel("% of variance"); plt.title("FAMD scree plot"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "scree.png", dpi=150); plt.close()

pi = np.sort(rng.choice(len(F), 10_000, replace=False))
plt.figure(figsize=(7, 6))
plt.scatter(F[pi, 0], F[pi, 1], s=3, c=labels[pi], cmap="tab10")
plt.title(f"FAMD clusters (k={N_CLUSTERS}) on the first two components")
plt.xlabel("FAMD 1"); plt.ylabel("FAMD 2")
plt.tight_layout(); plt.savefig(OUT / "famd_space.png", dpi=150); plt.close()
print("saved to", OUT)
