"""K-Means TUNED (Task C, clustering).

Input : data/processed/h2_clustering_stage5_frequency.csv  (12 columns: frequency-encoded categories, month as sin/cos)
Grid  : k = 2..10  x  init in {k-means++, random}  x  n_init in {10, 20}   (36 configurations, seed 42)
Choice: unsupervised only. A configuration is VALID if
          - the smallest cluster has >= 2% of the rows,
          - it is stable: mean ARI >= 0.8 over 5 other seeds,
          - it is not a trivial split: ARI < 0.9 against every single original column (a group-by is not a segmentation).
        Winner = highest silhouette (10k sample) among valid configurations (ties within 0.001: k-means++, then lowest Davies-Bouldin).
Also fits the baseline (k=3, k-means++, n_init=10) on the same input so baseline vs tuned is a fair comparison.
Run from the project root:  python clustering/kmeans/tuned/run.py
Writes to clustering/kmeans/tuned/outputs/: results.txt, grid.csv, labels.npy, k_curves.png, pca.png,
                                            kmeans_tuned_model.joblib
"""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score)

SEED, SAMPLE = 42, 10_000
K_RANGE = range(2, 11)
INITS = ["k-means++", "random"]
N_INITS = [10, 20]
STAB_SEEDS = [0, 1, 2, 3, 4]
MIN_SHARE, MIN_STAB, MAX_TRIVIAL = 2.0, 0.8, 0.9

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

X = pd.read_csv(ROOT / "data/processed/h2_clustering_stage5_frequency.csv").to_numpy(dtype=float)
stage3 = pd.read_csv(ROOT / "data/processed/h2_clustering_stage3.csv")      # original columns, same row order
stage2 = pd.read_csv(ROOT / "data/processed/h2_clustering_stage2.csv")      # untransformed values, for the profiles
assert len(X) == len(stage3) == len(stage2)
sil_idx = np.sort(np.random.default_rng(SEED).choice(len(X), SAMPLE, replace=False))
TRIVIAL_COLS = ["Adults", "MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType", "HasChildren"]


def fit(k, init, n_init, seed=SEED):
    return KMeans(n_clusters=k, init=init, n_init=n_init, random_state=seed).fit(X)


def evaluate(k, init, n_init, tag="grid"):
    t0 = time.perf_counter()
    km = fit(k, init, n_init)
    secs = time.perf_counter() - t0
    lab = km.labels_
    shares = np.bincount(lab, minlength=k) / len(lab) * 100
    stab = float(np.mean([adjusted_rand_score(lab, fit(k, init, n_init, s).labels_) for s in STAB_SEEDS]))
    closest = max(TRIVIAL_COLS, key=lambda c: adjusted_rand_score(stage3[c], lab))
    triv = float(adjusted_rand_score(stage3[closest], lab))
    row = dict(set=tag, k=k, init=init, n_init=n_init,
               silhouette=round(float(silhouette_score(X[sil_idx], lab[sil_idx])), 4),
               davies_bouldin=round(float(davies_bouldin_score(X, lab)), 4),
               calinski_harabasz=round(float(calinski_harabasz_score(X, lab)), 1),
               inertia=round(float(km.inertia_), 1), smallest_pct=round(float(shares.min()), 2),
               stability_ari=round(stab, 4), closest_column=closest, closest_column_ari=round(triv, 3),
               seconds=round(secs, 1))
    row["valid"] = bool(row["smallest_pct"] >= MIN_SHARE and stab >= MIN_STAB and triv < MAX_TRIVIAL)
    return row, km


print(f"K-Means tuned on {X.shape[0]} rows x {X.shape[1]} columns, 36 configurations ...")
rows, models = [], {}
for k in K_RANGE:
    for init in INITS:
        for n_init in N_INITS:
            row, km = evaluate(k, init, n_init)
            rows.append(row); models[(k, init, n_init)] = km
            print(f"  k={k:2d} {init:9s} n_init={n_init:2d}  silhouette={row['silhouette']:.4f}  "
                  f"stab={row['stability_ari']:.2f}  smallest={row['smallest_pct']:5.1f}%  "
                  f"closest={row['closest_column']}({row['closest_column_ari']:.2f})  valid={row['valid']}")

grid = pd.DataFrame(rows)
base_row, base_km = evaluate(3, "k-means++", 10, tag="baseline")
pd.concat([grid, pd.DataFrame([base_row])]).to_csv(OUT / "grid.csv", index=False)

pool = grid[grid.valid]
if pool.empty:
    print("No valid configuration - relax MIN_SHARE / MIN_STAB / MAX_TRIVIAL. Falling back to best silhouette overall.")
    pool = grid
near = pool[pool.silhouette >= pool.silhouette.max() - 0.001]          # silhouette ties within 0.001: prefer k-means++, then lowest Davies-Bouldin
best = near.assign(plus=(near.init != "k-means++")).sort_values(["plus", "davies_bouldin"]).iloc[0]
key = (int(best.k), best.init, int(best.n_init))
km = models[key]
labels = km.labels_
np.save(OUT / "labels.npy", labels)
joblib.dump(km, OUT / "kmeans_tuned_model.joblib")

# ---- profiles (describe the clusters in original units) ----
rawp = stage2.assign(cluster=labels, Nights=stage2["StaysInWeekNights"] + stage2["StaysInWeekendNights"],
                     HasChildren=(stage2["Children"] > 0).astype(int))
prof = rawp.groupby("cluster").agg(rows=("cluster", "size"), lead_time=("LeadTime", "mean"), nights=("Nights", "mean"),
                                   adults=("Adults", "mean"), pct_children=("HasChildren", "mean"))
prof["pct_children"] *= 100
prof.insert(1, "share_pct", prof["rows"] / len(labels) * 100)
for c in ["MarketSegment", "Country", "Meal", "CustomerType", "ReservedRoomType"]:
    prof["top_" + c] = rawp.groupby("cluster")[c].agg(lambda s: s.value_counts().index[0])

# ---- figure 1: the decision graph (silhouette and inertia vs k) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
for (init, n_init), g in grid.groupby(["init", "n_init"]):
    a1.plot(g.k, g.silhouette, marker="o", label=f"{init}, n_init={n_init}", alpha=.8)
    a2.plot(g.k, g.inertia, marker="o", alpha=.8)
bad = grid[~grid.valid]
a1.scatter(bad.k, bad.silhouette, s=90, facecolors="none", edgecolors="#C8433B", linewidths=1.5, label="not valid", zorder=5)
a1.scatter([best.k], [best.silhouette], s=260, marker="*", color="#2E8B57", zorder=6, label=f"chosen: k={int(best.k)}")
a2.axvline(best.k, color="#2E8B57", ls="--", lw=1)
a1.set(xlabel="k (number of clusters)", ylabel="silhouette (higher = better)", title="Silhouette per k")
a2.set(xlabel="k (number of clusters)", ylabel="inertia (lower = tighter)", title="Elbow: inertia per k")
a1.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "k_curves.png", dpi=140); plt.close(fig)

# ---- figure 2: clusters in 2-D PCA ----
pca = PCA(n_components=2, random_state=SEED).fit(X)
xy, cen = pca.transform(X), pca.transform(km.cluster_centers_)
show = np.random.default_rng(SEED).choice(len(X), 15000, replace=False)
fig, ax = plt.subplots(figsize=(9, 6.5))
for c in range(km.n_clusters):
    m = labels[show] == c
    ax.scatter(xy[show][m, 0], xy[show][m, 1], s=8, alpha=.35, label=f"Cluster {c} ({(labels == c).mean() * 100:.1f}%)")
ax.scatter(cen[:, 0], cen[:, 1], s=200, marker="X", color="black", zorder=5)
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% of variance)",
       ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% of variance)",
       title=f"K-Means tuned: k={key[0]}, init={key[1]}, n_init={key[2]}  (X = centroids)")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "pca.png", dpi=140); plt.close(fig)

# ---- summary ----
fmt = lambda r: (f"silhouette {r['silhouette']} | davies-bouldin {r['davies_bouldin']} | calinski-harabasz {r['calinski_harabasz']} | "
                 f"stability ARI {r['stability_ari']} | smallest cluster {r['smallest_pct']}%")
lines = ["K-MEANS: BASELINE vs TUNED (same input: frequency-encoded, 12 columns)", "",
         f"baseline  k=3, init=k-means++, n_init=10 : {fmt(base_row)}",
         f"tuned     k={key[0]}, init={key[1]}, n_init={key[2]} : {fmt(best)}",
         f"          closest single column: {best.closest_column} (ARI {best.closest_column_ari}) - below {MAX_TRIVIAL}, not a copy of one column", "",
         f"valid configurations: {int(grid.valid.sum())} of {len(grid)}",
         f"search space: k={K_RANGE.start}..{K_RANGE.stop - 1}, init={INITS}, n_init={N_INITS}; criterion: silhouette on a 10k sample, no labels used",
         f"validity rules: smallest cluster >= {MIN_SHARE}%, stability ARI >= {MIN_STAB}, closest-column ARI < {MAX_TRIVIAL}", "",
         "cluster sizes: " + json.dumps({int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()}),
         "", "Cluster profiles (original units):", prof.round(1).to_string()]
(OUT / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n" + "\n".join(lines))
print("\nSaved to", OUT)
