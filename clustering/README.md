# Clustering experiments

Input: data/processed/h2_clustering_stage5.csv (52,817 rows x 29 features, one-hot columns weighted 0.5).
Every script finds the project root by itself, so you can run it from any folder.

Layout:

    clustering/
        agglomerative/
            baseline/   run.py + outputs/   untuned: ward, euclidean, k=4, 10k sample
            tuned/      <- friend adds here
        dbscan/
            baseline/   run.py + outputs/   untuned: sklearn defaults (eps=0.5, min_samples=5), all rows
            tuned/      <- friend adds here

## Run (from the project root)

    python clustering/agglomerative/baseline/run.py
    python clustering/dbscan/baseline/run.py

## What each outputs/ folder holds (tuned versions should match, so the two are easy to compare)

| File | Content |
|---|---|
| results.txt | settings, cluster sizes, noise %, silhouette, Davies-Bouldin, Calinski-Harabasz |
| labels.npy | one cluster label per row, same order as the CSV (-1 = DBSCAN noise) |
| pca.png | clusters drawn in 2D PCA space |
| dendrogram.png | agglomerative only |
| k_distance.png | DBSCAN only |

Tuned folders should also save their search table as grid.csv.
Rules used everywhere: seed 42, no labels like IsCanceled used for any choice,
silhouette on a 10k subsample, DBSCAN scores computed on non-noise points only (report noise % next to them).
