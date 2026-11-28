# Clustering experiments

Input for every model: data/processed/h2_clustering_stage5_frequency.csv (52,817 rows x 12 columns:
frequency-encoded categories, month as sin/cos, numeric columns standardized). No PCA is used for fitting.
PCA is used only to draw the 2-D cluster plots.

Layout (each model has baseline/ and tuned/, each with run.py and outputs/):

    clustering/
        common.py          shared input, fixed 10k-row silhouette sample, single-column check, results writer
        compare_all.py     re-scores every labels.npy on the same ruler -> comparison.csv
        make_profiles.py   cluster_profiles.csv for every labels.npy (ADR and IsCanceled only describe clusters afterwards)
        kmeans/  agglomerative/  dbscan/  gmm/

Run from the project root, in this order:

    python clustering/kmeans/baseline/run.py
    python clustering/kmeans/tuned/run.py
    python clustering/agglomerative/baseline/run.py
    python clustering/agglomerative/tuned/run.py
    python clustering/dbscan/baseline/run.py
    python clustering/dbscan/tuned/run.py
    python clustering/gmm/baseline/run.py
    python clustering/gmm/tuned/run.py
    python clustering/make_profiles.py
    python clustering/compare_all.py

Rules used everywhere: seed 42; no IsCanceled or ADR in any choice; silhouette for comparison is computed on the
same fixed 10k-row sample in the 12-D space, with DBSCAN noise counted as its own group.
A tuned configuration is rejected when it is a copy of one input column (ARI >= 0.9 or NMI >= 0.5 against any
column, month included) or has a cluster under 2% of rows. Agglomerative and GMM also need stability ARI >= 0.8;
if none reaches it, the most stable one is reported with a WARNING in results.txt.
These thresholds were chosen by the team, not taken from a reference.

Each outputs/ folder: results.txt, labels.npy (-1 = DBSCAN noise), cluster_profiles.csv, plots.
Trained model files (.joblib) are ignored by git and must not be in the submission ZIP.
