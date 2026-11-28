# Hotel bookings (City Hotel H2): clustering part

## Environment setup

Repository: https://github.com/janaakarara/Hotel-ML-Project (branch clustering-fresh)

    python -m venv .venv
    .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux/macOS)
    pip install -r requirements.txt

Put H2.csv in data/raw/ before step 1.

Run everything from the project root, in this order. Every script has working defaults, so each command is just `python <file>`.

| # | Step | Command | Makes | Decision based on |
|---|---|---|---|---|
| 1 | Clean raw data | `python src/01_cleaning/clean.py` | data/processed/h2_clean.csv | data/processed/cleaning_log.csv |
| 2 | Remove outcome/future columns + zero-adult rows | `python src/02_preprocessing/stage1_drop_future.py` | ...stage1.csv | dataset paper (no figure) |
| 3 | Drop redundant columns | `python src/02_preprocessing/stage2a_redundancy_check.py` then `stage2b_drop_columns.py` | ...stage2.csv | outputs/02_redundancy/stage2a_constancy, cramers_v, eta |
| 4 | Group categories, fix skew | `stage3a_distributions_check.py`, `stage3b_group_and_fix_skew.py`, `stage3c_categories_after_grouping.py` (all in src/02_preprocessing) | ...stage3.csv | stage3a_categorical_shares, stage3a_numeric_skew |
| 5 | Scale numbers | `stage4a_scaling_check.py`, `stage4b_apply_scaling.py` | ...stage4.csv | stage4a_scaling_spread, stage4a_scaling_distributions |
| 6 | Encode categories | `python src/03_encoding/stage5_encode.py --encoding frequency` (add nothing for one-hot) | ...stage5_frequency.csv | step 7 below |
| 7 | Compare encodings, all models | `python experiments/comparing_experiments/run_all_models.py` then `run_grid.py` | experiments/comparing_experiments/outputs | all_models_encoding_comparison.png, grid/grid_final_comparison.png |
| 8 | Models (baseline, then tuned) | `python clustering/<kmeans or gmm or agglomerative or dbscan>/<baseline or tuned>/run.py` | clustering/*/outputs | per-model figures in each outputs folder |
| 9 | Cluster profiles and comparison table | `python clustering/make_profiles.py` then `python clustering/compare_all.py` | clustering/comparison.csv, cluster_profiles.csv | silhouette on one common ruler |

Other folders: `src/features` = reusable preprocessing object for the app, `experiments/comparing_experiments` = the encoding comparison for every algorithm (step 7); other side experiments live in `experiments`.

Known limitations: cluster structure in this data is weak (best silhouette about 0.17); the tuned DBSCAN has no valid segmentation; Agglomerative is fitted on a 10k sample. Details are in clustering/README.md and the clustering report.

File guide: data/processed/README.md (datasets) and outputs/README.md (figures and tables).





## Final audit (clustering)
- `python experiments/audit_check.py` re-derives cluster counts, noise, Davies-Bouldin and Calinski-Harabasz from each labels.npy and checks them against clustering/comparison.csv (77 checks, 0 failed).
- K-Means baseline silhouette is 0.1549 in its own results.txt (10k sample) and 0.1563 in comparison.csv (common ruler); both are correct for their sample.
- DBSCAN baseline: its results.txt reports 0.2814 (computed by the baseline script itself); comparison.csv re-scores every model on the same ruler with noise counted as its own group, which gives -0.1934 (47% noise, 2,098 clusters). Use the comparison.csv value when ranking models.

- Reproduce the rejected-idea tests: `python experiments/fix_test.py` (HasChildren weights, Meal ordinal) and `python experiments/improve_test.py` (drop-one-column, PCA, derived features, richer profiles). Outputs: experiments/fix_test_outputs and experiments/improve_test_outputs.
Submitted commit SHA (code state): 22fc90f
