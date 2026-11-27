# Hotel bookings (City Hotel H2): clustering part

Run everything from the project root, in this order. Every script has working defaults, so each command is just `python <file>`.

| # | Step | Command | Makes | Decision based on |
|---|---|---|---|---|
| 1 | Clean raw data | `python src/01_cleaning/clean.py` | data/processed/h2_clean.csv | data/processed/cleaning_log.csv |
| 2 | Remove outcome/future columns + zero-adult rows | `python src/02_preprocessing/stage1_drop_future.py` | ...stage1.csv | dataset paper (no figure) |
| 3 | Drop redundant columns | `python src/02_preprocessing/stage2a_redundancy_check.py` then `stage2b_drop_columns.py` | ...stage2.csv | outputs/figures/stage2a_constancy, cramers_v, eta |
| 4 | Group categories, fix skew | `stage3a_distributions_check.py`, `stage3b_group_and_fix_skew.py`, `stage3c_categories_after_grouping.py` (all in src/02_preprocessing) | ...stage3.csv | stage3a_categorical_shares, stage3a_numeric_skew |
| 5 | Scale numbers | `stage4a_scaling_check.py`, `stage4b_apply_scaling.py` | ...stage4.csv | stage4a_scaling_spread, stage4a_scaling_distributions |
| 6 | Encode categories | `python src/03_encoding/stage5_encode.py --encoding frequency` (add nothing for one-hot) | ...stage5_frequency.csv | step 7 below |
| 7 | Compare encodings, all models | `python src/04_encoding_comparison/run_all_models.py` then `run_grid.py` | src/04_encoding_comparison/outputs | all_models_encoding_comparison.png, grid/grid_final_comparison.png |
| 8 | Models (baseline, then tuned) | `python clustering/<kmeans or gmm or agglomerative or dbscan>/<baseline or tuned>/run.py` | clustering/*/outputs | per-model figures in each outputs folder |

Other folders: `src/features` = reusable preprocessing object for the app, `experiments` = side experiments.

Known issues (see chat notes): src/03_encoding/stage5_freq.py + stage5b_pca.py keep the month as raw 1-12 (unscaled), so the PCA file is dominated by month; baseline and tuned models read different input files; tests/ and two experiments import a module (src.clustering.gmm_untuned) that no longer exists.
