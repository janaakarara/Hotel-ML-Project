# outputs: where each figure and table is

One folder per pipeline step. The figure (png) and the table (csv) of the same analysis sit side by side
and share a name. Model results are not here: each model writes next to its own script
(`clustering/<model>/<baseline or tuned>/outputs`), and the encoding comparison is in
`src/04_encoding_comparison/outputs`.

| Folder | Made by | What it shows / decision it supports |
|---|---|---|
| `02_redundancy/` | `src/02_preprocessing/stage2a_redundancy_check.py`, `stage2b_drop_columns.py` | `stage2a_constancy` (share of zeros -> drop near-constant columns), `stage2a_cramers_v` (category pairs carrying the same information), `stage2a_eta` (numeric column explained by a category), `stage2a_spearman` (numeric pairs that move together), `stage2a_flagged_pairs.csv`, `stage2_drop_decisions.md` (final drop list) |
| `03_skew_and_categories/` | `stage3a_distributions_check.py`, `stage3c_categories_after_grouping.py` | `stage3a_numeric_skew` (best transform per column), `stage3a_categorical_shares` + `stage3a_rare_categories.csv` (rare categories to merge), `stage3c_categories_after_grouping` + `stage3c_category_shares.csv` (shares after merging), `stage3c_country_group_profile` (do the 4 Country groups differ?) |
| `04_scaling/` | `stage4a_scaling_check.py` | `stage4a_scaling_distributions` + `stage4a_scaling_summary.csv` (each column under each scaler), `stage4a_scaling_spread` + `stage4a_scaling_ranking.csv` (which scaler wins) |
| `05_encoding_balance/` | `src/03_encoding/stage5a_feature_balance_check.py` | `stage5a_balance_<tag>` (share of the distance each feature group controls) |
| `gmm/` | `clustering/gmm/baseline/run.py`, `clustering/gmm/tuned/run.py` | `clustering_gmm_untuned_*` (baseline), `clustering_gmm_tuned_*` (tuned model), `clustering_gmm_tuning_*` and `clustering_gmm_k_comparison` (search for k and covariance type), `clustering_gmm_comparison.csv` (untuned vs tuned) |
| `gmm/subclusters/` | `experiments/gmm_subclusters.py` | side analysis: cluster means and sub-groups of the largest GMM cluster (not part of the main pipeline) |

Naming: `stageNx_<what>` = pipeline step N, part x. Run order is in the root `README.md`.
