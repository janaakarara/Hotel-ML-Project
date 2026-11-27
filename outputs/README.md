# outputs: what each figure and table shows

Names start with the step that made them. Models write their own files next to their script (clustering/<model>/<baseline or tuned>/outputs), and the encoding comparison is in src/04_encoding_comparison/outputs.

| Step | Figure (png) | Table (csv) | Shows / decision it supports |
|---|---|---|---|
| 3 | stage2a_constancy | stage2a_constancy | share of zeros per column -> drop near-constant columns |
| 3 | stage2a_cramers_v | stage2a_cramers_v | category pairs that carry the same information |
| 3 | stage2a_eta | stage2a_eta | numeric column explained by a category |
| 3 | stage2a_spearman | stage2a_spearman | numeric pairs that move together |
| 3 | - | stage2a_flagged_pairs, stage2_drop_decisions.md | pairs above the threshold, final drop list |
| 4 | stage3a_numeric_skew | stage3a_numeric_skew | which transform gives the lowest skew per column |
| 4 | stage3a_categorical_shares | stage3a_rare_categories | rare categories to merge |
| 4 | stage3c_categories_after_grouping | stage3c_category_shares | category shares after merging |
| 4 | stage3c_country_group_profile | - | do the 4 Country groups differ? |
| 5 | stage4a_scaling_distributions | stage4a_scaling_summary | each column under each scaler |
| 5 | stage4a_scaling_spread | stage4a_scaling_ranking | which scaler wins |
| 6 | stage5a_balance_current | stage5a_balance_current | share of the distance each feature group controls |
| models | clustering_gmm_untuned_* , clustering_gmm_tuned_* | same names | GMM only: 2D PCA view, cluster profiles, covariance check (copies also in clustering/gmm/*/outputs) |
| models | clustering_gmm_tuning_curves, clustering_gmm_k_comparison | clustering_gmm_tuning_search, _bic_choice, _shortlist, _comparison | GMM tuning: choice of k and covariance type, untuned vs tuned |
| models | clustering_gmm_main_*, clustering_gmm_sub_* | clustering_gmm_sub_*, _main_feature_means, _segments | extra GMM analysis (gmm_subclusters.py) |
