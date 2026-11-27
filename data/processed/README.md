# data/processed: what each file is

Files are named after the step that made them. Read top to bottom = pipeline order.

| Step | File | What it is |
|---|---|---|
| 1 | cleaning_log.csv | one row per cleaning rule: rows before and after |
| 1 | column_summary.csv, consistency_checks.json | column profile and sanity checks on the raw data |
| 1 | h2_clean.csv | cleaned data, 53,036 rows |
| 2 | clustering_stage1_decisions.md | which columns and rows were removed, and why |
| 2 | h2_clustering_stage1.csv | 52,817 rows x 22 columns |
| 2 | h2_clustering_profile_cols.csv | removed columns (ADR, IsCanceled ...), kept only to describe clusters afterwards |
| 3 | stage2_drop_decisions.md | why 11 more columns were dropped |
| 3 | h2_clustering_stage2.csv | 52,817 x 11, original untransformed values (used for cluster profiles) |
| 4 | stage3_decisions.md | Country 167 -> 4 groups, rare merges, skew fixes |
| 4 | stage3_transform_params.json | fitted Yeo-Johnson / sqrt settings |
| 4 | stage3b_country_group_profile.csv | how the 4 Country groups differ |
| 4 | h2_clustering_stage3.csv | grouped and de-skewed, categories still text |
| 5 | stage4_scaler_params.json | mean and std used for scaling |
| 5 | h2_clustering_stage4.csv | scaled numbers, categories still text. INPUT of every encoding |
| 6 | h2_clustering_stage5.csv + stage5_encoding_columns.json | one-hot, 29 columns (category weight is stored in the json, currently 0.5) |
| 6 | h2_clustering_stage5_frequency.csv + stage5_frequency_maps.json | frequency, 12 columns, month as sin/cos (recommended) |
| 6 | h2_clustering_stage5_freq.csv + stage5_freq_encoding.json | frequency, older version, 11 columns, month as raw 1-12. Feeds the PCA file |
| 7 | h2_clustering_pca90.csv + pca90_metadata.json | PCA keeping 90% variance, 8 columns |
| - | stage6_decisions.md, stage6_weights.json, stage6_row_index.csv | idea to split family / non-family. No script, not used by any model |
