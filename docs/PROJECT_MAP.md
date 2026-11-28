# Project map

```
Hotel-ML-Project/  [3 files]  - CSE382 City Hotel (H2) project  { .gitignore, README.md, requirements.txt }
|-- artifacts/  [2 files]  - saved preprocessing params (models stay local)  { clustering_preprocessor.joblib, clustering_preprocessor_params.json }
|-- clustering/  [5 files]  - 4 algorithms x baseline/tuned, common ruler, comparison.csv  { README.md, common.py, compare_all.py, comparison.csv, make_profiles.py }
|   |-- agglomerative/  [0 files]
|   |   |-- baseline/  [1 files]  { run.py }
|   |   |   `-- outputs/  [5 files]  { cluster_profiles.csv, dendrogram.png, labels.npy, pca.png, results.txt }
|   |   `-- tuned/  [1 files]  { run.py }
|   |       `-- outputs/  [8 files]
|   |-- dbscan/  [0 files]
|   |   |-- baseline/  [1 files]  { run.py }
|   |   |   `-- outputs/  [5 files]  { cluster_profiles.csv, k_distance.png, labels.npy, pca.png, results.txt }
|   |   `-- tuned/  [1 files]  { run.py }
|   |       `-- outputs/  [6 files]  { cluster_profiles.csv, k_distance.png, labels.npy, pca.png, results.txt, search_table.csv }
|   |-- gmm/  [0 files]
|   |   |-- baseline/  [1 files]  { run.py }
|   |   |   `-- outputs/  [11 files]
|   |   `-- tuned/  [1 files]  { run.py }
|   |       `-- outputs/  [17 files]
|   `-- kmeans/  [0 files]
|       |-- baseline/  [1 files]  { run.py }
|       |   `-- outputs/  [7 files]
|       `-- tuned/  [1 files]  { run.py }
|           `-- outputs/  [7 files]
|-- data/  [0 files]  - raw input + every stage output
|   |-- processed/  [21 files]  - stage CSVs, *_decisions.md, *_params.json
|   `-- raw/  [1 files]  - H2.csv, untouched  { H2.csv }
|-- docs/  [1 files]  - explainer pages + this map  { pipeline_explained.html }
|-- experiments/  [1 files]  - side studies, not the main pipeline  { README.md }
|   |-- audit/  [1 files]  - audit_check.py + report  { audit_check.py }
|   |   `-- audit_output/  [1 files]  { audit_report.txt }
|   |-- comparing_experiments/  [3 files]  - encoding contest  { grid_common.py, run_all_models.py, run_grid.py }
|   |   `-- outputs/  [3 files]  { all_models_encoding_comparison.png, all_models_results.csv, all_models_winners.txt }
|   |       `-- grid/  [6 files]  { grid_final_comparison.png, grid_final_table.csv, grid_heatmaps.png, grid_k_curves.png, grid_results.csv, grid_winners.txt }
|   `-- rejected_ideas/  [2 files]  - fix_test + improve_test and their outputs  { fix_test.py, improve_test.py }
|       |-- fix_test_outputs/  [3 files]  { fix_test_chart.png, fix_test_results.csv, fix_test_summary.txt }
|       `-- improve_test_outputs/  [5 files]  { cluster_profiles_rich.csv, improve_test_chart.png, improve_test_results.csv, improve_test_summary.txt, profile_heatmap.png }
|-- outputs/  [1 files]  - figures that justify preprocessing decisions  { README.md }
|   |-- 02_redundancy/  [10 files]
|   |-- 03_skew_and_categories/  [7 files]
|   |-- 04_scaling/  [4 files]  { stage4a_scaling_distributions.png, stage4a_scaling_ranking.csv, stage4a_scaling_spread.png, stage4a_scaling_summary.csv }
|   `-- 05_encoding_balance/  [2 files]  { stage5a_balance_current.csv, stage5a_balance_current.png }
`-- src/  [0 files]  - pipeline scripts in running order
    |-- 01_cleaning/  [1 files]  - raw -> h2_clean.csv  { clean.py }
    |-- 02_preprocessing/  [8 files]  - stages 1-4: leakage, redundancy, skew/groups, scaling
    |-- 03_encoding/  [3 files]  - stage 5 frequency encoding  { stage5_encode.py, stage5a_feature_balance_check.py, stage5b_pca.py }
    `-- features/  [4 files]  - reusable pipeline for the app  { __init__.py, clustering_pipeline.py, country_groups.py, fit_clustering_pipeline.py }
```
