# Experiments

comparing_experiments/   Encoding comparison for K-Means, GMM, Agglomerative and DBSCAN (plus HDBSCAN and K-Prototypes in the grid).
                         It decided that every clustering model uses the frequency-encoded matrix.
    run_all_models.py    ranks every encoding for each algorithm -> outputs/all_models_winners.txt, all_models_results.csv
    run_grid.py          wider grid with PCA variants          -> outputs/grid/
    grid_common.py       shared code

Run from the project root:
    python experiments/comparing_experiments/run_all_models.py
    python experiments/comparing_experiments/run_grid.py
