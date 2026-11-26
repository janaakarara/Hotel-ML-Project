# FAMD experiment

Question: does handling the mixed (numeric + categorical) features with FAMD give better clusters
than the main preprocessing (z-score + one-hot x 0.5)?

## What FAMD does, in simple words

    numeric columns      -> standardized (mean 0, std 1)
    one-hot columns      -> centred, then divided by sqrt(share of that category)
                            (rare categories get a louder voice, so they are not drowned out)
    then PCA on all of it -> a few components that mix numeric and categorical information
    then cluster in that component space

## How it is run here

- Implemented by hand with numpy (the standard FAMD weighting), so nothing extra to install.
- Input: data/processed/h2_clustering_stage5.csv. The 0.5 category weight is undone first
  (FAMD does its own weighting), so one-hot columns go back to plain 0/1.
- Components kept: enough to explain 80% of the variance (VAR_KEPT in run.py).
- Clustering: same as the agglomerative baseline (ward, k=4, same 10k sample, seed 42, rest labelled by 5-NN),
  so the only thing that changes is the space the points live in.

## How to compare fairly

Scores computed in different spaces are not comparable with each other. So results.txt reports:
1. FAMD labels scored in the FAMD space (their own space)
2. FAMD labels scored in the main stage5 space
3. The baseline labels scored in that same stage5 space (reference)
4. Adjusted Rand Index between FAMD labels and baseline labels (1 = same clusters, 0 = unrelated)

Rows 2 and 3 are the fair comparison. Run:

    python experiments/famd_experiment/run.py

Write the conclusion at the bottom of this file after reading outputs/results.txt.
