# K-Nearest Neighbors (KNN)

## Setup

| Item | Setting |
|---|---|
| Split | 80/20, stratified, `random_state=42` |
| Cross-validation | 5-fold stratified, shuffled |
| Numeric columns | Median imputation, standard scaling |
| Text columns | Most-frequent imputation, one-hot encoding |
| `Country`, `Agent`, `Company` | Most-frequent imputation, target encoding, scaling |
| Model | Custom PyTorch KNN (`TorchKNNClassifier`), GPU if available |


## Trials

Each trial builds on the one before it. The "floor" rule picks the threshold with the highest recall while precision stays at 0.65 or more (training data only).

| # | Trial | What changed | Decision |
|---|---|---|---|
| 0 | Baseline | Defaults: `k=5`, uniform weights, Euclidean distance, no log transform, threshold 0.50 | Kept as the untuned model |
| 1 | Encoding and grid search | `Country`, `Agent`, `Company` changed from one-hot to target encoding, feature weights tuned, tuning goal changed from F1 to accuracy | Kept |
| 2 | Best-F1 threshold | Threshold chosen for the best F1 | **Rejected:** precision dropped too much |
| 3 | Precision-floor threshold | Floor rule, to increase recall | Kept: **final model** |


## Final Tuned model

| Setting | Value |
|---|---|
| `n_neighbors` (k) | 61 |
| `weights` | `distance` |
| Distance (`p`) | 1 (Manhattan) |
| Log transform | On (`LeadTime`, `ADR`) |
| Feature weights | `target` 3.0, `categorical` 0.5 |
| Threshold | 0.4444 |

## Results (test set)

| Metric | Baseline | Trial 1 | Final (trial 3) |
|---|---|---|---|
| Accuracy | 0.7707 | 0.8037 | 0.7996 |
| Precision | 0.6346 | 0.7109 | 0.6669 |
| Recall | 0.5591 | 0.5847 | 0.6656 |
| F1 | 0.5945 | 0.6416 | 0.6662 |
| ROC-AUC | 0.7903 | 0.8556 | 0.8556 |
| PR-AUC | 0.5942 | - | 0.7330 |

## Final Gain (test set, tuned vs. baseline)

| Metric | Change |
|---|---|
| recall | **+0.107** |
| F1 | **+0.072** |
| PR-AUC | **+0.139** |
| ROC-AUC | **+0.065** |
| accuracy | **+0.029** |
| precision | **+0.032** |
