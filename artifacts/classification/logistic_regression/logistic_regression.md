# Logistic Regression

## Setup 

| Item | Setting |
|---|---|
| Cross-validation | 5-fold stratified, shuffled |
| Numeric columns | Median imputation, standard scaling |
| Text columns | Most-frequent imputation, one-hot encoding |
| `Country`, `Agent`, `Company` | Most-frequent imputation, target encoding, scaling |
| Solver | `lbfgs`, L2 regularization, `max_iter=1000`, CPU |


## Trials

Each trial builds on the one before it. The "floor" rule picks the threshold with the highest recall while precision stays at 0.65 or more (training data only).

| # | Trial | What changed | Decision |
|---|---|---|---|
| 0 | Baseline | Defaults: `C=1`, no class weight, threshold 0.50 | Kept as the untuned model |
| 1 | Balanced + threshold | `class_weight="balanced"`, floor rule | Kept |
| 2 | Grid search | `C` from 0.01 to 100, log transform on/off (`LeadTime`, `ADR`), scored by PR-AUC | Kept |
| 3 | Wider `C` | Added 1000 and 10000 | No real change |
| 4 | Splines | Off, 5 or 10 knots on `LeadTime`, `ADR` | Kept (5 knots) |
| 5 | Interactions | Pairs from `LeadTime`, `ADR`, `TotalOfSpecialRequests`, `MarketSegment`, `DepositType`, `CustomerType` | Kept: **final model** |
| 6 | Accuracy threshold | Threshold chosen for best accuracy instead of the floor | **Rejected:** too much recall lost |


## Final Tuned model

| Setting | Value |
|---|---|
| `class_weight` | `balanced` |
| `C` | 1000 |
| Log transform | On |
| Spline knots | 5 |
| Interactions | On |
| Threshold | 0.6364 |

## Final Gain

| Setting | Increase|
|---|---|
|recall| **+0.126**|
|F1 |**+0.062**|
|PR-AUC |**+0.021**| 
|ROC-AUC |**+0.018**|
|accuracy |**+0.005**|
|precision |**-0.043**|

