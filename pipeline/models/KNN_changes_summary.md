# KNN Model: Major Changes

## What changed

| Change | Before | After |
|---|---|---|
| Random seed | 67 | 42 |
| `Country`, `Agent`, `Company` | One-hot encoding | Target encoding |
| Feature weights | Not tuned | Tuned |
| Tuning goal | F1 | Accuracy |
| Saved models | Tuned only | Baseline and tuned |

## Final settings

| Model | Settings |
|---|---|
| Baseline | k=5, uniform weights, Euclidean distance, no log transform |
| Tuned | k=61, distance weights, Manhattan distance, log transform, `target` weight 3.0, `categorical` weight 0.5 |

## Final results

| Metric | Baseline | Tuned |
|---|---|---|
| CV accuracy | 76.84% | **80.31%** |
| Test accuracy | 77.07% | **80.37%** |
| Test precision | 63.46% | 71.09% |
| Test recall | 55.91% | 58.47% |
| Test F1 | 0.5945 | 0.6416 |
| Test ROC-AUC | 0.7903 | 0.8556 |
