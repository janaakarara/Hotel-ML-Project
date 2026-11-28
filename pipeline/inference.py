from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from config import TARGET_CLASSIFICATION
from pipeline.preprocess import preprocess


def predict_uploaded_data(
    raw: pd.DataFrame, model: Pipeline, threshold: float
) -> tuple[pd.DataFrame, dict[str, Any] | None, dict[str, Any]]:
    """Preprocess uploaded rows and return predictions and optional metrics."""
    if not 0 <= threshold <= 1:
        raise ValueError("The decision threshold must be between 0 and 1.")

    processed, summary = preprocess(raw)
    if processed.empty:
        raise ValueError("No rows remain after preprocessing.")

    expected_features = list(model.feature_names_in_)
    features = processed.drop(columns=[TARGET_CLASSIFICATION], errors="ignore")
    missing = sorted(set(expected_features) - set(features.columns))
    if missing:
        raise ValueError(
            "The uploaded CSV is missing model features: "
            f"{', '.join(missing)}"
        )

    features = features.loc[:, expected_features]
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError("The selected model does not support class 1.")

    positive_probability = model.predict_proba(features)[:, classes.index(1)]
    predicted_class = (positive_probability >= threshold).astype(int)
    retained_rows = ~raw["DistributionChannel"].eq("Undefined").fillna(False)
    source_rows = np.flatnonzero(retained_rows.to_numpy()) + 1

    predictions = pd.DataFrame(
        {
            "Input row": source_rows,
            "Prediction": np.where(
                predicted_class == 1, "Canceled", "Not canceled"
            ),
            "Cancellation probability": positive_probability,
        }
    )

    uploaded_metrics = None
    if TARGET_CLASSIFICATION in processed.columns:
        actual = pd.to_numeric(
            processed[TARGET_CLASSIFICATION], errors="coerce"
        )
        if actual.isna().any() or not actual.isin([0, 1]).all():
            raise ValueError(
                f"If present, {TARGET_CLASSIFICATION} must contain only "
                "binary values 0 and 1."
            )

        predictions["Actual"] = np.where(
            actual.to_numpy() == 1, "Canceled", "Not canceled"
        )
        uploaded_metrics = {
            "accuracy": float(accuracy_score(actual, predicted_class)),
            "precision": float(
                precision_score(actual, predicted_class, zero_division=0)
            ),
            "recall": float(
                recall_score(actual, predicted_class, zero_division=0)
            ),
            "f1_score": float(
                f1_score(actual, predicted_class, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                actual, predicted_class, labels=[0, 1]
            ).tolist(),
        }
        if set(actual.unique()) == {0, 1}:
            uploaded_metrics["roc_auc"] = float(
                roc_auc_score(actual, positive_probability)
            )
            uploaded_metrics["pr_auc"] = float(
                average_precision_score(actual, positive_probability)
            )

    return predictions, uploaded_metrics, summary
