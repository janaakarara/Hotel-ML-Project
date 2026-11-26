
from __future__ import annotations
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import ARTIFACTS_DIR, DATA_FILE, RANDOM_STATE, TARGET_CLASSIFICATION
from pipeline.preprocess import preprocess


TEST_SIZE = 0.20
CV_FOLDS = 5
N_ESTIMATORS = 500
MODEL_NAME = "random_forest_classifier_optimized_0.3"

# --- Random forest hyperparameters ---
CLASS_WEIGHT = None         
MAX_FEATURES = "sqrt"       
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 2         

# --- Threshold selection ---
# "equal"            precision and recall as close as possible
# "f1"               threshold with the best F1 for the Canceled class
# "precision_target" highest recall that still reaches TARGET_PRECISION
THRESHOLD_MODE = "equal"
TARGET_PRECISION = 0.75      # only used when THRESHOLD_MODE == "precision_target"


def build_model_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=CLASS_WEIGHT,
        verbose=0,
        max_features=MAX_FEATURES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def choose_threshold(
    y_true: pd.Series,
    probability: np.ndarray,
    mode: str,
    target_precision: float,
) -> tuple[float, float, float]:
    """Return (threshold, precision, recall) picked on the given predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds

    if mode == "f1":
        score = 2 * precision * recall / (precision + recall + 1e-12)
        idx = int(np.argmax(score))
    elif mode == "equal":
        idx = int(np.argmin(np.abs(precision - recall)))
    elif mode == "precision_target":
        valid = np.where(precision >= target_precision)[0]
        if len(valid):
            # thresholds ascend and recall falls as they rise, so the first
            # valid index is the highest recall that still meets the target
            idx = int(valid[0])
        else:
            print(
                f"WARNING: precision {target_precision} is never reached; "
                "using the threshold with the highest precision instead."
            )
            idx = int(np.argmax(precision))
    else:
        raise ValueError(f"Unknown THRESHOLD_MODE: {mode!r}")

    return float(thresholds[idx]), float(precision[idx]), float(recall[idx])


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_probability: np.ndarray
) -> dict[str, float | list[int]]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_feature_importance(
    artifacts_dir: Path,
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Path:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    numeric_slice = preprocessor.output_indices_["numeric"]
    categorical_slice = preprocessor.output_indices_["categorical"]
    importances = classifier.feature_importances_
    feature_importance: dict[str, float] = {}

    for position, name in enumerate(numeric_features):
        feature_importance[name] = float(importances[numeric_slice][position])

    encoder = preprocessor.named_transformers_["categorical"].named_steps[
        "onehot"
    ]
    categorical_importances = importances[categorical_slice]
    offset = 0
    for name, categories in zip(
        categorical_features, encoder.categories_, strict=True
    ):
        width = len(categories)
        feature_importance[name] = float(
            categorical_importances[offset : offset + width].sum()
        )
        offset += width

    table = (
        pd.DataFrame(
            {
                "feature": list(feature_importance.keys()),
                "importance": list(feature_importance.values()),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = artifacts_dir / f"{MODEL_NAME}_feature_importance.csv"
    table.to_csv(csv_path, index=False)

    plot_data = table.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(plot_data))))
    ax.barh(plot_data["feature"], plot_data["importance"], color="steelblue")
    ax.set_title("Random Forest Feature Importance (Top 20)")
    ax.set_xlabel("Mean decrease in impurity")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(
        artifacts_dir / f"{MODEL_NAME}_feature_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    return csv_path


def main() -> None:
    artifacts_dir = Path(ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir = artifacts_dir / "classification" / "Randomforrestoptimized"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_FILE)
    processed, preprocess_summary = preprocess(raw)

    if TARGET_CLASSIFICATION not in processed.columns:
        raise ValueError(
            f"Target column {TARGET_CLASSIFICATION!r} not found "
            "after preprocessing"
        )

    features = processed.drop(columns=[TARGET_CLASSIFICATION])
    target = processed[TARGET_CLASSIFICATION]
    if target.isna().any():
        raise ValueError(
            f"Target column {TARGET_CLASSIFICATION!r} contains missing values"
        )
    if set(target.unique()) != {0, 1}:
        raise ValueError(
            f"{TARGET_CLASSIFICATION} must contain only binary values 0 and 1"
        )

    numeric_features = features.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_features = features.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    print(f"Rows after preprocessing: {len(processed):,}")
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(
        f"Categorical features ({len(categorical_features)}): "
        f"{categorical_features}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    print(f"\nTrain rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(
        "Train class balance: "
        f"{y_train.value_counts(normalize=True).round(3).to_dict()}"
    )
    print(
        "Test class balance: "
        f"{y_test.value_counts(normalize=True).round(3).to_dict()}"
    )

    model = build_model_pipeline(numeric_features, categorical_features)
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
        verbose=1,
    )

    cv_summary: dict[str, dict[str, float | list[float]]] = {}
    print(
        f"\n--- {CV_FOLDS}-fold stratified cross-validation "
        "(training set, threshold 0.5) ---"
    )
    for metric_name in scoring:
        scores = cv_results[f"test_{metric_name}"]
        cv_summary[metric_name] = {
            "fold_scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
        print(
            f"{metric_name:10s}: {scores.mean():.4f} ± "
            f"{scores.std():.4f} (folds: {np.round(scores, 4).tolist()})"
        )

    # --- Choose the decision threshold on out-of-fold training predictions ---
    oof_probability = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, oof_precision, oof_recall = choose_threshold(
        y_train, oof_probability, THRESHOLD_MODE, TARGET_PRECISION
    )
    print(
        f"\n--- Threshold selection (mode={THRESHOLD_MODE!r}) ---\n"
        f"Chosen threshold:          {threshold:.4f}\n"
        f"Out-of-fold precision:     {oof_precision:.4f}\n"
        f"Out-of-fold recall:        {oof_recall:.4f}"
    )

    # --- Fit on the full training set, evaluate once on the test set ---
    model.fit(X_train, y_train)
    y_probability = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_probability >= 0.5).astype(int)
    y_pred = (y_probability >= threshold).astype(int)

    default_metrics = compute_metrics(y_test, y_pred_default, y_probability)
    test_metrics = compute_metrics(y_test, y_pred, y_probability)

    print("\n--- Test set at the default 0.5 threshold ---")
    print(f"Precision: {default_metrics['precision']:.4f}")
    print(f"Recall:    {default_metrics['recall']:.4f}")
    print(f"F1-score:  {default_metrics['f1_score']:.4f}")

    print(f"\n--- Final test-set performance (threshold {threshold:.4f}) ---")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-score:  {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print("\nConfusion matrix ([[TN, FP], [FN, TP]]):")
    print(np.array(test_metrics["confusion_matrix"]))

    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        target_names=["Not Canceled", "Canceled"],
    )
    print("\nFull classification report:")
    print(report_text)

    cm_path = output_dir / f"{MODEL_NAME}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Not Canceled", "Canceled"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(
        f"Random Forest - Confusion Matrix (Test, threshold {threshold:.3f})"
    )
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    roc_path = output_dir / f"{MODEL_NAME}_roc_curve.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test, y_probability, ax=ax, name="Random Forest"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Chance",
    )
    ax.set_title(
        f"ROC Curve (AUC = {test_metrics['roc_auc']:.3f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pr_path = output_dir / f"{MODEL_NAME}_pr_curve.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_test, y_probability, ax=ax, name="Random Forest"
    )
    ax.scatter(
        [test_metrics["recall"]],
        [test_metrics["precision"]],
        color="red",
        zorder=5,
        label=f"Chosen threshold ({threshold:.3f})",
    )
    ax.legend()
    ax.set_title(
        f"Precision-Recall Curve (AP = {test_metrics['pr_auc']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    importance_csv = save_feature_importance(
        artifacts_dir,
        model,
        numeric_features,
        categorical_features,
    )
    model_path = output_dir / f"{MODEL_NAME}.joblib"
    joblib.dump(model, model_path)

    metrics_payload = {
        "model": "RandomForestClassifier",
        "target": TARGET_CLASSIFICATION,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "hyperparameters": {
            "class_weight": CLASS_WEIGHT,
            "max_features": MAX_FEATURES,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
        },
        "test_size": TEST_SIZE,
        "preprocess_summary": preprocess_summary,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "cross_validation": {
            "folds": CV_FOLDS,
            "metrics": cv_summary,
        },
        "decision_threshold": {
            "mode": THRESHOLD_MODE,
            "target_precision": (
                TARGET_PRECISION if THRESHOLD_MODE == "precision_target"
                else None
            ),
            "value": threshold,
            "out_of_fold_precision": oof_precision,
            "out_of_fold_recall": oof_recall,
        },
        "test_set_performance_at_0.5": default_metrics,
        "test_set_performance": test_metrics,
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            target_names=["Not Canceled", "Canceled"],
            output_dict=True,
        ),
    }
    metrics_path = output_dir / f"{MODEL_NAME}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved model to:             {model_path}")
    print(f"Saved metrics JSON to:      {metrics_path}")
    print(f"Saved confusion matrix to:  {cm_path}")
    print(f"Saved ROC curve to:         {roc_path}")
    print(f"Saved PR curve to:          {pr_path}")
    print(f"Saved feature importance to:{importance_csv}")
    print(
        f"\nReminder: model.predict() uses 0.5. In production use "
        f"model.predict_proba(X)[:, 1] >= {threshold:.4f}"
    )


if __name__ == "__main__":
    main()