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
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import ARTIFACTS_DIR, DATA_FILE, RANDOM_STATE, TARGET_CLASSIFICATION
from pipeline.preprocess import preprocess


TEST_SIZE = 0.20
CV_FOLDS = 5
MODEL_NAME = "MLP_1.8"
HIDDEN_LAYER_SIZES = (256, 128, 64)
CLASS_WEIGHT = {0: 1.0, 1: 2.5} 

# --- Epoch training settings ---
MAX_EPOCHS = 200
PATIENCE = 15                 # stop after this many epochs without improvement
MIN_DELTA = 1e-4              # smallest improvement that counts
BEST_METRIC = "pr_auc"            # "ap", "pr_auc", "f1", "val_loss" (lower is better)
VALIDATION_FRACTION = 0.15    # part of the training set used to pick epoch + threshold

# --- Threshold selection (done on the validation set) ---
# "recall_target" -> highest precision that still reaches TARGET_RECALL
# "f1"            -> threshold with the best F1 for the Canceled class
# "equal"         -> precision and recall as close as possible
THRESHOLD_MODE = "equal"
TARGET_RECALL = 0.70

# Cross-validation uses scikit-learn's own training (no per-epoch printing).
# Set to False to skip it and save time.
RUN_CROSS_VALIDATION = True


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_classifier(**overrides) -> MLPClassifier:
    params = dict(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        alpha=1e-3,                # L2 regularization
        batch_size=256,
        learning_rate_init=0.01,
        max_iter=MAX_EPOCHS,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return MLPClassifier(**params)


def build_cv_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "classifier",
                build_classifier(
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                ),
            ),
        ]
    )


def choose_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    mode: str,
    target_recall: float,
) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    precision, recall = precision[:-1], recall[:-1]   # align with thresholds

    if mode == "recall_target":
        # recall falls as the threshold rises, so the last index that still
        # meets the target has the highest precision
        valid = np.where(recall >= target_recall)[0]
        idx = int(valid[-1]) if len(valid) else 0
    elif mode == "f1":
        score = 2 * precision * recall / (precision + recall + 1e-12)
        idx = int(np.argmax(score))
    elif mode == "equal":
        idx = int(np.argmin(np.abs(precision - recall)))
    else:
        raise ValueError(f"Unknown THRESHOLD_MODE: {mode!r}")
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx])


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_probability: np.ndarray
) -> dict[str, float | list[int]]:
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true, y_probability
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        # area under the precision-recall curve (trapezoid rule)
        "pr_auc": float(auc(recall_curve, precision_curve)),
        # average precision (this is what earlier JSON files called "pr_auc")
        "ap": float(average_precision_score(y_true, y_probability)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def epoch_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    pred = (probability >= 0.5).astype(int)
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true, probability
    )
    return {
        "val_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "ap": float(average_precision_score(y_true, probability)),
    }


def print_epoch_header() -> None:
    print(
        f"\n{'epoch':>5} {'loss':>8} {'val_loss':>9} {'acc':>7} "
        f"{'prec':>7} {'recall':>7} {'f1':>7} {'pr_auc':>7} {'ap':>7}"
    )


def print_epoch_row(row: dict[str, float], improved: bool) -> None:
    print(
        f"{row['epoch']:>5d} {row['loss']:>8.4f} {row['val_loss']:>9.4f} "
        f"{row['accuracy']:>7.4f} {row['precision']:>7.4f} "
        f"{row['recall']:>7.4f} {row['f1']:>7.4f} "
        f"{row['pr_auc']:>7.4f} {row['ap']:>7.4f}" + ("  *" if improved else "")
    )


def train_with_epoch_log(
    classifier: MLPClassifier,
    X_fit,
    y_fit: np.ndarray,
    X_val,
    y_val: np.ndarray,
    sample_weight=None
) -> tuple[list[dict[str, float]], int]:
    """Train one epoch at a time, log metrics, restore the best weights."""
    lower_is_better = BEST_METRIC == "val_loss"
    best_score = np.inf if lower_is_better else -np.inf
    best_epoch, wait = 0, 0
    best_coefs = best_intercepts = None
    history: list[dict[str, float]] = []

    print_epoch_header()
    for epoch in range(1, MAX_EPOCHS + 1):
        classifier.partial_fit(
        X_fit, y_fit, classes=np.array([0, 1]), sample_weight=sample_weight
        )
        probability = classifier.predict_proba(X_val)[:, 1]
        row = {
            "epoch": epoch,
            "loss": float(classifier.loss_),  # training loss (includes L2 term)
            **epoch_metrics(y_val, probability),
        }
        history.append(row)

        score = row[BEST_METRIC]
        improved = (
            score < best_score - MIN_DELTA
            if lower_is_better
            else score > best_score + MIN_DELTA
        )
        if improved:
            best_score, best_epoch, wait = score, epoch, 0
            best_coefs = [w.copy() for w in classifier.coefs_]
            best_intercepts = [b.copy() for b in classifier.intercepts_]
        else:
            wait += 1
        print_epoch_row(row, improved)

        if wait >= PATIENCE:
            print(
                f"\nNo improvement in {BEST_METRIC} for {PATIENCE} epochs. "
                "Stopping."
            )
            break

    classifier.coefs_ = best_coefs
    classifier.intercepts_ = best_intercepts
    print(
        f"\nRestored weights from best epoch: {best_epoch} "
        f"({BEST_METRIC} = {best_score:.4f})"
    )
    return history, best_epoch


def save_feature_importance(
    output_dir: Path,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Path:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    table = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": result.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = output_dir / f"{MODEL_NAME}_feature_importance.csv"
    table.to_csv(csv_path, index=False)

    plot_data = table.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(plot_data))))
    ax.barh(plot_data["feature"], plot_data["importance"], color="steelblue")
    ax.set_title("MLP Permutation Importance (Top 20)")
    ax.set_xlabel("Drop in AP when the feature is shuffled")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{MODEL_NAME}_feature_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    return csv_path


def main() -> None:
    artifacts_dir = Path(ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir = artifacts_dir / "classification" / "MLP"
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
    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(sampling_strategy=0.8, random_state=RANDOM_STATE).fit_resample(X_fit_raw, y_fit)
    # X_train, y_train = ros.fit_resample(X_train, y_train)
    # X_test, y_test = ros.fit_resample(X_test, y_test)
    print(f"\nTrain rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(
        "Train class balance: "
        f"{y_train.value_counts(normalize=True).round(3).to_dict()}"
    )
    print(
        "Test class balance: "
        f"{y_test.value_counts(normalize=True).round(3).to_dict()}"
    )

    # --- Optional cross-validation (sklearn's own training) ---
    cv_summary: dict[str, dict[str, float | list[float]]] | None = None
    if RUN_CROSS_VALIDATION:
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
            "ap": "average_precision",
        }
        w_train = y_train.map(CLASS_WEIGHT).to_numpy(dtype=float)

        cv_results = cross_validate(
            build_cv_pipeline(numeric_features, categorical_features),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            params={"classifier__sample_weight": w_train},
            n_jobs=-1,
            return_train_score=False,
            verbose=3,
        )
        cv_summary = {}
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

    # --- Epoch-by-epoch training with best-epoch selection ---
    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    print(
        f"\nEpoch training: {len(X_fit_raw):,} fit rows | "
        f"{len(X_val_raw):,} validation rows"
    )
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_fit = preprocessor.fit_transform(X_fit_raw)   # fitted on fit rows only
    X_val = preprocessor.transform(X_val_raw)

    classifier = build_classifier(early_stopping=False)
    history, best_epoch = train_with_epoch_log(
        classifier, X_fit, y_fit.to_numpy(), X_val, y_val.to_numpy()
    )
    best_row = history[best_epoch - 1]
    print("\n--- Best epoch (validation set, threshold 0.5) ---")
    for key in (
        "epoch", "loss", "val_loss", "accuracy", "precision",
        "recall", "f1", "pr_auc", "ap",
    ):
        value = best_row[key]
        print(f"{key:10s}: {value}" if key == "epoch" else f"{key:10s}: {value:.4f}")

    pd.DataFrame(history).to_csv(
        output_dir / f"{MODEL_NAME}_epoch_history.csv", index=False
    )

    # --- Choose the decision threshold on the validation set (best weights) ---
    val_probability = classifier.predict_proba(X_val)[:, 1]
    threshold, val_precision, val_recall = choose_threshold(
        y_val.to_numpy(), val_probability, THRESHOLD_MODE, TARGET_RECALL
    )
    print(
        f"\n--- Threshold selection (mode={THRESHOLD_MODE!r}) ---\n"
        f"Chosen threshold:        {threshold:.4f}\n"
        f"Validation precision:    {val_precision:.4f}\n"
        f"Validation recall:       {val_recall:.4f}"
    )

    # The final model = fitted preprocessor + best-epoch classifier
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    y_probability = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_probability >= 0.5).astype(int)
    y_pred = (y_probability >= threshold).astype(int)

    default_metrics = compute_metrics(y_test, y_pred_default, y_probability)
    test_metrics = compute_metrics(y_test, y_pred, y_probability)

    print(f"\n--- Test set at the default 0.5 threshold (best epoch {best_epoch}) ---")
    print(f"Precision: {default_metrics['precision']:.4f}")
    print(f"Recall:    {default_metrics['recall']:.4f}")
    print(f"F1-score:  {default_metrics['f1_score']:.4f}")

    print(
        f"\n--- Final test-set performance "
        f"(best epoch {best_epoch}, threshold {threshold:.4f}) ---"
    )
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-score:  {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"AP:        {test_metrics['ap']:.4f}")
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
    ax.set_title(f"MLP - Confusion Matrix (Test, threshold {threshold:.3f})")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    roc_path = output_dir / f"{MODEL_NAME}_roc_curve.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test, y_probability, ax=ax, name="MLP"
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
        y_test, y_probability, ax=ax, name="MLP"
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
        f"Precision-Recall Curve (AP = {test_metrics['ap']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    importance_csv = save_feature_importance(
        output_dir, model, X_test, y_test
    )
    model_path = output_dir / f"{MODEL_NAME}.joblib"
    joblib.dump(model, model_path)

    metrics_payload = {
        "model": "MLP",
        "target": TARGET_CLASSIFICATION,
        "random_state": RANDOM_STATE,
        "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
        "test_size": TEST_SIZE,
        "preprocess_summary": preprocess_summary,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "cross_validation": (
            {"folds": CV_FOLDS, "metrics": cv_summary}
            if cv_summary is not None
            else None
        ),
        "epoch_training": {
            "fit_rows": len(X_fit_raw),
            "validation_rows": len(X_val_raw),
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "best_metric": BEST_METRIC,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "best_epoch_validation": best_row,
            "history": history,
        },
        "decision_threshold": {
            "mode": THRESHOLD_MODE,
            "target_recall": (
                TARGET_RECALL if THRESHOLD_MODE == "recall_target" else None
            ),
            "value": threshold,
            "validation_precision": val_precision,
            "validation_recall": val_recall,
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
    print(f"Saved epoch history to:     {output_dir / f'{MODEL_NAME}_epoch_history.csv'}")
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