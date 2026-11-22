"""Train, evaluate and persist the Linear Regression ADR pipeline.

Run from the project root::

    python -m src.regression.linear_regression_train
    python -m src.regression.linear_regression_train --strict
    python -m src.regression.linear_regression_train --show-plots

Workflow
--------
1. Load the shared clean CSV, audit leakage, and drop ``ADR <= 0`` rows.
2. Split once into train / test (stratified on ADR quantile bins).
3. Evaluate the Linear Regression baseline model using 5-fold cross-validation on the training set.
4. Fit the pipeline on the full training set.
5. Evaluate the model once on the held-out test set.
6. Save diagnostic and evaluation plots to ``artifacts/plots/``.
7. Save the fitted Linear Regression pipeline, metrics and metadata to ``artifacts/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.regression.pipeline import (
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    LIFESPAN_COLUMNS,
    STRICT_ONLY_COLUMNS,
    TARGET,
    build_pipeline,
    get_feature_names,
    get_numeric_features,
)

LOGGER = logging.getLogger("regression.linear_regression_train")

DEFAULT_DATA = Path("data/processed/h2_clean_v2_dedup.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_PATH = ARTIFACT_DIR / "linear_regression_pipeline.joblib"
PLOT_DIR = ARTIFACT_DIR / "plots"

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
METRICS = ("mae", "rmse", "r2")


# --------------------------------------------------------------------------
# Data loading and leakage audit
# --------------------------------------------------------------------------
def load_dataset(
    path: Path, strict: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the clean CSV and return ``(X, y)`` after the leakage audit.

    Rows with ``ADR <= 0`` are removed so that the log1p target transform is
    well-defined. Leaky and post-inquiry columns are explicitly dropped.
    """
    df = pd.read_csv(path)
    LOGGER.info("Loaded %s -> %d rows x %d columns", path, *df.shape)

    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in {path}")

    n_before = len(df)
    df = df[df[TARGET] > 0].reset_index(drop=True)
    LOGGER.info(
        "Removed %d rows with ADR <= 0 (complimentary/zero bookings)",
        n_before - len(df),
    )

    to_drop = list(LEAKAGE_COLUMNS) + list(LIFESPAN_COLUMNS)
    if strict:
        to_drop += list(STRICT_ONLY_COLUMNS)

    dropped = [column for column in to_drop if column in df.columns]
    LOGGER.info("Leakage audit - dropped columns: %s", dropped)

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET] + dropped)

    leftovers = set(X.columns) & set(LEAKAGE_COLUMNS)
    assert not leftovers, f"Leaky columns survived the audit: {leftovers}"

    return X, y


def stratify_bins(y: pd.Series, n_bins: int = 10) -> pd.Series:
    """Create quantile bins of the continuous target for stratification."""
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def regression_metrics(
    y_true: pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    """Calculate MAE, RMSE and R2 on the original ADR scale."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cv_metrics(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv, n_jobs: int
) -> dict[str, float]:
    """Return mean and standard deviation of CV metrics on the ADR scale."""
    results = cross_validate(
        pipe, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs
    )
    output: dict[str, float] = {}

    for metric in METRICS:
        scores = results[f"test_{metric}"]
        if metric != "r2":
            scores = -scores
        output[f"cv_{metric}"] = float(scores.mean())
        output[f"cv_{metric}_std"] = float(scores.std())

    return output


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    """Print result dictionaries as an aligned table."""
    df = pd.DataFrame(rows).set_index("model")
    print(f"\n=== {title} ===")
    print(df.round(3).to_string())


def save_visualizations(
    pipeline: Pipeline,
    y_test: pd.Series,
    predictions: np.ndarray,
    baseline_metrics: dict[str, float],
    test_metrics: dict[str, float],
    show_plots: bool = False,
) -> None:
    """Save plots explaining Linear Regression diagnostics and test performance."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))

    # 1. Actual vs predicted ADR; closer to the diagonal is better.
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.25, s=12, color="#1f77b4")
    ax.plot([low, high], [low, high], linestyle="--", color="crimson", linewidth=1.5)
    ax.set_xlabel("Actual ADR")
    ax.set_ylabel("Predicted ADR")
    ax.set_title("Linear Regression: Actual vs Predicted ADR (Test Set)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "linear_regression_actual_vs_predicted.png", dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    residuals = actual - predicted

    # 2. Residuals vs predictions; patternless cloud around zero is expected.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, residuals, alpha=0.25, s=12, color="#2ca02c")
    ax.axhline(0, linestyle="--", color="crimson", linewidth=1.5)
    ax.set_xlabel("Predicted ADR")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Linear Regression: Residual Plot (Test Set)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "linear_regression_residuals.png", dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    # 3. Error distribution (residual histogram).
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, edgecolor="black", color="#3b528b", alpha=0.8)
    ax.axvline(0, linestyle="--", color="crimson", linewidth=1.5)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Number of bookings")
    ax.set_title("Linear Regression: Distribution of Prediction Errors (Test Set)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "linear_regression_residual_histogram.png", dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    # 4. Compare baseline CV and test metrics.
    labels = ["Train CV", "Test Set"]
    metric_sets = [baseline_metrics, test_metrics]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title, color in zip(
        axes,
        ("mae", "rmse", "r2"),
        ("MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"),
        ("#4c72b0", "#55a868", "#c44e52"),
    ):
        values = [
            item.get(f"cv_{metric}", item.get(metric))
            for item in metric_sets
        ]
        bars = ax.bar(labels, values, color=color, alpha=0.85, width=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Linear Regression (Baseline) Performance Summary")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "linear_regression_metrics_comparison.png", dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    # 5. Top feature coefficients.
    try:
        feature_names = get_feature_names(pipeline)
        raw_regressor = pipeline.named_steps["model"].regressor_
        if hasattr(raw_regressor, "coef_"):
            coefs = raw_regressor.coef_
            coef_series = pd.Series(coefs, index=feature_names).sort_values()
            top_bottom = pd.concat([coef_series.head(10), coef_series.tail(10)])

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#d95f02" if c < 0 else "#1b9e77" for c in top_bottom.values]
            top_bottom.plot(kind="barh", ax=ax, color=colors)
            ax.set_xlabel("Coefficient Value (log1p scale)")
            ax.set_title("Linear Regression: Top 10 Negative & Positive Coefficients")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / "linear_regression_coefficients.png", dpi=160)
            if show_plots:
                plt.show()
            plt.close(fig)
    except Exception as e:
        LOGGER.warning("Could not generate coefficient plot: %s", e)

    print(f"\nSaved visualizations to: {PLOT_DIR.resolve()}")
    for filename in (
        "linear_regression_actual_vs_predicted.png",
        "linear_regression_residuals.png",
        "linear_regression_residual_histogram.png",
        "linear_regression_metrics_comparison.png",
        "linear_regression_coefficients.png",
    ):
        print(f"  - {filename}")


# --------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    """Train and evaluate the Linear Regression pipeline."""
    X, y = load_dataset(args.data, strict=args.strict)

    # --- single split: stratified on ADR quantile bins --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_bins(y),
    )
    LOGGER.info(
        "Split: train=%d  test=%d (test untouched until the end)",
        len(X_train),
        len(X_test),
    )

    # Stratified folds for cross-validation on the training set
    skf = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    cv = list(skf.split(X_train, stratify_bins(y_train)))

    # --- 1) baseline Linear Regression 5-fold CV on train ------------------
    pipeline = build_pipeline(
        LinearRegression(), scale_numeric=True, strict=args.strict
    )

    LOGGER.info("Evaluating Linear Regression baseline with %d-fold CV...", args.folds)
    start_time = time.time()
    baseline_metrics = cv_metrics(
        pipeline, X_train, y_train, cv, args.n_jobs
    )
    baseline_metrics["sec"] = time.time() - start_time
    print_table(
        f"Baseline Linear Regression {args.folds}-fold CV (default params)",
        [{"model": "LinearRegression (baseline)", **baseline_metrics}],
    )

    # --- 2) fit the pipeline on the full training set -----------------------
    LOGGER.info("Fitting Linear Regression on the full training set...")
    t_fit_start = time.time()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.time() - t_fit_start
    LOGGER.info("Fitted in %.2f seconds.", fit_seconds)

    # --- 3) final test evaluation, performed once -------------------------
    predictions = pipeline.predict(X_test)
    test_metrics = regression_metrics(y_test, predictions)
    test_metrics["sec"] = fit_seconds
    print_table(
        "Final TEST evaluation (held-out test set)",
        [{"model": "LinearRegression (baseline)", **test_metrics}],
    )

    # --- 4) create visual aids for diagnostics ------------------------------
    save_visualizations(
        pipeline=pipeline,
        y_test=y_test,
        predictions=predictions,
        baseline_metrics=baseline_metrics,
        test_metrics=test_metrics,
        show_plots=args.show_plots,
    )

    # --- 5) save the fitted model and its evaluation metadata ---------------
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.artifact, compress=3)

    metadata = {
        "model": "LinearRegression",
        "description": "Ordinary Least Squares baseline regressor",
        "baseline_cv_metrics": baseline_metrics,
        "test_metrics": test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "strict_mode": args.strict,
        "numeric_features": get_numeric_features(args.strict),
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": list(LEAKAGE_COLUMNS + LIFESPAN_COLUMNS)
        + (list(STRICT_ONLY_COLUMNS) if args.strict else []),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    (ARTIFACT_DIR / "linear_regression_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    pd.DataFrame([test_metrics]).to_csv(
        ARTIFACT_DIR / "linear_regression_test_metrics.csv", index=False
    )
    LOGGER.info("Saved Linear Regression pipeline to %s", args.artifact)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also drop borderline fields (parking, special requests)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="also open generated plots in windows (plots are always saved)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(parse_args())
