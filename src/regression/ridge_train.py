"""Train, tune and persist the Ridge Regression ADR pipeline.

Run from the project root::

    python -m src.regression.ridge_train
    python -m src.regression.ridge_train --quick
    python -m src.regression.ridge_train --strict

Workflow
--------
1. Load the shared clean CSV, audit leakage, and drop ``ADR <= 0`` rows.
2. Split once into train / test (stratified on ADR quantile bins).
3. Evaluate the default Ridge model using cross-validation on the training set.
4. Tune Ridge's ``alpha`` with GridSearchCV using the same folds.
5. Evaluate the tuned model once on the held-out test set.
6. Save the fitted Ridge pipeline, metrics and metadata to ``artifacts/``.
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
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.regression.pipeline import (
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    LIFESPAN_COLUMNS,
    STRICT_ONLY_COLUMNS,
    TARGET,
    build_pipeline,
    get_numeric_features,
)

LOGGER = logging.getLogger("regression.ridge_train")

DEFAULT_DATA = Path("data/processed/h2_clean_v2_dedup.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_PATH = ARTIFACT_DIR / "ridge_pipeline.joblib"
PLOT_DIR = ARTIFACT_DIR / "plots"

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
METRICS = ("mae", "rmse", "r2")
ALPHA_GRID = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


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


def search_cv_metrics(search: GridSearchCV) -> dict[str, float]:
    """Extract CV metrics for the best alpha candidate."""
    best_index = search.best_index_
    results = search.cv_results_
    output: dict[str, float] = {}

    for metric in METRICS:
        sign = 1.0 if metric == "r2" else -1.0
        output[f"cv_{metric}"] = sign * float(
            results[f"mean_test_{metric}"][best_index]
        )
        output[f"cv_{metric}_std"] = float(
            results[f"std_test_{metric}"][best_index]
        )

    return output


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    """Print result dictionaries as an aligned table."""
    df = pd.DataFrame(rows).set_index("model")
    print(f"\n=== {title} ===")
    print(df.round(3).to_string())




def save_visualizations(
    search: GridSearchCV,
    y_test: pd.Series,
    predictions: np.ndarray,
    baseline_metrics: dict[str, float],
    tuned_metrics: dict[str, float],
    test_metrics: dict[str, float],
    show_plots: bool = False,
) -> None:
    """Save plots explaining Ridge tuning and test-set performance."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Alpha vs cross-validation MAE.
    results = pd.DataFrame(search.cv_results_)
    alphas = results["param_model__regressor__alpha"].astype(float).to_numpy()
    cv_mae = -results["mean_test_mae"].to_numpy()
    order = np.argsort(alphas)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas[order], cv_mae[order], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Ridge alpha (log scale)")
    ax.set_ylabel("Mean CV MAE (ADR units)")
    ax.set_title("Ridge tuning: alpha vs cross-validation MAE")
    ax.grid(True, alpha=0.3)
    best_alpha = float(search.best_params_["model__regressor__alpha"])
    best_index = int(np.argmin(np.abs(alphas[order] - best_alpha)))
    ax.scatter([best_alpha], [cv_mae[order][best_index]], s=90, zorder=3)
    ax.annotate(
        f"Best alpha = {best_alpha:g}",
        (best_alpha, cv_mae[order][best_index]),
        xytext=(8, 8), textcoords="offset points",
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ridge_alpha_vs_cv_mae.png", dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))

    # 2. Actual vs predicted ADR; closer to the diagonal is better.
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.25, s=12)
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual ADR")
    ax.set_ylabel("Predicted ADR")
    ax.set_title("Ridge: actual vs predicted ADR (test set)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ridge_actual_vs_predicted.png", dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    residuals = actual - predicted

    # 3. Residuals vs predictions; ideally a patternless cloud around zero.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, residuals, alpha=0.25, s=12)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted ADR")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Ridge residual plot (test set)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ridge_residuals.png", dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    # 4. Error distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, edgecolor="black")
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Number of bookings")
    ax.set_title("Distribution of Ridge prediction errors (test set)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ridge_residual_histogram.png", dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    # 5. Compare baseline CV, tuned CV, and test metrics.
    labels = ["Baseline CV", "Tuned CV", "Test set"]
    metric_sets = [baseline_metrics, tuned_metrics, test_metrics]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(
        axes,
        ("mae", "rmse", "r2"),
        ("MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"),
    ):
        values = [
            item.get(f"cv_{metric}", item.get(metric))
            for item in metric_sets
        ]
        ax.bar(labels, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Ridge performance summary")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ridge_metrics_comparison.png", dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    print(f"\nSaved visualizations to: {PLOT_DIR.resolve()}")
    for filename in (
        "ridge_alpha_vs_cv_mae.png",
        "ridge_actual_vs_predicted.png",
        "ridge_residuals.png",
        "ridge_residual_histogram.png",
        "ridge_metrics_comparison.png",
    ):
        print(f"  - {filename}")



# --------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    """Train and evaluate the Ridge Regression pipeline."""
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

    # Use the same stratified folds for baseline and tuning.
    skf = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    cv = list(skf.split(X_train, stratify_bins(y_train)))

    # --- 1) baseline Ridge CV using the default alpha=1.0 ------------------
    baseline_pipe = build_pipeline(
        Ridge(random_state=args.seed), scale_numeric=True, strict=args.strict
    )
    start_time = time.time()
    baseline_metrics = cv_metrics(
        baseline_pipe, X_train, y_train, cv, args.n_jobs
    )
    baseline_metrics["sec"] = time.time() - start_time
    print_table(
        f"Baseline Ridge {args.folds}-fold CV (alpha=1.0)",
        [{"model": "Ridge (baseline)", **baseline_metrics}],
    )

    # --- 2) tune Ridge alpha using GridSearchCV ----------------------------
    alpha_values = [1.0, 10.0] if args.quick else ALPHA_GRID
    pipeline = build_pipeline(
        Ridge(random_state=args.seed), scale_numeric=True, strict=args.strict
    )
    param_grid = {"model__regressor__alpha": alpha_values}

    LOGGER.info("Tuning Ridge alpha over: %s", alpha_values)
    start_time = time.time()
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        refit="mae",
        cv=cv,
        n_jobs=args.n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)
    tuning_seconds = time.time() - start_time

    tuned_metrics = search_cv_metrics(search)
    tuned_metrics["sec"] = tuning_seconds
    print_table(
        f"Tuned Ridge {args.folds}-fold CV (GridSearchCV)",
        [{"model": "Ridge (tuned)", **tuned_metrics}],
    )
    LOGGER.info("Ridge best parameters: %s", search.best_params_)

    # --- 3) final test evaluation, performed once -------------------------
    predictions = search.best_estimator_.predict(X_test)
    test_metrics = regression_metrics(y_test, predictions)
    print_table("Final TEST evaluation", [{"model": "Ridge", **test_metrics}])

    # --- 4) create visual aids for tuning and model diagnostics -------------
    save_visualizations(
        search=search,
        y_test=y_test,
        predictions=predictions,
        baseline_metrics=baseline_metrics,
        tuned_metrics=tuned_metrics,
        test_metrics=test_metrics,
        show_plots=args.show_plots,
    )

    # --- 5) save the fitted model and its evaluation metadata ---------------
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, args.artifact, compress=3)

    metadata = {
        "model": "Ridge",
        "best_params": search.best_params_,
        "baseline_cv_metrics": baseline_metrics,
        "tuned_cv_metrics": tuned_metrics,
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

    (ARTIFACT_DIR / "ridge_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    pd.DataFrame([test_metrics]).to_csv(
        ARTIFACT_DIR / "ridge_test_metrics.csv", index=False
    )
    LOGGER.info("Saved tuned Ridge pipeline to %s", args.artifact)


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
        "--quick",
        action="store_true",
        help="use a small alpha grid (smoke test)",
    )
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
