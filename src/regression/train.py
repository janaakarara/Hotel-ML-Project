"""Train, tune, compare and persist ADR regression pipelines.
 
Run from the project root::
 
    python -m src.regression.train                 # full grids
    python -m src.regression.train --quick         # tiny grids (smoke test)
    python -m src.regression.train --strict        # also drop borderline fields
 
Workflow
--------
1. Load the shared clean CSV, audit leakage, drop ``ADR <= 0`` rows.
2. Split once into train / test (stratified on ADR quantile bins).
3. Baseline: 5-fold CV on the training set with default hyper-parameters.
4. Tuning: ``GridSearchCV`` (same folds) per model family.
5. Winner chosen by tuned CV MAE (never by test score).
6. Every tuned model is scored ONCE on the held-out test set.
7. Winner pipeline, metrics and metadata are saved to ``artifacts/``.
"""
 
from __future__ import annotations
 
import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
 
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
 
from src.regression.pipeline import (LEAKAGE_COLUMNS, LIFESPAN_COLUMNS,
                                      STRICT_ONLY_COLUMNS, TARGET,
                                      build_pipeline, get_numeric_features,
                                      CATEGORICAL_FEATURES)
 
LOGGER = logging.getLogger("regression.train")
 
DEFAULT_DATA = Path("data/processed/h2_clean_v2_dedup.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_PATH = ARTIFACT_DIR / "regression_pipeline.joblib"
 
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
METRICS = ("mae", "rmse", "r2")
 
 
# --------------------------------------------------------------------------
# Data loading and leakage audit
# --------------------------------------------------------------------------
def load_dataset(path: Path, strict: bool = False
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Load the clean CSV and return ``(X, y)`` after the leakage audit.
 
    * Rows with ``ADR <= 0`` (complimentary bookings) are removed so that
      ``log1p`` is well-defined and the target is strictly positive.
    * Leaky / post-inquiry columns are dropped explicitly.
    """
    df = pd.read_csv(path)
    LOGGER.info("Loaded %s -> %d rows x %d columns", path, *df.shape)
 
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in {path}")
 
    n_before = len(df)
    df = df[df[TARGET] > 0].reset_index(drop=True)
    LOGGER.info("Removed %d rows with ADR <= 0 (complimentary/zero bookings)",
                n_before - len(df))
 
    to_drop = list(LEAKAGE_COLUMNS) + list(LIFESPAN_COLUMNS)
    if strict:
        to_drop += list(STRICT_ONLY_COLUMNS)
    dropped = [c for c in to_drop if c in df.columns]
    LOGGER.info("Leakage audit - dropped columns: %s", dropped)
 
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET] + dropped)
    leftovers = set(X.columns) & set(LEAKAGE_COLUMNS)
    assert not leftovers, f"Leaky columns survived the audit: {leftovers}"
    return X, y
 
 
def stratify_bins(y: pd.Series, n_bins: int = 10) -> pd.Series:
    """Quantile bins of ``y`` used to stratify a continuous target."""
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
 
 
# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """A model family: estimator, preprocessing flag and tuning grid."""
 
    name: str
    estimator: RegressorMixin
    scale_numeric: bool
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
 
 
def _grid(grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Prefix grid keys so they address the model inside the TTR wrapper."""
    return {f"model__regressor__{k}": v for k, v in grid.items()}
 
 
def get_model_specs(seed: int, quick: bool = False) -> list[ModelSpec]:
    """Return the model families to compare (Linear Regression, Ridge).

    * ``LinearRegression`` is the un-regularised baseline (no hyper-parameters,
      so its grid is empty and ``GridSearchCV`` evaluates a single candidate).
    * ``Ridge`` adds L2 regularisation; ``alpha`` is tuned.
    """
    return [
        ModelSpec("LinearRegression (baseline)", LinearRegression(), True),
        ModelSpec(
            "Ridge", Ridge(random_state=seed), True,
            _grid({"alpha": [1.0, 10.0] if quick
                   else [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]}),
        ),
    ]
 
 
# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def regression_metrics(y_true: pd.Series, y_pred: np.ndarray
                       ) -> dict[str, float]:
    """MAE, RMSE and R2 on the original ADR scale."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
 
 
def cv_metrics(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv,
               n_jobs: int) -> dict[str, float]:
    """Cross-validate and return mean/std of MAE, RMSE, R2 (original scale)."""
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    out: dict[str, float] = {}
    for m in METRICS:
        scores = res[f"test_{m}"]
        scores = scores if m == "r2" else -scores
        out[f"cv_{m}"] = float(scores.mean())
        out[f"cv_{m}_std"] = float(scores.std())
    return out
 
 
def tune(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, cv, strict: bool,
         n_jobs: int) -> GridSearchCV:
    """Run ``GridSearchCV`` (refit on MAE) for one model family."""
    pipe = build_pipeline(spec.estimator, spec.scale_numeric, strict)
    search = GridSearchCV(pipe, spec.param_grid, scoring=SCORING,
                          refit="mae", cv=cv, n_jobs=n_jobs, verbose=1)
    return search.fit(X, y)
 
 
def search_cv_metrics(search: GridSearchCV) -> dict[str, float]:
    """Extract mean/std CV metrics of the best candidate of a grid search."""
    i = search.best_index_
    res = search.cv_results_
    out: dict[str, float] = {}
    for m in METRICS:
        sign = 1.0 if m == "r2" else -1.0
        out[f"cv_{m}"] = sign * float(res[f"mean_test_{m}"][i])
        out[f"cv_{m}_std"] = float(res[f"std_test_{m}"][i])
    return out
 
 
def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    """Pretty-print a list of result dicts as an aligned table."""
    df = pd.DataFrame(rows).set_index("model")
    print(f"\n=== {title} ===")
    print(df.round(3).to_string())
 
 
# --------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    """Execute the complete training workflow."""
    X, y = load_dataset(args.data, strict=args.strict)
 
    # --- single split: stratified on ADR quantile bins ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed,
        stratify=stratify_bins(y))
    LOGGER.info("Split: train=%d  test=%d (test untouched until the end)",
                len(X_train), len(X_test))
 
    # Same stratified folds for baseline and tuning -> fair comparison.
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed)
    cv = list(skf.split(X_train, stratify_bins(y_train)))
 
    specs = get_model_specs(args.seed, args.quick)
 
    # --- 1) baseline CV (default hyper-parameters) ----------------------------
    baseline_rows = []
    for spec in specs:
        t0 = time.time()
        pipe = build_pipeline(spec.estimator, spec.scale_numeric, args.strict)
        baseline_rows.append(
            {"model": spec.name,
             **cv_metrics(pipe, X_train, y_train, cv, args.n_jobs),
             "sec": time.time() - t0})
    print_table(f"Baseline {args.folds}-fold CV on TRAIN (default params)",
                baseline_rows)
 
    # --- 2) hyper-parameter tuning -------------------------------------------
    searches: dict[str, GridSearchCV] = {}
    tuned_rows = []
    for spec in specs:
        # LinearRegression has an empty grid -> a single candidate, so the
        # baseline goes through the same CV / test path as Ridge.
        LOGGER.info("Tuning %s ...", spec.name)
        t0 = time.time()
        search = tune(spec, X_train, y_train, cv, args.strict, args.n_jobs)
        searches[spec.name] = search
        tuned_rows.append(
            {"model": spec.name, **search_cv_metrics(search),
             "sec": time.time() - t0})
        LOGGER.info("%s best params: %s", spec.name, search.best_params_)
    print_table(f"Tuned {args.folds}-fold CV on TRAIN (GridSearchCV)",
                tuned_rows)
 
    # --- 3) pick winner on CV MAE, then ONE test evaluation -------------------
    winner = min(tuned_rows, key=lambda r: r["cv_mae"])["model"]
    test_rows = []
    for name, search in searches.items():
        preds = search.best_estimator_.predict(X_test)
        test_rows.append({"model": name + (" *" if name == winner else ""),
                          **regression_metrics(y_test, preds)})
    print_table("Final TEST evaluation (* = winner selected by CV MAE)",
                test_rows)
 
    # --- 4) persist artifacts --------------------------------------------------
    best_pipeline = searches[winner].best_estimator_
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, args.artifact, compress=3)
 
    winner_test = next(r for r in test_rows if r["model"].startswith(winner))
    meta = {
        "winner": winner,
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None)))
                            else str(v))
                        for k, v in searches[winner].best_params_.items()},
        "test_metrics": {m: winner_test[m] for m in METRICS},
        "cv_metrics": next(r for r in tuned_rows if r["model"] == winner),
        "n_train": len(X_train), "n_test": len(X_test),
        "strict_mode": args.strict,
        "numeric_features": get_numeric_features(args.strict),
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": list(LEAKAGE_COLUMNS + LIFESPAN_COLUMNS),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACT_DIR / "regression_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str))
    pd.DataFrame(test_rows).to_csv(ARTIFACT_DIR / "regression_test_metrics.csv",
                                   index=False)
    LOGGER.info("Saved winner '%s' to %s", winner, args.artifact)
 
 
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--quick", action="store_true",
                   help="use tiny hyper-parameter grids (smoke test)")
    p.add_argument("--strict", action="store_true",
                   help="also drop borderline fields (parking, special requests)")
    return p.parse_args()
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
 

