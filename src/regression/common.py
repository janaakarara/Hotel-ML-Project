"""Shared experiment protocol for every ADR regression model.

Everything that must be identical across Linear / Ridge / Random Forest /
XGBoost lives here -- data loading, the leakage audit, the train/test split, the
CV folds, the metrics, the plots and the artifact format -- so comparisons are
fair by construction (spec section 12).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.regression.pipeline import (
    DEFAULT_LOG_TARGET,
    LEAKAGE_COLUMNS,
    LIFESPAN_COLUMNS,
    MAX_ADR,
    REQUIRED_RAW_COLUMNS,
    TARGET,
    arrival_dates,
    build_pipeline,
    get_feature_names,
)

LOGGER = logging.getLogger("regression")

# One-hot levels missing from a CV fold are mapped to the 'infrequent' bucket on purpose.
warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)

#: Rows with 0 < ADR <= 10 are data errors / near-complimentary stays (93 of 53k on H2,
#: spread over all segments).  They dominate the log-space error of linear models.
DEFAULT_MIN_ADR = 10.0

DEFAULT_DATA = Path("data/processed/h2_clean_v2_dedup.csv")
ARTIFACT_DIR = Path("artifacts")

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
    path: Path,
    min_adr: float = DEFAULT_MIN_ADR,
    max_adr: float = MAX_ADR,
    drop_zero_guests: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the clean CSV and return ``(X, y)`` after row filters + leakage audit.

    Row filters (all logged, all deterministic functions of the row itself so
    applying them before the split cannot leak):

    * ``ADR <= min_adr``  -> complimentary / zero / near-zero bookings
    * ``ADR > max_adr``   -> data-entry outliers (also the prediction clip bound)
    * zero guests         -> Adults + Children + Babies == 0 is not a real stay
    """
    df = pd.read_csv(path)
    LOGGER.info("Loaded %s -> %d rows x %d columns", path, *df.shape)
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in {path}")

    n = len(df)
    df = df[(df[TARGET] > min_adr) & (df[TARGET] <= max_adr)]
    LOGGER.info("Removed %d rows outside %.2f < ADR <= %.0f", n - len(df), min_adr, max_adr)

    if drop_zero_guests:
        n = len(df)
        guests = sum(pd.to_numeric(df[c], errors="coerce").fillna(0) for c in ("Adults", "Children", "Babies"))
        df = df[guests > 0]
        LOGGER.info("Removed %d rows with zero guests", n - len(df))
    df = df.reset_index(drop=True)

    dropped = [c for c in dict.fromkeys(LEAKAGE_COLUMNS + LIFESPAN_COLUMNS) if c in df.columns]
    LOGGER.info("Leakage audit - dropped columns: %s", dropped)

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET] + dropped)

    leftovers = set(X.columns) & (set(LEAKAGE_COLUMNS) | set(LIFESPAN_COLUMNS))
    assert not leftovers, f"Leaky columns survived the audit: {leftovers}"
    absent = [c for c in REQUIRED_RAW_COLUMNS if c not in X.columns]
    assert not absent, f"Required input columns missing from data: {absent}"
    return X, y


def stratify_bins(y: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Quantile bins of the continuous target, used only to stratify splits."""
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop").to_numpy()


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    cv: list[tuple[np.ndarray, np.ndarray]]
    mode: str


def _group_keys(X: pd.DataFrame) -> np.ndarray:
    """Near-duplicate bookings (group / agent contracts) share these three fields."""
    day = arrival_dates(X).dt.strftime("%Y-%m-%d").fillna("NA").map(str)
    keys = X["Agent"].map(str) + "|" + day + "|" + X["ReservedRoomType"].map(str)
    return keys.to_numpy(dtype=object)


def make_split(
    X: pd.DataFrame, y: pd.Series, mode: str, test_size: float, folds: int, seed: int
) -> Split:
    """Single train/test split plus CV folds on the training part.

    ``random`` : stratified (ADR quantile bins) split + StratifiedKFold.
    ``group``  : like random, but all bookings sharing (Agent, arrival date,
                 room type) stay on the same side -- removes near-duplicate leakage.
    ``time``   : test = latest arrivals; CV = expanding-window TimeSeriesSplit.
                 The strictest check of "predict future bookings".
    """
    if mode == "random":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify_bins(y)
        )
        cv = list(
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(
                X_tr, stratify_bins(y_tr)
            )
        )
    elif mode == "group":
        k = max(2, round(1.0 / test_size))
        outer = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        tr_idx, te_idx = next(outer.split(X, stratify_bins(y), _group_keys(X)))
        X_tr, X_te, y_tr, y_te = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
        inner = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        cv = list(inner.split(X_tr, stratify_bins(y_tr), _group_keys(X_tr)))
    elif mode == "time":
        dates = arrival_dates(X)
        order = np.argsort(dates.fillna(dates.min()).to_numpy(), kind="stable")
        n_test = int(round(len(X) * test_size))
        tr_idx, te_idx = order[:-n_test], order[-n_test:]
        X_tr, X_te, y_tr, y_te = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
        cv = list(TimeSeriesSplit(n_splits=folds).split(X_tr))
    else:
        raise ValueError(f"Unknown split mode '{mode}'")

    LOGGER.info("Split[%s]: train=%d test=%d (test untouched until the end)", mode, len(X_tr), len(X_te))
    return Split(X_tr, X_te, y_tr, y_te, cv, mode)


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R2 on the original ADR scale."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cv_metrics(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv, n_jobs: int) -> dict[str, float]:
    """Mean/std of validation AND train metrics per fold, plus mean fit time.

    Train metrics expose over-fitting (essential for RF); ``fit_sec`` is the mean
    per-fold fit time, comparable with the final-fit time reported later.
    """
    res = cross_validate(
        pipe, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs, return_train_score=True
    )
    out: dict[str, float] = {}
    for m in METRICS:
        sign = 1.0 if m == "r2" else -1.0
        val, trn = sign * res[f"test_{m}"], sign * res[f"train_{m}"]
        out[f"cv_{m}"], out[f"cv_{m}_std"] = float(val.mean()), float(val.std())
        out[f"train_{m}"] = float(trn.mean())
    out["fit_sec"] = float(np.mean(res["fit_time"]))
    return out


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows).set_index("model")
    print(f"\n=== {title} ===")
    print(df.round(3).to_string())


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def feature_importance(pipeline: Pipeline) -> pd.Series | None:
    """Signed coefficients (linear) or impurity/gain importances (trees).

    ``None`` when a reducer is present (features are components) or the model
    exposes neither attribute.
    """
    if "reducer" in pipeline.named_steps:
        return None
    reg = pipeline.named_steps["model"].regressor_
    names = get_feature_names(pipeline)
    if hasattr(reg, "coef_"):
        values = np.ravel(reg.coef_)
    elif hasattr(reg, "feature_importances_"):
        values = np.ravel(reg.feature_importances_)
    else:
        return None
    return pd.Series(values, index=names) if len(values) == len(names) else None


def save_visualizations(
    name: str,
    pipeline: Pipeline,
    y_test: pd.Series,
    predictions: np.ndarray,
    cv_res: dict[str, float],
    test_res: dict[str, float],
    plot_dir: Path,
    show_plots: bool = False,
) -> None:
    """Save five diagnostic plots for ``name`` into ``plot_dir``."""
    if not show_plots:
        plt.switch_backend("Agg")
    plot_dir.mkdir(parents=True, exist_ok=True)
    title = name.replace("_", " ")

    def finish(fig, filename: str) -> None:
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}_{filename}.png", dpi=160)
        if show_plots:
            plt.show()
        plt.close(fig)

    actual, pred = np.asarray(y_test, float), np.asarray(predictions, float)
    resid = actual - pred
    lo, hi = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, pred, alpha=0.25, s=12, color="#1f77b4")
    ax.plot([lo, hi], [lo, hi], "--", color="crimson", lw=1.5)
    ax.set(xlabel="Actual ADR", ylabel="Predicted ADR", title=f"{title}: actual vs predicted (test)")
    ax.grid(alpha=0.25)
    finish(fig, "actual_vs_predicted")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred, resid, alpha=0.25, s=12, color="#2ca02c")
    ax.axhline(0, ls="--", color="crimson", lw=1.5)
    ax.set(xlabel="Predicted ADR", ylabel="Residual (actual - predicted)", title=f"{title}: residuals (test)")
    ax.grid(alpha=0.25)
    finish(fig, "residuals")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid, bins=40, edgecolor="black", color="#3b528b", alpha=0.8)
    ax.axvline(0, ls="--", color="crimson", lw=1.5)
    ax.set(xlabel="Residual (actual - predicted)", ylabel="Bookings", title=f"{title}: error distribution (test)")
    ax.grid(alpha=0.25)
    finish(fig, "residual_histogram")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m, label, color in zip(
        axes, METRICS,
        ("MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"),
        ("#4c72b0", "#55a868", "#c44e52"),
    ):
        vals = [cv_res[f"cv_{m}"], test_res[m]]
        errs = [cv_res[f"cv_{m}_std"], 0.0]
        bars = ax.bar(["CV (train folds)", "Test"], vals, yerr=errs, capsize=4, color=color, alpha=0.85, width=0.5)
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{title}: performance summary")
    finish(fig, "metrics_comparison")

    try:
        imp = feature_importance(pipeline)
        if imp is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            if hasattr(pipeline.named_steps["model"].regressor_, "coef_"):
                s = imp.sort_values()
                s = pd.concat([s.head(10), s.tail(10)])
                ax.barh(s.index, s.values, color=["#d95f02" if v < 0 else "#1b9e77" for v in s.values])
                ax.set_xlabel("Coefficient (log1p-ADR scale)")
                ax.set_title(f"{title}: 10 most negative / positive coefficients")
            else:
                s = imp.sort_values().tail(20)
                ax.barh(s.index, s.values, color="#1b9e77")
                ax.set_xlabel("Importance")
                ax.set_title(f"{title}: top 20 feature importances")
            ax.grid(axis="x", alpha=0.25)
            finish(fig, "feature_importance")
    except Exception as exc:  # plotting must never break training
        LOGGER.warning("Could not generate importance plot: %s", exc)

    print(f"\nSaved visualizations to: {plot_dir.resolve()}")


# --------------------------------------------------------------------------
# Artifacts and metadata
# --------------------------------------------------------------------------
def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifacts(
    name: str,
    pipeline: Pipeline,
    split: Split,
    cv_res: dict[str, float],
    test_res: dict[str, float],
    config: dict[str, Any],
    data_path: Path,
    artifact_dir: Path = ARTIFACT_DIR,
) -> Path:
    """Persist the fitted pipeline, metadata JSON and test-metrics CSV."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{name}_pipeline.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    steps = pipeline.named_steps
    n_out = (
        int(steps["reducer"].n_components_)
        if "reducer" in steps and hasattr(steps["reducer"], "n_components_")
        else len(get_feature_names(pipeline))
    )
    metadata = {
        "name": name,
        "regressor": repr(steps["model"].regressor_),
        "config": config,
        "baseline_cv_metrics": cv_res,
        "test_metrics": test_res,
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "split_mode": split.mode,
        "input_columns": list(split.X_train.columns),
        "required_raw_columns": list(REQUIRED_RAW_COLUMNS),
        "n_features_out": n_out,
        "excluded_columns": list(dict.fromkeys(LEAKAGE_COLUMNS + LIFESPAN_COLUMNS)),
        "data_sha256": _file_sha256(data_path),
        "git_commit": _git_sha(),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (artifact_dir / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    pd.DataFrame([test_res]).to_csv(artifact_dir / f"{name}_test_metrics.csv", index=False)
    return model_path


# --------------------------------------------------------------------------
# Shared command line and experiment runner (one thin file per model uses these)
# --------------------------------------------------------------------------
def pca_arg(value: str) -> float | int:
    """``0.95`` -> variance ratio, ``20`` -> number of components."""
    number = float(value)
    return int(number) if number >= 1 else number


def build_parser(description: str) -> argparse.ArgumentParser:
    """CLI flags shared by every regression training script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    p.add_argument("--split", choices=("group", "random", "time"), default="group",
                   help="group (default): near-duplicate bookings never straddle train/test; "
                        "random: stratified; time: test = latest arrivals")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--pca", type=pca_arg, default=None,
                   help="add PCA: a variance ratio (0.95) or a component count (20)")
    p.add_argument("--min-frequency", type=float, default=None,
                   help="pool one-hot levels rarer than this share (default: per family)")
    p.add_argument("--log-target", action=argparse.BooleanOptionalAction, default=None,
                   help="fit log1p(ADR) (default: per family)")
    p.add_argument("--min-adr", type=float, default=DEFAULT_MIN_ADR, help="keep rows with ADR > this")
    p.add_argument("--max-adr", type=float, default=MAX_ADR, help="keep rows with ADR <= this")
    p.add_argument("--keep-zero-guest", action="store_true",
                   help="keep rows with Adults+Children+Babies == 0")
    p.add_argument("--show-plots", action="store_true",
                   help="also open plots in windows (plots are always saved)")
    return p


def run_experiment(
    args: argparse.Namespace,
    key: str,
    family: str,
    make_estimator: Callable[..., Any],
    **estimator_kwargs: Any,
) -> None:
    """Baseline CV -> fit -> single test evaluation -> plots + artifacts.

    ``key`` names the artifacts (``<key>_pipeline.joblib``, ...), ``family`` is
    ``"linear"`` or ``"tree"`` and ``make_estimator(seed, n_jobs, **kwargs)`` returns
    the untuned regressor.
    """
    np.random.seed(args.seed)
    log_target = DEFAULT_LOG_TARGET[family] if args.log_target is None else args.log_target
    name = key + ("_pca" if args.pca else "")
    if args.log_target is not None:
        name += "_log" if args.log_target else "_nolog"
    label = f"{key} ({family} family{', PCA' if args.pca else ''}, log_target={log_target})"

    X, y = load_dataset(
        args.data, min_adr=args.min_adr, max_adr=args.max_adr,
        drop_zero_guests=not args.keep_zero_guest,
    )
    split = make_split(X, y, args.split, args.test_size, args.folds, args.seed)

    reducer = PCA(n_components=args.pca, random_state=args.seed) if args.pca else None
    pipeline = build_pipeline(
        make_estimator(args.seed, args.n_jobs, **estimator_kwargs),
        family=family, reducer=reducer, log_target=args.log_target,
        min_frequency=args.min_frequency, seed=args.seed,
    )

    # Tree models parallelise inside the estimator, linear models over CV folds --
    # never both (CPU oversubscription).
    cv_jobs = 1 if family == "tree" else args.n_jobs

    LOGGER.info("Baseline %d-fold CV for %s ...", args.folds, label)
    cv_res = cv_metrics(pipeline, split.X_train, split.y_train, split.cv, cv_jobs)
    print_table(
        f"Baseline {label}: {args.folds}-fold CV on train (split={args.split})",
        [{"model": name, **cv_res}],
    )

    t0 = time.perf_counter()
    pipeline.fit(split.X_train, split.y_train)
    fit_seconds = time.perf_counter() - t0
    LOGGER.info("Fitted in %.2f s", fit_seconds)

    predictions = pipeline.predict(split.X_test)          # test set touched exactly once
    test_res = regression_metrics(split.y_test, predictions)
    test_res["fit_sec"] = fit_seconds
    print_table("Final TEST evaluation (held-out)", [{"model": name, **test_res}])

    save_visualizations(
        name, pipeline, split.y_test, predictions, cv_res, test_res,
        args.artifact_dir / "plots", show_plots=args.show_plots,
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(family=family, log_target_used=log_target, estimator_kwargs=estimator_kwargs)
    path = save_artifacts(
        name, pipeline, split, cv_res, test_res,
        config=config, data_path=args.data, artifact_dir=args.artifact_dir,
    )
    LOGGER.info("Saved pipeline to %s", path)