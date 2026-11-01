
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# Make `config` and `pipeline` importable no matter where the script is launched.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR, DATA_FILE, RANDOM_STATE, TARGET_CLASSIFICATION  # noqa: E402
from pipeline.preprocess import preprocess  # noqa: E402

MODEL_NAME = "KNN"
TEST_SIZE = 0.2
N_SPLITS = 5
LOG_COLUMNS = ["LeadTime", "ADR"]  # right-skewed; log transform is a tuned option
ID_LIKE_COLUMNS = ["Agent", "Company"]  # IDs: must be treated as categories

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

FULL_GRID = {
    "classifier__n_neighbors": [3, 5, 11, 21, 31, 51],
    "classifier__weights": ["uniform", "distance"],
    "classifier__p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
    "preprocessor__log_numeric__log__apply": [False, True],
}
QUICK_GRID = {
    "classifier__n_neighbors": [5, 21],
    "classifier__weights": ["uniform"],
    "classifier__p": [2],
    "preprocessor__log_numeric__log__apply": [False, True],
}


# --------------------------------------------------------------------------- #
# Custom transformer
# --------------------------------------------------------------------------- #
class LogTransformer(BaseEstimator, TransformerMixin):
    """log1p transform that can be switched on/off, so GridSearch can tune it."""

    def __init__(self, apply: bool = False) -> None:
        self.apply = apply

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log1p(np.clip(X, 0, None)) if self.apply else X


# --------------------------------------------------------------------------- #
# GPU KNN classifier (sklearn-compatible)
# --------------------------------------------------------------------------- #
class TorchKNNClassifier(ClassifierMixin, BaseEstimator):


    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        p: float = 2,
        batch_size: int = 1024,
        device: str | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.batch_size = batch_size
        self.device = device

    # ------------------------------------------------------------------ utils
    def _get_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_tensors(self, dev: torch.device):
        """Copy the training data to the device once and cache it."""
        key = str(dev)
        if key not in self._cache:
            self._cache[key] = (
                torch.from_numpy(self._fit_X).to(dev),
                torch.from_numpy(self._fit_y).to(dev),
            )
        return self._cache[key]

    def __getstate__(self):
        # Never pickle GPU tensors: keep only the numpy training data.
        state = dict(super().__getstate__())
        if "_cache" in state:
            state["_cache"] = {}
        return state

    # -------------------------------------------------------------------- fit
    def fit(self, X, y):
        if self.weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        if self.p < 1:
            raise ValueError("p must be >= 1")
        X, y = check_X_y(X, y, dtype=np.float32)
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors is larger than the number of training rows")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self._fit_X = np.ascontiguousarray(X)
        self._fit_y = y_idx.astype(np.int64)
        self._cache = {}
        return self

    # ---------------------------------------------------------------- predict
    @staticmethod
    def _distance_weights(dist: torch.Tensor) -> torch.Tensor:
        # Same rule as sklearn: if a query has an exact match (distance 0),
        # only the exact matches vote; otherwise weight = 1 / distance.
        zero = dist == 0
        any_zero = zero.any(dim=1, keepdim=True)
        inv = 1.0 / dist.clamp_min(torch.finfo(dist.dtype).tiny)
        return torch.where(any_zero, zero.to(dist.dtype), inv)

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        dev = self._get_device()
        train_X, train_y = self._train_tensors(dev)
        n_classes = len(self.classes_)
        out = np.empty((len(X), n_classes), dtype=np.float64)

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                q = torch.from_numpy(X[start : start + self.batch_size]).to(dev)
                d = torch.cdist(
                    q, train_X, p=float(self.p),
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                dist, idx = torch.topk(d, k=self.n_neighbors, dim=1, largest=False)
                labels = train_y[idx]
                if self.weights == "distance":
                    w = self._distance_weights(dist)
                else:
                    w = torch.ones_like(dist)
                votes = torch.zeros(len(q), n_classes, device=dev, dtype=dist.dtype)
                votes.scatter_add_(1, labels, w)
                votes /= votes.sum(dim=1, keepdim=True)
                out[start : start + len(q)] = votes.cpu().numpy()
        return out

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(DATA_FILE)
    df, _ = preprocess(raw)

    # Strip stray whitespace in text columns; force ID columns to be categories.
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype(str).str.strip()
    for col in ID_LIKE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    y = df[TARGET_CLASSIFICATION].astype(int)
    X = df.drop(columns=[TARGET_CLASSIFICATION])
    return X, y


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Full model: preprocessing + KNN. Reusable by the inference code."""
    log_cols = [c for c in LOG_COLUMNS if c in X.columns]
    numeric_cols = [
        c for c in X.select_dtypes(include="number").columns if c not in log_cols
    ]
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "log_numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("log", LogTransformer(apply=False)),
                        ("scale", StandardScaler()),
                    ]
                ),
                log_cols,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        sparse_threshold=0,  # dense output: faster distance computation for KNN
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", TorchKNNClassifier())]
    )


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def cv_metrics(model: Pipeline, X, y, cv) -> dict:
    # n_jobs=1: the GPU is shared, so folds run one after another.
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    out = {}
    for name in SCORING:
        out[f"cv_{name}_mean"] = float(res[f"test_{name}"].mean())
        out[f"cv_{name}_std"] = float(res[f"test_{name}"].std())
    return out


def test_metrics(model: Pipeline, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_recall": recall_score(y, pred, zero_division=0),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y, proba),
        "test_pr_auc": average_precision_score(y, proba),
    }
    return {k: float(v) for k, v in metrics.items()}, pred, proba


def short_params(model: Pipeline) -> dict:
    knn = model.named_steps["classifier"]
    log_apply = model.named_steps["preprocessor"].named_transformers_["log_numeric"][
        "log"
    ].apply
    return {
        "n_neighbors": knn.n_neighbors,
        "weights": knn.weights,
        "p": knn.p,
        "log_transform": bool(log_apply),
    }


def save_results(rows: list[dict], path: Path) -> None:
    """Shared results table for all classification models (one row per model+config)."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["model"] != MODEL_NAME]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def save_plots(out: Path, y_test, results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (label, r) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, r["pred"], display_labels=["Not canceled", "Canceled"],
            ax=ax, colorbar=False,
        )
        ax.set_title(f"KNN {label} - confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(out / "knn_confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, r in results.items():
        RocCurveDisplay.from_predictions(y_test, r["proba"], name=f"KNN {label}", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("KNN ROC curve (test)")
    fig.tight_layout()
    fig.savefig(out / "knn_roc_curve.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def verify_against_sklearn(X_train, y_train, X_test) -> None:
    """Sanity check: the torch KNN must agree with sklearn's KNeighborsClassifier."""
    pre = build_pipeline(X_train).named_steps["preprocessor"]
    Xt = pre.fit_transform(X_train.iloc[:8000])
    yt = y_train.iloc[:8000]
    Xq = pre.transform(X_test.iloc[:1000])
    for k, w, p in [(5, "uniform", 2), (11, "distance", 1), (21, "distance", 2)]:
        ref = KNeighborsClassifier(n_neighbors=k, weights=w, p=p).fit(Xt, yt)
        ours = TorchKNNClassifier(n_neighbors=k, weights=w, p=p).fit(Xt, yt)
        diff = np.abs(ref.predict_proba(Xq) - ours.predict_proba(Xq)).max()
        agree = (ref.predict(Xq) == ours.predict(Xq)).mean()
        print(f"k={k:<2} weights={w:<8} p={p}: max proba diff={diff:.2e} | "
              f"prediction agreement={agree:.2%}")


class ProgressF1:
    """F1 scorer that also prints one self-updating progress line."""

    def __init__(self, total: int) -> None:
        self.total, self.done, self.start = total, 0, time.perf_counter()

    def __call__(self, estimator, X, y) -> float:
        score = f1_score(y, estimator.predict(X), zero_division=0)
        self.done += 1
        elapsed = time.perf_counter() - self.start
        eta = elapsed / self.done * (self.total - self.done)
        print(f"\r  Grid search: {self.done}/{self.total} fits | "
              f"elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min   ",
              end="", flush=True)
        return score

def main(quick: bool = False, verify: bool = False) -> None:
    gpu = (
        "GPU - " + torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU (CUDA not available!)"
    )
    print(f"Compute device: {gpu}")
    out = Path(ARTIFACTS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    print(f"Rows: {len(X):,} | features: {X.shape[1]} | cancel rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    if verify:
        verify_against_sklearn(X_train, y_train, X_test)
        return
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ---------------- Baseline (sklearn defaults: k=5, uniform, Euclidean) ------
    print("\n[1/2] Baseline KNN ...")
    baseline = build_pipeline(X_train)
    base_cv = cv_metrics(baseline, X_train, y_train, cv)
    t0 = time.perf_counter()
    baseline.fit(X_train, y_train)
    base_fit_time = time.perf_counter() - t0
    base_test, base_pred, base_proba = test_metrics(baseline, X_test, y_test)
    print(f"  CV F1 = {base_cv['cv_f1_mean']:.4f} | test F1 = {base_test['test_f1']:.4f}")

    # ---------------- Tuned (GridSearchCV, scoring = F1) -----------------------
    print("\n[2/2] GridSearchCV ...")
    grid_space = QUICK_GRID if quick else FULL_GRID
    progress = ProgressF1(len(ParameterGrid(grid_space)) * N_SPLITS)
    grid = GridSearchCV(
        build_pipeline(X_train),
        param_grid=grid_space,
        scoring=progress,
        cv=cv,
        n_jobs=1,  # GPU is shared; parallel workers would fight over it
        refit=True,
        verbose=0,
    )
    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    print()  # finish the progress line
    search_time = time.perf_counter() - t0
    tuned = grid.best_estimator_
    tuned_cv = cv_metrics(tuned, X_train, y_train, cv)
    tuned_test, tuned_pred, tuned_proba = test_metrics(tuned, X_test, y_test)
    best_params = short_params(tuned)
    print(f"  Best params: {best_params}")
    print(f"  CV F1 = {tuned_cv['cv_f1_mean']:.4f} | test F1 = {tuned_test['test_f1']:.4f}")

    # ---------------- Save everything -----------------------------------------
    rows = [
        {
            "task": "classification", "model": MODEL_NAME, "config": "baseline",
            "params": json.dumps(short_params(baseline)), "selection_metric": "",
            "fit_time_s": base_fit_time, "search_time_s": 0.0,
            **base_cv, **base_test,
        },
        {
            "task": "classification", "model": MODEL_NAME, "config": "tuned",
            "params": json.dumps(best_params), "selection_metric": "f1 (5-fold CV)",
            "fit_time_s": float(grid.refit_time_), "search_time_s": search_time,
            **tuned_cv, **tuned_test,
        },
    ]
    save_results(rows, out / "knn_classification_results.csv")

    keep = [c for c in grid.cv_results_ if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time",
        "mean_score_time",
    ]
    (
        pd.DataFrame(grid.cv_results_)[keep]
        .rename(columns={"mean_test_score": "mean_cv_f1", "std_test_score": "std_cv_f1"})
        .sort_values("rank_test_score")
        .to_csv(out / "knn_grid_search_results.csv", index=False)
    )
    (out / "knn_best_params.json").write_text(
        json.dumps(
            {"best_params": best_params, "search_space": {k: list(v) for k, v in grid_space.items()},
             "scoring": "f1", "cv_folds": N_SPLITS, "random_state": RANDOM_STATE},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    save_plots(
        out, y_test,
        {"baseline": {"pred": base_pred, "proba": base_proba},
         "tuned": {"pred": tuned_pred, "proba": tuned_proba}},
    )
    joblib.dump(tuned, out / "knn_classifier.joblib", compress=3)

    print("\n=== KNN results ===")
    print(pd.DataFrame(rows)[["config", "cv_f1_mean", "test_accuracy", "test_precision",
                              "test_recall", "test_f1", "test_roc_auc", "test_pr_auc",
                              "fit_time_s"]].round(4).to_string(index=False))
    print(f"\nSaved to: {out.resolve()}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Train KNN for IsCanceled.")
    parser.add_argument("--quick", action="store_true", help="use a tiny grid (smoke test)")
    parser.add_argument("--verify", action="store_true",
                        help="only check that the torch KNN matches sklearn's KNN")
    args = parser.parse_args()
    main(quick=args.quick, verify=args.verify)


if __name__ == "__main__":
    # Re-import this file under its real module name so the classes inside the
    # saved model are recorded as `pipeline.models.KNN.*` instead of `__main__.*`.
    from pipeline.models.KNN import cli as _cli

    _cli()