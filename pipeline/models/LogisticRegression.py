"""Baseline and tuned logistic regression for hotel booking cancellation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR, RANDOM_STATE
from pipeline.data import SCORING, get_cv, load_data, split_data
from pipeline.transformers import LogTransformer

MODEL_NAME = "LogisticRegression"
LOG_COLUMNS = ["LeadTime", "ADR"]  # right-skewed; log transform is off in the baseline
HIGH_CARD_COLUMNS = ["Country", "Agent", "Company"]  # target encoded
MAX_ITER = 1000
# Columns combined in pairs when interactions are on
INTERACTION_NUMERIC = ["LeadTime", "ADR", "TotalOfSpecialRequests"]
INTERACTION_CATEGORICAL = ["MarketSegment", "DepositType", "CustomerType"]
PRECISION_FLOOR = 0.65  # same threshold rule as KNN
GRID_SCORING = "average_precision"  # PR-AUC: ranking quality, threshold-free

def build_interactions(X: pd.DataFrame) -> Pipeline:
    """Pairwise products of the selected columns (numeric ones scaled first)."""
    numeric = [c for c in INTERACTION_NUMERIC if c in X.columns]
    categorical = [c for c in INTERACTION_CATEGORICAL if c in X.columns]
    base = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ])
    return Pipeline([
        ("base", base),
        # degree=(2, 2): pairs only, since single columns are already in the model
        ("pairs", PolynomialFeatures(
            degree=(2, 2), interaction_only=True, include_bias=False
        )),
        ("scale", StandardScaler(with_mean=False)),
    ])


PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10, 100, 1000, 10000],
    "preprocessor__log_numeric__log__apply": [False, True],
    # Curved effects for LOG_COLUMNS: off, or cubic splines with 5 or 10 knots
    "preprocessor__log_numeric__spline": [
        "passthrough",
        SplineTransformer(n_knots=5, degree=3, knots="quantile"),
        SplineTransformer(n_knots=10, degree=3, knots="quantile"),
    ],
}


# Pipeline
def build_pipeline(X: pd.DataFrame, class_weight: str | None = None) -> Pipeline:
    """Full model: preprocessing + logistic regression."""
    log_cols = [c for c in LOG_COLUMNS if c in X.columns]
    numeric_cols = [
        c for c in X.select_dtypes(include="number").columns if c not in log_cols
    ]
    text_cols = X.select_dtypes(exclude="number").columns.tolist()
    high_card = [c for c in HIGH_CARD_COLUMNS if c in text_cols]
    categorical_cols = [c for c in text_cols if c not in high_card]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "log_numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("log", LogTransformer(apply=False)),
                    ("spline", "passthrough"),  # off in the baseline
                    ("scale", StandardScaler()),
                ]),
                log_cols,
            ),
            (
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
            (
                "target",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", TargetEncoder(
                        target_type="binary",
                        cv=StratifiedKFold(
                            n_splits=5, shuffle=True, random_state=RANDOM_STATE
                        ),
                    )),
                    ("scale", StandardScaler()),
                ]),
                high_card,
            ),
            ("interactions", "drop", list(X.columns)),  # off in the baseline
        ]
    )
    # L2 regularization is the scikit-learn default.
    classifier = LogisticRegression(
        C=1.0, solver="lbfgs", class_weight=class_weight, max_iter=MAX_ITER
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


# Evaluation
def cv_metrics(model: Pipeline, X, y, cv) -> dict:
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING)
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


def short_params(model: Pipeline, threshold: float = 0.5) -> dict:
    clf = model.named_steps["classifier"]
    steps = model.named_steps["preprocessor"].transformers[1][1].named_steps
    spline = steps["spline"]
    interactions = model.named_steps["preprocessor"].transformers[-1][1]
    return {
        "C": clf.C,
        "regularization": "l2",
        "solver": clf.solver,
        "class_weight": clf.class_weight,
        "max_iter": clf.max_iter,
        "log_transform": bool(steps["log"].apply),
        "spline_knots": None if spline == "passthrough" else spline.n_knots,
        "interactions": interactions != "drop",
        "threshold": threshold,
    }


def save_results(rows: list[dict], path: Path) -> None:
    """Write the rows, keeping any other configs already saved in the file."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["config"].isin(new["config"])]
        new = pd.concat([new, old], ignore_index=True)
    new.to_csv(path, index=False, float_format="%.4f")


def save_plots(out: Path, y_test, results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (label, r) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, r["pred"], display_labels=["Not canceled", "Canceled"],
            ax=ax, colorbar=False,
        )
        ax.set_title(f"LogReg {label} - confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(out / "logreg_confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, r in results.items():
        RocCurveDisplay.from_predictions(
            y_test, r["proba"], name=f"LogReg {label}", ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("Logistic regression ROC curve (test)")
    fig.tight_layout()
    fig.savefig(out / "logreg_roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, r in results.items():
        PrecisionRecallDisplay.from_predictions(
            y_test, r["proba"], name=f"LogReg {label}", ax=ax
        )
    # Operating point of each model at its own threshold
    for label, r in results.items():
        ax.scatter(
            recall_score(y_test, r["pred"]),
            precision_score(y_test, r["pred"], zero_division=0),
            s=60, zorder=3, label=f"{label} threshold {r['threshold']:.2f}",
        )
    rate = float(y_test.mean())
    ax.axhline(rate, color="k", linestyle="--", alpha=0.4,
               label=f"random guess ({rate:.2f})")
    ax.set_title("Logistic regression precision-recall curve (test)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "logreg_pr_curve.png", dpi=150)
    plt.close(fig)


def recall_at_precision_floor(y_true, y_pred) -> float:
    """Recall if precision meets the floor, otherwise 0."""
    if precision_score(y_true, y_pred, zero_division=0) < PRECISION_FLOOR:
        return 0.0
    return recall_score(y_true, y_pred, zero_division=0)


def main() -> None:
    out = ROOT / ARTIFACTS_DIR / "classification" / "logistic_regression"
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    print(f"Rows: {len(X):,} | features: {X.shape[1]} | cancel rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    cv = get_cv()

    # ---------------- Baseline (default settings, threshold 0.50) ----------------
    print("\n[1/2] Baseline logistic regression ...")
    baseline = build_pipeline(X_train)
    base_cv = cv_metrics(baseline, X_train, y_train, cv)
    t0 = time.perf_counter()
    baseline.fit(X_train, y_train)
    base_fit_time = time.perf_counter() - t0
    base_test, base_pred, base_proba = test_metrics(baseline, X_test, y_test)

    # ------- Tuned (balanced classes + grid search + tuned threshold) ------------
    print("[2/2] Tuned logistic regression: grid search ...")
    grid = GridSearchCV(
        build_pipeline(X_train, class_weight="balanced"),
        param_grid={
            **PARAM_GRID,
            # Interactions: off or on (depends on the data columns)
            "preprocessor__interactions": ["drop", build_interactions(X_train)],
        },
        scoring=GRID_SCORING,
        cv=cv,
        n_jobs=-1,  # run the fits in parallel on the CPU cores
        verbose=1,
    )
    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    grid_time = time.perf_counter() - t0
    best = short_params(grid.best_estimator_)
    print(f"  Best: C={best['C']}, log={best['log_transform']}, "
          f"spline_knots={best['spline_knots']}, interactions={best['interactions']} "
          f"| CV PR-AUC = {grid.best_score_:.4f}")

    print(f"  Tuning decision threshold (precision floor {PRECISION_FLOOR}) ...")
    tuned = TunedThresholdClassifierCV(
        grid.best_estimator_,
        scoring=make_scorer(recall_at_precision_floor),
        cv=cv,
    )
    t0 = time.perf_counter()
    tuned.fit(X_train, y_train)  # threshold is chosen on training data only
    threshold_time = time.perf_counter() - t0
    search_time = grid_time + threshold_time
    threshold = float(tuned.best_threshold_)
    if tuned.best_score_ == 0:
        print(f"  WARNING: no threshold reached precision {PRECISION_FLOOR}")
    print(f"  Chosen threshold: {threshold:.4f}")
    tuned_cv = cv_metrics(tuned, X_train, y_train, cv)
    tuned_test, tuned_pred, tuned_proba = test_metrics(tuned, X_test, y_test)

    # ---------------- Save -------------------------------------------------------
    rows = [
        {
            "task": "classification", "model": MODEL_NAME, "config": "baseline",
            "params": json.dumps(short_params(baseline)), "selection_metric": "",
            "fit_time_s": base_fit_time, "search_time_s": 0.0,
            **base_cv, **base_test,
        },
        {
            "task": "classification", "model": MODEL_NAME, "config": "tuned",
            "params": json.dumps(short_params(tuned.estimator_, threshold)),
            "selection_metric": f"PR-AUC (grid search), recall at precision >= {PRECISION_FLOOR} (threshold)",
            # the tuned wrapper's fit includes the threshold search
            "fit_time_s": threshold_time, "search_time_s": search_time,
            **tuned_cv, **tuned_test,
        },
    ]
    save_results(rows, out / "logreg_classification_results.csv")
    results = dict(grid.cv_results_)
    results["param_preprocessor__interactions"] = [
        "off" if str(v) == "drop" else "on"
        for v in results["param_preprocessor__interactions"]
    ]
    keep = [c for c in results if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time",
    ]
    (
        pd.DataFrame(results)[keep]
        .rename(columns={"mean_test_score": "mean_cv_pr_auc",
                         "std_test_score": "std_cv_pr_auc"})
        .sort_values("rank_test_score")
        .to_csv(out / "logreg_grid_search_results.csv", index=False)
    )
    save_plots(out, y_test, {
        "baseline": {"pred": base_pred, "proba": base_proba, "threshold": 0.5},
        "tuned": {"pred": tuned_pred, "proba": tuned_proba, "threshold": threshold},
    })
    joblib.dump(baseline, out / "logreg_classifier_baseline.joblib", compress=3)
    joblib.dump(tuned, out / "logreg_classifier.joblib", compress=3)

    print("\n=== Logistic regression results ===")
    print(f"  {'metric':<10} {'base CV':>8} {'tuned CV':>9} {'base test':>10} {'tuned test':>11}")
    for name in SCORING:
        print(f"  {name:<10} {base_cv[f'cv_{name}_mean']:>8.4f} "
              f"{tuned_cv[f'cv_{name}_mean']:>9.4f} "
              f"{base_test[f'test_{name}']:>10.4f} "
              f"{tuned_test[f'test_{name}']:>11.4f}")
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    # Run through the package path so the saved models can be loaded elsewhere.
    from pipeline.models.LogisticRegression import main as _main

    _main()