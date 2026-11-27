"""What every model does, in the same way. Each model's train.py calls train() with its config.py.

    baseline  the model with its default settings, scored with 5-fold cross-validation (CV) on the
              training set
    tuned     a grid search tries every combination of settings in the model's GRID, also with
              5-fold CV on the training set, and keeps the one with the lowest CV MAE
    test      the tuned model is scored once on the test set (the test set never chooses anything)
    save      both models (.joblib) and the result tables and figure, in the model's own folder
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate

from config import INPUTS, TARGET
from data import train_test

# MAE = average error in money; RMSE punishes big errors more; R2 = share of ADR variation explained.
SCORING = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}


def cv_columns(scores, prefix):
    """scikit-learn treats 'higher is better', so it reports errors as negative numbers
    (scorers named 'neg_...'). Flip those back to normal positive errors."""
    return {f"cv_{name}": (-1 if scorer.startswith("neg_") else 1) * scores[prefix + name]
            for name, scorer in SCORING.items()}


def test_scores(model, X, y):
    predicted = model.predict(X)
    mse = mean_squared_error(y, predicted)
    return {"test_MAE": mean_absolute_error(y, predicted), "test_MSE": mse,
            "test_RMSE": mse ** 0.5, "test_R2": r2_score(y, predicted)}


def train(model):
    """Train one model. model = that model's config.py (its NAME, FOLDER, MODEL and GRID)."""
    outputs, artifacts = model.FOLDER / "outputs", model.FOLDER / "artifacts"
    outputs.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    train_rows, test_rows = train_test()
    X_train, y_train = train_rows[INPUTS], train_rows[TARGET]
    X_test, y_test = test_rows[INPUTS], test_rows[TARGET]

    print(f"{model.NAME}: baseline (default settings), 5-fold cross-validation...")
    start = time.perf_counter()
    scores = cross_validate(model.MODEL, X_train, y_train, cv=5, scoring=SCORING)
    means = {key: values.mean() for key, values in scores.items()}
    rows = [{"model": model.NAME, "version": "baseline", "settings": "defaults",
             **cv_columns(means, "test_"), "seconds": round(time.perf_counter() - start, 1)}]
    baseline = clone(model.MODEL).fit(X_train, y_train)  # a fitted copy on all training rows, for /predict

    print(f"{model.NAME}: tuning (grid search over {model.GRID})...")
    start = time.perf_counter()
    # "model__" points each setting at the pipeline's step named "model" (see shared_pipeline.py).
    search = GridSearchCV(model.MODEL, {"model__" + key: values for key, values in model.GRID.items()},
                          cv=5, scoring=SCORING, refit="MAE")
    search.fit(X_train, y_train)  # also refits the best settings on the whole training set
    best = {key: values[search.best_index_] for key, values in search.cv_results_.items()}
    settings = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    rows.append({"model": model.NAME, "version": "tuned", "settings": str(settings),
                 **cv_columns(best, "mean_test_"), "seconds": round(time.perf_counter() - start, 1)})
    tuned = search.best_estimator_  # the best settings, fitted on all training rows

    # Every setting the grid search tried, with its cross-validated scores (rank 1 = lowest MAE).
    table = pd.DataFrame(search.cv_results_)
    tried = pd.DataFrame({
        "settings": [str({key.removeprefix("model__"): value for key, value in params.items()})
                     for params in table["params"]],
        **cv_columns(table, "mean_test_"),
        "rank": table["rank_test_MAE"]})

    # The test set: used once, on the tuned model, after every choice has been made.
    test_table = pd.DataFrame([{"model": model.NAME, **test_scores(tuned, X_test, y_test)}])

    joblib.dump(baseline, artifacts / "baseline.joblib", compress=3)
    joblib.dump(tuned, artifacts / "tuned.joblib", compress=3)
    results = pd.DataFrame(rows)
    results.to_csv(outputs / "results.csv", index=False)
    tried.to_csv(outputs / "search.csv", index=False)
    test_table.to_csv(outputs / "test_results.csv", index=False)
    (outputs / "tuning.json").write_text(json.dumps({
        "model": model.NAME,
        "optimization metric": "MAE, 5-fold cross-validation on the training set",
        "search space": model.GRID,
        "best settings": settings,
    }, indent=2, default=str))  # default=str: write objects such as Ridge() as text

    # How good is the tuned model on the test set? Points near the red line = good.
    predicted = tuned.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, predicted, s=3, alpha=0.3)
    axes[0].plot([0, 500], [0, 500], color="red")
    axes[0].set(xlabel="actual ADR", ylabel="predicted ADR", title=f"{model.NAME} (tuned): test set")
    axes[1].hist(y_test - predicted, bins=60)
    axes[1].set(xlabel="actual - predicted ADR", title="Errors (residuals) on the test set")
    plt.tight_layout()
    plt.savefig(outputs / "test_predictions.png")
    plt.close("all")

    folder = model.FOLDER.name
    print("\nCROSS-VALIDATION ON THE TRAINING SET\n" + results.round(3).to_string(index=False))
    print("\nTEST SET (tuned model)\n" + test_table.round(3).to_string(index=False))
    print(f"\nSaved: {folder}/artifacts/baseline.joblib and tuned.joblib; tables and figure in {folder}/outputs/")
