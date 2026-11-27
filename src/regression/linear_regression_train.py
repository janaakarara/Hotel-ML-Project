"""Train, evaluate and persist the Linear Regression ADR pipeline (baseline model).

Run from the project root::

    python -m src.regression.linear_regression_train
    python -m src.regression.linear_regression_train --pca 0.95
    python -m src.regression.linear_regression_train --split random
    python -m src.regression.linear_regression_train --split time

Preprocessing, split, CV, metrics, plots and artifacts are shared across models
through ``src.regression.common`` and ``src.regression.pipeline``.
"""

from __future__ import annotations

import logging

from sklearn.linear_model import LinearRegression

from src.regression.common import build_parser, run_experiment

KEY = "linear_regression"
FAMILY = "linear"


def make_estimator(seed: int, n_jobs: int) -> LinearRegression:
    """Untuned Ordinary Least Squares (OLS) baseline regressor."""
    return LinearRegression()


def main() -> None:
    parser = build_parser(__doc__.splitlines()[0])
    args = parser.parse_args()
    run_experiment(args, KEY, FAMILY, make_estimator)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
