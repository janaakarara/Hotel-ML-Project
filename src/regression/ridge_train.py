"""Train, evaluate and persist the Ridge Regression ADR pipeline.

Run from the project root::

    python -m src.regression.ridge_train                   # default alpha=1.0 (baseline)
    python -m src.regression.ridge_train --alpha 10.0      # tuned regularized variant
    python -m src.regression.ridge_train --pca 0.95        # with PCA dimensionality reduction
    python -m src.regression.ridge_train --split random    # random stratified split
    python -m src.regression.ridge_train --split time      # time-based split

Preprocessing, split, CV, metrics, plots and artifacts are shared across models
through ``src.regression.common`` and ``src.regression.pipeline``.
"""

from __future__ import annotations

import logging

from sklearn.linear_model import Ridge

from src.regression.common import build_parser, run_experiment

KEY = "ridge"
FAMILY = "linear"


def make_estimator(seed: int, n_jobs: int, alpha: float = 1.0) -> Ridge:
    """L2-regularized Ridge regressor."""
    return Ridge(alpha=alpha, random_state=seed)


def main() -> None:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="L2 regularization strength (default: 1.0; use e.g. 10.0 for tuned model)",
    )
    args = parser.parse_args()
    run_experiment(args, KEY, FAMILY, make_estimator, alpha=args.alpha)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
