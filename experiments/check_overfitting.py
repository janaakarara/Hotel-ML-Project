"""Held-out log-likelihood check: does a larger k overfit?

For k = 1..K a GMM is fitted on 70% of the rows (PCA fitted on the same 70% only) and scored on the
other 30%. A model that generalises keeps a small train-test gap and its test log-likelihood stops
improving once k is large enough. A growing gap, or a falling test log-likelihood, means overfitting.

Run on four data sets: all rows and de-duplicated rows, for the full data and for the largest cluster
of the saved tuned model. (One-hot data has many identical rows, so the all-rows split leaks a little
and is slightly optimistic; the de-duplicated split does not, but measures distinct booking profiles.)

Usage (repo root, after gmm_tuned has run):
  python -m src.clustering.gmm_heldout_check
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from src.clustering.gmm_untuned import base_arg_parser, load_data, to_model_space

SEED = 42
K_MAX = 8
REG_COVARS = (1e-4, 1e-2)


def evaluate(X, name, dedup):
    Xc = np.unique(X, axis=0) if dedup else X
    print(f"\n=== {name} | {'unique rows' if dedup else 'all rows'}: {len(Xc)} of {len(X)} ===")
    Xtr, Xte = train_test_split(Xc, test_size=0.3, random_state=SEED)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=SEED).fit(Xtr)   # train only
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    print(f"PCA dims kept: {pca.n_components_}")
    for reg in REG_COVARS:
        print(f"\nreg_covar = {reg:g}")
        print(f"{'k':>2} | {'train LL':>9} | {'test LL':>9} | {'gap':>7}")
        print("-" * 36)
        tests = []
        for k in range(1, K_MAX + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                g = GaussianMixture(n_components=k, covariance_type="full", reg_covar=reg, n_init=5,
                                    init_params="k-means++", max_iter=300, random_state=SEED).fit(Ztr)
            tr, te = g.score(Ztr), g.score(Zte)
            tests.append(te)
            print(f"{k:>2} | {tr:>9.4f} | {te:>9.4f} | {tr - te:>7.4f}")
        print(f"best test LL at k = {int(np.argmax(tests)) + 1}")


def main():
    args = base_arg_parser("Held-out log-likelihood check.").parse_args()
    X_df, X, raw_df, prof_df = load_data(args.features, args.raw, args.profile_cols)

    pipe = joblib.load(Path(args.artifacts_dir) / "gmm_tuned_model.joblib")   # the model you reported
    labels = pipe.named_steps["gmm"].predict(to_model_space(pipe, X))
    big = int(np.argmax(np.bincount(labels)))
    print(f"Largest cluster of the saved tuned model: C{big} ({(labels == big).mean() * 100:.1f}%)")

    for dedup in (False, True):
        evaluate(X, "Full data", dedup)
        evaluate(X[labels == big], f"Largest cluster C{big}", dedup)


if __name__ == "__main__":
    main()