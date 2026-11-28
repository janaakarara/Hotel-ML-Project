"""Custom transformers shared by the classification models."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log(1 + x) when apply=True; negative values are clipped to 0."""

    def __init__(self, apply: bool = False) -> None:
        self.apply = apply

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log1p(np.clip(X, 0, None)) if self.apply else X
