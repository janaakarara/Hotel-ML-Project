"""Checks for the untuned / tuned GMM clustering artifacts.

Run after `python -m src.clustering.gmm_untuned` and `python -m src.clustering.gmm_tuned`.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:          # the tests can also be called directly without pytest
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering.gmm_untuned import n_free_parameters  # noqa: E402
from src.features.clustering_pipeline import ClusteringPreprocessor  # noqa: E402

STAGE5 = ROOT / "data/processed/h2_clustering_stage5.csv"
RAW = ROOT / "data/processed/h2_clustering_stage2.csv"
PREP_PARAMS = ROOT / "artifacts/clustering_preprocessor_params.json"
UNTUNED_MODEL = ROOT / "artifacts/gmm_untuned_model.joblib"
TUNED_MODEL = ROOT / "artifacts/gmm_tuned_model.joblib"
COMP_CSV = ROOT / "outputs/gmm/clustering_gmm_comparison.csv"

_ready = STAGE5.exists() and UNTUNED_MODEL.exists() and TUNED_MODEL.exists() and COMP_CSV.exists()
if pytest is not None:
    needs_artifacts = pytest.mark.skipif(not _ready, reason="run gmm_untuned and gmm_tuned first")
else:
    def needs_artifacts(fn):
        return fn

BOOKING = {"Adults": 2, "MarketSegment": "Online TA", "Meal": "BB", "Country": "FRA",
           "CustomerType": "Transient", "ReservedRoomType": "A", "ArrivalDateMonth": "August",
           "LeadTime": 95, "StaysInWeekNights": 4, "StaysInWeekendNights": 2, "Children": 0}


@needs_artifacts
def test_models_load_and_predict_on_feature_matrix():
    X = pd.read_csv(STAGE5).head(300).values
    for path in (UNTUNED_MODEL, TUNED_MODEL):
        pipe = joblib.load(path)
        labels, probs = pipe.predict(X), pipe.predict_proba(X)
        k = pipe.named_steps["gmm"].n_components
        assert labels.shape == (300,) and labels.min() >= 0 and labels.max() < k
        assert probs.shape == (300, k) and np.allclose(probs.sum(axis=1), 1.0)


@needs_artifacts
def test_tuned_beats_untuned_on_internal_metrics():
    comp = pd.read_csv(COMP_CSV).set_index("configuration")
    base, tuned = comp.loc["untuned_baseline"], comp.loc["tuned"]
    assert tuned["silhouette"] > base["silhouette"]
    assert tuned["davies_bouldin"] < base["davies_bouldin"]


@needs_artifacts
def test_free_parameter_count_matches_formula():
    gmm = joblib.load(UNTUNED_MODEL).named_steps["gmm"]
    k, d = gmm.means_.shape
    assert n_free_parameters(gmm) == k * d * (d + 1) // 2 + k * d + k - 1


@needs_artifacts
def test_new_booking_end_to_end():
    cat_weight = json.loads(PREP_PARAMS.read_text())["cat_weight"] if PREP_PARAMS.exists() else 1.0
    prep = ClusteringPreprocessor(cat_weight=cat_weight).fit(pd.read_csv(RAW))
    z = prep.transform(pd.DataFrame([BOOKING]))
    assert list(z.columns) == list(pd.read_csv(STAGE5, nrows=1).columns)
    pipe = joblib.load(TUNED_MODEL)
    assert 0 <= int(pipe.predict(z.values)[0]) < pipe.named_steps["gmm"].n_components


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
