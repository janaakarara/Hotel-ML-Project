import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.clustering_pipeline import ClusteringPreprocessor  # noqa: E402

RAW = ROOT / "data/processed/h2_clustering_stage2.csv"
REF = ROOT / "data/processed/h2_clustering_stage5.csv"
needs_data = pytest.mark.skipif(not (RAW.exists() and REF.exists()), reason="processed data not found")

BOOKING = {"Adults": 2, "MarketSegment": "Online TA", "Meal": "BB", "Country": "DEU",
           "CustomerType": "Transient", "ReservedRoomType": "A", "ArrivalDateMonth": "July",
           "LeadTime": 40, "StaysInWeekNights": 3, "StaysInWeekendNights": 2, "Children": 0}


@pytest.fixture(scope="module")
def fitted():
    return ClusteringPreprocessor().fit(pd.read_csv(RAW))


@needs_data
def test_matches_stage_files(fitted):
    Z, ref = fitted.transform(pd.read_csv(RAW)), pd.read_csv(REF)
    assert list(Z.columns) == list(ref.columns)
    assert np.abs(Z.values - ref.values).max() < 1e-5


@needs_data
def test_single_booking(fitted):
    z = fitted.transform(pd.DataFrame([BOOKING]))
    assert z.shape == (1, 36) and not z.isna().any().any()
    assert z.loc[0, "Country_DEU"] == 1 and z.loc[0, "Meal_BB"] == 1


@needs_data
def test_unknown_values_do_not_crash(fitted):
    z = fitted.transform(pd.DataFrame([{**BOOKING, "Country": "JPN", "Meal": "ZZ", "Children": np.nan}]))
    assert z.loc[0, "Country_Asia"] == 1 and z.loc[0, "HasChildren"] < 0
    assert z[[c for c in z.columns if c.startswith("Meal_")]].sum(axis=1).iloc[0] == 0
