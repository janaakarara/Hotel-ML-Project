"""Quick checks for the clustering pipeline. Run with:  python -m pytest -q"""
import pandas as pd

from src.clustering.kmeans_baseline import build_kmeans_pipeline
from src.config import load_config, project_path
from src.data.load import load_clean_data
from src.features.clustering_features import CLUSTER_FEATURES, FORBIDDEN_FEATURES, add_clustering_features

CONFIG = load_config()
SAMPLE = load_clean_data(project_path(CONFIG["paths"]["clean_data"])).sample(2000, random_state=0)


def test_no_outcome_column_is_a_clustering_input():
    assert not set(CLUSTER_FEATURES) & set(FORBIDDEN_FEATURES)


def test_zero_night_booking_gets_weekend_share_zero():
    row = SAMPLE.head(1).assign(StaysInWeekendNights=0, StaysInWeekNights=0)
    assert add_clustering_features(row)["WeekendShare"].iloc[0] == 0


def test_pipeline_gives_one_label_per_booking():
    labels = build_kmeans_pipeline(3, CONFIG["seed"]).fit_predict(SAMPLE)
    assert len(labels) == len(SAMPLE)
    assert set(labels) == {0, 1, 2}


def test_same_seed_gives_same_clusters():
    first = build_kmeans_pipeline(3, CONFIG["seed"]).fit_predict(SAMPLE)
    second = build_kmeans_pipeline(3, CONFIG["seed"]).fit_predict(SAMPLE)
    assert (first == second).all()


def test_changing_outcome_columns_does_not_change_clusters():
    pipeline = build_kmeans_pipeline(3, CONFIG["seed"]).fit(SAMPLE)
    flipped = SAMPLE.assign(IsCanceled=1 - SAMPLE["IsCanceled"], ADR=0.0, ReservationStatus="Check-Out")
    assert (pipeline.predict(SAMPLE) == pipeline.predict(flipped)).all()


def test_unseen_category_does_not_crash():
    pipeline = build_kmeans_pipeline(3, CONFIG["seed"]).fit(SAMPLE)
    new = SAMPLE.head(1).assign(MarketSegment="Brand new segment")
    assert pipeline.predict(new)[0] in {0, 1, 2}
