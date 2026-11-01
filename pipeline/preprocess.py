from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.config import DATA


DROPPED_COLUMNS = [
    "ReservationStatus",
    "ReservationStatusDate",
    "RequiredCarParkingSpaces",
]
UNDEFINED_DISTRIBUTION_CHANNEL = "Undefined"
BOOKING_CHANGES_COLUMN = "BookingChanges"


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DropColumnsTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.columns) - set(X.columns))
        if missing:
            raise ValueError(f"Missing columns to drop: {missing}")
        return X.drop(columns=self.columns).copy()


class DropUndefinedDistributionChannelTransformer(
    BaseEstimator, TransformerMixin
):
    def fit(self, X: pd.DataFrame, y: Any = None) -> (
        "DropUndefinedDistributionChannelTransformer"
    ):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if "DistributionChannel" not in X.columns:
            raise ValueError("Missing required column: DistributionChannel")
        mask = X["DistributionChannel"].eq(UNDEFINED_DISTRIBUTION_CHANNEL)
        return X.loc[~mask].copy()


class BinarizeBookingChangesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Any = None) -> (
        "BinarizeBookingChangesTransformer"
    ):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if BOOKING_CHANGES_COLUMN not in X.columns:
            raise ValueError(
                f"Missing required column: {BOOKING_CHANGES_COLUMN}"
            )
        result = X.copy()
        result[BOOKING_CHANGES_COLUMN] = (
            result[BOOKING_CHANGES_COLUMN].fillna(0).ne(0).astype("int8")
        )
        return result


def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "drop_columns",
                DropColumnsTransformer(columns=DROPPED_COLUMNS),
            ),
            (
                "drop_undefined_distribution_channel",
                DropUndefinedDistributionChannelTransformer(),
            ),
            (
                "binarize_booking_changes",
                BinarizeBookingChangesTransformer(),
            ),
        ]
    )


def preprocess(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the sklearn preprocessing pipeline and return its summary."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    missing = sorted(
        (set(DROPPED_COLUMNS) | {"DistributionChannel", BOOKING_CHANGES_COLUMN})
        - set(data.columns)
    )
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    original_rows, original_columns = data.shape
    original_booking_changes = (
        data[BOOKING_CHANGES_COLUMN].value_counts(dropna=False).sort_index()
    )
    undefined_rows = int(
        data["DistributionChannel"].eq(UNDEFINED_DISTRIBUTION_CHANNEL).sum()
    )

    pipeline = build_preprocessing_pipeline()
    result = pipeline.fit_transform(data).reset_index(drop=True)
    columns_after_drop = result.columns.tolist()

    summary = {
        "input": {"rows": original_rows, "columns": original_columns},
        "output": {"rows": len(result), "columns": len(result.columns)},
        "dropped_columns": DROPPED_COLUMNS,
        "undefined_distribution_channel_rows_removed": undefined_rows,
        "booking_changes_original_value_counts": {
            str(k): int(v) for k, v in original_booking_changes.items()
        },
        "booking_changes_binarized_value_counts": {
            int(k): int(v)
            for k, v in result[BOOKING_CHANGES_COLUMN]
            .value_counts()
            .sort_index()
            .items()
        },
        "columns_after_drops": columns_after_drop,
        "verification": {
            "dropped_columns_absent": all(
                column not in result.columns for column in DROPPED_COLUMNS
            ),
            "undefined_rows_absent": not bool(
                result["DistributionChannel"]
                .eq(UNDEFINED_DISTRIBUTION_CHANNEL)
                .any()
            ),
            "booking_changes_binary": bool(
                set(result[BOOKING_CHANGES_COLUMN].dropna().unique()).issubset(
                    {0, 1}
                )
            ),
        },
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the sklearn H2 preprocessing pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DATA),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    raw = pd.read_csv(args.input)
    processed, summary = preprocess(raw)

    print(
        f"Preprocessed {len(raw):,} rows into {len(processed):,} rows "
        f"and {len(processed.columns)} columns."
    )
    print(f"Pipeline: {build_preprocessing_pipeline()}")


if __name__ == "__main__":
    main()
