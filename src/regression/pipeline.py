"""Feature engineering and preprocessing for the ADR regression task.

Everything that learns from data (imputation medians, scaling statistics,
one-hot categories) lives inside a scikit-learn ``Pipeline`` so it is fitted
on the training split only -- including inside every cross-validation fold.

Public API
----------
``HotelFeatureEngineer``   stateless transformer creating derived features
``build_preprocessor``     ``ColumnTransformer`` for numeric / categorical data
``build_pipeline``         full ``Pipeline`` ending in a log1p-target regressor
"""

from __future__ import annotations

import calendar
from typing import Final, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Target and leakage / lifespan audit
# --------------------------------------------------------------------------
TARGET: Final[str] = "ADR"

#: Outcome leaks and check-in-dependent fields: never allowed as features.
LEAKAGE_COLUMNS: Final[tuple[str, ...]] = (
    "ReservationStatus",       # fatal outcome leak
    "ReservationStatusDate",   # fatal outcome leak
    "IsCanceled",              # outcome unknown at quote time
    "AssignedRoomType",        # assigned at/after check-in -> use ReservedRoomType
)

#: Fields that only materialise AFTER the initial inquiry (lifespan audit).
LIFESPAN_COLUMNS: Final[tuple[str, ...]] = (
    "BookingChanges",     # amendments made between booking and arrival
    "DaysInWaitingList",  # known only once the booking is processed
)

#: Borderline fields (often added after the first quote). Kept by default,
#: dropped when ``strict=True`` / ``--strict`` is used.
STRICT_ONLY_COLUMNS: Final[tuple[str, ...]] = (
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
)

# --------------------------------------------------------------------------
# Feature lists (single source of truth for the ColumnTransformer)
# --------------------------------------------------------------------------
REQUIRED_RAW_COLUMNS: Final[tuple[str, ...]] = (
    "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "LeadTime",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType", "Agent",
)

CATEGORICAL_FEATURES: Final[list[str]] = [
    "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType", "Agent",
]

_BASE_NUMERIC_FEATURES: Final[list[str]] = [
    "LeadTime", "LogLeadTime", "ArrivalDateYear",
    "StaysInWeekendNights", "StaysInWeekNights", "TotalNights", "WeekendShare",
    "Adults", "Children", "Babies", "TotalGuests", "HasChildren",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "HasCompany",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "week_sin", "week_cos",
]

_MONTHS: Final[dict[str, int]] = {
    name: i for i, name in enumerate(calendar.month_name) if name
}


def get_numeric_features(strict: bool = True) -> list[str]:
    """Return numeric feature names; borderline fields are added if not strict."""
    features = list(_BASE_NUMERIC_FEATURES)
    if not strict:
        features += list(STRICT_ONLY_COLUMNS)
    return features


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def _add_cyclical(out: pd.DataFrame, values: pd.Series, period: float,
                  name: str) -> None:
    """Append ``<name>_sin`` / ``<name>_cos`` columns encoding a cyclic value."""
    angle = 2.0 * np.pi * values.astype(float) / period
    out[f"{name}_sin"] = np.sin(angle)
    out[f"{name}_cos"] = np.cos(angle)


def _month_number(series: pd.Series) -> pd.Series:
    """Convert month names ('July') or numbers to integers 1-12 (NaN if bad)."""
    numeric = pd.to_numeric(series, errors="coerce")
    named = series.astype(str).str.strip().str.title().map(_MONTHS)
    return numeric.fillna(named)


class HotelFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived booking-stage features.

    The transformer is stateless (nothing is learned in ``fit``), so it can
    never leak information from the validation or test data.

    Created features
    ----------------
    * ``month_*``, ``dow_*``, ``week_*``: sine/cosine encodings of arrival
      month, arrival weekday and ISO week, so December and January are close.
    * ``TotalNights``, ``WeekendShare``: stay length and share of weekend nights.
    * ``TotalGuests``, ``HasChildren``: party size indicators.
    * ``LogLeadTime``: log1p of the lead time (reduces skew for linear models).
    * ``HasCompany``: 1 if the booking is attached to a company.
    """

#This guarantees that it can never cause data leakage across cross-validation folds or train/test splits.
    def fit(self, X: pd.DataFrame, y=None) -> "HotelFeatureEngineer":
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with the engineered columns appended."""
        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        out = X.copy()

        # --- arrival date -> cyclical features --------------------------------
        month = _month_number(out["ArrivalDateMonth"])
        arrival = pd.to_datetime(
            {
                "year": pd.to_numeric(out["ArrivalDateYear"], errors="coerce"),
                "month": month,
                "day": pd.to_numeric(out["ArrivalDateDayOfMonth"], errors="coerce"),
            },
            errors="coerce",
        )
        _add_cyclical(out, arrival.dt.month, 12.0, "month")
        _add_cyclical(out, arrival.dt.dayofweek, 7.0, "dow")
        _add_cyclical(out, arrival.dt.isocalendar().week, 53.0, "week")

        # --- stay length and party size ---------------------------------------
        weekend = pd.to_numeric(out["StaysInWeekendNights"], errors="coerce")
        week = pd.to_numeric(out["StaysInWeekNights"], errors="coerce")
        children = pd.to_numeric(out["Children"], errors="coerce").fillna(0)
        babies = pd.to_numeric(out["Babies"], errors="coerce").fillna(0)
        adults = pd.to_numeric(out["Adults"], errors="coerce")

        out["Children"] = children
        out["Babies"] = babies
        out["TotalNights"] = weekend + week
        out["WeekendShare"] = weekend / out["TotalNights"].replace(0, np.nan)
        out["TotalGuests"] = adults + children + babies
        out["HasChildren"] = ((children + babies) > 0).astype(int)

        # --- misc numeric ------------------------------------------------------
        lead = pd.to_numeric(out["LeadTime"], errors="coerce")
        #ln(1 + \text{LeadTime})$ to compress extreme lead-time outliers and reduce distribution skewness.
        out["LogLeadTime"] = np.log1p(lead.clip(lower=0))
        if "Company" in out.columns:
            company = pd.to_numeric(out["Company"], errors="coerce").fillna(0)
            out["HasCompany"] = (company > 0).astype(int)
        else:
            out["HasCompany"] = 0

        # --- categorical clean-up (agent IDs are labels, not magnitudes) -------
        agent = pd.to_numeric(out["Agent"], errors="coerce")
        out["Agent"] = agent.fillna(-1).astype(int).astype(str)
        for col in CATEGORICAL_FEATURES:
            if col != "Agent":
                out[col] = out[col].astype(str).str.strip().replace(
                    {"nan": "Unknown", "NULL": "Unknown", "": "Unknown"}
                )
        return out


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def build_preprocessor(scale_numeric: bool = True,
                       strict: bool = False) -> ColumnTransformer:
    """Build the ``ColumnTransformer`` (unfitted).

    Parameters
    ----------
    scale_numeric:
        Standardise numeric features (needed for Ridge; harmless for trees).
    strict:
        If True, drop the borderline columns in ``STRICT_ONLY_COLUMNS``.

    Notes
    -----
    ``remainder="drop"`` means any column not listed here (including every
    leaky column) is discarded even if it is present in the input frame.
    """

    #ist of tuples representing the pipeline sequence for numerical data
    numeric_steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.005,       # rare levels (e.g. odd countries) pooled
            sparse_output=True,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), get_numeric_features(strict)),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model: RegressorMixin, scale_numeric: bool = True,
                   strict: bool = False) -> Pipeline:
    """Assemble ``features -> preprocessor -> log1p-target model``.

    The model is wrapped in ``TransformedTargetRegressor`` so it trains on
    ``log1p(ADR)`` while ``predict`` returns values on the original ADR scale.
    """
    return Pipeline([
        ("features", HotelFeatureEngineer()),
        ("preprocessor", build_preprocessor(scale_numeric, strict)),
        ("model", TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )),
    ])


def get_feature_names(pipeline: Pipeline) -> Sequence[str]:
    """Return post-preprocessing feature names of a fitted pipeline."""
    return pipeline.named_steps["preprocessor"].get_feature_names_out()