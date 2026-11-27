"""Feature engineering and preprocessing for the ADR regression task.

One module, three models (Linear/Ridge, Random Forest, XGBoost).  The models
want different inputs, so ``build_pipeline`` takes a ``family`` argument:

===============  ==========================================  ================================================
aspect           ``family="linear"``                         ``family="tree"``
===============  ==========================================  ================================================
numeric scaling  StandardScaler                              none
exact-sum cols   dropped (TotalNights, TotalGuests)          kept (trees cannot split on sums)
heavy tails      log1p(PreviousCancellations, ...)           raw counts
time             ArrivalTimeIdx + sin/cos month/week/dow     ArrivalTimeIdx + raw month/week/day-of-year/dow
lead time        LeadTime + LogLeadTime                      LeadTime only (log is monotonic = no gain)
low-card cats    one-hot, ``drop="first"``                   one-hot
Country / Agent  one-hot (rare levels pooled)                TargetEncoder (cross-fitted, leak-safe)
interactions     RoomMonth (room type x month) one-hot       learned by the trees
log1p(ADR)       yes (default)                               no (default; measured better on H2)
min_frequency    0.001                                       0.005 (affects low-cardinality one-hot only)
output           dense                                       dense
===============  ==========================================  ================================================

Everything that learns from data (imputation medians, scaling statistics, one-hot
levels, target encodings, PCA) lives inside the sklearn ``Pipeline`` and is
therefore fitted on training folds only.  ``HotelFeatureEngineer`` is stateless.

Public API
----------
``HotelFeatureEngineer``     stateless derived-feature transformer
``build_preprocessor``       ``ColumnTransformer`` for a model family
``build_pipeline``           features -> preprocessor -> [scaler -> reducer] -> log1p-target model
``prefix_model_params``      adds the ``model__regressor__`` prefix for tuning grids
``get_numeric_features`` / ``get_categorical_features`` / ``get_feature_names``
``arrival_dates``            arrival date as ``datetime64`` (also used for time/group splits)
"""

from __future__ import annotations

import calendar
from typing import Any, Final, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

Family = Literal["linear", "tree"]

# --------------------------------------------------------------------------
# Target and leakage audit
# --------------------------------------------------------------------------
TARGET: Final[str] = "ADR"

#: Rows above this ADR are dropped at load time and predictions are clipped to it.
MAX_ADR: Final[float] = 1000.0

#: Outcome leaks and check-in-dependent fields: never allowed as features.
LEAKAGE_COLUMNS: Final[tuple[str, ...]] = (
    "ReservationStatus",       # fatal outcome leak
    "ReservationStatusDate",   # fatal outcome leak (date the final status was set)
    "IsCanceled",              # classification target, unknown at quote time
    "AssignedRoomType",        # assigned at/after check-in; use ReservedRoomType
)

#: Fields that materialise AFTER the booking is created (prediction point =
#: "booking just confirmed"): amendments, waiting list, parking and special
#: requests all accumulate between booking and arrival.
LIFESPAN_COLUMNS: Final[tuple[str, ...]] = (
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
)

# --------------------------------------------------------------------------
# Feature lists (single source of truth for the ColumnTransformer)
# --------------------------------------------------------------------------
#: Raw columns the engineer needs.  Missing ones raise instead of silently
#: defaulting (``Company`` used to default to 0, creating train/serve skew).
REQUIRED_RAW_COLUMNS: Final[tuple[str, ...]] = (
    "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "LeadTime",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType", "Agent", "Company",
)

_LOW_CARD: Final[list[str]] = [
    "Meal", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType",
]
_HIGH_CARD: Final[list[str]] = ["Country", "Agent"]

#: Raw categorical inputs that are cleaned by the engineer.
CATEGORICAL_FEATURES: Final[list[str]] = _LOW_CARD + _HIGH_CARD

_COMMON_NUMERIC: Final[list[str]] = [
    "ArrivalTimeIdx",          # continuous months since Jul-2015: price level drifts ~13%/yr
    "StaysInWeekendNights", "StaysInWeekNights", "WeekendShare",
    "Adults", "Children", "Babies", "HasChildren",
    "IsRepeatedGuest", "HasCompany",
    "LeadTime",
]
_LINEAR_NUMERIC: Final[list[str]] = _COMMON_NUMERIC + [
    "LogLeadTime", "LogPreviousCancellations", "LogPreviousBookingsNotCanceled",
    "month_sin", "month_cos", "week_sin", "week_cos", "dow_sin", "dow_cos",
]
_TREE_NUMERIC: Final[list[str]] = _COMMON_NUMERIC + [
    "PreviousCancellations", "PreviousBookingsNotCanceled",
    "TotalNights", "TotalGuests",
    "ArrivalMonth", "ArrivalWeek", "ArrivalDayOfYear", "ArrivalDayOfWeek",
]

# RoomMonth (= ReservedRoomType x arrival month) already spans the ReservedRoomType
# main effect: keeping both makes the one-hot block exactly collinear, so the plain
# room-type column is left out of the linear family.
_LINEAR_CATEGORICAL: Final[list[str]] = (
    [c for c in _LOW_CARD if c != "ReservedRoomType"] + _HIGH_CARD + ["RoomMonth"]
)

#: Per-family defaults chosen by 5-fold grouped CV on H2 (train split only).
#: Linear underfits and profits from more Agent/Country levels; trees do not.
DEFAULT_MIN_FREQUENCY: Final[dict[str, float]] = {"linear": 0.001, "tree": 0.005}
#: log1p(ADR) helps linear models (MAE 14.8 vs 15.6) but hurts boosted trees
#: (10.4 vs 10.2) because trees already model the skew through splits.
DEFAULT_LOG_TARGET: Final[dict[str, bool]] = {"linear": True, "tree": False}

_MONTHS: Final[dict[str, int]] = {
    name: i for i, name in enumerate(calendar.month_name) if name
}
#: Fixed epoch for the continuous time index (first arrival date in H2).
_EPOCH: Final[pd.Timestamp] = pd.Timestamp("2015-07-01")


def get_numeric_features(family: Family = "tree") -> list[str]:
    """Numeric feature names used by ``family``."""
    return list(_LINEAR_NUMERIC if family == "linear" else _TREE_NUMERIC)


def get_categorical_features(family: Family = "tree") -> list[str]:
    """Categorical feature names used by ``family`` (incl. engineered ones)."""
    return list(_LINEAR_CATEGORICAL if family == "linear" else CATEGORICAL_FEATURES)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _month_number(series: pd.Series) -> pd.Series:
    """Convert month names ('July') or numbers (7, '7') to integers 1-12."""
    numeric = pd.to_numeric(series, errors="coerce")
    named = series.astype(str).str.strip().str.title().map(_MONTHS)
    return numeric.fillna(named)


def arrival_dates(X: pd.DataFrame) -> pd.Series:
    """Arrival date as ``datetime64`` (``NaT`` for invalid components)."""
    return pd.to_datetime(
        {
            "year": pd.to_numeric(X["ArrivalDateYear"], errors="coerce"),
            "month": _month_number(X["ArrivalDateMonth"]),
            "day": pd.to_numeric(X["ArrivalDateDayOfMonth"], errors="coerce"),
        },
        errors="coerce",
    )


def _add_cyclical(out: pd.DataFrame, values: pd.Series, period: float, name: str) -> None:
    angle = 2.0 * np.pi * values.astype(float) / period
    out[f"{name}_sin"] = np.sin(angle)
    out[f"{name}_cos"] = np.cos(angle)


def _count(out: pd.DataFrame, col: str) -> pd.Series:
    """Numeric count column: coerce, NaN -> 0, non-negative."""
    return pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)


def _clean_categorical_series(series: pd.Series, is_agent: bool = False) -> pd.Series:
    """Cast to clean strings; every missing variant becomes ``'Unknown'``.

    Agent IDs are identifiers: ``9``, ``9.0`` and ``'9'`` all become ``'9'``;
    ``'NULL'``/NaN/non-numeric become ``'Unknown'`` (= no travel agency).
    """
    if is_agent:
        numeric_agent = pd.to_numeric(series, errors="coerce")
        valid = numeric_agent.notna() & (numeric_agent >= 0)
        result = pd.Series("Unknown", index=series.index, dtype=object)
        result[valid] = numeric_agent[valid].astype(int).astype(str)
        return result

    clean = series.astype(object).where(series.notna(), "Unknown").astype(str).str.strip()
    missing_tokens = {"", "unknown", "nan", "none", "null", "<na>", "na", "undefined"}
    clean[clean.str.lower().isin(missing_tokens)] = "Unknown"
    return clean


# --------------------------------------------------------------------------
# Feature engineering (stateless)
# --------------------------------------------------------------------------
class HotelFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the union of derived booking-stage features for both families.

    The ``ColumnTransformer`` later selects the subset a given family needs, so
    this class has no hyper-parameters and learns nothing in ``fit`` -- it can
    never leak information across CV folds or train/test splits.

    Missing-value policy (single, explicit): counts and lead time -> 0 (a missing
    count means "none recorded"); arrival-date parts that cannot be parsed stay
    NaN and are median-imputed by the numeric pipeline (never happens on H2).
    ``Agent``/``Company`` arrive as the string ``'none'`` in the cleaned CSV; both
    become "no agent" / ``HasCompany = 0``.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "HotelFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        out = X.copy()

        # 1. Calendar features from the arrival date ------------------------
        arrival = arrival_dates(out)
        month = arrival.dt.month.astype(float)
        week = arrival.dt.isocalendar().week.astype(float)
        dow = arrival.dt.dayofweek.astype(float)
        out["ArrivalMonth"] = month
        out["ArrivalWeek"] = week
        out["ArrivalDayOfYear"] = arrival.dt.dayofyear.astype(float)
        out["ArrivalDayOfWeek"] = dow
        _add_cyclical(out, month, 12.0, "month")
        _add_cyclical(out, week, 53.0, "week")
        _add_cyclical(out, dow, 7.0, "dow")

        out["ArrivalTimeIdx"] = (arrival - _EPOCH).dt.days.astype(float) / 30.4375  # months since Jul-2015

        # 2. Stay structure ---------------------------------------------------
        weekend = _count(out, "StaysInWeekendNights")
        weeknights = _count(out, "StaysInWeekNights")
        out["StaysInWeekendNights"] = weekend
        out["StaysInWeekNights"] = weeknights
        total_nights = weekend + weeknights
        out["TotalNights"] = total_nights
        out["WeekendShare"] = np.where(total_nights > 0, weekend / total_nights.where(total_nights > 0, 1.0), 0.0)

        # 3. Party composition -------------------------------------------------
        adults, children, babies = (_count(out, c) for c in ("Adults", "Children", "Babies"))
        out["Adults"], out["Children"], out["Babies"] = adults, children, babies
        out["TotalGuests"] = adults + children + babies
        out["HasChildren"] = ((children + babies) > 0).astype(int)

        # 4. Lead time and customer history -------------------------------------
        lead = pd.to_numeric(out["LeadTime"], errors="coerce").fillna(0.0).clip(lower=0.0)
        out["LeadTime"] = lead
        out["LogLeadTime"] = np.log1p(lead)
        prev_c = _count(out, "PreviousCancellations")
        prev_n = _count(out, "PreviousBookingsNotCanceled")
        out["PreviousCancellations"], out["PreviousBookingsNotCanceled"] = prev_c, prev_n
        out["LogPreviousCancellations"] = np.log1p(prev_c)
        out["LogPreviousBookingsNotCanceled"] = np.log1p(prev_n)
        out["IsRepeatedGuest"] = pd.to_numeric(out["IsRepeatedGuest"], errors="coerce").fillna(0)

        # 5. Company indicator ----------------------------------------------------
        company = pd.to_numeric(out["Company"], errors="coerce").fillna(0)
        out["HasCompany"] = (company > 0).astype(int)

        # 6. Categorical normalisation + room x season interaction -----------------
        for col in CATEGORICAL_FEATURES:
            out[col] = _clean_categorical_series(out[col], is_agent=(col == "Agent"))
        month_label = month.astype("Int64").astype(object).where(month.notna(), "NA").map(str)
        out["RoomMonth"] = out["ReservedRoomType"] + "_" + month_label

        return out


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def _target_encoder(seed: int) -> TargetEncoder:
    """Seeded, cross-fitted target encoder for any sklearn >= 1.3.

    sklearn 1.9 deprecates ``random_state`` in favour of passing a splitter via ``cv``.
    """
    major_minor = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
    if major_minor >= (1, 9):
        return TargetEncoder(
            target_type="continuous", smooth="auto",
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        )
    return TargetEncoder(
        target_type="continuous", smooth="auto", cv=5, shuffle=True, random_state=seed
    )


def build_preprocessor(
    family: Family = "tree",
    scale_numeric: bool | None = None,
    min_frequency: float | None = None,
    seed: int = 42,
) -> ColumnTransformer:
    """Build the ``ColumnTransformer`` for ``family``.

    Parameters
    ----------
    scale_numeric : default ``None`` -> scale for ``"linear"``, not for ``"tree"``.
    min_frequency : one-hot levels rarer than this share are pooled into one
        ``infrequent`` column (default per family, see ``DEFAULT_MIN_FREQUENCY``).
        Tunable via ``preprocessor__cat__min_frequency``.
    seed : seed of the cross-fitted ``TargetEncoder`` (tree family).

    Output is always dense (``sparse_threshold=0``): sklearn's ``LinearRegression``
    silently switches to an iterative LSQR solver on sparse input, and PCA needs
    dense data.  ``remainder="drop"`` discards every column not listed here.
    """
    if scale_numeric is None:
        scale_numeric = family == "linear"
    if min_frequency is None:
        min_frequency = DEFAULT_MIN_FREQUENCY[family]

    num_steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    def ohe(drop: str | None) -> OneHotEncoder:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            drop=drop,
            sparse_output=False,
        )

    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", Pipeline(num_steps), get_numeric_features(family))
    ]
    if family == "linear":
        # drop="first" removes the dummy-variable trap (OLS needs full column rank)
        transformers.append(("cat", ohe("first"), _LINEAR_CATEGORICAL))
    else:
        transformers.append(("cat", ohe(None), _LOW_CARD))
        # Cross-fitted inside fit_transform -> no target leakage within a fold.
        transformers.append(("cat_high", _target_encoder(seed), _HIGH_CARD))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------
# Target transform and pipeline assembly
# --------------------------------------------------------------------------
def _inv_log1p_clipped(z: np.ndarray) -> np.ndarray:
    """expm1 with the log-space prediction clipped to [0, log1p(MAX_ADR)].

    Module-level (not a lambda) so fitted pipelines pickle with joblib.
    Prevents extrapolating linear models from emitting negative or astronomic ADR.
    """
    return np.expm1(np.clip(z, 0.0, np.log1p(MAX_ADR)))


def _identity(y: np.ndarray) -> np.ndarray:
    return y


def _clip_adr(y: np.ndarray) -> np.ndarray:
    return np.clip(y, 0.0, MAX_ADR)


def build_pipeline(
    model: RegressorMixin,
    family: Family = "tree",
    reducer: BaseEstimator | None = None,
    log_target: bool | None = None,
    min_frequency: float | None = None,
    seed: int = 42,
) -> Pipeline:
    """Assemble ``features -> preprocessor -> [post_scaler -> reducer] -> model``.

    ``model`` is wrapped in ``TransformedTargetRegressor`` so tuning parameters
    always carry the prefix ``model__regressor__`` (see ``prefix_model_params``).

    ``reducer`` (e.g. ``PCA``) is inserted after a ``StandardScaler`` over *all*
    preprocessed columns so no single block dominates the components.
    ``log_target=None`` uses the family default (``DEFAULT_LOG_TARGET``);
    ``False`` keeps ADR on its original scale (still clipped to ``[0, MAX_ADR]``).
    """
    steps: list[tuple[str, Any]] = [
        ("features", HotelFeatureEngineer()),
        ("preprocessor", build_preprocessor(family, min_frequency=min_frequency, seed=seed)),
    ]
    if reducer is not None:
        steps += [("post_scaler", StandardScaler()), ("reducer", reducer)]

    if log_target is None:
        log_target = DEFAULT_LOG_TARGET[family]
    if log_target:
        func, inverse = np.log1p, _inv_log1p_clipped
    else:
        func, inverse = _identity, _clip_adr
    steps.append((
        "model",
        TransformedTargetRegressor(
            regressor=model, func=func, inverse_func=inverse, check_inverse=False
        ),
    ))
    return Pipeline(steps)


MODEL_PREFIX: Final[str] = "model__regressor__"


def prefix_model_params(params: dict[str, Any]) -> dict[str, Any]:
    """``{'n_estimators': ...}`` -> ``{'model__regressor__n_estimators': ...}``."""
    return {f"{MODEL_PREFIX}{k}": v for k, v in params.items()}


def get_feature_names(pipeline: Pipeline) -> Sequence[str]:
    """Post-preprocessing feature names of a fitted pipeline."""
    return pipeline.named_steps["preprocessor"].get_feature_names_out()