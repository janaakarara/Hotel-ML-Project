"""The ONE pipeline all three models use:

    the teammate's extra features (pipeline.py) -> our preprocessing -> the model

Everything in the preprocessing LEARNS from data (medians, means/std, category
lists), so it sits inside a scikit-learn Pipeline: during cross-validation and training it is fitted on
the training rows only, and then simply applied to validation/test/new rows.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from pipeline import HotelFeatureEngineer  # the teammate's file, unchanged

# The 12 columns the teammate's HotelFeatureEngineer adds to every booking: stay length, party size,
# log of the lead time, and the arrival month / weekday / week as sin/cos waves.
EXTRA_FEATURES = ["LogLeadTime", "TotalNights", "WeekendShare", "TotalGuests", "HasChildren", "HasCompany",
                  "month_sin", "month_cos", "dow_sin", "dow_cos", "week_sin", "week_cos"]


def preprocessor():
    """Turns the table of bookings into a table of numbers a model can use."""
    numeric_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),         # a safety net for gaps in new bookings
        ("scale", StandardScaler()),                                 # mean 0, std 1 for every column
    ])
    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="constant", fill_value="Unknown")),  # e.g. 3 missing Country
        # One 0/1 column per category. Categories seen in under 0.5% of training rows are grouped
        # into one "infrequent" column; a category never seen before also goes there.
        ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                  sparse_output=False)),  # dense: same results as sparse, much faster
    ])
    return ColumnTransformer([
        ("numeric", numeric_steps, NUMERIC_FEATURES + EXTRA_FEATURES),
        ("categorical", categorical_steps, CATEGORICAL_FEATURES),
    ])  # every column not listed (e.g. the leaky ones) is dropped


def regression_pipeline(model):
    """Each model's config.py calls this with its own model."""
    return Pipeline([
        ("features", HotelFeatureEngineer()),
        ("prep", preprocessor()),
        ("model", model),
    ])
