"""Turn the chosen booking columns into one all-numeric table that K-Means can measure distances on."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.clustering_features import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_clustering_preprocessor():
    # Numbers: fill any gap with the median, then rescale to mean 0 and std 1,
    # so "days of lead time" and "number of adults" count equally in a distance.
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    # 0/1 columns stay 0/1. Scaling a rare flag (3% repeated guests) would blow
    # its 1s up to about 5.6 and let that one column dominate the distances.
    binary = SimpleImputer(strategy="most_frequent")
    # Text categories become one 0/1 column per value. "ignore" lets a new booking
    # with a value we never saw get all zeros instead of crashing.
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bin", binary, BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])  # every other column is dropped, so leaky columns can never sneak in
