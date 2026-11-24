"""Reusable preprocessing for the clustering task.

One object that holds every rule from the preprocessing stages, so it can be fitted once,
saved, and reloaded to transform new bookings (the app) exactly like the training data.

Input : the 11 clustering source columns (RAW_COLUMNS), one row per booking.
Output: the 36 numeric features used for clustering (same order as h2_clustering_stage5.csv).

Rules (from the stage scripts):
  - Country: top 5 countries stay, the rest grouped by continent, unknown codes -> Other
  - Meal, CustomerType, MarketSegment, ReservedRoomType: rare values merged (MERGES)
  - LeadTime, StaysInWeekNights: Yeo-Johnson (lambda learned in fit); StaysInWeekendNights: sqrt
  - Children -> HasChildren (0/1);  Adults, the 3 transformed columns -> StandardScaler
  - ArrivalDateMonth -> sin / cos;  5 categorical columns -> one-hot
Fitting learns only the Yeo-Johnson lambdas, the scaler mean/std and the category lists.
"""
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson
from sklearn.base import BaseEstimator, TransformerMixin

RAW_COLUMNS = ["Adults", "MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType",
               "ArrivalDateMonth", "LeadTime", "StaysInWeekNights", "StaysInWeekendNights", "Children"]
CATEGORICAL = ["MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType"]
NUMERIC_OUT = ["Adults", "LeadTime_yj", "StaysInWeekNights_yj", "StaysInWeekendNights_sqrt"]
MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}

TOP_COUNTRIES = ["PRT", "FRA", "GBR", "DEU", "ESP"]
EUROPE = [
    "ALB", "AND", "AUT", "BEL", "BGR", "BIH", "BLR", "CHE", "CZE", "DEU", "DNK", "ESP", "EST",
    "FIN", "FRA", "FRO", "GBR", "GGY", "GIB", "GRC", "HRV", "HUN", "IMN", "IRL", "ISL", "ITA",
    "JEY", "LIE", "LTU", "LUX", "LVA", "MCO", "MKD", "MLT", "MNE", "NLD", "NOR", "POL", "PRT",
    "ROU", "RUS", "SRB", "SVK", "SVN", "SWE", "UKR",
]
ASIA = [
    "ARE", "ARM", "AZE", "BGD", "BHR", "CHN", "CN", "CYP", "GEO", "HKG", "IDN", "IND", "IRN", "IRQ",
    "ISR", "JOR", "JPN", "KAZ", "KHM", "KOR", "KWT", "LAO", "LBN", "LKA", "MAC", "MDV", "MMR",
    "MYS", "OMN", "PAK", "PHL", "QAT", "SAU", "SGP", "SYR", "THA", "TJK", "TUR", "TWN", "UZB", "VNM",
]
SOUTH_AMERICA = ["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PER", "PRY", "SUR", "URY", "VEN"]
NORTH_AMERICA = [
    "ABW", "AIA", "BRB", "CRI", "CUB", "DMA", "DOM", "GLP", "GTM", "HND", "JAM", "KNA", "LCA",
    "MEX", "NIC", "PAN", "PRI", "SLV", "USA", "VGB",
]
AFRICA = [
    "AGO", "BEN", "BFA", "CAF", "CIV", "CMR", "COM", "CPV", "DZA", "EGY", "ETH", "GAB", "GHA",
    "GNB", "KEN", "LBY", "MAR", "MLI", "MOZ", "MRT", "MUS", "MYT", "NAM", "NGA", "RWA", "SDN",
    "SEN", "SLE", "STP", "SYC", "TGO", "TUN", "TZA", "UGA", "ZAF", "ZMB", "ZWE",
]
CONTINENTS = {"Rest of Europe": EUROPE, "Asia": ASIA, "South America": SOUTH_AMERICA,
              "North America": NORTH_AMERICA, "Africa": AFRICA}
CODE_TO_GROUP = {code: name for name, codes in CONTINENTS.items() for code in codes}
# anything not listed (Oceania, Unknown, TMP, ATA ...) becomes "Other"

MERGES = {
    "Meal": {"FB": "HB/FB", "HB": "HB/FB"},
    "CustomerType": {"Group": "Transient-Party/Group", "Transient-Party": "Transient-Party/Group"},
    "MarketSegment": {"Complementary": "Other", "Aviation": "Other"},
    "ReservedRoomType": {"G": "Other", "C": "Other"},
}


def _text(v):
    return "" if pd.isna(v) else str(v).strip()


def _month(v):
    if isinstance(v, str) and not v.strip().isdigit():
        if v.strip().title() not in MONTHS:
            raise ValueError(f"Unknown month: {v!r}")
        return MONTHS[v.strip().title()]
    return int(v)


class ClusteringPreprocessor(BaseEstimator, TransformerMixin):
    """cat_weight multiplies the one-hot columns (1.0 = no change)."""

    def __init__(self, cat_weight=1.0):
        self.cat_weight = cat_weight

    # ---------- steps that need no fitting ----------
    @staticmethod
    def _prepare(X):
        df = pd.DataFrame(X).copy()
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df[RAW_COLUMNS].reset_index(drop=True) if not isinstance(X, pd.DataFrame) else df[RAW_COLUMNS].copy()
        for c in CATEGORICAL:
            df[c] = df[c].map(_text).astype(object)
        df["Country"] = df["Country"].replace({"": "Unknown", "NULL": "Unknown"})
        df["Country"] = df["Country"].where(df["Country"].isin(TOP_COUNTRIES),
                                            df["Country"].map(CODE_TO_GROUP).fillna("Other"))
        for col, mapping in MERGES.items():
            df[col] = df[col].replace(mapping)
        df["ArrivalDateMonth"] = df["ArrivalDateMonth"].map(_month)
        df["Children"] = pd.to_numeric(df["Children"], errors="coerce").fillna(0)
        for c in ["Adults", "LeadTime", "StaysInWeekNights", "StaysInWeekendNights"]:
            df[c] = pd.to_numeric(df[c]).astype(float)
        return df

    def _numeric(self, g):
        return pd.DataFrame({
            "Adults": g["Adults"].values,
            "LeadTime_yj": yeojohnson(g["LeadTime"].values, lmbda=self.lambdas_["LeadTime_yj"]),
            "StaysInWeekNights_yj": yeojohnson(g["StaysInWeekNights"].values,
                                               lmbda=self.lambdas_["StaysInWeekNights_yj"]),
            "StaysInWeekendNights_sqrt": np.sqrt(g["StaysInWeekendNights"].values),
        })

    # ---------- fit / transform ----------
    def fit(self, X, y=None):
        g = self._prepare(X)
        self.lambdas_ = {"LeadTime_yj": float(yeojohnson(g["LeadTime"].values)[1]),
                         "StaysInWeekNights_yj": float(yeojohnson(g["StaysInWeekNights"].values)[1])}
        num = self._numeric(g)
        self.mean_ = {c: float(num[c].mean()) for c in NUMERIC_OUT}
        self.std_ = {c: float(num[c].std(ddof=0)) for c in NUMERIC_OUT}
        self.categories_ = {c: sorted(g[c].unique().tolist()) for c in CATEGORICAL}
        self.feature_names_ = (NUMERIC_OUT + ["HasChildren", "ArrivalMonth_sin", "ArrivalMonth_cos"]
                               + [f"{c}_{v}" for c in CATEGORICAL for v in self.categories_[c]])
        return self

    def transform(self, X):
        g = self._prepare(X)
        num = self._numeric(g)
        out = pd.DataFrame({c: (num[c] - self.mean_[c]) / self.std_[c] for c in NUMERIC_OUT})
        out["HasChildren"] = (g["Children"] > 0).astype(int).values
        angle = 2 * np.pi * (g["ArrivalDateMonth"].values - 1) / 12
        out["ArrivalMonth_sin"] = np.sin(angle)
        out["ArrivalMonth_cos"] = np.cos(angle)
        for c in CATEGORICAL:
            for v in self.categories_[c]:          # a value not seen in fit gives all zeros
                out[f"{c}_{v}"] = (g[c] == v).astype(int).values * self.cat_weight
        return out[self.feature_names_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

    def to_dict(self):
        """Readable copy of everything that was learned (for the report)."""
        return {"cat_weight": self.cat_weight, "yeo_johnson_lambda": self.lambdas_,
                "scaler_mean": self.mean_, "scaler_std": self.std_,
                "categories": self.categories_, "feature_names": self.feature_names_}
