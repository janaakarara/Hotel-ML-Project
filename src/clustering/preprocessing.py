"""Look at every column of the clustering CSV and decide how it must be prepared.

Each column ends up in exactly ONE of three groups:

  ENCODE   text categories           -> one 0/1 column per category (one-hot)
  SCALE    numbers on a ruler        -> (log1p if long right tail) then StandardScaler
  BOOLEAN  skewed "almost always 0"  -> a 0/1 flag ("is it different from the usual value?")

Columns that are already 0/1 are left alone (scaling a rare flag would blow its 1s up).

Run from the project root:
    python -m src.preprocessing.feature_router
    python -m src.preprocessing.feature_router --input data/processed/h2_clustering_raw_features.csv

It prints what it decided and what changed, and writes:
    data/processed/h2_clustering_preprocessed.csv   <- the clean, all-numeric table
    outputs/tables/preprocessing_decisions.csv      <- one row per column: group + reason
    artifacts/clustering_preprocessor.joblib        <- fitted transformer, reused at inference
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# Project root = two folders above this file (src/clustering/preprocessing.py -> project root)
ROOT = Path(__file__).resolve().parents[2]

# ---- the rules (change a number here and the whole decision table follows) ----
DOMINANT_SHARE = 0.85    # one value covers >= 85% of rows ...
SKEW_LIMIT = 1.0         # ... and |skew| >= 1  -> BOOLEAN flag
RARE_CATEGORY = 0.01     # categories below 1% of rows are merged into one "infrequent" column

DEFAULT_INPUT = "data/processed/h2_clustering_raw_features.csv"
DEFAULT_OUTPUT = "data/processed/h2_clustering_preprocessed_v0.csv"
DEFAULT_TABLE = "outputs/tables/preprocessing_decisions.csv"
DEFAULT_MODEL = "artifacts/clustering_preprocessor.joblib"


class NotUsualValueFlag(BaseEstimator, TransformerMixin):
    """1 if the value differs from the most common value seen in fit(), else 0.

    The usual value is learned in fit(), so a new booking is judged against the
    same reference as the training rows (and the transformer can be saved).
    """

    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else None
        X = np.asarray(X, dtype=float)
        self.usual_ = np.array([pd.Series(X[:, j]).mode().iloc[0] for j in range(X.shape[1])])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) != self.usual_).astype(int)

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.columns_
        return np.array([f"{c}_flag" for c in names])


# ------------------------------------------------------------------ decisions
def profile_and_decide(df):
    """Return one row per column: its statistics, the chosen group and the reason."""
    rows = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        row = {"column": col, "dtype": str(s.dtype), "missing": missing, "unique": int(s.nunique())}

        if not pd.api.types.is_numeric_dtype(s):
            shares = s.value_counts(normalize=True)
            rare = int((shares < RARE_CATEGORY).sum())
            row.update(group="ENCODE", skew=np.nan, dominant_share=round(float(shares.iloc[0]), 3),
                       log_first=False,
                       reason=f"text column, {row['unique']} categories -> one-hot"
                              + (f"; {rare} rare (<{RARE_CATEGORY:.0%}) merged into 'infrequent'" if rare else ""))
        else:
            values = s.dropna()
            skew = float(values.skew())
            dominant = float(values.value_counts(normalize=True).iloc[0])
            row.update(skew=round(skew, 2), dominant_share=round(dominant, 3), log_first=False)
            if set(values.unique()) <= {0, 1}:
                row.update(group="KEEP 0/1", reason="already a 0/1 flag -> left as is (scaling would inflate its rare 1s)")
            elif dominant >= DOMINANT_SHARE and abs(skew) >= SKEW_LIMIT:
                row.update(group="BOOLEAN",
                           reason=f"{dominant:.0%} of rows share one value and skew = {skew:.2f} -> keep only 'different from usual?'")
            elif skew >= SKEW_LIMIT:
                row.update(group="SCALE", log_first=True,
                           reason=f"spread-out values but long right tail (skew {skew:.2f}) -> log1p, then standardize")
            else:
                row.update(group="SCALE", reason=f"fairly symmetric (skew {skew:.2f}) -> standardize")
        if missing:
            row["reason"] += f"  [{missing} missing -> filled with the "
            row["reason"] += "most common category]" if row["group"] == "ENCODE" else "median]"
        rows.append(row)
    return pd.DataFrame(rows)


def build_preprocessor(decisions):
    """Build the sklearn transformer that applies the decisions."""
    by = lambda g: list(decisions.loc[decisions.group == g, "column"])
    scale_log = list(decisions.loc[(decisions.group == "SCALE") & decisions.log_first, "column"])
    scale_plain = list(decisions.loc[(decisions.group == "SCALE") & ~decisions.log_first, "column"])

    steps = []
    if scale_plain:
        steps.append(("scale", Pipeline([("fill", SimpleImputer(strategy="median")),
                                         ("std", StandardScaler())]), scale_plain))
    if scale_log:
        steps.append(("log_scale", Pipeline([("fill", SimpleImputer(strategy="median")),
                                             ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                             ("std", StandardScaler())]), scale_log))
    if by("BOOLEAN"):
        steps.append(("boolean", Pipeline([("fill", SimpleImputer(strategy="median")),
                                           ("flag", NotUsualValueFlag())]), by("BOOLEAN")))
    if by("KEEP 0/1"):
        steps.append(("keep01", SimpleImputer(strategy="most_frequent"), by("KEEP 0/1")))
    if by("ENCODE"):
        steps.append(("encode", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                                   min_frequency=RARE_CATEGORY,
                                                                   sparse_output=False))]), by("ENCODE")))
    return ColumnTransformer(steps, verbose_feature_names_out=False).set_output(transform="pandas")


# ------------------------------------------------------------------ reporting
def line(char="-", n=100):
    print(char * n)


def print_decisions(decisions):
    print("\nSTEP 1 - what group does each column belong to?")
    line()
    print(f"{'column':22s}{'type':9s}{'skew':>7s}{'top-value':>11s}   {'GROUP':10s} reason")
    line()
    for r in decisions.itertuples():
        skew = "" if pd.isna(r.skew) else f"{r.skew:.2f}"
        print(f"{r.column:22s}{r.dtype[:8]:9s}{skew:>7s}{r.dominant_share:>11.1%}   {r.group:10s} {r.reason}")
    line()
    for g in ["ENCODE", "SCALE", "BOOLEAN", "KEEP 0/1"]:
        cols = list(decisions.loc[decisions.group == g, "column"])
        print(f"  {g:9s} ({len(cols)}): {', '.join(cols) if cols else '-'}")


def print_changes(raw, out, decisions):
    print("\nSTEP 2 - what changed?")
    line()
    print(f"shape before: {raw.shape[0]} rows x {raw.shape[1]} columns   ->   after: {out.shape[0]} rows x {out.shape[1]} columns")
    print(f"missing values before: {int(raw.isna().sum().sum())}   ->   after: {int(out.isna().sum().sum())}")
    print()
    for r in decisions.itertuples():
        if r.group == "ENCODE":
            new = [c for c in out.columns if c.startswith(r.column + "_")]
            print(f"  {r.column:22s} 1 text column -> {len(new)} 0/1 columns: {', '.join(c[len(r.column) + 1:] for c in new)}")
        elif r.group == "BOOLEAN":
            print(f"  {r.column:22s} count 0..{raw[r.column].max():.0f} -> {r.column}_flag (1 = not the usual value)  "
                  f"mean {out[r.column + '_flag'].mean():.3f}")
        elif r.group == "SCALE":
            c = r.column
            how = "log1p + standardize" if r.log_first else "standardize"
            print(f"  {c:22s} {how:20s} mean {raw[c].mean():8.3f} -> {out[c].mean():6.3f}   std {raw[c].std():8.3f} -> {out[c].std():6.3f}   "
                  f"range [{raw[c].min():.0f}, {raw[c].max():.0f}] -> [{out[c].min():.2f}, {out[c].max():.2f}]")
        else:
            print(f"  {r.column:22s} unchanged (already 0/1)")


def tidy_columns(out, original_order):
    """Rename merged rare categories to '<col>_Other' and put columns in the original CSV order."""
    out = out.rename(columns=lambda c: c.replace("_infrequent_sklearn", "_Other"))
    def origin(c):
        for i, o in enumerate(original_order):
            if c == o or c == o + "_flag" or c.startswith(o + "_"):
                return i
        return len(original_order)
    return out[sorted(out.columns, key=origin)]


def check_output(out):
    """Fail loudly if the result is not clean."""
    assert out.isna().sum().sum() == 0, "NaN left in the output"
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes), "non-numeric column left in the output"
    assert np.isfinite(out.to_numpy(dtype=float)).all(), "inf left in the output"


# ------------------------------------------------------------------ main
def run(input_path, output_path, table_path, model_path):
    raw = pd.read_csv(ROOT / input_path)
    print(f"Loaded {input_path}: {raw.shape[0]} rows x {raw.shape[1]} columns")

    decisions = profile_and_decide(raw)
    print_decisions(decisions)

    pre = build_preprocessor(decisions)
    out = tidy_columns(pre.fit_transform(raw), raw.columns)
    check_output(out)
    print_changes(raw, out, decisions)

    for rel, writer in [(output_path, lambda p: out.to_csv(p, index=False)),
                        (table_path, lambda p: decisions.drop(columns="log_first").to_csv(p, index=False)),
                        (model_path, lambda p: joblib.dump(pre, p))]:
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)

    print("\nSTEP 3 - saved")
    line()
    print(f"  clean data   : {output_path}   ({out.shape[0]} x {out.shape[1]}, no missing, all numeric)")
    print(f"  decisions    : {table_path}")
    print(f"  fitted model : {model_path}   (load it to transform new bookings the same way)")
    return out


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # never crash on a Windows console
    p = argparse.ArgumentParser(description="Route each clustering column to ENCODE / SCALE / BOOLEAN and write a clean CSV")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    a = p.parse_args(argv)
    run(a.input, a.output, a.table, a.model)


if __name__ == "__main__":
    main()