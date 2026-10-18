"""
Stage-1 cleaning for the CSE382 City Hotel (H2) dataset.

Applies only rule-based (deterministic) fixes that learn nothing from the
data, so it is safe to run on the full file BEFORE the train/test split.
Anything data-dependent (rare-category grouping, log1p, scaling, encoding,
median imputation, PCA/LDA) belongs in the sklearn pipeline, fitted on the
training split only.

Column names are kept exactly as the source labels in Section 4 of the
project specification. The only structural change is Company -> has_company.

Usage:
    python src/data/clean.py --input data/raw/H2.csv --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/afaq-ahmad/"
    "Hotel-booking-demand-Case-Study/master/H2.csv"
)

EXPECTED_ROWS = 79_330
EXPECTED_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

# Final dtypes of the cleaned table (Company is replaced by has_company).
TARGET_DTYPES = {
    "IsCanceled": "int8",
    "LeadTime": "int64",
    "ArrivalDateYear": "int16",
    "ArrivalDateMonth": "int8",
    "ArrivalDateWeekNumber": "int8",
    "ArrivalDateDayOfMonth": "int8",
    "StaysInWeekendNights": "int8",
    "StaysInWeekNights": "int8",
    "Adults": "int8",
    "Children": "int8",
    "Babies": "int8",
    "Meal": "category",
    "Country": "category",
    "MarketSegment": "category",
    "DistributionChannel": "category",
    "IsRepeatedGuest": "int8",
    "PreviousCancellations": "int8",
    "PreviousBookingsNotCanceled": "int8",
    "ReservedRoomType": "category",
    "AssignedRoomType": "category",
    "BookingChanges": "int8",
    "DepositType": "category",
    "Agent": "category",
    "has_company": "int8",
    "DaysInWaitingList": "int16",
    "CustomerType": "category",
    "ADR": "float64",
    "RequiredCarParkingSpaces": "int8",
    "TotalOfSpecialRequests": "int8",
    "ReservationStatus": "category",
    "ReservationStatusDate": "datetime64[ns]",
}

# Columns that must never be used as model features (documented here, but
# kept in the cleaned file -- feature selection happens per task).
LEAKY = {
    "ReservationStatus": "equals the target IsCanceled exactly",
    "ReservationStatusDate": "set when the booking reaches its final status",
    "AssignedRoomType": "only known at check-in",
    "BookingChanges": "accumulates until check-in/cancellation (classification)",
}

ADR_OUTLIER_THRESHOLD = 1000  # only one row (5400) exceeds this; next is 510
MAX_BABIES = 2


class CleaningLog:
    """Records every step with the row count after it, for the report."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add(self, step: str, df: pd.DataFrame, removed: int = 0,
            changed: int = 0, note: str = "") -> None:
        self.steps.append({
            "step": step, "rows_after": len(df), "rows_removed": removed,
            "values_changed": changed, "note": note,
        })
        extra = f" (-{removed} rows)" if removed else ""
        extra += f" ({changed} values changed)" if changed else ""
        print(f"[{len(self.steps):02d}] {step:<45s} rows={len(df):>6d}{extra}"
              + (f"  | {note}" if note else ""))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# ---------------------------------------------------------------- general --

def load_raw(path: Path) -> pd.DataFrame:
    # Read everything as text first so whitespace padding and the literal
    # "NULL" are visible and handled explicitly (not silently by pandas).
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def verify_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(EXPECTED_COLUMNS)
        raise ValueError(f"Schema mismatch. Missing: {missing}  Extra: {extra}")
    if len(df) != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows, found {len(df)}")


def strip_whitespace(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    stripped = df.apply(lambda s: s.str.strip())
    changed = int((stripped != df).sum().sum())
    return stripped, changed


def standardise_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    tokens = ["NULL", "NA", ""]
    n = int(df.isin(tokens).sum().sum())
    return df.replace(tokens, np.nan), n


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c, t in TARGET_DTYPES.items()
               if t.startswith(("int", "float")) and c in df.columns
               and c != "ArrivalDateMonth"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df


def consistency_checks(df: pd.DataFrame) -> dict:
    """Read-only checks. Results go into the report; nothing is dropped."""
    num = df.select_dtypes("number")
    status_vs_target = pd.crosstab(df["ReservationStatus"], df["IsCanceled"])
    leak_exact = bool(
        (df["IsCanceled"] == df["ReservationStatus"].isin(["Canceled", "No-Show"])
         .astype(int)).all())

    arrival = pd.to_datetime(
        df["ArrivalDateYear"].astype(int).astype(str) + "-"
        + df["ArrivalDateMonth"].astype(str) + "-"
        + df["ArrivalDateDayOfMonth"].astype(int).astype(str),
        format="%Y-%B-%d", errors="coerce")
    status_date = pd.to_datetime(df["ReservationStatusDate"], errors="coerce")
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    checkout = df["ReservationStatus"] == "Check-Out"
    bad_checkout = int((checkout & (status_date != arrival
                                    + pd.to_timedelta(nights, unit="D"))).sum())

    return {
        "constant_columns": [c for c in df if df[c].nunique(dropna=False) <= 1],
        "fully_empty_rows": int(df.isna().all(axis=1).sum()),
        "negative_values": {c: int((num[c] < 0).sum())
                            for c in num if (num[c] < 0).any()},
        "invalid_arrival_dates": int(arrival.isna().sum()),
        "invalid_status_dates": int(status_date.isna().sum()),
        "status_matches_target_exactly": leak_exact,
        "status_vs_target": status_vs_target.to_dict(),
        "checkout_date_mismatch_rows": bad_checkout,
    }


# ------------------------------------------------------ row-level rules --

def apply_row_rules(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    rules = {
        "zero total guests (Adults+Children+Babies=0)": guests == 0,
        "zero total nights (weekend+week=0)": nights == 0,
        f"Babies > {MAX_BABIES} (outliers 9, 10)": df["Babies"] > MAX_BABIES,
        f"ADR > {ADR_OUTLIER_THRESHOLD} (single 5400 outlier)":
            df["ADR"] > ADR_OUTLIER_THRESHOLD,
        "MarketSegment == Undefined": df["MarketSegment"] == "Undefined",
        "DistributionChannel == Undefined":
            df["DistributionChannel"] == "Undefined",
    }
    # Apply sequentially so each rule's count is what it removed on its own
    # turn (rules overlap, e.g. zero guests and zero nights).
    for name, mask in rules.items():
        mask = mask.reindex(df.index, fill_value=False)
        n = int(mask.sum())
        df = df.loc[~mask]
        log.add(f"drop rows: {name}", df, removed=n)
    return df


# ---------------------------------------------------- per-column fixes --

def fix_columns(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = df.copy()

    n = int(df["Children"].isna().sum())
    df["Children"] = df["Children"].fillna(0)
    log.add("Children: NaN -> 0", df, changed=n)

    n = int(df["Country"].isna().sum())
    df["Country"] = df["Country"].fillna("Unknown")
    log.add("Country: NaN -> 'Unknown'", df, changed=n)

    n = int(df["Agent"].isna().sum())
    df["Agent"] = df["Agent"].fillna("none")
    log.add("Agent: NaN -> 'none' (kept as categorical ID)", df, changed=n)

    df.insert(df.columns.get_loc("Company"), "has_company",
              df["Company"].notna().astype("int8"))
    n = int(df["Company"].isna().sum())
    df = df.drop(columns="Company")
    log.add("Company -> has_company flag (column dropped)", df,
            note=f"{n} rows had no company")

    n = int((df["RequiredCarParkingSpaces"] > 1).sum())
    df["RequiredCarParkingSpaces"] = df["RequiredCarParkingSpaces"].clip(upper=1)
    log.add("RequiredCarParkingSpaces: cap at 1", df, changed=n)

    bad = set(df["ArrivalDateMonth"].unique()) - set(MONTHS)
    if bad:
        raise ValueError(f"Unknown month names: {bad}")
    df["ArrivalDateMonth"] = df["ArrivalDateMonth"].map(MONTHS)
    log.add("ArrivalDateMonth: name -> 1..12", df)

    df["ReservationStatusDate"] = pd.to_datetime(
        df["ReservationStatusDate"], format="%Y-%m-%d")
    log.add("ReservationStatusDate: parse to datetime", df)
    return df


def apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in TARGET_DTYPES.items():
        if dtype.startswith("int"):
            info = np.iinfo(dtype)
            if df[col].min() < info.min or df[col].max() > info.max:
                raise OverflowError(f"{col} does not fit in {dtype}")
        if dtype.startswith("datetime"):
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(dtype)
    return df


def load_clean(path: str | Path) -> pd.DataFrame:
    """Load the cleaned CSV and restore the dtypes (CSV does not store them)."""
    df = pd.read_csv(path, dtype={"Agent": str})
    return apply_dtypes(df[list(TARGET_DTYPES)])


# ---------------------------------------------------------------- report --

def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        r = {"column": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
             "n_unique": int(s.nunique()), "leaky": LEAKY.get(c, "")}
        if pd.api.types.is_numeric_dtype(s):
            r.update(min=s.min(), median=s.median(), max=s.max())
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main --

def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog, dict]:
    log = CleaningLog()

    verify_schema(raw)
    log.add("load + verify schema (79,330 x 31)", raw)

    df, changed = strip_whitespace(raw)
    log.add("strip whitespace (all string columns)", df, changed=changed)

    df, changed = standardise_missing(df)
    log.add("'NULL' / empty -> NaN", df, changed=changed)

    df = to_numeric(df)
    log.add("numeric columns -> numbers", df)

    before = len(df)
    df = df.drop_duplicates(keep="first")
    log.add("drop exact duplicate rows", df, removed=before - len(df),
            note="no booking ID; copies would leak across train/test")

    checks = consistency_checks(df)
    log.add("consistency checks (read-only)", df,
            note=f"checkout date mismatches: "
                 f"{checks['checkout_date_mismatch_rows']} (kept)")

    df = apply_row_rules(df, log)
    df = fix_columns(df, log)
    df = apply_dtypes(df).reset_index(drop=True)
    log.add("apply final dtypes", df)

    leftover_dups = int(df.duplicated().sum())
    if leftover_dups:
        print(f"note: {leftover_dups} rows became identical after fixes (kept)")
    if df.isna().any().any():
        raise AssertionError(
            f"NaN left after cleaning: {df.columns[df.isna().any()].tolist()}")
    return df, log, checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean the H2 City Hotel dataset.")
    ap.add_argument("--input", default="data/raw/H2.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--download", action="store_true",
                    help="download H2.csv from the spec's mirror if missing")
    args = ap.parse_args()

    src, out = Path(args.input), Path(args.out_dir)
    if not src.exists():
        if not args.download:
            raise SystemExit(f"{src} not found (use --download to fetch it)")
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SOURCE_URL}")
        pd.read_csv(SOURCE_URL, dtype=str, keep_default_na=False).to_csv(
            src, index=False)

    df, log, checks = clean(load_raw(src))

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "h2_clean.csv", index=False)
    try:
        df.to_parquet(out / "h2_clean.parquet", index=False)
        parquet = " + h2_clean.parquet"
    except ImportError:
        parquet = " (install pyarrow to also write parquet)"
    log.to_frame().to_csv(out / "cleaning_log.csv", index=False)
    column_summary(df).to_csv(out / "column_summary.csv", index=False)
    (out / "consistency_checks.json").write_text(
        json.dumps(checks, indent=2, default=str))

    print(f"\nclean data: {len(df)} rows x {df.shape[1]} cols "
          f"-> {out}/h2_clean.csv{parquet}")
    print("also wrote cleaning_log.csv, column_summary.csv, consistency_checks.json")
    print("leaky columns kept in file (drop per task):", ", ".join(LEAKY))


if __name__ == "__main__":
    main()
