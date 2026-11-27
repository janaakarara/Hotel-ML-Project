"""Load and verify H2.csv, clean it, split it, and the data analysis (EDA).

    python data.py      runs the data analysis (the same as /analyze in the app)
"""
import json
import urllib.request

import matplotlib
matplotlib.use("Agg")  # draw figures straight into .png files (no pop-up windows)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_FILE, DATA_URL, LEAKY_COLUMNS, OUTPUTS, SEED, SOURCE_COLUMNS


def load_raw():
    """Read H2.csv (downloading it first if missing) and check it is the right file."""
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    assert df.shape == (79330, 31), f"expected 79,330 rows x 31 columns, got {df.shape}"
    assert list(df.columns) == SOURCE_COLUMNS, "column names differ from the H2 source labels"
    return df


def tidy(df):
    """Fix the text values of a table in H2.csv's format; no row is removed.
    Used by clean() and by /predict for a CSV file of bookings."""
    df = df.copy()
    for col in df.columns:  # text columns are padded with spaces: "HB       " -> "HB"
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    # "NULL" agent/company means the booking had no agent/company: keep it as its own category.
    # "CN" is not a 3-letter country code; China is "CHN". (A column the table lacks is skipped.)
    return df.replace({"Agent": {"NULL": "none"}, "Company": {"NULL": "none"}, "Country": {"CN": "CHN"}})


def clean(df):
    """Rule-based cleaning. It learns nothing from the data, so it is safe before the split.
    It rebuilds the team's Milestone 1 file H2_clean_v2_dedup.csv (53,239 rows).
    Returns the clean table and a log of how many rows are left after each step."""
    df = tidy(df)
    log = [("H2.csv as loaded", len(df))]

    df = df.drop_duplicates()  # no booking ID exists, so identical rows would leak into the test set
    log.append(("remove exact duplicate rows", len(df)))
    df = df[df["Adults"] + df["Children"].fillna(0) + df["Babies"] > 0]
    log.append(("remove bookings with 0 guests", len(df)))
    df = df[df["Babies"] <= 2]
    log.append(("remove bookings with more than 2 babies (9 and 10 are typos)", len(df)))
    df = df[df["ADR"] <= 1000]
    log.append(("remove ADR above 1000 (one 5,400 value; next highest is 510)", len(df)))
    df = df[df["StaysInWeekendNights"] + df["StaysInWeekNights"] <= 24]
    log.append(("remove stays longer than 24 nights", len(df)))
    return df.reset_index(drop=True), pd.DataFrame(log, columns=["step", "rows_left"])


def load_clean():
    """The clean table only."""
    df, _ = clean(load_raw())
    return df


def split(df):
    """One fixed 80% train / 20% test split. It is the same split the team's classification models
    use: stratify keeps the same share of canceled bookings (about 30%) in both parts."""
    return train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["IsCanceled"])


def train_test():
    """The training and test bookings all three models use."""
    train, test = split(load_clean())
    # Free bookings (ADR = 0, complimentary stays) are left out, as the teammate's script does: their
    # price is not set by the booking details. Done after the split, so the test bookings are still
    # a part of the same test set that the team's classification models use.
    return train[train["ADR"] > 0], test[test["ADR"] > 0]


def analyze():
    """EDA: data quality checks, summary statistics, skewness, leakage evidence and plots."""
    out = OUTPUTS / "eda"
    out.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df, log = clean(raw)
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]

    # 1) Data quality of the RAW file (what the cleaning had to fix).
    quality = {
        "rows, columns": list(raw.shape),
        "exact duplicate rows": int(raw.duplicated().sum()),
        "missing values per column": {k: int(v) for k, v in raw.isna().sum().items() if v > 0},
        "'NULL' in Agent (no agent)": int((raw["Agent"].str.strip() == "NULL").sum()),
        "'NULL' in Company (no company)": int((raw["Company"].str.strip() == "NULL").sum()),
        "bookings with 0 guests": int((raw["Adults"] + raw["Children"].fillna(0) + raw["Babies"] == 0).sum()),
        "bookings with more than 2 babies": int((raw["Babies"] > 2).sum()),
        "ADR above 1000": int((raw["ADR"] > 1000).sum()),
        "ADR equal to 0 (free/complimentary stays; kept, but not used by the models)": int((raw["ADR"] == 0).sum()),
        "stays longer than 24 nights": int((raw["StaysInWeekendNights"] + raw["StaysInWeekNights"] > 24).sum()),
        "Country code 'CN' (fixed to 'CHN')": int((raw["Country"] == "CN").sum()),
    }
    # 2) Per-column summary of the CLEAN data: type, missing, number of categories, range, skew.
    summary = pd.DataFrame({
        "type": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique_values": df.nunique(),
        "min": df.min(numeric_only=True),
        "median": df.median(numeric_only=True),
        "max": df.max(numeric_only=True),
        "skewness": df.skew(numeric_only=True).round(2),  # > 1 means a long right tail
    })
    # 3) Leakage evidence: when a "later" column has a value vs. when it doesn't, how different is
    #    the mean ADR (the answer)? A big gap = the column gives the answer away.
    price = lambda mask: round(df.loc[mask, "ADR"].mean(), 1)
    conditions = {  # condition name -> True/False for every booking
        "RequiredCarParkingSpaces > 0": df["RequiredCarParkingSpaces"] > 0,
        "AssignedRoomType differs from ReservedRoomType": df["AssignedRoomType"] != df["ReservedRoomType"],
        "BookingChanges > 0": df["BookingChanges"] > 0,
        "TotalOfSpecialRequests > 0": df["TotalOfSpecialRequests"] > 0,
        "ReservationStatus is Canceled or No-Show": df["ReservationStatus"] != "Check-Out",
    }
    evidence = pd.DataFrame([(name, price(mask), price(~mask))  # ~ = "not"
                             for name, mask in conditions.items()],
                            columns=["condition", "mean_ADR_if_true", "mean_ADR_if_false"])

    log.to_csv(out / "cleaning_log.csv", index=False)
    summary.to_csv(out / "column_summary.csv")
    evidence.to_csv(out / "leakage_evidence.csv", index=False)
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2))

    # 4) Plots.
    df.select_dtypes("number").hist(bins=40, figsize=(16, 12))
    plt.tight_layout()
    plt.savefig(out / "numeric_histograms.png")
    plt.close("all")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    raw.boxplot(column="ADR", ax=axes[0])
    axes[0].set_title("ADR before cleaning (outlier 5,400)")
    df.boxplot(column=["ADR", "LeadTime"], ax=axes[1])
    axes[1].set_title("ADR and LeadTime after cleaning")
    nights.value_counts().sort_index().plot.bar(ax=axes[2], title="Total nights per booking")
    plt.tight_layout()
    plt.savefig(out / "outliers.png")
    plt.close("all")

    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    df["ADR"].plot.hist(bins=60, ax=axes[0], title="ADR (price per night)")
    df.groupby("ArrivalDateMonth")["ADR"].mean().reindex(months).plot.bar(ax=axes[1], title="Mean ADR by arrival month")
    df.groupby("ReservedRoomType")["ADR"].mean().sort_values().plot.barh(ax=axes[2], title="Mean ADR by reserved room type")
    plt.tight_layout()
    plt.savefig(out / "adr.png")
    plt.close("all")

    # 5) Print everything for the terminal.
    print("DATA QUALITY (raw H2.csv)")
    for key, value in quality.items():
        print(f"  {key}: {value}")
    print("\nCLEANING STEPS\n" + log.to_string(index=False))
    print("\nCOLUMN SUMMARY (clean data)\n" + summary.to_string())
    print("\nLEAKAGE EVIDENCE (mean ADR)\n" + evidence.to_string(index=False))
    print("\nColumns never used by the models:")
    for col, reason in LEAKY_COLUMNS.items():
        print(f"  {col}: {reason}")
    print(f"\nTarget: ADR mean {df['ADR'].mean():.2f}, median {df['ADR'].median():.2f};"
          f" {(df['ADR'] <= 0).sum():,} free stays (ADR 0) are left out of training and testing")
    print(f"Tables and figures saved in {out}")


if __name__ == "__main__":  # run analyze() only when this file is started directly
    analyze()
