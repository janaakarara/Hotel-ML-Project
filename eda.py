
import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_data
from src.config import BASE_DIR


OUTPUT_REPORT_PATH = os.path.join(
    BASE_DIR, "outputs", "eda_data_report.md"
)


def generate_eda_report():
    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)

    # Always analyze the raw source file for this report.
    data_file = os.path.join("data", "raw", "H2.csv")

    if not os.path.exists(data_file):
        print(f"[ERROR] CSV file not found: {data_file}")
        return

    df = pd.read_csv(data_file)

    total_rows, total_cols = df.shape

    if total_rows == 0:
        print("[ERROR] The CSV file contains no rows.")
        return

    duplicates_count = int(df.duplicated().sum())
    duplicates_pct = (duplicates_count / total_rows) * 100

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("# Exploratory Data Analysis & Audit Report: City Hotel (H2)")
    log(
        f"**Loaded File:** `{data_file}` | "
        f"**Total Records:** {total_rows:,} | "
        f"**Features Count:** {total_cols} | "
        f"**Duplicate Rows:** {duplicates_count:,} "
        f"({duplicates_pct:.2f}%)"
    )
    log("\n---\n")

    # =========================================================
    # 1. Feature Types & Nulls Overview Table
    # =========================================================

    log("## 1. Column Metadata, Missing Values & Data Types\n")

    log(
        "| # | Column Name | Data Type | Null Count | "
        "Null % | Unique Count | Top Sample / Mode |"
    )
    log(
        "|---|-------------|-----------|------------|"
        "--------|--------------|-------------------|"
    )

    null_summary = {}

    for idx, col in enumerate(df.columns, start=1):
        dtype = str(df[col].dtype)

        null_cnt = int(df[col].isnull().sum())
        null_pct = (null_cnt / total_rows) * 100

        n_unique = df[col].nunique(dropna=True)

        if null_cnt > 0:
            null_summary[col] = (null_cnt, null_pct)

        mode_series = df[col].mode(dropna=True)

        top_val = (
            mode_series.iloc[0]
            if not mode_series.empty
            else "N/A"
        )

        if isinstance(top_val, str) and len(top_val) > 18:
            top_val = top_val[:15] + "..."

        log(
            f"| {idx} | `{col}` | `{dtype}` | "
            f"{null_cnt:,} | {null_pct:.2f}% | "
            f"{n_unique:,} | {top_val} |"
        )

    log("\n---\n")

    # =========================================================
    # 2. EVERY Unique Value & Percentage Distribution
    #    For ALL columns: categorical, numerical, and binary
    # =========================================================

    log(
        "## 2. Every Column's Unique Values "
        "& Percentage Distribution\n"
    )

    for col in df.columns:

        log(f"### Column: `{col}`")
        log(f"Data Type: `{df[col].dtype}`")
        log(f"Total Rows: {total_rows:,}")
        log(
            f"Unique Values (excluding missing): "
            f"{df[col].nunique(dropna=True):,}\n"
        )

        # Count every unique value, including missing values
        raw_counts = df[col].value_counts(dropna=False)

        log("| Unique Value | Count | Percentage (%) |")
        log("|--------------|-------|----------------|")

        for value, count in raw_counts.items():

            percentage = (count / total_rows) * 100

            # Display missing values clearly
            if pd.isna(value):
                value_label = "*NULL / MISSING*"
            else:
                value_label = str(value)

            # Escape pipes so Markdown tables remain valid
            value_label = value_label.replace("|", r"\|")

            log(
                f"| `{value_label}` | "
                f"{count:,} | {percentage:.2f}% |"
            )

        log("")

    log("\n---\n")

    # =========================================================
    # 3. Numerical Features Statistics & Outlier Screening
    # =========================================================

    log("## 3. Numerical Columns Summary Statistics & Range\n")

    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    log(
        "| Column Name | Min | 25% | Median | Mean | "
        "75% | Max | Std Dev | Skewness |"
    )
    log(
        "|-------------|-----|-----|--------|------|"
        "-----|-----|---------|----------|"
    )

    for col in num_cols:

        series = df[col].dropna()

        if series.empty:
            log(
                f"| `{col}` | N/A | N/A | N/A | "
                "N/A | N/A | N/A | N/A | N/A |"
            )
            continue

        skew_val = (
            float(series.skew())
            if len(series) > 1
            else 0.0
        )

        std_val = series.std()

        log(
            f"| `{col}` | "
            f"{series.min():.1f} | "
            f"{series.quantile(0.25):.1f} | "
            f"{series.median():.1f} | "
            f"{series.mean():.1f} | "
            f"{series.quantile(0.75):.1f} | "
            f"{series.max():.1f} | "
            f"{std_val:.1f} | "
            f"{skew_val:.2f} |"
        )

    log("\n---\n")

    # =========================================================
    # 4. Per-Column Domain Validity & Pathology Audit
    # =========================================================

    log("## 4. Per-Column Domain Validity & Pathology Checks\n")

    log(
        "| Column Name | Spec Expected Type/Role | "
        "Check / Pathology Description | "
        "Count / Scope | Risk / Status |"
    )
    log(
        "|-------------|--------------------------|"
        "--------------------------------|"
        "---------------|---------------|"
    )

    def audit_row(
        col_name,
        spec_role,
        check_desc,
        count_val,
        status
    ):
        log(
            f"| `{col_name}` | {spec_role} | "
            f"{check_desc} | {count_val} | "
            f"**{status}** |"
        )

    # Negative value checks for all numeric columns
    for col in num_cols:

        neg_count = int((df[col] < 0).sum())

        if neg_count > 0:
            audit_row(
                col,
                "Numeric/Integer",
                "Impossible negative values found",
                f"{neg_count:,} rows",
                "HIGH"
            )

    # Whitespace padding check on text columns
    cat_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    for col in cat_cols:

        padded = (
            df[col]
            .dropna()
            .astype(str)
            .str.contains(r"^\s+|\s+$")
            .sum()
        )

        if padded > 0:
            audit_row(
                col,
                "Categorical",
                "Unstripped fixed-width SQL spaces",
                f"{padded:,} rows",
                "MEDIUM"
            )

    # Target & leakage checks
    if "ReservationStatus" in df.columns:

        audit_row(
            "ReservationStatus",
            "Categorical (Target Leak)",
            "Matches booking cancellation state (Post-event)",
            f"{total_rows:,} rows",
            "FATAL LEAK"
        )

    if "ReservationStatusDate" in df.columns:

        audit_row(
            "ReservationStatusDate",
            "Date (Temporal Leak)",
            "Date when cancellation/checkout occurred",
            f"{total_rows:,} rows",
            "LEAK"
        )

    # Occupancy & stay zero checks
    if all(
        k in df.columns
        for k in ["Adults", "Children", "Babies"]
    ):

        zero_guests = int(
            (
                (df["Adults"] == 0)
                & (df["Children"].fillna(0) == 0)
                & (df["Babies"] == 0)
            ).sum()
        )

        audit_row(
            "Adults/Children/Babies",
            "Integer (Guests)",
            "Zero total guests (Posting Master / Dummy)",
            f"{zero_guests:,} rows",
            "INVALID" if zero_guests > 0 else "CLEAN"
        )

        adults_zero = int(
            (
                (df["Adults"] == 0)
                & (
                    (df["Children"].fillna(0) > 0)
                    | (df["Babies"] > 0)
                )
            ).sum()
        )

        audit_row(
            "Adults",
            "Integer",
            "Zero adults but children/babies present (Adjoining room)",
            f"{adults_zero:,} rows",
            "INFO"
        )

    if all(
        k in df.columns
        for k in [
            "StaysInWeekendNights",
            "StaysInWeekNights"
        ]
    ):

        zero_nights = int(
            (
                (df["StaysInWeekendNights"] == 0)
                & (df["StaysInWeekNights"] == 0)
            ).sum()
        )

        audit_row(
            "StaysInNights",
            "Integer (Stay Length)",
            "Zero total nights stayed (Day-use or cancelled)",
            f"{zero_nights:,} rows",
            "INVALID" if zero_nights > 0 else "CLEAN"
        )

    # Specific outliers
    if "Babies" in df.columns:

        high_babies = int((df["Babies"] > 2).sum())

        audit_row(
            "Babies",
            "Integer",
            "Extreme infant count (Typo anomaly e.g. 9 or 10)",
            f"{high_babies:,} rows",
            "OUTLIER" if high_babies > 0 else "CLEAN"
        )

    if "ADR" in df.columns:

        adr_outliers = int((df["ADR"] > 1000).sum())

        audit_row(
            "ADR",
            "Numeric",
            "Extreme ADR outlier (> 1000 e.g. 5400.0)",
            f"{adr_outliers:,} rows",
            "OUTLIER" if adr_outliers > 0 else "CLEAN"
        )

        zero_adr = int((df["ADR"] == 0).sum())

        audit_row(
            "ADR",
            "Numeric",
            "Zero ADR (Complimentary / promotion)",
            f"{zero_adr:,} rows",
            "INFO"
        )

    if "RequiredCarParkingSpaces" in df.columns:

        high_parking = int(
            (df["RequiredCarParkingSpaces"] > 1).sum()
        )

        audit_row(
            "RequiredCarParkingSpaces",
            "Integer",
            "Parking spaces > 1 (City hotel rare tail)",
            f"{high_parking:,} rows",
            "CAP SUGGESTION" if high_parking > 0 else "CLEAN"
        )

    # Categorical domain flags
    if "MarketSegment" in df.columns:

        undef_ms = int(
            df["MarketSegment"]
            .isin(["Undefined", "undefined"])
            .sum()
        )

        audit_row(
            "MarketSegment",
            "Categorical",
            "Undefined market segment entries",
            f"{undef_ms:,} rows",
            "INVALID" if undef_ms > 0 else "CLEAN"
        )

    if "DistributionChannel" in df.columns:

        undef_dc = int(
            df["DistributionChannel"]
            .isin(["Undefined", "undefined"])
            .sum()
        )

        audit_row(
            "DistributionChannel",
            "Categorical",
            "Undefined distribution channel entries",
            f"{undef_dc:,} rows",
            "INVALID" if undef_dc > 0 else "CLEAN"
        )

    if "ReservedRoomType" in df.columns:

        p_room = int(
            (df["ReservedRoomType"] == "P").sum()
        )

        audit_row(
            "ReservedRoomType",
            "Categorical",
            "Phantom room type 'P' (Billing dummy)",
            f"{p_room:,} rows",
            "ANOMALY" if p_room > 0 else "CLEAN"
        )

    if "AssignedRoomType" in df.columns:

        audit_row(
            "AssignedRoomType",
            "Categorical",
            "Known only at check-in (Check-in time leakage)",
            f"{total_rows:,} rows",
            "LEAK WARNING"
        )

    if "Company" in df.columns:

        company_nulls = (
            int(df["Company"].isnull().sum())
            + int(
                (
                    df["Company"]
                    .astype(str)
                    .str.upper() == "NULL"
                ).sum()
            )
        )

        audit_row(
            "Company",
            "Categorical / ID",
            "Missing corporate ID (Individual private bookings)",
            f"{company_nulls:,} rows "
            f"({(company_nulls / total_rows) * 100:.1f}%)",
            "HIGH SPARSITY"
        )

    if "Agent" in df.columns:

        agent_nulls = (
            int(df["Agent"].isnull().sum())
            + int(
                (
                    df["Agent"]
                    .astype(str)
                    .str.upper() == "NULL"
                ).sum()
            )
        )

        audit_row(
            "Agent",
            "Categorical / ID",
            "Missing travel agency ID (Direct / Non-agency)",
            f"{agent_nulls:,} rows "
            f"({(agent_nulls / total_rows) * 100:.1f}%)",
            "SPARSE ID"
        )

    log("\n---\n")

    # =========================================================
    # 5. Critical Targets Summary
    # =========================================================

    log("## 5. Modeling Targets & Evaluation Context\n")

    if "ADR" in df.columns:

        log(
            f"- **Task A Target (`ADR`):** "
            f"Mean = {df['ADR'].mean():.2f}, "
            f"Median = {df['ADR'].median():.2f}, "
            f"Max = {df['ADR'].max():.2f}"
        )

    if "IsCanceled" in df.columns:

        counts = df["IsCanceled"].value_counts()

        not_canceled = counts.get(0, 0)
        canceled = counts.get(1, 0)

        log(
            f"- **Task B Target (`IsCanceled`):** "
            f"Not Canceled (0): {not_canceled:,} "
            f"({not_canceled / total_rows * 100:.1f}%), "
            f"Canceled (1): {canceled:,} "
            f"({canceled / total_rows * 100:.1f}%)"
        )

    # =========================================================
    # 6. Export Markdown Report
    # =========================================================

    with open(
        OUTPUT_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(report_lines))

    print(
        f"\n[DONE] Full Markdown audit report "
        f"successfully exported to: {OUTPUT_REPORT_PATH}"
    )


if __name__ == "__main__":
    generate_eda_report()