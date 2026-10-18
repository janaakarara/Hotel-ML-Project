import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_data
from src.config import BASE_DIR

OUTPUT_REPORT_PATH = os.path.join(BASE_DIR, "outputs", "eda_data_report.md")


def generate_eda_report():
    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    # Satr el gded:
    clean_data_path = os.path.join("data", "processed", "h2_clean.csv") # Aw masar el file 3andek
    df = pd.read_csv(clean_data_path)

    total_rows, total_cols = df.shape
    duplicates_count = int(df.duplicated().sum())
    duplicates_pct = (duplicates_count / total_rows) * 100

    report_lines = []
    
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("# Exploratory Data Analysis & Audit Report: City Hotel (H2)")
    log(f"**Total Records:** {total_rows:,} | **Features Count:** {total_cols} | **Duplicate Rows:** {duplicates_count:,} ({duplicates_pct:.2f}%)")
    log("\n---\n")

    # 1. Feature Types & Nulls Overview Table
    log("## 1. Column Metadata, Missing Values & Data Types\n")
    log("| # | Column Name | Data Type | Null Count | Null % | Unique Count | Top Sample / Mode |")
    log("|---|-------------|-----------|------------|--------|--------------|-------------------|")

    null_summary = {}
    for idx, col in enumerate(df.columns, start=1):
        dtype = str(df[col].dtype)
        null_cnt = int(df[col].isnull().sum())
        null_pct = (null_cnt / total_rows) * 100
        n_unique = df[col].nunique(dropna=True)

        if null_cnt > 0:
            null_summary[col] = (null_cnt, null_pct)

        # Get top value representation
        mode_series = df[col].mode(dropna=True)
        top_val = mode_series.iloc[0] if not mode_series.empty else "N/A"
        if isinstance(top_val, str) and len(top_val) > 18:
            top_val = top_val[:15] + "..."

        log(f"| {idx} | `{col}` | `{dtype}` | {null_cnt:,} | {null_pct:.2f}% | {n_unique:,} | {top_val} |")

    log("\n---\n")

    # 2. Detailed Categorical Features Cardinality & Distribution
    log("## 2. Categorical Columns Detailed Frequency Breakdown\n")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        n_unq = df[col].nunique(dropna=True)
        log(f"### Feature: `{col}` (Unique categories: {n_unq})")
        val_counts = df[col].value_counts(dropna=False, normalize=True) * 100
        raw_counts = df[col].value_counts(dropna=False)

        top_cats = val_counts.head(8)
        log("| Category Value | Absolute Count | Percentage (%) |")
        log("|----------------|----------------|----------------|")
        for cat, pct in top_cats.items():
            cat_label = str(cat) if pd.notnull(cat) else "*NULL / MISSING*"
            cnt = raw_counts[cat]
            log(f"| `{cat_label}` | {cnt:,} | {pct:.2f}% |")

        if n_unq > 8:
            log(f"*... and {n_unq - 8} additional categories omitted for brevity.*")
        log("")

    log("\n---\n")

    # 3. Numerical Features Statistical Distribution & Outlier Screening
    log("## 3. Numerical Columns Summary Statistics & Range\n")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    log("| Column Name | Min | 25% | Median | Mean | 75% | Max | Std Dev | Skewness |")
    log("|-------------|-----|-----|--------|------|-----|-----|---------|----------|")

    for col in num_cols:
        series = df[col].dropna()
        skew_val = float(series.skew())
        log(
            f"| `{col}` | {series.min():.1f} | {series.quantile(0.25):.1f} | "
            f"{series.median():.1f} | {series.mean():.1f} | {series.quantile(0.75):.1f} | "
            f"{series.max():.1f} | {series.std():.1f} | {skew_val:.2f} |"
        )

    log("\n---\n")

    # 4. Critical Leakage & Target Columns Specific Audit
    log("## 4. Leakage & Target Audit Warnings\n")
    
    # Task A Target: ADR
    if "ADR" in df.columns:
        neg_adr = (df["ADR"] < 0).sum()
        zero_adr = (df["ADR"] == 0).sum()
        max_adr = df["ADR"].max()
        log(f"- **Task A Target (`ADR`):** Contains {neg_adr} negative values, {zero_adr:,} zero values, and an extreme maximum of {max_adr:.2f}. A clean regression pipeline should filter/handle non-positive ADR or extreme outlier bookings.")

    # Task B Target: IsCanceled
    if "IsCanceled" in df.columns:
        counts = df["IsCanceled"].value_counts()
        log(f"- **Task B Target (`IsCanceled`):** Class 0 (Kept): {counts.get(0, 0):,} ({counts.get(0, 0)/total_rows*100:.1f}%), Class 1 (Canceled): {counts.get(1, 0):,} ({counts.get(1, 0)/total_rows*100:.1f}%).")

    # ReservationStatus (Fatal Leakage)
    if "ReservationStatus" in df.columns:
        log("- **CRITICAL LEAKAGE DETECTED (`ReservationStatus`):**")
        for val, cnt in df["ReservationStatus"].value_counts().items():
            log(f"  - `{val}`: {cnt:,}")
        log("  - *Action:* Must be dropped prior to modeling for `IsCanceled`, as 'Canceled' vs 'Check-Out' maps 1-to-1 with the outcome.")

    # Write report file to disk
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[DONE] Full Markdown audit report successfully exported to: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    generate_eda_report()