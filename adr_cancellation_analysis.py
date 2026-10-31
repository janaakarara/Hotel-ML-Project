from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pointbiserialr


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "H2.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SUMMARY_CSV = ARTIFACTS_DIR / "adr_cancellation_summary.csv"
ANALYSIS_MD = ARTIFACTS_DIR / "adr_cancellation_analysis.md"
RATE_CHART = ARTIFACTS_DIR / "adr_cancellation_rate_chart.png"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    required = {"ADR", "IsCanceled"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["ADR"] = pd.to_numeric(df["ADR"], errors="raise")
    df["IsCanceled"] = pd.to_numeric(df["IsCanceled"], errors="raise")
    if not df["IsCanceled"].isin([0, 1]).all():
        raise ValueError("IsCanceled must contain only 0 and 1 values")

    overall_rate = float(df["IsCanceled"].mean())
    zero_mask = df["ADR"].eq(0)
    zero_df = df.loc[zero_mask]
    nonzero_df = df.loc[~zero_mask]

    zero_rows = int(len(zero_df))
    zero_canceled = int(zero_df["IsCanceled"].sum())
    zero_not_canceled = zero_rows - zero_canceled
    zero_cancellation_rate = zero_canceled / zero_rows if zero_rows else 0.0
    nonzero_rows = int(len(nonzero_df))
    nonzero_canceled = int(nonzero_df["IsCanceled"].sum())
    nonzero_not_canceled = nonzero_rows - nonzero_canceled
    nonzero_cancellation_rate = nonzero_canceled / nonzero_rows if nonzero_rows else 0.0

    clean_adr = df["ADR"].dropna()
    clean_cancel = df.loc[clean_adr.index, "IsCanceled"]
    point_biserial = pointbiserialr(clean_adr, clean_cancel)

    bins = [-float("inf"), 0, 50, 100, 150, 200, 300, 500, float("inf")]
    labels = ["0", ">0-50", ">50-100", ">100-150", ">150-200", ">200-300", ">300-500", ">500"]
    df["ADR_Band"] = pd.cut(df["ADR"], bins=bins, labels=labels, right=True)
    summary = df.groupby("ADR_Band", observed=False).agg(
        Total_Bookings=("ADR", "size"),
        Not_Canceled=("IsCanceled", lambda x: int((x == 0).sum())),
        Canceled=("IsCanceled", "sum"),
    ).reset_index()
    summary["Cancellation_Rate"] = summary["Canceled"] / summary["Total_Bookings"]
    summary["Cancellation_Percent"] = summary["Cancellation_Rate"] * 100
    summary["Rate_vs_Overall_Percentage_Point"] = (
        summary["Cancellation_Rate"] - overall_rate
    ) * 100
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.6f")

    comparison = pd.DataFrame([
        {
            "ADR_Group": "ADR = 0",
            "Total_Bookings": zero_rows,
            "Canceled": zero_canceled,
            "Not_Canceled": zero_not_canceled,
            "Cancellation_Rate": zero_cancellation_rate,
            "Cancellation_Percent": zero_cancellation_rate * 100,
        },
        {
            "ADR_Group": "ADR != 0",
            "Total_Bookings": nonzero_rows,
            "Canceled": nonzero_canceled,
            "Not_Canceled": nonzero_not_canceled,
            "Cancellation_Rate": nonzero_cancellation_rate,
            "Cancellation_Percent": nonzero_cancellation_rate * 100,
        },
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(comparison["ADR_Group"], comparison["Cancellation_Percent"])
    ax.axhline(overall_rate * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"Overall rate: {overall_rate:.1%}")
    ax.set_title("Cancellation Rate: ADR = 0 vs ADR != 0")
    ax.set_ylabel("Canceled bookings (%)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in comparison["Cancellation_Percent"]], padding=3)
    fig.tight_layout()
    fig.savefig(RATE_CHART, dpi=150, bbox_inches="tight")
    plt.close(fig)

    always_canceled = zero_rows > 0 and zero_not_canceled == 0
    report = [
        "# ADR vs. Cancellation Analysis",
        "",
        f"- **Data file:** `{DATA_PATH.relative_to(PROJECT_ROOT)}`",
        f"- **Rows analyzed:** {len(df):,}",
        f"- **Overall cancellation rate:** {overall_rate:.2%}",
        "",
        "## Direct answer: ADR = 0",
        "",
        f"- Rows with ADR = 0: **{zero_rows:,}**",
        f"- Canceled among those rows: **{zero_canceled:,}**",
        f"- Not canceled among those rows: **{zero_not_canceled:,}**",
        f"- Cancellation rate for ADR = 0: **{zero_cancellation_rate:.2%}**",
        "",
        f"### {'Yes' if always_canceled else 'No'} — ADR = 0 does {'always' if always_canceled else 'not always'} imply IsCanceled = 1.",
        "",
        "This is calculated directly from all observed rows. A zero ADR is associated with a smaller "
        "cancellation rate than ADR values that are not zero, but it does not determine cancellation.",
        "",
        "## ADR = 0 vs ADR != 0",
        "",
        "| ADR group | Canceled | Not canceled | Total | Cancellation rate |",
        "|---|---:|---:|---:|---:|",
        f"| ADR = 0 | {zero_canceled:,} | {zero_not_canceled:,} | {zero_rows:,} | {zero_cancellation_rate:.2%} |",
        f"| ADR != 0 | {nonzero_canceled:,} | {nonzero_not_canceled:,} | {nonzero_rows:,} | {nonzero_cancellation_rate:.2%} |",
        "",
        "## Numeric ADR relationship",
        "",
        f"- Point-biserial correlation with IsCanceled: **{point_biserial.statistic:.4f}**",
        f"- P-value: **{point_biserial.pvalue:.3e}**",
        "",
        "The point-biserial correlation measures the linear association between numeric ADR and the binary target.",
        "",
        "## Cancellation rate by ADR band",
        "",
        "| ADR band | Canceled | Not canceled | Total | Cancellation rate | Rate difference (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        report.append(
            f"| {row['ADR_Band']} | {int(row['Canceled'])} | {int(row['Not_Canceled'])} | "
            f"{int(row['Total_Bookings'])} | {row['Cancellation_Percent']:.2f}% | "
            f"{row['Rate_vs_Overall_Percentage_Point']:+.2f} |"
        )

    report.extend([
        "",
        "## Artifact files",
        "",
        f"- Full ADR-band summary: [`{SUMMARY_CSV.name}`](./{SUMMARY_CSV.name})",
        f"- Comparison chart: [`{RATE_CHART.name}`](./{RATE_CHART.name})",
        "",
    ])

    ANALYSIS_MD.write_text("\n".join(report), encoding="utf-8")

    print(f"Analyzed {len(df):,} rows")
    print(f"ADR = 0: {zero_rows:,} rows, {zero_canceled:,} canceled, {zero_cancellation_rate:.2%}")
    print(f"ADR = 0 always canceled: {'Yes' if always_canceled else 'No'}")
    print(f"ADR != 0 cancellation rate: {nonzero_cancellation_rate:.2%}")
    print(f"Point-biserial correlation: {point_biserial.statistic:.4f}; p={point_biserial.pvalue:.3e}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {ANALYSIS_MD}")
    print(f"Wrote {RATE_CHART}")


if __name__ == "__main__":
    main()
