from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "H2.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SUMMARY_CSV = ARTIFACTS_DIR / "agent_cancellation_summary.csv"
ANALYSIS_MD = ARTIFACTS_DIR / "agent_cancellation_analysis.md"
RATE_CHART = ARTIFACTS_DIR / "agent_cancellation_rates.png"
MIN_BOOKINGS_FOR_RATE_RANKING = 30


def cramer_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = int(table.to_numpy().sum())
    if n == 0:
        return 0.0

    phi2 = chi2 / n
    r, k = table.shape
    phi2_corrected = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    k_corrected = k - ((k - 1) ** 2) / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    if denominator <= 0:
        return 0.0
    return float((phi2_corrected / denominator) ** 0.5)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    required = {"Agent", "IsCanceled"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["Agent"] = df["Agent"].fillna("UNKNOWN").astype(str).str.strip()
    df["IsCanceled"] = pd.to_numeric(df["IsCanceled"], errors="raise")
    if not df["IsCanceled"].isin([0, 1]).all():
        raise ValueError("IsCanceled must contain only 0 and 1 values")

    contingency = pd.crosstab(df["Agent"], df["IsCanceled"])
    contingency.columns = ["Not_Canceled", "Canceled"]

    chi2, p_value, degrees_of_freedom, expected = chi2_contingency(contingency)
    minimum_expected = float(expected.min())
    low_expected_cells = int((expected < 5).sum())

    summary = contingency.reset_index()
    summary["Total_Bookings"] = summary["Not_Canceled"] + summary["Canceled"]
    summary["Cancellation_Rate"] = summary["Canceled"] / summary["Total_Bookings"]
    summary["Cancellation_Percent"] = summary["Cancellation_Rate"] * 100
    summary["Share_of_All_Bookings"] = summary["Total_Bookings"] / len(df) * 100
    overall_rate = df["IsCanceled"].mean()
    summary["Rate_vs_Overall_Percentage_Point"] = (
        summary["Cancellation_Rate"] - overall_rate
    ) * 100
    summary = summary.sort_values(["Total_Bookings", "Agent"], ascending=[False, True])
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.6f")

    ranked = summary.loc[
        summary["Total_Bookings"] >= MIN_BOOKINGS_FOR_RATE_RANKING
    ].sort_values("Cancellation_Rate", ascending=False)
    top_10_by_rate = ranked.head(10).copy()
    top_10_by_volume = summary.head(10).copy()

    chart_data = summary.head(20).sort_values("Cancellation_Rate")
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(chart_data["Agent"].astype(str), chart_data["Cancellation_Percent"])
    ax.axvline(overall_rate * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"Overall rate: {overall_rate:.1%}")
    ax.set_title("Hotel Cancellation Rate by Agent — Top 20 by Booking Volume")
    ax.set_xlabel("Canceled bookings (%)")
    ax.set_ylabel("Agent")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in chart_data["Cancellation_Percent"]],
                 padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(RATE_CHART, dpi=150, bbox_inches="tight")
    plt.close(fig)

    v = cramer_v(contingency)
    report = [
        "# Agent vs. Cancellation Analysis",
        "",
        f"- **Data file:** `{DATA_PATH.relative_to(PROJECT_ROOT)}`",
        f"- **Rows analyzed:** {len(df):,}",
        f"- **Agents (including UNKNOWN):** {df['Agent'].nunique():,}",
        f"- **Overall cancellation rate:** {overall_rate:.2%}",
        "",
        "## Statistical relationship",
        "",
        f"- Chi-square statistic: **{chi2:,.2f}**",
        f"- Degrees of freedom: **{degrees_of_freedom:,}**",
        f"- P-value: **{p_value:.3e}**",
        f"- Bias-corrected Cramer's V: **{v:.4f}**",
        f"- Minimum expected cell count: **{minimum_expected:.2f}**",
        f"- Cells with expected count below 5: **{low_expected_cells:,}** of {expected.size:,}",
        "",
        "The chi-square test evaluates whether cancellation and agent are independent. "
        "Cramer's V describes the strength of the association; 0 means no association and "
        "larger values indicate stronger association.",
        "",
        "> With this dataset's large sample size, a very small p-value is not by itself proof ",
        "> of a practically large effect. Interpret the cancellation rates and Cramer's V together.",
        "",
        "## Top 10 agents by cancellation rate",
        "",
        f"Agents are shown only if they have at least {MIN_BOOKINGS_FOR_RATE_RANKING} bookings, "
        "to avoid overemphasizing very small samples.",
        "",
        "| Agent | Canceled | Not canceled | Total | Cancellation rate | Rate difference (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in top_10_by_rate.iterrows():
        report.append(
            f"| {row['Agent']} | {int(row['Canceled'])} | {int(row['Not_Canceled'])} | "
            f"{int(row['Total_Bookings'])} | {row['Cancellation_Percent']:.2f}% | "
            f"{row['Rate_vs_Overall_Percentage_Point']:+.2f} |"
        )

    report.extend([
        "",
        "## Top 10 agents by booking volume",
        "",
        "| Agent | Canceled | Not canceled | Total | Share of bookings | Cancellation rate | Rate difference (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for _, row in top_10_by_volume.iterrows():
        report.append(
            f"| {row['Agent']} | {int(row['Canceled'])} | {int(row['Not_Canceled'])} | "
            f"{int(row['Total_Bookings'])} | {row['Share_of_All_Bookings']:.2f}% | "
            f"{row['Cancellation_Percent']:.2f}% | "
            f"{row['Rate_vs_Overall_Percentage_Point']:+.2f} |"
        )

    report.extend([
        "",
        "## Artifact files",
        "",
        f"- Full agent summary: [`{SUMMARY_CSV.name}`](./{SUMMARY_CSV.name})",
        f"- Cancellation-rate chart: [`{RATE_CHART.name}`](./{RATE_CHART.name})",
        "",
    ])

    ANALYSIS_MD.write_text("\n".join(report), encoding="utf-8")

    print(f"Analyzed {len(df):,} rows across {df['Agent'].nunique():,} agents.")
    print(f"Overall cancellation rate: {overall_rate:.2%}")
    print(f"Chi-square p-value: {p_value:.3e}; Cramer's V: {v:.4f}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {ANALYSIS_MD}")
    print(f"Wrote {RATE_CHART}")


if __name__ == "__main__":
    main()
