"""Stage 2b: Drop columns selected from Stage 2a redundancy analysis."""

import argparse
from pathlib import Path
import pandas as pd

DECISIONS = {
    "Babies": ("Near-constant", "99.32% of rows contain 0.", "stage2a_constancy.png"),
    "DaysInWaitingList": ("Near-constant", "98.65% of rows contain 0.", "stage2a_constancy.png"),
    "PreviousCancellations": ("Near-constant", "97.79% of rows contain 0.", "stage2a_constancy.png"),
    "PreviousBookingsNotCanceled": ("Near-constant", "97.13% of rows contain 0.", "stage2a_constancy.png"),
    "IsRepeatedGuest": ("Near-constant", "97.08% of rows contain 0.", "stage2a_constancy.png"),
    "DistributionChannel": ("Redundancy", "Redundant with MarketSegment (Cramer's V = 0.77), Agent (V = 0.86), and has_company (V = 0.82).", "stage2a_cramers_v.png"),
    "Agent": ("Redundancy", "Redundant with DistributionChannel (Cramer's V = 0.86). It also has 224 categories.", "stage2a_cramers_v.png"),
    "has_company": ("Redundancy", "Redundant with MarketSegment (Cramer's V = 0.85). Additionally, 94.81% of rows contain 0.", "stage2a_cramers_v.png"),
    "ArrivalDateWeekNumber": ("Redundancy", "Almost completely explained by ArrivalDateMonth (eta = 0.995).", "stage2a_eta.png"),
    "ArrivalDateDayOfMonth": ("Judgment-based removal", "The day is spread relatively evenly throughout the month and showed no meaningful relationship with other columns.", "stage2a_spearman.png"),
    "ArrivalDateYear": ("Judgment-based removal", "The dataset covers only July 2015 to August 2017. Year was treated as a calendar artifact rather than a useful clustering feature.", "Dataset date coverage"),
}

KEEP = [
    "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "MarketSegment", "Meal", "Country",
    "CustomerType", "ReservedRoomType", "ArrivalDateMonth"
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage1.csv")
    ap.add_argument("--output", default="data/processed/h2_clustering_stage2.csv")
    ap.add_argument("--report", default="outputs/tables/stage2_drop_decisions.md")
    a = ap.parse_args()

    df = pd.read_csv(a.input, dtype={"Agent": str})

    missing = [c for c in DECISIONS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}")

    out = df.drop(columns=list(DECISIONS))

    if sorted(out.columns) != sorted(KEEP):
        raise SystemExit(
            f"Unexpected columns left: {sorted(set(out.columns) ^ set(KEEP))}"
        )

    out = out[KEEP]

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)

    report = [
        "# Stage 2b: Column Removal Decisions",
        "",
        "## 1. Objective",
        "",
        "Stage 2b removes the 11 columns selected for removal based on the Stage 2a redundancy analysis.",
        "The purpose is to reduce redundant and low-information features before clustering.",
        "",
        "## 2. Input and Output",
        "",
        f"- **Input:** `{a.input}`",
        f"- **Output:** `{a.output}`",
        f"- **Rows:** {len(df)}",
        f"- **Original columns:** {len(df.columns)}",
        f"- **Dropped columns:** {len(DECISIONS)}",
        f"- **Retained columns:** {len(out.columns)}",
        "",
        "## 3. Reasons for Dropping Each Column",
        "",
        "| Column | Reason | Evidence | Stage 2a reference |",
        "|---|---|---|---|",
    ]

    for column, (category, reason, evidence) in DECISIONS.items():
        report.append(
            f"| `{column}` | {category} | {reason} | `{evidence}` |"
        )

    report += [
        "",
        "## 4. Retained Columns",
        "",
    ]

    for column in KEEP:
        report.append(f"- `{column}`")

    report += [
        "",
        "## 5. Final Result",
        "",
        f"Stage 2b reduced the dataset from **{len(df.columns)} to {len(out.columns)} columns**, "
        f"while preserving all **{len(df)} rows** and their original order.",
        "",
        "No encoding, scaling, grouping, or additional statistical analysis is performed in this stage.",
        "",
        "## 6. Methodological Note",
        "",
        "The removal decisions are based on the Stage 2a analysis and the documented judgment calls above. "
        "Stage 2b only applies these decisions; it does not recalculate the Stage 2a statistics.",
        "",
    ]

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text("\n".join(report), encoding="utf-8")

    print("\n========== STAGE 2b COMPLETED ==========")
    print(f"Rows preserved: {len(out)}")
    print(f"Columns dropped: {len(DECISIONS)}")
    print(f"Columns retained: {len(out.columns)}")
    print(f"CSV saved: {a.output}")
    print(f"Markdown report saved: {a.report}")
    print("=======================================")


if __name__ == "__main__":
    main()
