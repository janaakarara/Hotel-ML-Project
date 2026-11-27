"""Stage 1 for clustering: drop zero-adult rows, then drop outcome and future-event columns.

Logic-based (see the dataset paper's variable definitions). No statistics are used here.
Redundancy, near-constant columns, skew and grouping are decided later from the plots.

Rows: bookings with Adults = 0 are removed (they all have children but no adult, so they are not a
meaningful customer segment). This is done for clustering only; h2_clean.csv is not changed.

Usage: python src/02_preprocessing/stage1_drop_future.py --input data/processed/h2_clean.csv --out-dir data/processed
Writes: h2_clustering_stage1.csv           (the 22 columns still in play)
        h2_clustering_profile_cols.csv     (the 9 dropped columns, same rows and order, for describing clusters)
        clustering_stage1_decisions.md     (readable table of every decision)
"""
import argparse
from pathlib import Path
import pandas as pd

DROP = {
 "IsCanceled": ("outcome", "The classification target. Clusters must be found without it."),
 "ReservationStatus": ("outcome", "Final status of the booking: a copy of IsCanceled."),
 "ReservationStatusDate": ("outcome", "Date the final status was set. Minus arrival date it predicts cancellation perfectly."),
 "AssignedRoomType": ("future event", "Paper: assigned for hotel operation reasons, near check-in."),
 "BookingChanges": ("future event", "Paper: changes counted 'until the moment of check-in or cancellation'."),
 "DepositType": ("future event", "Paper: computed from payments 'before arrival or cancellation date'."),
 "RequiredCarParkingSpaces": ("suspected future", "Not documented as future, but 0 of 1,886 bookings with parking were canceled."),
 "TotalOfSpecialRequests": ("suspected future", "Timing unclear; kept bookings show more requests at the same lead time."),
 "ADR": ("outcome-like", "Regression target; paper defines it from lodging transactions. Use it only to describe clusters."),
}

def write_markdown(columns, path, rows_before, rows_after):
    kept = [c for c in columns if c not in DROP]
    lines = ["# Clustering stage 1: zero-adult rows, outcome and future columns", "",
             "## Rows removed", "",
             f"**{rows_before - rows_after} rows removed** ({rows_before} -> {rows_after}): bookings with `Adults = 0`. "
             "All of them have children but no adult, so they are most likely data errors and not a meaningful customer segment. "
             "This is done for clustering only; `h2_clean.csv` is unchanged.", "",
             f"**Dropped columns: {len(DROP)}**  |  **Kept for the graph stage: {len(kept)}**  |  Total: {len(columns)}", "",
             "## Dropped", "", "| Column | Type | Reason |", "|---|---|---|"]
    lines += [f"| `{c}` | {t} | {r} |" for c, (t, r) in DROP.items()]
    lines += ["", "The dropped columns are saved for the kept rows in `h2_clustering_profile_cols.csv` (same row order) "
              "and are used only to describe the clusters afterwards.", "",
              "## Kept for now", "",
              "These are not outcomes or future values. Redundancy, skew and grouping are decided from the plots.", "",
              "| Column |", "|---|"]
    lines += [f"| `{c}` |" for c in kept]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clean.csv")
    ap.add_argument("--out-dir", default="data/processed")
    a = ap.parse_args()
    df = pd.read_csv(a.input, dtype={"Agent": str})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows_before = len(df)
    df = df[df["Adults"] > 0].reset_index(drop=True)
    df.drop(columns=list(DROP)).to_csv(out / "h2_clustering_stage1.csv", index=False)
    df[list(DROP)].to_csv(out / "h2_clustering_profile_cols.csv", index=False)
    write_markdown(list(df.columns), out / "clustering_stage1_decisions.md", rows_before, len(df))
    print(f"removed {rows_before - len(df)} rows with Adults = 0")
    print(f"kept {df.shape[1]-len(DROP)} columns, dropped {len(DROP)}; rows {len(df)}")

if __name__ == "__main__":
    main()
