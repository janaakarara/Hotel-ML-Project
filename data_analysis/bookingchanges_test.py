import pandas as pd

df = pd.read_csv(r"data\raw\H2.csv")

# 1. Distribution of booking changes
print("Distribution of booking changes:")
print(df["BookingChanges"].value_counts().sort_index())

# 2. Cancellation rate by number of changes (capped at 3+ so small groups merge)
df["changes_capped"] = df["BookingChanges"].clip(upper=3)
by_changes = (
    df.groupby("changes_capped")["IsCanceled"]
    .agg(cancellation_rate="mean", bookings="count")
)
by_changes["cancellation_rate"] = (by_changes["cancellation_rate"] * 100).round(1)
by_changes["share_of_bookings"] = (by_changes["bookings"] / len(df) * 100).round(1)
print("\nCancellation rate by booking changes (%):")
print(by_changes)
by_changes.to_csv("cancellation_by_booking_changes.csv")

# 3. Leakage check: mean changes by ReservationStatus
print("\nMean booking changes by ReservationStatus:")
print(df.groupby("ReservationStatus")["BookingChanges"].agg(["mean", "median", "count"]).round(2))

# 4. Correlation with cancellation
corr = df["BookingChanges"].corr(df["IsCanceled"])
print(f"\nPearson correlation (BookingChanges vs IsCanceled): {corr:.3f}")