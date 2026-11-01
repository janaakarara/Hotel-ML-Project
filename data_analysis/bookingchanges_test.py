import os
import pandas as pd
from config import DATA_FILE

df = pd.read_csv(DATA_FILE)
    
os.makedirs("outputs", exist_ok=True)

# ---------- Normal (multi-level) analysis ----------

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
by_changes.to_csv(os.path.join("outputs", "cancellation_by_booking_changes.csv"))

# 3. Mean booking changes for canceled vs non-canceled bookings
print("\nMean booking changes by IsCanceled:")
print(df.groupby("IsCanceled")["BookingChanges"].agg(["mean", "median", "count"]).round(2))

# 4. Correlation with cancellation (count version)
corr = df["BookingChanges"].corr(df["IsCanceled"])
print(f"\nPearson correlation (BookingChanges vs IsCanceled): {corr:.3f}")

# ---------- Binary analysis ----------

# 5. Binary flag: 1 if changed at least once, 0 otherwise
df["has_booking_change"] = (df["BookingChanges"] > 0).astype(int)

binary = (
    df.groupby("has_booking_change")["IsCanceled"]
    .agg(cancellation_rate="mean", bookings="count")
)
binary["cancellation_rate"] = (binary["cancellation_rate"] * 100).round(1)
binary["share_of_bookings"] = (binary["bookings"] / len(df) * 100).round(1)
binary.index = binary.index.map({0: "No changes", 1: "At least one change"})
print("\nCancellation rate by booking changes (binary, %):")
print(binary)
binary.to_csv(os.path.join("outputs", "cancellation_by_booking_changes_binary.csv"))

# 6. Correlation of the binary flag with cancellation
corr_binary = df["has_booking_change"].corr(df["IsCanceled"])
print(f"\nPearson correlation (has_booking_change vs IsCanceled): {corr_binary:.3f}")