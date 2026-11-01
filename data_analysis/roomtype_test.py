import os
import pandas as pd
from config import DATA_FILE

df = pd.read_csv(DATA_FILE)
os.makedirs("outputs", exist_ok=True)

# 1. Distribution of assigned room types
print("Distribution of assigned room types:")
print(df["AssignedRoomType"].value_counts(dropna=False))

# 2. Cancellation rate per assigned room type
by_assigned = (
    df.groupby("AssignedRoomType")["IsCanceled"]
    .agg(cancellation_rate="mean", bookings="count")
    .sort_values("bookings", ascending=False)
)
by_assigned["cancellation_rate"] = (by_assigned["cancellation_rate"] * 100).round(1)
by_assigned["share_of_bookings"] = (by_assigned["bookings"] / len(df) * 100).round(1)
print("\nCancellation rate by assigned room type (%):")
print(by_assigned)
by_assigned.to_csv(os.path.join("outputs", "cancellation_by_assigned_room_type.csv"))

# 3. Share of bookings where the assigned room differs from the reserved room
mismatch = (df["AssignedRoomType"] != df["ReservedRoomType"])
print(f"\nShare of bookings where assigned room differs from reserved: {mismatch.mean() * 100:.1f}%")