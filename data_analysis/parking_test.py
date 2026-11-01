import os
import pandas as pd
from data_analysis.config import DATA

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA)

# 1. Distribution of parking spaces requested
print("Distribution of required parking spaces:")
print(df["RequiredCarParkingSpaces"].value_counts().sort_index())

# 2. Cancellation rate by number of parking spaces (capped at 2+ so small groups merge)
df["parking_capped"] = df["RequiredCarParkingSpaces"].clip(upper=2)
by_parking = (
    df.groupby("parking_capped")["IsCanceled"]
    .agg(cancellation_rate="mean", bookings="count")
)
by_parking["cancellation_rate"] = (by_parking["cancellation_rate"] * 100).round(1)
by_parking["share_of_bookings"] = (by_parking["bookings"] / len(df) * 100).round(1)
print("\nCancellation rate by parking spaces (%):")
print(by_parking)
by_parking.to_csv(os.path.join("outputs", "cancellation_by_parking_spaces.csv"))

# 3. Mean parking spaces for canceled vs non-canceled bookings
print("\nMean parking spaces by IsCanceled:")
print(df.groupby("IsCanceled")["RequiredCarParkingSpaces"].agg(["mean", "median", "count"]).round(3))

# 4. Correlation with cancellation
corr = df["RequiredCarParkingSpaces"].corr(df["IsCanceled"])
print(f"\nPearson correlation (RequiredCarParkingSpaces vs IsCanceled): {corr:.3f}")