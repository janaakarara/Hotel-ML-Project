import os
import pandas as pd
from config import DATA_FILE

df = pd.read_csv(DATA_FILE)

os.makedirs("outputs", exist_ok=True)

# 1. Distribution of distribution channels
print("Distribution of distribution channels:")
print(df["DistributionChannel"].value_counts(dropna=False))

# 2. Cancellation rate per distribution channel
by_channel = (
    df.groupby("DistributionChannel")["IsCanceled"]
    .agg(cancellation_rate="mean", bookings="count")
    .sort_values("bookings", ascending=False)
)
by_channel["cancellation_rate"] = (by_channel["cancellation_rate"] * 100).round(1)
by_channel["share_of_bookings"] = (by_channel["bookings"] / len(df) * 100).round(1)
print("\nCancellation rate by distribution channel (%):")
print(by_channel)
by_channel.to_csv(os.path.join("outputs", "cancellation_by_distribution_channel.csv"))

# 3. Check for inconsistent labels (spaces, case differences)
print("\nRaw unique values:", [repr(v) for v in df["DistributionChannel"].unique()])