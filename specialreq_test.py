import pandas as pd
import numpy as np

df = pd.read_csv("Hotel_Data.csv")

# 1. Basic distribution
print("Distribution of special requests:")
print(df["TotalOfSpecialRequests"].value_counts().sort_index())

# 2. Cancellation rate per number of special requests
by_requests = (
    df.groupby("TotalOfSpecialRequests")["IsCanceled"]
    .agg(cancellation_rate="mean", bookings="count")
)
by_requests["cancellation_rate"] = (by_requests["cancellation_rate"] * 100).round(1)
by_requests["share_of_bookings"] = (by_requests["bookings"] / len(df) * 100).round(1)
print("\nCancellation rate by number of special requests (%):")
print(by_requests)
by_requests.to_csv("cancellation_by_special_requests.csv")