import pandas as pd

df = pd.read_csv("data\\raw\\H2.csv")

print(df["LeadTime"].describe())

# Cancellation rate by lead-time band
bins = [-1, 7, 30, 90, 180, 365, df["LeadTime"].max()]
labels = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"]
df["lead_band"] = pd.cut(df["LeadTime"], bins=bins, labels=labels)

bands = df.groupby("lead_band", observed=True)["IsCanceled"].agg(
    cancellation_rate="mean", bookings="count"
)
bands["cancellation_rate"] = (bands["cancellation_rate"] * 100).round(1)
print(bands)