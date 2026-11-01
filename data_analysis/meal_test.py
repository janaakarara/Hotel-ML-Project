import pandas as pd

df = pd.read_csv(r"data\raw\H2.csv")

# 1. Distribution of meal types
print("Distribution of meal types:")
print(df["Meal"].value_counts(dropna=False))

# 2. Cancellation rate per meal type
by_meal = (
    df.groupby("Meal")["IsCanceled"]
    .agg(cancellation_rate="mean", bookings="count")
    .sort_values("bookings", ascending=False)
)
by_meal["cancellation_rate"] = (by_meal["cancellation_rate"] * 100).round(1)
by_meal["share_of_bookings"] = (by_meal["bookings"] / len(df) * 100).round(1)
print("\nCancellation rate by meal type (%):")
print(by_meal)
by_meal.to_csv("cancellation_by_meal.csv")

# 3. Check for inconsistent labels (spaces, case differences)
print("\nRaw unique values:", [repr(v) for v in df["Meal"].unique()])