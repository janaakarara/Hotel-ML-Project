"""Which columns the clustering may see, and the new columns made from them.

Every new column uses only values from the same row, so it can be computed the
same way for a brand-new booking.
"""
import numpy as np

# Inputs to the clustering, grouped by how they are prepared (see clustering_preprocessor.py).
NUMERIC_FEATURES = ["LogLeadTime", "LogTotalNights", "WeekendShare", "Adults"]

# Meal and HasKids were tested and removed from the model inputs. Against a shuffled-column
# null they added noise: dropping them raised the gap over the null from 0.021 to 0.042 and
# the seed stability (ARI) from 0.42 to 0.86. Meal is 77% one value (BB) and HasKids is
# about 90% zeros. They are still computed and reported in the cluster profiles, only they
# no longer influence the distances.
BINARY_FEATURES = ["IsRepeatedGuest"]
CATEGORICAL_FEATURES = ["MarketSegment", "CustomerType"]
CLUSTER_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Never allowed as clustering inputs: they are outcomes, or are set/updated
# after the booking is made (see the leakage audit in the guide).
FORBIDDEN_FEATURES = [
    "IsCanceled", "ReservationStatus", "ReservationStatusDate", "AssignedRoomType",
    "BookingChanges", "RequiredCarParkingSpaces", "TotalOfSpecialRequests", "ADR",
]


def add_clustering_features(df):
    df = df.copy()
    total_nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    kids = df["Children"].fillna(0) + df["Babies"]
    # LeadTime and nights have long right tails; log1p pulls the tail in so a few
    # very long values do not dominate the distances.
    df["LogLeadTime"] = np.log1p(df["LeadTime"])
    df["LogTotalNights"] = np.log1p(total_nights)
    # Share of the stay that falls on a weekend: 0 = weekday (business-like), 1 = weekend trip.
    # Day-use bookings have 0 nights; dividing by 1 instead of 0 gives them a share of 0.
    df["WeekendShare"] = df["StaysInWeekendNights"] / total_nights.replace(0, 1)
    # Not a model input any more (see above), but kept for describing the clusters.
    df["HasKids"] = (kids > 0).astype(int)
    return df