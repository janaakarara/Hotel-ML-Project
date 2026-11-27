"""Settings shared by all three models: paths, the random seed and the input columns.
(Each model's own settings are in its folder: linear_regression/config.py, random_forest/config.py,
xgboost_model/config.py.)"""
from pathlib import Path

# Fixed random seed: every split, model and sample is random in the same way each run,
# so anyone who runs the project gets exactly the same results (reproducibility).
SEED = 42

ROOT = Path(__file__).resolve().parent  # this folder
DATA_FILE = ROOT / "data" / "H2.csv"
DATA_URL = ("https://raw.githubusercontent.com/afaq-ahmad/"
            "Hotel-booking-demand-Case-Study/master/H2.csv")
# Results shared by the 3 models (the comparison, the data analysis, predictions from files).
# Each model's own results and saved models are in its own folder: outputs/ and artifacts/.
OUTPUTS = ROOT / "outputs"

# The 31 source labels, exactly as written in Section 4 of the project description.
SOURCE_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]

# Prediction point: the moment the booking is made.
# These columns are only known LATER, so no model may use them (leakage).
LEAKY_COLUMNS = {
    "ReservationStatus": "final status (Canceled / No-Show / Check-Out): only known at the end",
    "ReservationStatusDate": "date the final status was set: only exists after cancel or check-out",
    "AssignedRoomType": "room given at check-in (a different room means the guest came)",
    "BookingChanges": "counts changes made until arrival or cancellation",
    "RequiredCarParkingSpaces": "recorded on arrival: 0 of the 1,890 bookings with parking were canceled",
    "TotalOfSpecialRequests": "requests are often added after booking or at check-in",
}
# IsCanceled is also never an input: whether the guest cancels is only known later.

TARGET = "ADR"  # what the models predict: the average daily rate (price per night)

# The booking-time inputs: 12 number + 9 text columns (ADR is not one of them: it is the answer).
NUMERIC_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    # DaysInWaitingList is left out, as in the teammate's pipeline: it is only known once processed.
]
CATEGORICAL_FEATURES = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "Agent", "CustomerType",
]
INPUTS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Company"]  # Company: only for HasCompany (0/1)
