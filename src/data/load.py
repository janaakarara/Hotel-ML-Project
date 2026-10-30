"""Load the already-cleaned H2 file and check it still has the 31 source columns."""
import pandas as pd

SOURCE_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children",
    "Babies", "Meal", "Country", "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType", "AssignedRoomType",
    "BookingChanges", "DepositType", "Agent", "Company", "DaysInWaitingList", "CustomerType", "ADR",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]


def load_clean_data(path):
    df = pd.read_csv(path)
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cleaned file is missing columns: {missing}")
    return df
