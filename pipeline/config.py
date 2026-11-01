import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "H2.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


# Random Seed for reproducibility
RANDOM_STATE = 42

DROPPED_COLS = ["ReservationStatus", "ReservationStatusDate","RequiredCarParkingSpaces"]

# Targets
TARGET_CLASSIFICATION = "IsCanceled"