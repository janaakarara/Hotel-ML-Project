import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_FILE = os.path.join(DATA_DIR, "H2.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Dataset URL
H2_URL = "https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv"

# Random Seed for reproducibility
RANDOM_STATE = 42

# Strict Leakage Columns to drop from predictors
# ReservationStatus w ReservationStatusDate b-elzat fatal leakage fel classification
LEAKAGE_COLS = ["ReservationStatus", "ReservationStatusDate"]

# Targets
TARGET_CLASSIFICATION = "IsCanceled"
