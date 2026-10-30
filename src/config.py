#Constants, paths, seed raqam wahed (RANDOM_STATE = 42) 3lshan el reproducibility, w el target columns.
import os
from pathlib import Path

import yaml

# Paths
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
TARGET_REGRESSION = "ADR"
TARGET_CLASSIFICATION = "IsCanceled"

# The project root is the folder that contains config.yaml (one level above src/).
ROOT = Path(__file__).resolve().parent.parent


def load_config(path=ROOT / "config.yaml"):
    """Read config.yaml once, so every module uses the same seed and paths."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative):
    """Turn a path from config.yaml into a full path, and make sure its folder exists."""
    full = ROOT / relative
    full.parent.mkdir(parents=True, exist_ok=True)
    return full
