from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "H2.csv"
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
