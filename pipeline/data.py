from __future__ import annotations
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from config import DATA_FILE, RANDOM_STATE, TARGET_CLASSIFICATION
from pipeline.preprocess import preprocess

TEST_SIZE = 0.2
N_SPLITS = 5
ID_LIKE_COLUMNS = ["Agent", "Company"]  # IDs: must be treated as categories

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load H2, clean it, and separate the features from the target."""
    raw = pd.read_csv(DATA_FILE)
    df, _ = preprocess(raw)

    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype(str).str.strip()
    for col in ID_LIKE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    y = df[TARGET_CLASSIFICATION].astype(int)
    X = df.drop(columns=[TARGET_CLASSIFICATION])
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """Return X_train, X_test, y_train, y_test (stratified, fixed seed)."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def get_cv() -> StratifiedKFold:
    """Return the cross-validation splitter used by every model."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)