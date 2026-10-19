"""
Leakage & ablation tests for the H2 hotel-cancellation data.
Usage:  python leakage_tests.py            (expects H2.csv in the same folder)
Needs:  pandas, scikit-learn
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

TARGET = "IsCanceled"
FEATURE_TO_TEST = "PreviousBookingsNotCanceled"
LEAKY = ["ReservationStatus", "ReservationStatusDate",
         "RequiredCarParkingSpaces", "AssignedRoomType"]

# ---------- 1. Load, deduplicate, drop known leaky columns ----------
df = pd.read_csv("data/raw/H2.csv")
print(f"Rows before dedup: {len(df):,}")
#df = df.drop_duplicates()
#print(f"Rows after dedup:  {len(df):,}")

# Direct leakage evidence (run BEFORE dropping, on the raw file)
raw = pd.read_csv("data/raw/H2.csv")
print("\n--- ReservationStatus vs target (perfect mapping = leakage) ---")
print(pd.crosstab(raw["ReservationStatus"], raw[TARGET]))

df = df.drop(columns=LEAKY)
for c in df.select_dtypes(exclude="number").columns:
    df[c] = df[c].astype("category")

# ---------- 2. Time-based split ----------
train = df[df["ArrivalDateYear"] < 2017]
test = df[df["ArrivalDateYear"] == 2017]
all_cols = [c for c in df.columns if c != TARGET]


def fit_auc(cols, seed=0):
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(train[cols], train[TARGET])
    return roc_auc_score(test[TARGET], model.predict_proba(test[cols])[:, 1])


# ---------- 3. Ablation: with vs without the feature (several seeds) ----------
def ablate(feature, seeds=(0, 1, 2, 3, 4)):
    with_f = [fit_auc(all_cols, s) for s in seeds]
    without_f = [fit_auc([c for c in all_cols if c != feature], s) for s in seeds]
    w, wo = sum(with_f) / len(seeds), sum(without_f) / len(seeds)
    print(f"\n--- Ablation: {feature} ---")
    print(f"AUC with:    {w:.4f}")
    print(f"AUC without: {wo:.4f}")
    print(f"Difference:  {w - wo:+.4f}")


ablate(FEATURE_TO_TEST)

# ---------- 4. Single-feature AUC (numeric columns) ----------
print("\n--- Single-feature AUC (flag anything > 0.9 or < 0.1) ---")
for c in df.select_dtypes("number").columns.drop(TARGET):
    print(f"{c:32s} {roc_auc_score(df[TARGET], df[c].fillna(0)):.3f}")

# ---------- 5. Stability over time ----------
print(f"\n--- {FEATURE_TO_TEST}: cancel rate by year (only rows > 0) ---")
sub = df[df[FEATURE_TO_TEST] > 0]
print(sub.groupby("ArrivalDateYear")[TARGET].agg(["mean", "size"]))
print("\nShare of rows > 0 by year:")
print(df.groupby("ArrivalDateYear")[FEATURE_TO_TEST].apply(lambda s: (s > 0).mean()))

# ---------- 6. Cancel rate by feature bucket ----------
print(f"\n--- Cancel rate by {FEATURE_TO_TEST} bucket ---")
buckets = pd.cut(df[FEATURE_TO_TEST], [-1, 0, 2, 10, 100])
print(df.groupby(buckets, observed=True)[TARGET].agg(["mean", "size"]))

# ---------- 7. Consistency with IsRepeatedGuest ----------
print("\n--- IsRepeatedGuest vs has_history ---")
print(pd.crosstab(df["IsRepeatedGuest"], df[FEATURE_TO_TEST] > 0))