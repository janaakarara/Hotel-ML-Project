import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

TARGET = "IsCanceled"
TIME_COL = "ArrivalDateYear"
FEATURES_TO_TEST = ["DistributionChannel", "MarketSegment"]
ALWAYS_DROP = ["ReservationStatus", "ReservationStatusDate", "AssignedRoomType"]
SEEDS = (0, 1, 2, 3, 4)

# ---------- 1. Load and leakage check (on raw data, before dropping) ----------
raw = pd.read_csv("data/raw/H2.csv")
print(f"Rows: {len(raw):,}  |  exact duplicates: {raw.duplicated().sum():,}")

print("\n--- ReservationStatus vs target (perfect mapping = leakage) ---")
print(pd.crosstab(raw["ReservationStatus"], raw[TARGET]))

# ---------- 2. Prepare data ----------
df = raw.drop(columns=ALWAYS_DROP)
cat_cols = df.select_dtypes(exclude="number").columns
df[cat_cols] = df[cat_cols].astype("category")
all_cols = [c for c in df.columns if c != TARGET]

# ---------- 3. Time-based split ----------
train = df[df[TIME_COL] < 2017]
test = df[df[TIME_COL] == 2017]
assert len(train) and len(test), "Time split produced an empty set"
print(f"\nTrain rows: {len(train):,}  |  Test rows (2017): {len(test):,}")


def fit_auc(cols, seed):
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(train[cols], train[TARGET])
    return roc_auc_score(test[TARGET], model.predict_proba(test[cols])[:, 1])


def score(cols):
    scores = [fit_auc(cols, s) for s in SEEDS]
    return np.mean(scores), np.std(scores)


# ---------- 4. Baseline and ablations ----------
base_m, base_sd = score(all_cols)
print(f"\nBaseline AUC (all features): {base_m:.4f} ± {base_sd:.4f}")

print("\n--- Ablations (Δ = with − without; positive = feature helps) ---")
for f in FEATURES_TO_TEST:
    m, sd = score([c for c in all_cols if c != f])
    print(f"without {f:22s} AUC = {m:.4f} ± {sd:.4f}  Δ = {base_m - m:+.4f}")

m, sd = score([c for c in all_cols if c not in FEATURES_TO_TEST])
print(f"without both                 AUC = {m:.4f} ± {sd:.4f}  Δ = {base_m - m:+.4f}")

# ---------- 5. Single-feature AUC (numeric columns, direction-aware) ----------
print("\n--- Single-feature AUC (flag if strength > 0.9) ---")
for c in df.select_dtypes("number").columns.drop(TARGET):
    auc = roc_auc_score(df[TARGET], df[c].fillna(0))
    strength = max(auc, 1 - auc)
    flag = "  <-- check for leak" if strength > 0.9 else ""
    print(f"{c:32s} AUC={auc:.3f}  strength={strength:.3f}{flag}")


# ---------- 6. Stability and patterns, per tested feature ----------
def describe(feature):
    print(f"\n--- {feature} ---")
    if pd.api.types.is_numeric_dtype(df[feature]):
        buckets = pd.qcut(df[feature], 4, duplicates="drop")
        print(df.groupby(buckets, observed=True)[TARGET].agg(["mean", "size"]))
    else:
        print(df.groupby(feature, observed=True)[TARGET]
                .agg(["mean", "size"]).sort_values("size", ascending=False))
        print("\nCancel rate by year:")
        print(df.pivot_table(index=feature, columns=TIME_COL, values=TARGET,
                             aggfunc="mean", observed=True).round(3))


for f in FEATURES_TO_TEST:
    describe(f)