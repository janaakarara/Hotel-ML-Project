import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv('data/processed/H2_clean_v2_dedup.csv')

# Explicitly check for known leakage columns
potential_leaks = [
    'ReservationStatus',
    'ReservationStatusDate',
    'IsCanceled',
    'DaysInWaitingList',
]
print(f'[*] Auditing dataset shape: {df.shape}')

# Separate Target and Features
y = df['ADR']
drop_cols = ['ADR'] + [c for c in potential_leaks if c in df.columns]
X_cols = [c for c in df.columns if c not in drop_cols]
X = df[X_cols].copy()

# Robust encoding for text/categorical columns
for c in X.columns:
  if not pd.api.types.is_numeric_dtype(X[c]):
    X[c] = pd.factorize(X[c].fillna('Unknown').astype(str))[0]
  else:
    X[c] = X[c].fillna(0)

# Train/Test Split to avoid data leakage during feature evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit benchmark model
rf = RandomForestRegressor(
    n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

# 2. Compute Permutation Importance (Gold Standard for validation drop)
perm_importance = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
perm_scores = perm_importance.importances_mean

# 3. Compute Mutual Information (Non-linear statistical dependency)
# Subsample for speed if dataset is large
mi_scores = mutual_info_regression(X_train, y_train, random_state=42)

# 4. Compute Pearson Correlation with Target
pearson_corr = X.corrwith(y).abs()

# Compile Master Audit DataFrame
audit_df = pd.DataFrame({
    'Feature': X_cols,
    'PermutationImpact': perm_scores,
    'MutualInfo': mi_scores,
    'AbsCorrelation': pearson_corr.values,
}).sort_values(by='PermutationImpact', ascending=False)


# 5. Automated Decision Classification Logic
def classify_feature(row):
  if row['PermutationImpact'] > 0.01:
    return 'Core Driver (Keep)'
  elif row['PermutationImpact'] > 0.001:
    return 'Grey Zone (Evaluate)'
  else:
    return 'Noise / Drop'


audit_df['Decision'] = audit_df.apply(classify_feature, axis=1)

print('\n' + '=' * 80)
print('STATISTICAL FEATURE AUDIT & DECISION MATRIX')
print('=' * 80)
print(audit_df.to_string(index=False))

# 6. Correlation Matrix / Multicollinearity check for top features
print('\n' + '=' * 80)
print('MULTICOLLINEARITY CHECK (Top Features Correlation)')
print('=' * 80)
top_features = audit_df['Feature'].head(8).tolist()
print(X[top_features].corr().to_string())