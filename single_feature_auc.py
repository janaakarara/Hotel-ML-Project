# Single-feature AUC for ALL 30 predictors (numeric + categorical), out-of-fold.
# Usage: python single_feature_auc.py   (H2.csv in the same folder; change the path if needed)
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "IsCanceled"
full = pd.read_csv("data/raw/H2.csv")
# turn text columns into categories; the date string into a number
full["ReservationStatusDate"] = pd.to_datetime(full["ReservationStatusDate"]).map(pd.Timestamp.toordinal)
for c in full.select_dtypes(exclude="number").columns:
    full[c] = full[c].astype(str).str.strip().astype("category")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
rows = []
for c in full.columns.drop(TARGET):
    # one-feature model, scored out-of-fold (so high-cardinality columns can't memorise)
    p = cross_val_predict(
        HistGradientBoostingClassifier(random_state=0, categorical_features="from_dtype"),
        full[[c]], full[TARGET], cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(full[TARGET], p)
    rows.append((c, str(full[c].dtype), auc))

res = pd.DataFrame(rows, columns=["feature", "type", "AUC"]).sort_values("AUC", ascending=False)
res["flag"] = res["AUC"].apply(lambda a: "LEAK?" if a > 0.9 else ("check" if a > 0.7 else ""))
print(res.to_string(index=False, formatters={"AUC": "{:.3f}".format}))