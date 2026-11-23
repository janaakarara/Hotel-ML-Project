"""Phase 1: ALL EDA figures in one script. Read-only.

Run from the project root:   python src/eda_plots_all.py
Data: data/raw/H2.csv with exact duplicates removed (nothing else is changed).
Figures: p1_* (skew, dominant values, categories) and p1b_01 .. p1b_11 in outputs/figures/.
Tables:  p1b_column_table.csv and p1b_timing_table.csv in outputs/tables/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)
RAW = Path("data/raw/H2.csv")
FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)
TAB = Path("outputs/tables"); TAB.mkdir(parents=True, exist_ok=True)

# ---------- load: strip, NULL -> NaN, drop exact duplicates (nothing else) ----------
df = pd.read_csv(RAW, dtype=str)
df = df.apply(lambda s: s.str.strip()).replace({"NULL": np.nan, "": np.nan})
NUM = ["LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
       "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
       "PreviousCancellations", "PreviousBookingsNotCanceled", "BookingChanges",
       "DaysInWaitingList", "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
       "IsRepeatedGuest", "IsCanceled"]
for c in NUM:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.drop_duplicates().reset_index(drop=True)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
df["Month"] = df.ArrivalDateMonth.map({m: i + 1 for i, m in enumerate(MONTHS)})
df["TotalNights"] = df.StaysInWeekendNights + df.StaysInWeekNights
df["TotalGuests"] = df.Adults + df.Children.fillna(0) + df.Babies
CAT = ["Meal", "Country", "MarketSegment", "DistributionChannel", "ReservedRoomType",
       "AssignedRoomType", "DepositType", "Agent", "CustomerType"]
print(f"rows after removing duplicates: {len(df)}")

# ---------- 0. The three first figures (skew, dominant values, categories) ----------
cols0 = ["LeadTime", "TotalNights", "StaysInWeekNights"]
fig, ax = plt.subplots(2, 3, figsize=(11, 5.5))
for j, c in enumerate(cols0):
    ax[0, j].hist(df[c].dropna(), bins=50, color="#2F5DA8")
    ax[0, j].set_title(f"{c} (raw), skew {df[c].skew():.2f}")
    ax[1, j].hist(np.log1p(df[c].dropna()), bins=50, color="#0E8A8A")
    ax[1, j].set_title(f"log1p({c}), skew {np.log1p(df[c]).skew():.2f}")
plt.tight_layout(); plt.savefig(FIG / "p1_skew_raw_vs_log.png", dpi=140); plt.close()

share = {c: df[c].value_counts(normalize=True, dropna=True).iloc[0] for c in df.columns[:31] if c != "ADR"}
sh = pd.Series(share).sort_values()
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(sh.index, sh.values, color=["#C8433B" if v > 0.9 else "#2F5DA8" for v in sh.values])
ax.axvline(0.9, color="#C8433B", ls="--", lw=1)
ax.set_title("Share of rows holding the most common value (red: over 90%)")
plt.tight_layout(); plt.savefig(FIG / "p1_dominant_value_share.png", dpi=140); plt.close()

cats0 = ["MarketSegment", "CustomerType", "DistributionChannel", "Meal"]
fig, ax = plt.subplots(2, 2, figsize=(11, 6))
for a, c in zip(ax.ravel(), cats0):
    df[c].value_counts().plot.bar(ax=a, color="#6B4FB3"); a.set_title(c); a.set_xlabel("")
    a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(FIG / "p1_categories.png", dpi=140); plt.close()

# ---------- 1-2. Scales and outliers ----------
cols = ["LeadTime", "TotalNights", "Adults", "Children", "Babies", "BookingChanges",
        "DaysInWaitingList", "PreviousCancellations", "PreviousBookingsNotCanceled",
        "TotalOfSpecialRequests", "ADR"]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].boxplot([df[c].dropna() for c in cols], vert=False, flierprops=dict(markersize=2))
ax[0].set_yticks(range(1, len(cols) + 1)); ax[0].set_yticklabels(cols)
ax[0].set_xscale("symlog"); ax[0].set_title("Raw ranges (log-style axis): the scales differ a lot")
out = {}
for c in cols:
    x = df[c].dropna(); q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    out[c] = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() * 100
o = pd.Series(out).sort_values()
ax[1].barh(o.index, o.values, color=["#C8433B" if v > 10 else "#2F5DA8" for v in o.values])
ax[1].set_title("% of rows outside 1.5 x IQR (red: over 10%)")
plt.tight_layout(); plt.savefig(FIG / "p1b_01_scales_outliers.png", dpi=140); plt.close()

# ---------- 3-4. Few-value numbers and zero-heavy columns ----------
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
for a, c in zip(ax[:3], ["Adults", "Children", "Babies"]):
    df[c].value_counts().sort_index().plot.bar(ax=a, color="#6B4FB3"); a.set_title(f"{c}: few values"); a.set_xlabel("")
zc = ["Children", "Babies", "DaysInWaitingList", "PreviousCancellations", "PreviousBookingsNotCanceled",
      "BookingChanges", "RequiredCarParkingSpaces", "TotalOfSpecialRequests", "IsRepeatedGuest"]
z = pd.Series({c: (df[c] == 0).mean() * 100 for c in zc}).sort_values()
ax[3].barh(z.index, z.values, color="#0E8A8A"); ax[3].set_title("% of rows equal to 0")
plt.tight_layout(); plt.savefig(FIG / "p1b_02_discrete_and_zeros.png", dpi=140); plt.close()

# ---------- 5. Spearman correlation (numeric) ----------
nc = ["LeadTime", "TotalNights", "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
      "BookingChanges", "DaysInWaitingList", "PreviousCancellations", "PreviousBookingsNotCanceled",
      "TotalOfSpecialRequests", "RequiredCarParkingSpaces", "IsRepeatedGuest", "ArrivalDateWeekNumber"]
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(df[nc].corr(method="spearman"), cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 6}, ax=ax)
ax.set_title("Spearman correlation: look for pairs above 0.6 (duplicate information)")
plt.tight_layout(); plt.savefig(FIG / "p1b_03_spearman.png", dpi=140); plt.close()

# ---------- 6. Cramer's V (categorical) ----------
def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.values.sum(); r, k = t.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0
cv = pd.DataFrame(index=CAT, columns=CAT, dtype=float)
sub = df[CAT].fillna("missing")
for a in CAT:
    for b in CAT:
        cv.loc[a, b] = 1.0 if a == b else cramers_v(sub[a], sub[b])
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(cv, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
ax.set_title("Cramer's V between categorical columns (0 none, 1 identical)")
plt.tight_layout(); plt.savefig(FIG / "p1b_04_cramers_v.png", dpi=140); plt.close()

# ---------- 7. Category versus number ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (cat, num) in zip(ax, [("MarketSegment", "LeadTime"), ("DepositType", "LeadTime"), ("CustomerType", "ADR")]):
    d = df[df.ADR < 1000] if num == "ADR" else df
    sns.boxplot(data=d, x=cat, y=num, ax=a, fliersize=1, color="#9DB7E0")
    a.set_title(f"{num} by {cat}"); a.tick_params(axis="x", rotation=35); a.set_xlabel("")
plt.tight_layout(); plt.savefig(FIG / "p1b_05_category_vs_number.png", dpi=140); plt.close()

# ---------- 8. Duplicates after encoding on candidate columns ----------
cands = {
    "all 31 columns": list(df.columns[:31]),
    "no ADR/status/dates (28)": [c for c in df.columns[:31] if c not in ("ADR", "ReservationStatus", "ReservationStatusDate", "IsCanceled")],
    "7 core columns": ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Adults", "IsRepeatedGuest", "MarketSegment", "CustomerType"],
    "4 numeric only": ["LeadTime", "TotalNights", "Adults", "BookingChanges"],
}
dist = {k: len(df[v].drop_duplicates()) for k, v in cands.items()}
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.barh(list(dist.keys()), list(dist.values()), color="#B8800A")
ax.axvline(len(df), color="k", ls="--"); ax.text(len(df), -0.45, f" rows: {len(df)}", fontsize=8)
for i, v in enumerate(dist.values()): ax.text(v, i, f" {v}", va="center", fontsize=8)
ax.set_title("Distinct rows on chosen columns (fewer = more ties for DBSCAN)")
plt.tight_layout(); plt.savefig(FIG / "p1b_06_duplicates_after_selection.png", dpi=140); plt.close()

# ---------- 9. Impossible values and missingness ----------
rules = {
    "Adults+Children+Babies = 0": (df.TotalGuests == 0).sum(),
    "TotalNights = 0": (df.TotalNights == 0).sum(),
    "ADR <= 0": (df.ADR <= 0).sum(),
    "ADR > 1000": (df.ADR > 1000).sum(),
    "Babies > 2": (df.Babies > 2).sum(),
    "MarketSegment = Undefined": (df.MarketSegment == "Undefined").sum(),
    "DistributionChannel = Undefined": (df.DistributionChannel == "Undefined").sum(),
}
miss = df.isna().mean() * 100
miss = miss[miss > 0].sort_values()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
r = pd.Series(rules).sort_values()
ax[0].barh(r.index, r.values, color="#C8433B"); ax[0].set_title("Rows breaking a sanity rule")
for i, v in enumerate(r.values): ax[0].text(v, i, f" {v}", va="center", fontsize=8)
ax[1].barh(miss.index, miss.values, color="#2F5DA8"); ax[1].set_title("% missing per column")
plt.tight_layout(); plt.savefig(FIG / "p1b_07_impossible_and_missing.png", dpi=140); plt.close()
mm = df[["Agent", "Company", "Country", "Children"]].isna()
print("missing together (rows):\n", mm.value_counts().to_string())

# ---------- 10. Rare categories (cumulative coverage) ----------
fig, ax = plt.subplots(figsize=(7.5, 4))
for c in ["Country", "Agent"]:
    s = df[c].fillna("missing").value_counts(normalize=True).cumsum().values
    ax.plot(np.arange(1, len(s) + 1), s * 100, label=f"{c} ({len(s)} values)")
ax.axhline(90, color="k", ls="--", lw=.8); ax.set_xscale("log")
ax.set_xlabel("number of top categories kept"); ax.set_ylabel("% of rows covered"); ax.legend()
ax.set_title("How many categories cover 90% of rows?")
plt.tight_layout(); plt.savefig(FIG / "p1b_08_rare_categories.png", dpi=140); plt.close()

# ---------- 11-12. Seasonality and cyclical month ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
(df.groupby(["Month", "ArrivalDateYear"]).size().unstack().plot(ax=ax[0], marker="o"))
ax[0].set_title("Bookings per arrival month and year"); ax[0].set_xlabel("month")
m = df.Month.value_counts().sort_index()
ang = 2 * np.pi * (m.index - 1) / 12
ax[1].scatter(np.sin(ang), np.cos(ang), s=m.values / m.values.max() * 600, color="#0E8A8A")
for a_, i in zip(ang, m.index): ax[1].text(np.sin(a_) * 1.18, np.cos(a_) * 1.18, str(i), ha="center", va="center")
ax[1].set_aspect("equal"); ax[1].axis("off"); ax[1].set_title("Months on a circle (12 sits next to 1)")
plt.tight_layout(); plt.savefig(FIG / "p1b_09_seasonality_cyclical.png", dpi=140); plt.close()

# ---------- 13. Timing table (fill the last column yourself) ----------
timing = pd.DataFrame([
    ("IsCanceled", "outcome", "Never use"),
    ("ReservationStatus", "outcome (copy of IsCanceled)", "Never use"),
    ("ReservationStatusDate", "after the booking", "Never use"),
    ("ADR", "Task A target", "Exclude"),
    ("AssignedRoomType", "set at check-in", "Uncertain, exclude to be safe"),
    ("BookingChanges", "grows until check-in", "Uncertain, exclude to be safe"),
    ("RequiredCarParkingSpaces", "may change later", "Uncertain"),
    ("TotalOfSpecialRequests", "may change later", "Uncertain"),
    ("DaysInWaitingList", "known by confirmation", "Probably fine"),
    ("LeadTime, Adults, Meal, MarketSegment, ...", "known at booking", "Fine"),
], columns=["column", "when it is known", "your decision (edit this)"])
timing.to_csv(TAB / "p1b_timing_table.csv", index=False)

# ---------- 14. Cluster tendency: Hopkins + kNN distance ----------
core = ["LeadTime", "TotalNights", "Adults", "Children", "Babies", "BookingChanges", "DaysInWaitingList",
        "PreviousCancellations", "PreviousBookingsNotCanceled", "TotalOfSpecialRequests", "RequiredCarParkingSpaces",
        "IsRepeatedGuest"]
X = df[core].fillna(0).copy()
X["LeadTime"] = np.log1p(X.LeadTime); X["TotalNights"] = np.log1p(X.TotalNights)
X = X.drop_duplicates()   # identical rows give nearest-neighbour distance 0 and fake a high Hopkins value
print(f"distinct rows on these 12 columns: {len(X)} of {len(df)}")
Xs = StandardScaler().fit_transform(X)
def hopkins(A, m=1000, seeds=5):
    vals = []
    for s in range(seeds):
        r = np.random.default_rng(SEED + s)
        idx = r.choice(len(A), m, replace=False)
        nn = NearestNeighbors(n_neighbors=2).fit(A)
        w = nn.kneighbors(A[idx], 2)[0][:, 1]
        U = r.uniform(A.min(0), A.max(0), size=(m, A.shape[1]))
        u = nn.kneighbors(U, 1)[0][:, 0]
        vals.append(u.sum() / (u.sum() + w.sum()))
    return float(np.mean(vals)), float(np.std(vals))
h, hs = hopkins(Xs)
# Fair comparison: shuffle each column on its own (keeps every column's shape, destroys relations between columns)
Xsh = np.column_stack([rng.permutation(Xs[:, j]) for j in range(Xs.shape[1])])
h0, _ = hopkins(Xsh)
print(f"Hopkins statistic (12 numeric columns): {h:.3f} +- {hs:.3f};  shuffled-columns baseline: {h0:.3f}")
kn = NearestNeighbors(n_neighbors=10).fit(Xs)
d10 = np.sort(kn.kneighbors(Xs[rng.choice(len(Xs), 8000, replace=False)])[0][:, -1])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(["shuffled columns\n(no relations)", "real data"], [h0, h], color=["#999", "#0E8A8A"])
ax[0].set_ylim(0, 1); ax[0].set_title(f"Hopkins: real {h:.2f} vs shuffled {h0:.2f}\n(compare the two, not with 0.5)")
ax[1].plot(d10, color="#2F5DA8"); ax[1].set_title("Sorted distance to 10th neighbour (DBSCAN eps hint)")
ax[1].set_xlabel("points, sorted"); ax[1].set_ylabel("distance")
plt.tight_layout(); plt.savefig(FIG / "p1b_10_cluster_tendency.png", dpi=140); plt.close()

# ---------- 15. PCA 2D, for VIEWING only ----------
S = df.sample(8000, random_state=SEED)
Xv = pd.concat([pd.DataFrame(StandardScaler().fit_transform(
        np.column_stack([np.log1p(S.LeadTime), np.log1p(S.TotalNights),
                         S[["Adults", "Children", "Babies", "BookingChanges", "DaysInWaitingList",
                            "PreviousCancellations", "TotalOfSpecialRequests"]].fillna(0).values]))),
                pd.get_dummies(S[["MarketSegment", "CustomerType", "DepositType"]]).astype(float).reset_index(drop=True) * 0.5],
               axis=1)
Xv.columns = Xv.columns.astype(str)
pca = PCA(2, random_state=SEED).fit(Xv); P = pca.transform(Xv)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(P[:, 0], P[:, 1], s=3, alpha=.3, color="#2F5DA8"); ax[0].set_title(f"PCA 2D ({pca.explained_variance_ratio_.sum()*100:.0f}% of variance kept)")
for k, col in zip(S.MarketSegment.fillna("NA").unique(), sns.color_palette("tab10", 10)):
    mk = (S.MarketSegment.fillna("NA") == k).values
    ax[1].scatter(P[mk, 0], P[mk, 1], s=3, alpha=.4, color=col, label=k)
ax[1].legend(markerscale=4, fontsize=7); ax[1].set_title("Same plot coloured by MarketSegment (viewing only)")
plt.tight_layout(); plt.savefig(FIG / "p1b_11_pca_view.png", dpi=140); plt.close()

# ---------- column summary table ----------
rows = []
for c in df.columns[:31]:
    x = df[c]
    isn = pd.api.types.is_numeric_dtype(x)
    rows.append(dict(column=c, type="numeric" if isn else "categorical", missing_pct=round(x.isna().mean() * 100, 2),
                     unique=x.nunique(), dominant_share=round(x.value_counts(normalize=True).iloc[0], 3),
                     skew=round(x.skew(), 2) if isn else "", outlier_pct=round(
                         ((x < x.quantile(.25) - 1.5 * (x.quantile(.75) - x.quantile(.25))) |
                          (x > x.quantile(.75) + 1.5 * (x.quantile(.75) - x.quantile(.25)))).mean() * 100, 1) if isn else "",
                     role="(fill in: use / exclude / undecided)"))
pd.DataFrame(rows).to_csv(TAB / "p1b_column_table.csv", index=False)
print("figures in outputs/figures (p1b_*), tables in outputs/tables (p1b_*)")