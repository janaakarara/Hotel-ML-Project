"""
Clustering feature audit + cleaning (CSE382 Task C, stage 1).

    python -m src.clustering.feature_audit --input data/processed/H2_clean_v2_dedup.csv

What it does (no manual column editing):
  1. Loads the file, normalises placeholders ('NULL', 'none', blanks).
  2. Drops columns by MEANING (leakage: outcome / known-only-later). A model cannot
     decide this - it depends on timing - so this list is the only hand-written rule.
  3. Audits every remaining column with DATA evidence and draws graphs:
       missing share, cardinality, dominant-value share, skew,
       redundancy between columns, leave-one-out cluster impact, surrogate importance.
  4. Applies thresholds (see THRESHOLDS) -> DROP / KEEP, each with a written reason.
  5. Cleans rows (impossible rows, undefined categories, rare categories -> 'Other').
     Derived feature: Kids = Children + Babies (Babies alone is 99% zeros).
  6. Saves the clustering-ready file + decision tables + graphs.

IsCanceled is read ONLY by the leakage screen (step 2b), to detect columns that secretly
encode the outcome. It is never used to choose useful features, k, or hyperparameters.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, silhouette_score

SEED = 42
K_REF = 4                 # fixed reference k, so leave-one-out runs are comparable
SAMPLE = 8000             # rows for silhouette / redundancy (speed)

THRESHOLDS = {
    "max_missing_share": 0.50,    # drop if more than 50% placeholder/missing
    "max_cardinality": 30,        # drop categoricals with more levels than this
    "max_dominant_share": 0.98,   # drop if one value covers >= 98% of rows
    "redundancy": 0.80,           # |corr| / Cramer's V / eta above this = same info
    "rare_category_share": 0.01,  # categories under 1% are grouped into 'Other'
    "skew_transform": 2.0,        # |skew| above this -> log1p before scaling
    "loo_noise_margin": 0.03,     # LOO ARI within this of the re-seeding ARI = no impact
    "leak_min_group": 100,        # leakage screen: ignore groups smaller than this
    "leak_extreme_rate": 0.01,    # a group with cancel rate <=1% or >=99% = suspicious
    "max_importance_share": 0.20,  # one feature explaining >20% of the cluster structure
}                                  # hijacks the clustering -> dropped, then re-checked

# ---------------- meaning-based rules (the only hand-written decisions) ---------------
LEAKAGE = {
    "IsCanceled": "outcome (the thing we must not hand to the clusters)",
    "ReservationStatus": "outcome: final status (Canceled / No-Show / Check-Out)",
    "ReservationStatusDate": "outcome: date the final status was set (future event)",
    "AssignedRoomType": "assigned at/near check-in, not known when booking is made",
    "BookingChanges": "accumulates between booking and check-in (future information)",
    "ADR": "price outcome: clusters would become 'cheap vs expensive'",
    "TotalOfSpecialRequests": "accumulates between booking and check-in (like BookingChanges); "
                              "within every lead-time bin, non-canceled bookings have ~60% more "
                              "requests, consistent with canceled bookings stopping early",
}
CALENDAR = ["ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
            "ArrivalDateMonth"]   # describe WHEN, not WHO -> used to interpret clusters later
NUMERIC = ["LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
           "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Kids",
           "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
           "DaysInWaitingList", "RequiredCarParkingSpaces", "TotalOfSpecialRequests"]
CATEGORICAL = ["ArrivalDateMonth", "Meal", "Country", "MarketSegment",
               "DistributionChannel", "ReservedRoomType", "DepositType", "Agent",
               "Company", "CustomerType"]
PLACEHOLDERS = ["NULL", "NA", "nan", "none", "None", ""]


# ---------------------------------- helpers ---------------------------------------
def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.apply(lambda s: s.str.strip())
    return df.replace({p: np.nan for p in PLACEHOLDERS})


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a, b)
    n = t.values.sum()
    exp = np.outer(t.sum(1), t.sum(0)) / n
    chi2 = ((t.values - exp) ** 2 / np.where(exp == 0, 1, exp)).sum()
    r, k = t.shape
    return float(np.sqrt(chi2 / (n * max(1, min(r, k) - 1))))


def eta(cat: pd.Series, num: pd.Series) -> float:
    """correlation ratio: share of the numeric variance explained by the category."""
    grand = num.mean()
    g = num.groupby(cat)
    ss_between = (g.count() * (g.mean() - grand) ** 2).sum()
    ss_total = ((num - grand) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total else 0.0


def build_blocks(df: pd.DataFrame, feats: list[str]) -> dict[str, np.ndarray]:
    """One block of columns per original feature, so a feature can be removed whole."""
    blocks = {}
    for f in feats:
        if f in NUMERIC:
            x = df[f].astype(float)
            if abs(x.skew()) > THRESHOLDS["skew_transform"] and x.min() >= 0:
                x = np.log1p(x)
            sd = x.std()
            blocks[f] = ((x - x.mean()) / (sd if sd else 1)).to_numpy()[:, None]
        else:
            blocks[f] = pd.get_dummies(df[f], dtype=float).to_numpy()
    return blocks


def stack(blocks, feats):
    return np.hstack([blocks[f] for f in feats])


def kmeans_labels(X, seed=SEED):
    return KMeans(K_REF, n_init=5, random_state=seed).fit_predict(X)


def sil(X, lab, rng):
    idx = rng.choice(len(X), min(SAMPLE, len(X)), replace=False)
    return silhouette_score(X[idx], lab[idx])


def reference(df, surv):
    """Blocks, matrix, reference K-Means labels, surrogate importance per feature."""
    blocks = build_blocks(df, surv)
    X = stack(blocks, surv)
    ref = kmeans_labels(X)
    Xs = pd.concat([df[[c for c in surv if c in NUMERIC]].astype(float),
                    pd.get_dummies(df[[c for c in surv if c in CATEGORICAL]], dtype=float)],
                   axis=1)
    rf = RandomForestClassifier(200, max_depth=12, n_jobs=-1, random_state=SEED).fit(Xs, ref)
    raw_imp = pd.Series(rf.feature_importances_, index=Xs.columns)
    imp = pd.Series({f: raw_imp[[c for c in Xs.columns if c == f or c.startswith(f + "_")]].sum()
                     for f in surv}).sort_values(ascending=False)
    return blocks, X, ref, imp


def leakage_screen(raw, cand_num, cand_cat, target="IsCanceled"):
    """LEAKAGE DETECTION ONLY. For every candidate column, compute the cancellation rate
    inside each value group. A sizeable group where (almost) nobody or (almost) everybody
    cancels means the column is probably filled in AFTER the outcome is known."""
    y = pd.to_numeric(raw[target], errors="coerce")
    rows = []
    for c in cand_num + cand_cat:
        x = raw[c]
        if c in cand_num and x.nunique() > 15:
            g = pd.qcut(x, 10, duplicates="drop").astype(str)
        else:
            g = x.astype(str).fillna("MISSING")
        t = pd.DataFrame({"g": g, "y": y}).groupby("g").y.agg(["size", "mean"])
        for k, r in t.iterrows():
            rows.append({"column": c, "group": k, "rows": int(r["size"]),
                         "cancel_rate": round(float(r["mean"]), 4)})
    tab = pd.DataFrame(rows)
    lo, hi = THRESHOLDS["leak_extreme_rate"], 1 - THRESHOLDS["leak_extreme_rate"]
    tab["suspicious"] = ((tab["rows"] >= THRESHOLDS["leak_min_group"]) &
                         ((tab["cancel_rate"] <= lo) | (tab["cancel_rate"] >= hi)))
    return tab


def barh(series, title, path, xlabel="", vline=None, color=None):
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(series))))
    series.sort_values().plot.barh(ax=ax, color=color)
    if vline is not None:
        ax.axvline(vline, color="red", ls="--", lw=1)
    ax.set_title(title); ax.set_xlabel(xlabel)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# -------------------------------------- main ---------------------------------------
def main(input_path: str, out_dir="outputs/clustering/audit",
         clean_out="data/processed/h2_cluster_ready.csv"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    raw = load(input_path)
    n0 = len(raw)
    n_cols0 = raw.shape[1]

    # ---- 0. derived feature: Kids = Children + Babies --------------------------------
    for c in ["Children", "Babies"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["Children"] = raw["Children"].fillna(0)          # 4 blanks -> 0
    raw["Kids"] = raw["Children"] + raw["Babies"]        # Babies alone is ~99% zeros
    reasons: dict[str, list[str]] = {}
    evidence: dict[str, dict] = {}

    def drop(col, why):
        reasons.setdefault(col, []).append(why)

    # ---- 1. meaning-based drops -----------------------------------------------------
    for c, why in LEAKAGE.items():
        if c in raw:
            drop(c, f"LEAKAGE: {why}")

    cand_num = [c for c in NUMERIC if c in raw]
    cand_cat = [c for c in CATEGORICAL if c in raw]
    for c in cand_num:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # ---- 2b. leakage screen (uses IsCanceled ONLY to detect leaks) ----------------------
    if "IsCanceled" in raw:
        leak = leakage_screen(raw, cand_num, cand_cat)
        leak.to_csv(out / "leakage_screen.csv", index=False)
        for c, grp in leak[leak.suspicious].groupby("column"):
            if c in reasons:
                continue
            ex = grp.sort_values("rows", ascending=False).iloc[0]
            drop(c, f"SUSPECTED INDIRECT LEAKAGE: value '{ex.group}' ({ex.rows} rows) has "
                    f"cancellation rate {ex.cancel_rate:.1%}, so the column is probably "
                    f"recorded after the outcome is known")

    # ---- 2. data-evidence audit (graphs 1-4) ------------------------------------------
    cand = cand_num + cand_cat
    miss = raw[cand].isna().mean()
    card = pd.Series({c: raw[c].nunique() for c in cand_cat})
    dom = pd.Series({c: raw[c].value_counts(normalize=True, dropna=True).iloc[0]
                     for c in cand})
    skew = raw[cand_num].skew()
    for c in cand:
        evidence[c] = {"missing_share": round(miss[c], 3),
                       "cardinality": int(card.get(c, raw[c].nunique())),
                       "dominant_share": round(dom[c], 3),
                       "skew": round(skew[c], 2) if c in skew else np.nan}

    barh(miss, "Share of missing / placeholder values", out / "01_missing.png",
         vline=THRESHOLDS["max_missing_share"], xlabel="share of rows")
    barh(card, "Categorical cardinality (levels)", out / "02_cardinality.png",
         vline=THRESHOLDS["max_cardinality"], xlabel="distinct values")
    barh(dom, "Share of rows taken by the single most common value",
         out / "03_dominant_value.png", vline=THRESHOLDS["max_dominant_share"])
    barh(skew, "Skewness of numeric columns", out / "04_skew.png",
         vline=THRESHOLDS["skew_transform"], xlabel="skew (|x|>2 -> log transform)")

    for c in cand:
        if c in reasons:
            continue
        if miss[c] > THRESHOLDS["max_missing_share"]:
            drop(c, f"MISSING: {miss[c]:.0%} of rows are blank/'none'")
        if c in cand_cat and card[c] > THRESHOLDS["max_cardinality"]:
            drop(c, f"HIGH CARDINALITY: {card[c]} levels -> one-hot would swamp distances")
        if dom[c] >= THRESHOLDS["max_dominant_share"]:
            drop(c, f"NEAR-CONSTANT: one value covers {dom[c]:.1%} of rows")

    # ---- 3. row cleaning needed before any modelling ------------------------------------
    log = []
    df = raw.copy()

    def remove(mask, why):
        nonlocal df
        k = int(mask.sum())
        log.append({"step": why, "rows_removed": k})
        df = df[~mask]

    df["Children"] = df["Children"].fillna(0) if "Children" in df else 0
    log.append({"step": "Children blank -> 0", "rows_removed": 0})
    guests = df["Adults"] + df["Children"] + df["Babies"]
    remove(guests.fillna(0) <= 0, "impossible: zero guests")
    remove((df["StaysInWeekendNights"] + df["StaysInWeekNights"]) <= 0,
           "impossible: zero nights stayed")
    for c in ["MarketSegment", "DistributionChannel"]:
        remove(df[c].eq("Undefined"), f"undefined {c}")
    remove(df[cand_cat].drop(columns=[c for c in cand_cat if c in reasons],
                             errors="ignore").isna().any(axis=1), "blank categorical")
    df = df.reset_index(drop=True)

    # rare categories -> 'Other' (kept as information, no tiny one-hot columns)
    survivors_cat = [c for c in cand_cat if c not in reasons]
    for c in survivors_cat:
        share = df[c].value_counts(normalize=True)
        rare = share[share < THRESHOLDS["rare_category_share"]].index
        if len(rare):
            df[c] = df[c].where(~df[c].isin(rare), "Other")
            log.append({"step": f"{c}: grouped {len(rare)} rare level(s) into 'Other'",
                        "rows_removed": 0})

    # ---- 4a. evidence: what drives a clustering built on ALL surviving columns ---------------
    surv = [c for c in cand if c not in reasons]
    blocks, X, ref, first_imp = reference(df, surv)
    barh(first_imp, "Surrogate importance with ALL surviving columns\n"
         "(RandomForest predicting the cluster label)",
         out / "07_surrogate_importance.png", xlabel="importance",
         vline=THRESHOLDS["max_importance_share"])
    cal = [c for c in CALENDAR if c in surv]
    cal_share = float(first_imp[cal].sum()) if cal else 0.0
    for c in cal:
        drop(c, f"CALENDAR: describes when, not who. Together the calendar columns explain "
                f"{cal_share:.0%} of the cluster structure, so the clusters would mean "
                f"'which period' instead of 'which customer'. Use them to INTERPRET clusters later.")
    surv = [c for c in surv if c not in reasons]

    # ---- 4b. redundancy (stable statistic, no clustering involved) ---------------------------
    blocks, X, ref, imp = reference(df, surv)
    ref_sil = sil(X, ref, rng)
    idx = rng.choice(len(df), min(SAMPLE, len(df)), replace=False)
    sm = df.iloc[idx]
    pairs = []
    for a_, b_ in combinations(surv, 2):
        if a_ in NUMERIC and b_ in NUMERIC:
            v = abs(sm[a_].astype(float).corr(sm[b_].astype(float), method="spearman"))
        elif a_ in CATEGORICAL and b_ in CATEGORICAL:
            v = cramers_v(sm[a_], sm[b_])
        else:
            cat, num = (a_, b_) if a_ in CATEGORICAL else (b_, a_)
            v = eta(sm[cat], sm[num].astype(float))
        pairs.append((a_, b_, v))
    red = pd.DataFrame(pairs, columns=["a", "b", "strength"])
    mat = red.pivot(index="a", columns="b", values="strength").reindex(index=surv, columns=surv)
    full = mat.combine_first(mat.T).fillna(1.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(full.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(surv))); ax.set_xticklabels(surv, rotation=90, fontsize=7)
    ax.set_yticks(range(len(surv))); ax.set_yticklabels(surv, fontsize=7)
    fig.colorbar(im, label="association (0-1)")
    ax.set_title("Redundancy between columns")
    fig.tight_layout(); fig.savefig(out / "05_redundancy_heatmap.png", dpi=130); plt.close(fig)
    for _, r in red[red.strength >= THRESHOLDS["redundancy"]].sort_values(
            "strength", ascending=False).iterrows():
        if r.a in reasons or r.b in reasons:
            continue
        loser = r.a if imp[r.a] < imp[r.b] else r.b
        keep_ = r.b if loser == r.a else r.a
        drop(loser, f"REDUNDANT with {keep_} (association {r.strength:.2f}) - same "
                    f"information counted twice")

    # ---- 5. ADVISORY evidence on the final set (flags, never automatic drops) ------------------
    surv = [c for c in surv if c not in reasons]
    blocks, X, ref, imp = reference(df, surv)
    ref_sil = sil(X, ref, rng)
    seed_pairs = [(1, 2), (3, 4)]
    ref_runs = {s_: kmeans_labels(X, s_) for pair in seed_pairs for s_ in pair}
    noise = float(np.mean([adjusted_rand_score(ref_runs[a_], ref_runs[b_])
                           for a_, b_ in seed_pairs]))
    loo = {}
    for f in surv:
        Xr = stack(blocks, [c for c in surv if c != f])
        labs = {b_: kmeans_labels(Xr, b_) for _, b_ in seed_pairs}
        loo[f] = {"ARI_vs_full": float(np.mean(
            [adjusted_rand_score(ref_runs[a_], labs[b_]) for a_, b_ in seed_pairs]))}
    loo = pd.DataFrame(loo).T
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(loo))))
    (1 - loo["ARI_vs_full"]).sort_values().plot.barh(ax=ax)
    ax.axvline(1 - noise, color="red", ls="--", lw=1)
    ax.set_title("Leave-one-out: how much the clusters change when a feature is removed\n"
                 "(red line = change caused by re-seeding alone)")
    ax.set_xlabel("1 - ARI"); fig.tight_layout()
    fig.savefig(out / "06_leave_one_out.png", dpi=130); plt.close(fig)

    review = {}
    for f in surv:
        notes = []
        if loo.loc[f, "ARI_vs_full"] >= noise - THRESHOLDS["loo_noise_margin"]:
            notes.append("removing it changes clusters no more than re-seeding")
        if imp[f] >= THRESHOLDS["max_importance_share"]:
            notes.append(f"explains {imp[f]:.0%} of the cluster structure (dominant)")
        review[f] = "; ".join(notes)
    stable = noise >= 0.7
    print(f"re-seeding ARI = {noise:.2f} -> clustering evidence is "
          f"{'reasonably stable' if stable else 'UNSTABLE: treat the review flags as hints only'}")

    # ---- 6. final tables, cleaned file ------------------------------------------------------
    keep = [c for c in cand if c not in reasons]
    rows = []
    for c in dict.fromkeys(list(LEAKAGE) + cand):   # no duplicates
        if c not in raw:
            continue
        ev = evidence.get(c, {})
        rows.append({"feature": c,
                     "decision": "DROP" if c in reasons else "KEEP",
                     "reason": " | ".join(reasons.get(c, ["passed every check"])),
                     "importance": round(float(first_imp.get(c, np.nan)), 4),
                     "loo_ARI": round(float(loo["ARI_vs_full"].get(c, np.nan)), 3),
                     "review_flag": review.get(c, ""),
                     **ev})
    for c in ["Children", "Babies"]:
        rows.append({"feature": c, "decision": "MERGED",
                     "reason": "merged into Kids = Children + Babies (Babies alone is 99% zeros)"})
    dec = pd.DataFrame(rows)
    dec.to_csv(out / "feature_decisions.csv", index=False)
    pd.DataFrame(log).to_csv(out / "cleaning_log.csv", index=False)

    final = df[keep].copy()
    Path(clean_out).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(clean_out, index=False)
    (out / "kept_features.json").write_text(json.dumps(
        {"numeric": [c for c in keep if c in NUMERIC],
         "categorical": [c for c in keep if c in CATEGORICAL],
         "thresholds": THRESHOLDS, "reference_k": K_REF}, indent=2))

    print(f"rows: {n0} -> {len(final)}   features: {n_cols0} -> {len(keep)} (Children + Babies merged into Kids)")
    print(f"reference silhouette (k={K_REF}): {ref_sil:.3f} | re-seeding ARI: {noise:.3f}\n")
    print(dec[["feature", "decision", "reason"]].to_string(index=False, max_colwidth=100))
    print("\nREVIEW FLAGS (advice only):")
    print(dec[dec.review_flag.fillna("") != ""][["feature", "review_flag"]].to_string(index=False))
    print("\n", pd.DataFrame(log).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs/clustering/audit")
    ap.add_argument("--clean-out", default="data/processed/h2_cluster_ready.csv")
    a = ap.parse_args()
    main(a.input, a.out, a.clean_out)