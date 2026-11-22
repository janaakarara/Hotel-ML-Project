"""
Task C - K-Means tuning on the preprocessed table (h2_clustering_preprocessed_v0.csv).

    python -m src.clustering.preprocessedcsv_trials.kmeans_tuning              # best 2 k values
    python -m src.clustering.preprocessedcsv_trials.kmeans_tuning --top 3      # best 3 k values
    python -m src.clustering.preprocessedcsv_trials.kmeans_tuning --k 5 6      # exactly these k values

Every k-specific output is named preprocessed_kmeans_k<k>_...  (so different k never overwrite each other).

What it does
  1. K SCAN      k = 2..10, each k fitted with 5 different seeds (10 restarts each) and averaged,
                 so the choice does not depend on one lucky seed.   (done once)
  2. CHOOSE k    inside k = 3..8, drop any k whose smallest cluster is under 3% or whose result
                 changes with the seed (stability below 0.90). Rank the rest by silhouette
                 (higher is better) and Davies-Bouldin (lower is better). The BEST 2 (lowest average
                 rank; a tie goes to the smaller k) are carried forward.   (done once)
  Then, for EACH of the chosen k:
  3. SEEDING     try 2 initialisations (k-means++ and random) x 20 seeds, one start each,
                 and keep the (init, seed) with the lowest inertia.
  4. TRAIN/TEST  fit on 80% of the rows, then assign the unseen 20% to the nearest centre and
                 compare the scores. Similar scores = the clusters are not specific to the training rows.
  5. FINAL       refit on all rows with the best settings and save the model.
  6. EXPLAIN     say which cluster each feature BELONGS TO: for every feature, the one cluster where
                 it differs most from the hotel average, HIGH or LOW, with the numbers in plain words.
                 Also saves the cluster profiles in real units, a heatmap, and a CSV with the cluster
                 of every booking.

  "belongs to" = the cluster where that feature differs most from the hotel average (at least 0.5
                 standard deviations away, in either direction). If no cluster stands out, the feature
                 is spread evenly and belongs to none.

No outcome column (IsCanceled, ADR ...) is read anywhere. Only distances between bookings are used.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[3]            # project folder (src/clustering/preprocessedcsv_trials/ -> 3 levels up)
PROCESSED = ROOT / "data" / "processed"
PREP_CSV = ROOT / "data" / "processed" / "h2_clustering_preprocessed_v0.csv"
RAW_CSV = ROOT / "data" / "processed" / "h2_clustering_raw_features.csv"   # only to describe clusters
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
ARTIFACTS = ROOT / "artifacts"

SEED = 42                       # fixed seed for the split, the silhouette sample and the plots
K_SCAN = list(range(2, 11))     # k values that are tried
K_MIN, K_MAX = 3, 8             # k = 2 only halves the hotel; more than 8 is too many to describe
MIN_CLUSTER_SHARE = 0.03        # a cluster under 3% of the bookings is too small to act on
MIN_STABILITY = 0.90            # ARI between runs with different seeds must be at least this
SCAN_SEEDS = [0, 1, 2, 3, 4]    # seeds averaged for every k in the scan
SEED_TRIALS = list(range(20))   # seeds tried for the final model
INITS = ["k-means++", "random"]
SILHOUETTE_SAMPLE = 10_000      # silhouette compares every pair of points: score a fixed sample
TEST_SIZE = 0.20
TOP_K = 2                       # how many of the best k values are carried forward
NOTICEABLE = 0.5                # |z-diff| at or above this is reported as a defining feature
PLOT_SAMPLE = 5_000


# ----------------------------------------------------------------------------- scoring
def score(X, labels, sample_idx=None):
    """Internal scores. They only look at the prepared data and the cluster numbers."""
    if sample_idx is None:
        sample_idx = np.random.RandomState(SEED).choice(
            len(X), min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    return {
        "silhouette": silhouette_score(X[sample_idx], labels[sample_idx]),   # higher = better
        "davies_bouldin": davies_bouldin_score(X, labels),                   # lower = better
        "calinski_harabasz": calinski_harabasz_score(X, labels),             # higher = better
    }


def smallest_share(labels):
    return np.bincount(labels).min() / len(labels)


# ----------------------------------------------------------------------------- 1. k scan
def scan_k(X, sample_idx):
    rows = []
    for k in K_SCAN:
        start = time.perf_counter()
        runs = []
        for seed in SCAN_SEEDS:
            model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
            runs.append((model.inertia_, model.labels_))
        # stability: does a different seed give the same clusters? (1.0 = identical)
        ref = runs[0][1]
        stability = float(np.mean([adjusted_rand_score(ref, lab) for _, lab in runs[1:]]))
        per_seed = [score(X, lab, sample_idx) for _, lab in runs]
        rows.append({
            "k": k,
            "inertia": np.mean([r[0] for r in runs]),
            "silhouette": np.mean([s["silhouette"] for s in per_seed]),
            "davies_bouldin": np.mean([s["davies_bouldin"] for s in per_seed]),
            "calinski_harabasz": np.mean([s["calinski_harabasz"] for s in per_seed]),
            "smallest_cluster_pct": 100 * np.mean([smallest_share(lab) for _, lab in runs]),
            "stability_ari": stability,
            "seconds": round(time.perf_counter() - start, 1),
        })
        print(f"  k={k:2d}  silhouette={rows[-1]['silhouette']:.3f}  "
              f"DB={rows[-1]['davies_bouldin']:.3f}  smallest={rows[-1]['smallest_cluster_pct']:.1f}%  "
              f"stability={stability:.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. choose k
def choose_k(scan):
    in_range = scan[(scan.k >= K_MIN) & (scan.k <= K_MAX)].copy()
    ok = in_range[(in_range.smallest_cluster_pct >= 100 * MIN_CLUSTER_SHARE) &
                  (in_range.stability_ari >= MIN_STABILITY)].copy()
    note = ""
    if ok.empty:                                    # never leave the user without an answer
        ok = in_range.copy()
        note = "WARNING: no k passed the size/stability guards; ranking all k in the range."
    ok["rank_silhouette"] = ok.silhouette.rank(ascending=False)
    ok["rank_davies_bouldin"] = ok.davies_bouldin.rank(ascending=True)
    ok["mean_rank"] = (ok.rank_silhouette + ok.rank_davies_bouldin) / 2
    best = ok.sort_values(["mean_rank", "k"]).iloc[0]          # tie -> smaller k
    return int(best.k), ok.sort_values(["mean_rank", "k"]), note


# ----------------------------------------------------------------------------- 3. seeding
def search_seed(X, k, sample_idx):
    rows, labels = [], {}
    for init in INITS:
        for seed in SEED_TRIALS:
            model = KMeans(n_clusters=k, init=init, n_init=1, random_state=seed).fit(X)
            labels[(init, seed)] = model.labels_
            rows.append({"init": init, "seed": seed, "inertia": model.inertia_,
                         "silhouette": score(X, model.labels_, sample_idx)["silhouette"],
                         "iterations": model.n_iter_})
    table = pd.DataFrame(rows).sort_values(["inertia", "silhouette"], ascending=[True, False])
    best = table.iloc[0]
    best_labels = labels[(best.init, int(best.seed))]
    # how typical is the winner? ARI against every other run (1.0 = every start finds it)
    typical = float(np.mean([adjusted_rand_score(best_labels, lab)
                             for key, lab in labels.items() if key != (best.init, int(best.seed))]))
    return table, best.init, int(best.seed), typical


# ----------------------------------------------------------------------------- 4. train / test
def train_test(X, k, init, seed):
    X_tr, X_te = train_test_split(X, test_size=TEST_SIZE, random_state=SEED)
    model = KMeans(n_clusters=k, init=init, n_init=1, random_state=seed).fit(X_tr)
    lab_tr, lab_te = model.labels_, model.predict(X_te)          # test rows are only ASSIGNED
    rows = []
    for name, data, lab in [("train", X_tr, lab_tr), ("test", X_te, lab_te)]:
        rows.append({"split": name, "rows": len(data), **score(data, lab),
                     "smallest_cluster_pct": 100 * smallest_share(lab)})
    shares = pd.DataFrame({
        "cluster": range(k),
        "train_share_pct": 100 * np.bincount(lab_tr, minlength=k) / len(lab_tr),
        "test_share_pct": 100 * np.bincount(lab_te, minlength=k) / len(lab_te)}).round(2)
    return pd.DataFrame(rows), shares


# ----------------------------------------------------------------------------- profiles
def profile_clusters(raw, labels):
    """Describe the clusters with the ORIGINAL (unscaled) columns."""
    d = raw.copy()
    d["cluster"] = labels
    out = pd.DataFrame({"bookings": d.groupby("cluster").size()})
    out["share_pct"] = (100 * out.bookings / out.bookings.sum()).round(1)
    for col in d.select_dtypes("number").columns.drop("cluster"):
        out[f"{col}_mean"] = d.groupby("cluster")[col].mean().round(2)
    if {"StaysInWeekendNights", "StaysInWeekNights"} <= set(d.columns):
        out["nights_mean"] = (d.StaysInWeekendNights + d.StaysInWeekNights).groupby(d.cluster).mean().round(2)
    if "LeadTime" in d:
        out["LeadTime_median"] = d.groupby("cluster").LeadTime.median()
    for col in d.select_dtypes(exclude="number").columns:
        top = d.groupby("cluster")[col].agg(lambda s: f"{s.mode().iloc[0]} ({100 * (s == s.mode().iloc[0]).mean():.0f}%)")
        out[f"top_{col}"] = top
    return out


def raw_is_aligned(raw, prep):
    """The raw table must be the same bookings in the same order as the preprocessed table."""
    if len(raw) != len(prep) or "LeadTime" not in raw or "LeadTime" not in prep:
        return False
    return raw["LeadTime"].corr(prep["LeadTime"], method="spearman") > 0.99


# ----------------------------------------------------------------------------- 6. explain
def zdiff_table(prep, labels):
    """cluster x feature: (cluster mean - overall mean) / overall std  (+ = higher than the hotel average)."""
    overall_mean, overall_std = prep.mean(), prep.std(ddof=0).replace(0, np.nan)
    return ((prep.groupby(labels).mean() - overall_mean) / overall_std).round(3)


def pretty(feature, raw_columns):
    """MarketSegment_Direct -> 'MarketSegment = Direct', Children_flag -> 'Children (has any)'."""
    for col in sorted(raw_columns, key=len, reverse=True):
        if feature == col:
            return col
        if feature == col + "_flag":
            return f"{col} (has any)"
        if feature.startswith(col + "_"):
            value = feature[len(col) + 1:]
            return f"{col} = {'Other (rare types)' if value == 'Other' else value}"
    return feature


def feature_to_cluster(prep, labels, z, raw):
    """One row per feature: the ONE cluster it belongs to (where it differs most from the hotel
    average), whether it is HIGH or LOW there, and WHY in plain words."""
    means = prep.groupby(labels).mean()
    overall = prep.mean()
    raw_columns = list(raw.columns) if raw is not None else []
    rows = []
    for f in z.columns:
        zf = z[f].dropna()
        if zf.empty:
            continue
        c = int(zf.abs().idxmax())
        strength = float(abs(zf[c]))
        name = pretty(f, raw_columns)
        if strength < NOTICEABLE:
            rows.append({"belongs_to_cluster": "none", "feature": name, "direction": "", "strength": round(strength, 2),
                         "why": "about the same in every cluster"})
            continue
        high = zf[c] > 0
        if set(prep[f].unique()) <= {0, 1, 0.0, 1.0}:
            share, hotel = 100 * means.loc[c, f], 100 * overall[f]
            why = f"{share:.0f}% of this cluster vs {hotel:.0f}% of the whole hotel"
            if high:
                why += f"; {100 * prep.loc[labels == c, f].sum() / prep[f].sum():.0f}% of all such bookings are in this cluster"
        else:
            src = raw[f] if raw is not None and f in raw else None
            if src is not None:
                per = src.groupby(labels).mean()
                why = f"average {per[c]:.1f} here vs {src.mean():.1f} for the whole hotel"
                other = int(zf.idxmin() if high else zf.idxmax())          # the opposite extreme
                if other != c and abs(zf[other]) >= NOTICEABLE:
                    why += f"; the {'lowest' if high else 'highest'} is cluster {other} (average {per[other]:.1f})"
            else:
                why = f"{'above' if high else 'below'} the hotel average (scaled units)"
        rows.append({"belongs_to_cluster": c, "feature": name, "direction": "HIGH" if high else "LOW",
                     "strength": round(strength, 2), "why": why})
    table = pd.DataFrame(rows)
    table["_o"] = table.belongs_to_cluster.map(lambda v: 99 if v == "none" else v)
    return table.sort_values(["_o", "strength"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def print_simple_answer(table, labels, k):
    """Grouped by cluster: each feature, HIGH or LOW, and why."""
    sizes = pd.Series(labels).value_counts().sort_index()
    width = max(len(f) for f in table.feature) + 2
    print("\n" + "#" * 100)
    print(f"# k = {k}: WHICH FEATURE BELONGS TO WHICH CLUSTER, AND WHY")
    print("#   HIGH = this feature is higher in the cluster than in the whole hotel, LOW = lower")
    print("#" * 100)
    for c in range(k):
        mine = table[table.belongs_to_cluster == c]
        print(f"\nCLUSTER {c}   ({sizes[c]} bookings, {100 * sizes[c] / len(labels):.0f}% of the hotel)")
        if mine.empty:
            print("   no feature stands out: these are the 'typical' bookings")
        for r in mine.itertuples():
            print(f"   {r.feature:<{width}}{r.direction:<5}  {r.why}")
    none = table[table.belongs_to_cluster == "none"].feature.tolist()
    print(f"\nNO CLUSTER (about the same everywhere): {', '.join(none) if none else '-'}")


def plot_heatmap(z, k, tag):
    data = z.T.fillna(0)                                   # features down, clusters across
    fig, ax = plt.subplots(figsize=(1.2 * k + 5, 0.32 * len(data) + 2))
    im = ax.imshow(data.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(k), [f"cluster {c}" for c in data.columns])
    ax.set_yticks(range(len(data)), data.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 1.2 else "black")
    fig.colorbar(im, ax=ax, label="z-diff vs hotel average (red = higher, blue = lower)", shrink=0.6)
    ax.set_title(f"What defines each cluster (k = {k})")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{tag}_centroid_heatmap.png", dpi=140)
    plt.close(fig)



# ----------------------------------------------------------------------------- plots
def plot_scan(scan, ok, k, tag):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, title in zip(ax, ["inertia", "silhouette", "davies_bouldin"],
                             ["Inertia (look for an elbow)", "Silhouette (higher is better)",
                              "Davies-Bouldin (lower is better)"]):
        a.plot(scan.k, scan[col], "o-")
        a.axvspan(K_MIN - .5, K_MAX + .5, color="grey", alpha=.12)
        a.axvline(k, color="red", ls="--", lw=1.2, label=f"chosen k = {k}")
        a.set_title(title); a.set_xlabel("k")
    ax[0].legend()
    fig.suptitle("K-Means k scan (mean of 5 seeds; grey band = allowed range)")
    fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_k_scan.png", dpi=130); plt.close(fig)


def plot_pca(X, labels, centers, tag):
    idx = np.random.RandomState(SEED).choice(len(X), min(PLOT_SAMPLE, len(X)), replace=False)
    pca = PCA(2, random_state=SEED).fit(X)
    p, c = pca.transform(X[idx]), pca.transform(centers)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(p[:, 0], p[:, 1], c=labels[idx], s=4, cmap="tab10", alpha=.5)
    ax.scatter(c[:, 0], c[:, 1], c="black", marker="X", s=120, label="centres")
    ax.set_title(f"Clusters in PCA 2-D ({100 * pca.explained_variance_ratio_.sum():.0f}% of variance shown)")
    ax.legend(*sc.legend_elements(), title="cluster", loc="best")
    fig.tight_layout(); fig.savefig(FIGURES / f"{tag}_pca_2d.png", dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- one k, steps 3-6
def run_one_k(k, prep, X, scan, ranked, sample_idx, raw):
    tag = f"preprocessed_kmeans_k{k}"          # every k-specific file starts with this
    print("\n" + "=" * 100)
    print(f"k = {k}")
    print("=" * 100)

    print(f"\n3) Seeding for k = {k}: {len(INITS)} inits x {len(SEED_TRIALS)} seeds, one start each")
    seeds, init, seed, typical = search_seed(X, k, sample_idx)
    seeds.round(4).to_csv(TABLES / f"{tag}_seed_search.csv", index=False)
    summary = seeds.groupby("init").inertia.agg(["min", "mean", "std"]).round(1)
    print(summary.to_string())
    print(f"   -> best init = {init}, best seed = {seed}  (lowest inertia; "
          f"agreement with the other runs = {typical:.3f})")

    print("\n4) Train / test (fit on 80%, assign the unseen 20%):")
    tt, shares = train_test(X, k, init, seed)
    print(tt.round(3).to_string(index=False))
    gap = abs(tt.iloc[0].silhouette - tt.iloc[1].silhouette)
    print(f"   silhouette gap train vs test = {gap:.3f}  "
          f"({'clusters carry over to unseen bookings' if gap < 0.02 else 'check: clusters may be specific to the training rows'})")
    print(shares.to_string(index=False))
    tt.round(4).to_csv(TABLES / f"{tag}_train_test.csv", index=False)

    print("\n5) Final model on all rows")
    model = KMeans(n_clusters=k, init=init, n_init=1, random_state=seed).fit(X)
    labels = model.labels_
    final = {"algorithm": "kmeans", "k": k, "init": init, "seed": seed, "rows": len(X),
             "columns": X.shape[1], **score(X, labels, sample_idx), "inertia": float(model.inertia_),
             "smallest_cluster_pct": float(100 * smallest_share(labels)),
             "train_test_silhouette_gap": float(gap)}
    pd.DataFrame([final]).round(4).to_csv(TABLES / f"{tag}_final_metrics.csv", index=False)
    print({key: (round(v, 4) if isinstance(v, float) else v) for key, v in final.items()})
    joblib.dump({"model": model, "columns": list(prep.columns), "k": k, "init": init, "seed": seed},
                ARTIFACTS / f"{tag}.joblib")

    # ---- 6. explain: which cluster does each feature belong to?
    print(f"\n6) Which cluster does each feature belong to? (k = {k})  -> printed at the very end, saved to {tag}_feature_to_cluster.csv")
    z = zdiff_table(prep, labels)
    table = feature_to_cluster(prep, labels, z, raw)
    table.to_csv(TABLES / f"{tag}_feature_to_cluster.csv", index=False)
    if raw is not None:
        profile = profile_clusters(raw, labels)
        profile.to_csv(TABLES / f"{tag}_cluster_profiles.csv")
        print("\n   Cluster profiles (original, unscaled columns):")
        print(profile.T.to_string())
        labeled = raw.copy()
        labeled["cluster"] = labels
        labeled.to_csv(PROCESSED / f"h2_clustering_labeled_k{k}.csv", index=False)
    else:
        print("(Profiles and the labeled CSV skipped: raw feature file missing or not in the same row order.)")

    plot_scan(scan, ranked, k, tag)
    plot_pca(X, labels, model.cluster_centers_, tag)
    plot_heatmap(z, k, tag)
    print(f"\nSaved for k = {k}: tables {tag}_*.csv, figures {tag}_k_scan / _pca_2d / _centroid_heatmap .png, "
          f"model artifacts/{tag}.joblib" + (f", data/processed/h2_clustering_labeled_k{k}.csv" if raw is not None else ""))
    return final, table, labels


# ----------------------------------------------------------------------------- main
def main(force_ks=None, top=TOP_K):
    for folder in (TABLES, FIGURES, ARTIFACTS, PROCESSED):
        folder.mkdir(parents=True, exist_ok=True)
    prep = pd.read_csv(PREP_CSV)
    assert not prep.isna().any().any(), "the preprocessed table must have no missing values"
    X = prep.to_numpy(dtype=float)
    sample_idx = np.random.RandomState(SEED).choice(len(X), min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    print(f"Loaded {X.shape[0]} bookings x {X.shape[1]} columns from {PREP_CSV.name}\n")

    print("1) k scan (5 seeds per k, 10 restarts each):")
    scan = scan_k(X, sample_idx)
    scan.round(4).to_csv(TABLES / "preprocessed_kmeans_k_scan.csv", index=False)

    chosen_all, ranked, note = choose_k(scan)
    best = [int(k) for k in ranked.k.head(top)]              # ranked is already sorted best-first
    ks = force_ks or best
    print(f"\n2) Choice of k (range {K_MIN}-{K_MAX}, smallest cluster >= {MIN_CLUSTER_SHARE:.0%}, "
          f"stability >= {MIN_STABILITY}):")
    print(ranked[["k", "silhouette", "davies_bouldin", "calinski_harabasz", "smallest_cluster_pct",
                  "stability_ari", "mean_rank"]].round(3).to_string(index=False))
    if note:
        print(note)
    print(f"   -> best {top} k values: {best}" + (f"   (forced to k = {force_ks})" if force_ks else ""))
    ranked.round(4).to_csv(TABLES / "preprocessed_kmeans_k_ranking.csv", index=False)

    raw = None
    if RAW_CSV.exists():
        candidate = pd.read_csv(RAW_CSV)
        raw = candidate if raw_is_aligned(candidate, prep) else None

    results = [run_one_k(k, prep, X, scan, ranked, sample_idx, raw) for k in ks]
    finals = [r[0] for r in results]

    print("\n" + "=" * 100)
    print("Final comparison of the chosen k values (all rows)")
    print("=" * 100)
    comparison = pd.DataFrame(finals).set_index("k")[
        ["silhouette", "davies_bouldin", "calinski_harabasz", "inertia", "smallest_cluster_pct",
         "train_test_silhouette_gap", "init", "seed"]]
    print(comparison.round(4).to_string())
    print(f"\nTables in {TABLES}, figures in {FIGURES}, models in {ARTIFACTS}")

    for k, (_, table, labels) in zip(ks, results):          # the easy answer, last, so it is what you see
        print_simple_answer(table, labels, k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tune K-Means on the preprocessed table, then train and explain the best k values")
    ap.add_argument("--k", type=int, nargs="+", default=None, help="use exactly these k values instead of the best ones")
    ap.add_argument("--top", type=int, default=TOP_K, help="how many of the best k values to carry forward")
    args = ap.parse_args()
    main(args.k, args.top)