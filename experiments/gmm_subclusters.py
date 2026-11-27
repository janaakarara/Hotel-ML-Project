"""Profile the tuned GMM clusters and test whether the large cluster contains sub-groups.

Part 1  What is in each main cluster
  - mean of every feature per cluster (the features are mostly one-hot, so a mean is the % of bookings
    in that category) and a standardised difference per feature (which features define each cluster)
  - heatmap of cluster means vs the overall mean

Part 2  Does the largest cluster have sub-groups?
  - take only the bookings of the largest cluster, fit a new PCA (90% variance) on them
  - fit GMMs with k = 1..K (full / tied / diag, two reg_covar values; BIC picks the covariance per k)
  - for each k report: BIC, silhouette (model space and 29-D), seed stability (ARI), and a NULL silhouette
    obtained by shuffling every PCA column independently (same marginals, no joint structure).
    A real sub-group structure should beat the null clearly; a continuum is split about as well by
    shuffled data, so its gap is small.
  - verdict + profiles of the chosen sub-clusters

Usage (repo root, after gmm_untuned and gmm_tuned have run):
  python -m src.clustering.gmm_subclusters
  python -m src.clustering.gmm_subclusters --k-max 6
"""
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from src.clustering.gmm_untuned import (
    PALETTE, RANDOM_STATE, base_arg_parser, build_cluster_profiles, cluster_metrics, load_data,
    make_dirs, n_free_parameters, to_model_space,
)

COV_TYPES = ["full", "tied", "diag"]
REG_COVARS = [1e-4, 1e-2]          # 1e-6 is left out: it lets near-singular fits win on BIC
MIN_SHARE_PCT = 2.0
STABILITY_SEEDS = [1, 2, 3, 4, 5]
MIN_STABLE_ARI = 0.80
N_NULL = 5
SIL_WEAK = 0.25                    # rule of thumb: silhouette below ~0.25 = no substantial structure
GAP_MIN = 0.05                     # real silhouette must beat the shuffled-data silhouette by this much


def fit_gmm(Z, k, cov, reg, n_init, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return GaussianMixture(n_components=k, covariance_type=cov, reg_covar=reg, n_init=n_init,
                               init_params="k-means++", max_iter=300, random_state=seed).fit(Z)


def sil(Z, labels, idx):
    return round(float(silhouette_score(Z[idx], labels[idx])), 4) if len(np.unique(labels[idx])) > 1 else np.nan


def stability(Z, k, cov, reg, ref):
    aris = [adjusted_rand_score(ref, fit_gmm(Z, k, cov, reg, 3, s).predict(Z)) for s in STABILITY_SEEDS]
    return round(float(np.mean(aris)), 4), round(float(np.min(aris)), 4)


def null_silhouette(Z, k, cov, reg, idx):
    """Silhouette of the same model type on data whose columns were shuffled independently."""
    rng = np.random.RandomState(RANDOM_STATE)
    vals = []
    for i in range(N_NULL):
        Zn = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        lab = fit_gmm(Zn, k, cov, reg, 2, i).predict(Zn)
        s = sil(Zn, lab, idx)
        if not np.isnan(s):
            vals.append(s)
    return round(float(np.mean(vals)), 4) if vals else np.nan


def feature_drivers(X_df, labels):
    """Standardised difference of each feature: (mean inside cluster - mean outside) / overall std."""
    sd = X_df.std().replace(0, np.nan)
    out = {}
    for c in np.unique(labels):
        m = labels == c
        out[f"C{c}"] = ((X_df[m].mean() - X_df[~m].mean()) / sd).fillna(0.0)
    return pd.DataFrame(out)


def plot_heatmap(X_df, labels, out_path, title):
    sd = X_df.std().replace(0, np.nan)
    z = ((X_df.groupby(labels).mean() - X_df.mean()) / sd).fillna(0.0).T
    z = z.loc[z.abs().max(axis=1).sort_values(ascending=False).index]
    lim = max(1.0, float(np.abs(z.values).max()))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * z.shape[1] + 4), 0.32 * z.shape[0] + 2))
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(z.shape[1]))
    ax.set_xticklabels([f"C{c}" for c in z.columns])
    ax.set_yticks(range(z.shape[0]))
    ax.set_yticklabels(z.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="cluster mean vs overall mean (std units)")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = base_arg_parser("Profile tuned GMM clusters and test the largest cluster for sub-groups.")
    ap.add_argument("--k-max", type=int, default=8)
    args = ap.parse_args()
    art_dir, fig_dir, tab_dir = make_dirs(args.artifacts_dir, args.figures_dir, args.tables_dir)

    X_df, X, raw_df, prof_df = load_data(args.features, args.raw, args.profile_cols)
    pipe = joblib.load(art_dir / "gmm_tuned_model.joblib")
    labels = pipe.named_steps["gmm"].predict(to_model_space(pipe, X))
    counts = pd.Series(labels).value_counts()
    big = int(counts.index[0])
    print("Main clusters:", {int(c): f"{n} ({n / len(labels) * 100:.1f}%)" for c, n in counts.sort_index().items()})

    # ---------------- part 1: what is in each main cluster
    means = X_df.groupby(labels).mean().T
    means.columns = [f"C{c}_mean" for c in means.columns]
    means["overall_mean"] = X_df.mean()
    drivers = feature_drivers(X_df, labels).add_suffix("_std_diff")
    out1 = pd.concat([means, drivers], axis=1)
    out1.to_csv(tab_dir / "clustering_gmm_main_feature_means.csv")
    plot_heatmap(X_df, labels, fig_dir / "clustering_gmm_main_heatmap.png",
                 "Main clusters: feature means vs overall (red = above, blue = below)")
    print("\nTop features separating the clusters (standardised difference):")
    first = drivers.columns[0]
    print(drivers.reindex(drivers[first].abs().sort_values(ascending=False).index).head(10).round(2).to_string())

    # ---------------- part 2: sub-groups inside the largest cluster
    idx = np.where(labels == big)[0]
    Xb = X[idx]
    pca = PCA(n_components=0.90, svd_solver="full", random_state=RANDOM_STATE)
    Zb = pca.fit_transform(Xb)
    print(f"\nLargest cluster C{big}: {len(idx)} bookings -> own PCA {Zb.shape[1]}-D "
          f"({pca.explained_variance_ratio_.sum() * 100:.1f}% variance)")
    rng = np.random.RandomState(RANDOM_STATE)
    sil_b = np.sort(rng.choice(len(Zb), size=min(10_000, len(Zb)), replace=False))

    rows = []
    for k in range(1, args.k_max + 1):
        for cov in (["full"] if k == 1 else COV_TYPES):
            for reg in REG_COVARS:
                g = fit_gmm(Zb, k, cov, reg, 3, RANDOM_STATE)
                lab = g.predict(Zb)
                uniq, cnt = np.unique(lab, return_counts=True)
                row = {"k": k, "cov_type": cov, "reg": reg, "converged": bool(g.converged_),
                       "n_params": n_free_parameters(g), "bic": round(float(g.bic(Zb)), 2),
                       "n_found": len(uniq), "min_share_pct": round(float(cnt.min() / len(lab) * 100), 2)}
                row["valid"] = bool(row["converged"] and row["n_found"] == k and row["min_share_pct"] >= MIN_SHARE_PCT)
                if k >= 2 and len(uniq) >= 2:
                    row["silhouette_model"] = sil(Zb, lab, sil_b)
                    row["silhouette_29d"] = cluster_metrics(Xb, lab, sil_b)["silhouette"]
                rows.append(row)
        print(f"  k={k} fitted")
    search = pd.DataFrame(rows)
    search.to_csv(tab_dir / "clustering_gmm_sub_search.csv", index=False)

    best = (search[search["valid"]].sort_values("bic").groupby("k", as_index=False).first())
    ari_mean, ari_min, null_sil = [], [], []
    for r in best.itertuples():
        if r.k < 2:
            ari_mean.append(np.nan); ari_min.append(np.nan); null_sil.append(np.nan)
            continue
        ref = fit_gmm(Zb, r.k, r.cov_type, r.reg, 10, RANDOM_STATE).predict(Zb)
        a, b = stability(Zb, r.k, r.cov_type, r.reg, ref)
        ari_mean.append(a); ari_min.append(b)
        null_sil.append(null_silhouette(Zb, r.k, r.cov_type, r.reg, sil_b))
    best["ari_mean"], best["ari_min"], best["silhouette_null"] = ari_mean, ari_min, null_sil
    best["silhouette_gap"] = (best["silhouette_model"] - best["silhouette_null"]).round(4)
    best["delta_bic"] = best["bic"].diff()
    best.to_csv(tab_dir / "clustering_gmm_sub_best_per_k.csv", index=False)
    cols = ["k", "cov_type", "reg", "bic", "delta_bic", "silhouette_model", "silhouette_null", "silhouette_gap",
            "ari_mean", "min_share_pct"]
    print("\nBest model per k (covariance chosen by BIC):")
    print(best[cols].to_string(index=False))

    cand_all = best[best["k"] >= 2]
    if cand_all.empty:
        print("\nNo valid k>=2 fit. The largest cluster is treated as a single group.")
        return
    stable = cand_all[cand_all["ari_mean"] >= MIN_STABLE_ARI]
    pool = stable if not stable.empty else cand_all.sort_values("ari_mean", ascending=False).head(1)
    chosen = pool.sort_values("silhouette_model", ascending=False).iloc[0]
    supported = bool(not stable.empty and chosen["silhouette_model"] >= SIL_WEAK and chosen["silhouette_gap"] >= GAP_MIN)
    kc, cc, rc = int(chosen["k"]), chosen["cov_type"], float(chosen["reg"])
    print(f"\nCandidate: k={kc}, {cc}, reg={rc:g}: silhouette {chosen['silhouette_model']:.3f} "
          f"(shuffled-data {chosen['silhouette_null']:.3f}), stability ARI {chosen['ari_mean']:.2f}")
    if supported:
        print("VERDICT: sub-groups are supported (stable, silhouette above 0.25, clearly above the null).")
    else:
        print("VERDICT: no convincing sub-groups. The large cluster behaves like one continuous segment; "
              "the profiles below are descriptive only and should not be reported as distinct segments.")

    g = fit_gmm(Zb, kc, cc, rc, 20, RANDOM_STATE)
    sub = g.predict(Zb)
    probs = g.predict_proba(Zb)
    profiles = build_cluster_profiles(raw_df.iloc[idx].reset_index(drop=True),
                                      prof_df.iloc[idx].reset_index(drop=True), sub, probs)
    profiles.to_csv(tab_dir / "clustering_gmm_sub_profiles.csv", index=False)
    sub_means = pd.DataFrame(Xb).set_axis(X_df.columns, axis=1).groupby(sub).mean().T
    sub_means.columns = [f"S{c}_mean" for c in sub_means.columns]
    pd.concat([sub_means, feature_drivers(pd.DataFrame(Xb, columns=X_df.columns), sub).add_suffix("_std_diff")],
              axis=1).to_csv(tab_dir / "clustering_gmm_sub_feature_means.csv")
    plot_heatmap(pd.DataFrame(Xb, columns=X_df.columns), sub, fig_dir / "clustering_gmm_sub_heatmap.png",
                 f"Sub-clusters of C{big} (k={kc}): feature means vs the cluster's own mean")

    seg = pd.DataFrame({"row": np.arange(len(X)), "main_cluster": labels, "sub_cluster": -1})
    seg.loc[idx, "sub_cluster"] = sub
    seg["segment"] = np.where(seg["sub_cluster"] >= 0, "C" + seg["main_cluster"].astype(str) + "." + seg["sub_cluster"].astype(str),
                              "C" + seg["main_cluster"].astype(str))
    seg.to_csv(tab_dir / "clustering_gmm_segments.csv", index=False)

    # ---------------- figure: BIC, silhouette vs null, and the sub-clusters in the sub-PCA plane
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(best["k"], best["bic"], marker="o", color="#2F5DA8")
    ax[0].set_title("BIC vs k (lower = better; look for the elbow)", loc="left", fontsize=10.5, fontweight="bold")
    c2 = best[best["k"] >= 2]
    ax[1].plot(c2["k"], c2["silhouette_model"], marker="o", color="#2E8B57", label="real data")
    ax[1].plot(c2["k"], c2["silhouette_null"], marker="s", ls="--", color="#C8433B", label="shuffled columns (null)")
    ax[1].axhline(SIL_WEAK, color="grey", ls=":", lw=1)
    ax[1].set_title("Silhouette: real vs null", loc="left", fontsize=10.5, fontweight="bold")
    ax[1].legend(frameon=True)
    show = rng.choice(len(Zb), size=min(15000, len(Zb)), replace=False)
    for c in np.unique(sub):
        m = sub[show] == c
        ax[2].scatter(Zb[show][m, 0], Zb[show][m, 1], s=8, alpha=0.35, edgecolors="none",
                      color=PALETTE[int(c) % len(PALETTE)], label=f"S{c} ({(sub == c).mean() * 100:.1f}%)")
    ax[2].set_xlabel("PC1 (within the large cluster)")
    ax[2].set_ylabel("PC2 (within the large cluster)")
    ax[2].set_title(f"Sub-clusters of C{big}, k={kc}", loc="left", fontsize=10.5, fontweight="bold")
    ax[2].legend(frameon=True)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_xlabel("k"); ax[1].set_xlabel("k")
    fig.tight_layout()
    fig.savefig(fig_dir / "clustering_gmm_sub_analysis.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("\nSub-cluster profiles:")
    print(profiles[["Cluster", "Count", "Share_Pct", "LeadTime_Mean", "TotalNights_Mean", "HasChildren_Pct",
                    "Top_MarketSegment", "Top_CountryGroup", "PostHoc_ADR_Mean", "PostHoc_CancelRate_Pct"]].to_string(index=False))


if __name__ == "__main__":
    main()