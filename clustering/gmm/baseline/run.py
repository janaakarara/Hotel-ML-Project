"""Untuned (baseline) Gaussian Mixture Model for Task C (clustering).

Baseline = scikit-learn's GaussianMixture defaults, with only the number of components fixed to 3:
  covariance_type='full', reg_covar=1e-6, n_init=1, init_params='kmeans', max_iter=100,
  fitted directly on the preprocessed clustering features (h2_clustering_stage5.csv, 29 columns).

What is reported:
  - Internal metrics in the 29-D feature space: Silhouette (fixed 10,000-row sample), Davies-Bouldin,
    Calinski-Harabasz. These are the numbers to compare with the tuned model.
  - The GMM's own criteria: BIC, AIC and the number of free parameters (degrees of freedom).
  - Covariance diagnostics: eigenvalues and condition numbers of each fitted covariance matrix, and
    how many eigenvalues sit at the reg_covar floor. The one-hot groups each sum to a constant, so the
    data covariance has exact zero eigenvalues; a full covariance then collapses in those directions.
  - Posterior confidence and entropy, as a diagnostic only (high confidence is NOT evidence of good
    clusters in high dimensions).
ADR, IsCanceled and the other held-out columns are used only afterwards, to describe the clusters.

Usage (from the repo root):
  python -m src.clustering.gmm_untuned
Writes:
  artifacts/gmm_untuned_model.joblib            (fitted pipeline; ignored by git)
  artifacts/gmm_untuned_metadata.json
  outputs/gmm/clustering_gmm_untuned_metrics.csv
  outputs/gmm/clustering_gmm_untuned_profiles.csv
  outputs/gmm/clustering_gmm_untuned_covariance.csv
  outputs/gmm/clustering_gmm_untuned_pca_2d.png
  outputs/gmm/clustering_gmm_untuned_profiles.png
  outputs/gmm/clustering_gmm_untuned_cov_eigenvalues.png
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from src.features.country_groups import country_group

RANDOM_STATE = 42
SIL_SAMPLE = 10_000          # silhouette is O(n^2); a fixed sample keeps it fast and comparable
NEAR_FLOOR_FACTOR = 10.0     # an eigenvalue <= 10 * reg_covar counts as "at the regularisation floor"
FLOOR_COL = "n_eig_at_floor"           # eigenvalues <= NEAR_FLOOR_FACTOR * reg_covar
PALETTE = ["#2F5DA8", "#2E8B57", "#C8433B", "#E08E2B", "#7A49A5", "#2B9EB3", "#B55D60", "#5975A4",
           "#8C6D31", "#3B7A57", "#A0522D", "#6A5ACD"]

BASELINE_PARAMS = dict(n_components=3, covariance_type="full", reg_covar=1e-6, n_init=1,
                       init_params="kmeans", max_iter=100, random_state=RANDOM_STATE)


# ---------------------------------------------------------------- data and model helpers
def load_data(features, raw, profile_cols):
    """Feature matrix used for clustering + the raw columns and held-out columns for profiling."""
    X_df = pd.read_csv(features)
    raw_df = pd.read_csv(raw)
    prof_df = pd.read_csv(profile_cols)
    assert len(X_df) == len(raw_df) == len(prof_df), "row counts differ between the input files"
    return X_df, X_df.values.astype(np.float64), raw_df, prof_df


def silhouette_sample_idx(n_rows, size=SIL_SAMPLE):
    """The same random rows are used for every model, so silhouettes are comparable."""
    rng = np.random.RandomState(RANDOM_STATE)
    return np.sort(rng.choice(n_rows, size=min(size, n_rows), replace=False))


def to_model_space(pipe, X):
    """Apply every step except the GMM (PCA or passthrough)."""
    reducer = pipe.named_steps["reduce"]
    return X if reducer == "passthrough" else reducer.transform(X)


def n_free_parameters(gmm):
    """Degrees of freedom of a fitted GMM: weights + means + covariance parameters."""
    k, d = gmm.means_.shape
    cov = {"full": k * d * (d + 1) / 2, "tied": d * (d + 1) / 2, "diag": k * d, "spherical": k}
    return int(cov[gmm.covariance_type] + k * d + k - 1)


def component_cov(gmm, k):
    """Full (d x d) covariance of component k in the space the GMM was fitted in."""
    d = gmm.means_.shape[1]
    c = gmm.covariances_
    return {"full": lambda: c[k], "tied": lambda: c, "diag": lambda: np.diag(c[k]),
            "spherical": lambda: np.eye(d) * float(c[k])}[gmm.covariance_type]()


def cluster_metrics(X, labels, sil_idx):
    """Internal metrics, always computed in the original feature space (so every model is comparable)."""
    uniq, counts = np.unique(labels, return_counts=True)
    out = {"n_clusters_found": int(len(uniq)),
           "min_cluster_share_pct": round(float(counts.min() / len(labels) * 100), 2),
           "max_cluster_share_pct": round(float(counts.max() / len(labels) * 100), 2)}
    if len(uniq) < 2 or len(np.unique(labels[sil_idx])) < 2:
        out.update(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
        return out
    out["silhouette"] = round(float(silhouette_score(X[sil_idx], labels[sil_idx])), 4)
    out["davies_bouldin"] = round(float(davies_bouldin_score(X, labels)), 4)
    out["calinski_harabasz"] = round(float(calinski_harabasz_score(X, labels)), 2)
    return out


def posterior_diagnostics(probs):
    """Mean max-posterior and mean normalised entropy (0 = hard assignment, 1 = uniform)."""
    k = probs.shape[1]
    p = np.clip(probs, 1e-300, 1.0)
    ent = -(p * np.log(p)).sum(axis=1) / np.log(k) if k > 1 else np.zeros(len(p))
    return {"mean_max_posterior": round(float(probs.max(axis=1).mean()), 4),
            "mean_normalised_entropy": round(float(ent.mean()), 4)}


def summarize_model(pipe, X, sil_idx, fit_time, configuration):
    """One row of metrics for a fitted pipeline."""
    gmm = pipe.named_steps["gmm"]
    Z = to_model_space(pipe, X)
    labels = gmm.predict(Z)
    probs = gmm.predict_proba(Z)
    reducer = pipe.named_steps["reduce"]
    row = {
        "configuration": configuration,
        "feature_space": "raw" if reducer == "passthrough" else f"pca{Z.shape[1]}",
        "model_dims": int(Z.shape[1]),
        "pca_variance_kept": 1.0 if reducer == "passthrough" else round(float(reducer.explained_variance_ratio_.sum()), 4),
        "n_components": int(gmm.n_components),
        "covariance_type": gmm.covariance_type,
        "reg_covar": float(gmm.reg_covar),
        "init_params": gmm.init_params,
        "n_init": int(gmm.n_init),
        "max_iter": int(gmm.max_iter),
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "n_free_parameters": n_free_parameters(gmm),
    }
    row.update(cluster_metrics(X, labels, sil_idx))
    row.update({"bic_model_space": round(float(gmm.bic(Z)), 2),
                "aic_model_space": round(float(gmm.aic(Z)), 2),
                "mean_log_likelihood": round(float(gmm.score(Z)), 4)})
    row.update(posterior_diagnostics(probs))
    shares = (pd.Series(labels).value_counts(normalize=True).sort_index() * 100).round(2)
    row["cluster_shares_pct"] = json.dumps({int(k): float(v) for k, v in shares.items()})
    row["training_time_sec"] = round(float(fit_time), 3)
    return row, labels, probs


def covariance_diagnostics(gmm):
    """Eigenvalue summary of each fitted covariance (in the GMM's own space)."""
    reg = float(gmm.reg_covar)
    comps = [0] if gmm.covariance_type == "tied" else range(gmm.n_components)
    rows = []
    for k in comps:
        ev = np.linalg.eigvalsh(component_cov(gmm, k))
        near = int((ev <= NEAR_FLOOR_FACTOR * reg).sum())
        rows.append({
            "component": "all (tied)" if gmm.covariance_type == "tied" else int(k),
            "dims": int(len(ev)),
            "eig_min": float(ev.min()),
            "eig_max": float(ev.max()),
            "condition_number": float(ev.max() / max(ev.min(), 1e-300)),
            FLOOR_COL: near,
            "reg_covar": reg,
        })
    return pd.DataFrame(rows)


def data_zero_eigenvalues(X, tol=1e-10):
    """Exact linear dependencies in the data (each one-hot group sums to a constant)."""
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    return int((ev < tol).sum()), ev


def build_cluster_profiles(raw_df, prof_df, labels, probs):
    """Describe each cluster with readable booking traits; held-out outcomes are post-hoc only."""
    df = raw_df.copy()
    for col in prof_df.columns:
        df[col] = prof_df[col].values
    df["Cluster"] = labels
    df["MaxPosterior"] = probs.max(axis=1)
    df["TotalNights"] = df["StaysInWeekNights"] + df["StaysInWeekendNights"]
    df["HasChildren"] = (pd.to_numeric(df["Children"], errors="coerce").fillna(0) > 0).astype(int)
    df["CountryGroup"] = df["Country"].map(country_group)

    def top(s):
        vc = s.value_counts(normalize=True)
        return f"{vc.index[0]} ({vc.iloc[0] * 100:.1f}%)"

    rows = []
    for cid, g in df.groupby("Cluster"):
        rows.append({
            "Cluster": int(cid),
            "Count": int(len(g)),
            "Share_Pct": round(len(g) / len(df) * 100, 2),
            "Mean_Max_Posterior": round(float(g["MaxPosterior"].mean()), 4),
            "LeadTime_Mean": round(float(g["LeadTime"].mean()), 2),
            "LeadTime_Median": round(float(g["LeadTime"].median()), 1),
            "TotalNights_Mean": round(float(g["TotalNights"].mean()), 2),
            "WeekNights_Mean": round(float(g["StaysInWeekNights"].mean()), 2),
            "WeekendNights_Mean": round(float(g["StaysInWeekendNights"].mean()), 2),
            "NoWeekendNight_Pct": round(float((g["StaysInWeekendNights"] == 0).mean() * 100), 2),
            "Adults_Mean": round(float(g["Adults"].mean()), 2),
            "HasChildren_Pct": round(float(g["HasChildren"].mean() * 100), 2),
            "Top_MarketSegment": top(g["MarketSegment"]),
            "Top_CustomerType": top(g["CustomerType"]),
            "Top_CountryGroup": top(g["CountryGroup"]),
            "Top_Meal": top(g["Meal"]),
            "Top_ReservedRoom": top(g["ReservedRoomType"]),
            "PostHoc_ADR_Mean": round(float(g["ADR"].mean()), 2),
            "PostHoc_CancelRate_Pct": round(float(g["IsCanceled"].mean() * 100), 2),
            "PostHoc_SpecialRequests_Mean": round(float(g["TotalOfSpecialRequests"].mean()), 2),
        })
    return pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)


# ---------------------------------------------------------------- plots
def gaussians_in_feature_space(pipe):
    """Mean and covariance of every component expressed in the original feature space."""
    gmm, reducer = pipe.named_steps["gmm"], pipe.named_steps["reduce"]
    out = []
    for k in range(gmm.n_components):
        mu, cov = gmm.means_[k], component_cov(gmm, k)
        if reducer != "passthrough":            # x = pca.mean_ + z @ V  ->  Sigma_x = V^T Sigma_z V
            V = reducer.components_
            mu, cov = reducer.mean_ + mu @ V, V.T @ cov @ V
        out.append((mu, cov))
    return out


def _draw_ellipses(ax, mean_2d, cov_2d, color):
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    w, h = 2.0 * np.sqrt(np.maximum(vals, 1e-12))
    for n_std, alpha, lw, ls in [(1.0, 0.18, 2.0, "-"), (2.0, 0.07, 1.5, "--")]:
        ax.add_patch(Ellipse(mean_2d, w * n_std, h * n_std, angle=angle, facecolor=color,
                             edgecolor="none", alpha=alpha, zorder=3))
        ax.add_patch(Ellipse(mean_2d, w * n_std, h * n_std, angle=angle, facecolor="none",
                             edgecolor=color, linewidth=lw, linestyle=ls, zorder=4))


def plot_pca_2d(X, labels, pipe, out_path, title):
    """2-D PCA view of the clusters with each fitted Gaussian projected as 1-sigma / 2-sigma ellipses."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    coords = pca.transform(X)
    W = pca.components_
    var1, var2 = pca.explained_variance_ratio_ * 100
    rng = np.random.RandomState(RANDOM_STATE)
    show = rng.choice(len(X), size=min(15000, len(X)), replace=False)
    gaussians = gaussians_in_feature_space(pipe)

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    for cid in np.unique(labels):
        color = PALETTE[int(cid) % len(PALETTE)]
        m = labels[show] == cid
        ax.scatter(coords[show][m, 0], coords[show][m, 1], s=10, alpha=0.35, color=color, edgecolors="none",
                   label=f"Cluster {cid} ({(labels == cid).mean() * 100:.1f}%)", zorder=2)
        mu, cov = gaussians[int(cid)]
        mean_2d, cov_2d = W @ (mu - pca.mean_), W @ cov @ W.T
        _draw_ellipses(ax, mean_2d, cov_2d, color)
        ax.scatter(*mean_2d, s=220, color=color, edgecolors="black", linewidths=1.8, marker="X", zorder=5)
    ax.set_xlabel(f"PC1 ({var1:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% of variance)")
    ax.set_title(f"{title}\nX = Gaussian means; solid / dashed = 1σ / 2σ ellipses. "
                 f"This view keeps only {var1 + var2:.1f}% of the variance: clusters that overlap here "
                 f"may still separate in the other dimensions.", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="best")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_profiles(profile_tab, out_path, title):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    metrics = [("Share_Pct", "Cluster share (% of bookings)"), ("LeadTime_Mean", "Mean lead time (days)"),
               ("TotalNights_Mean", "Mean total stay (nights)"), ("HasChildren_Pct", "Bookings with children (%)"),
               ("PostHoc_ADR_Mean", "Post-hoc mean ADR (EUR)"), ("PostHoc_CancelRate_Pct", "Post-hoc cancellation rate (%)")]
    names = [f"C{int(c)}" for c in profile_tab["Cluster"]]
    colors = [PALETTE[int(c) % len(PALETTE)] for c in profile_tab["Cluster"]]
    for ax, (col, label) in zip(axes.flat, metrics):
        vals = profile_tab[col].values
        top = max(vals.max(), 1e-9)
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(label, fontsize=10.5, fontweight="bold", loc="left")
        ax.set_ylim(0, top * 1.22)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(title + "\n(ADR and cancellation were NOT used for clustering; shown only to describe the clusters)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cov_eigenvalues(gmm, data_eigs, out_path, title):
    """Sorted eigenvalues of every fitted covariance (log scale) against the reg_covar floor."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    comps = [0] if gmm.covariance_type == "tied" else range(gmm.n_components)
    for k in comps:
        ev = np.sort(np.linalg.eigvalsh(component_cov(gmm, k)))[::-1]
        lab = "tied covariance" if gmm.covariance_type == "tied" else f"component {k}"
        ax.semilogy(np.arange(1, len(ev) + 1), np.maximum(ev, 1e-12), marker="o", ms=4, lw=1.8,
                    color=PALETTE[k % len(PALETTE)], label=lab)
    if data_eigs is not None:
        de = np.sort(data_eigs)[::-1]
        ax.semilogy(np.arange(1, len(de) + 1), np.maximum(de, 1e-12), color="#8A8A8A", ls=":", lw=1.6,
                    label="covariance of the data (this space)")
    ax.axhline(gmm.reg_covar, color="black", ls="--", lw=1.2, label=f"reg_covar = {gmm.reg_covar:g}")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.5, loc="best")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
    return [Path(p) for p in paths]


def base_arg_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--features", default="data/processed/h2_clustering_stage5.csv")
    ap.add_argument("--raw", default="data/processed/h2_clustering_stage2.csv")
    ap.add_argument("--profile-cols", default="data/processed/h2_clustering_profile_cols.csv")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--figures-dir", default="outputs/gmm")
    ap.add_argument("--tables-dir", default="outputs/gmm")
    return ap


# ---------------------------------------------------------------- main
def main():
    args = base_arg_parser("Untuned baseline GMM (scikit-learn defaults, k=3).").parse_args()
    art_dir, fig_dir, tab_dir = make_dirs(args.artifacts_dir, args.figures_dir, args.tables_dir)

    X_df, X, raw_df, prof_df = load_data(args.features, args.raw, args.profile_cols)
    sil_idx = silhouette_sample_idx(len(X))
    print(f"Untuned GMM {BASELINE_PARAMS} on {X.shape[0]} bookings x {X.shape[1]} features")

    pipe = Pipeline([("reduce", "passthrough"), ("gmm", GaussianMixture(**BASELINE_PARAMS))])
    t0 = time.perf_counter()
    pipe.fit(X)
    fit_time = time.perf_counter() - t0

    metrics, labels, probs = summarize_model(pipe, X, sil_idx, fit_time, "untuned_baseline")
    gmm = pipe.named_steps["gmm"]
    n_zero, data_eigs = data_zero_eigenvalues(X)
    cov_diag = covariance_diagnostics(gmm)
    metrics["data_zero_eigenvalues"] = n_zero
    metrics["max_condition_number"] = float(cov_diag["condition_number"].max())
    metrics["min_eig_at_floor_per_component"] = int(cov_diag[FLOOR_COL].min())
    profiles = build_cluster_profiles(raw_df, prof_df, labels, probs)

    joblib.dump(pipe, art_dir / "gmm_untuned_model.joblib")
    (art_dir / "gmm_untuned_metadata.json").write_text(json.dumps(
        {"params": BASELINE_PARAMS, "feature_names": list(X_df.columns), "metrics": metrics}, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(tab_dir / "clustering_gmm_untuned_metrics.csv", index=False)
    profiles.to_csv(tab_dir / "clustering_gmm_untuned_profiles.csv", index=False)
    cov_diag.to_csv(tab_dir / "clustering_gmm_untuned_covariance.csv", index=False)

    tag = f"GMM untuned baseline (raw {X.shape[1]}-D, k=3, full, reg_covar=1e-6)"
    plot_pca_2d(X, labels, pipe, fig_dir / "clustering_gmm_untuned_pca_2d.png", tag)
    plot_cluster_profiles(profiles, fig_dir / "clustering_gmm_untuned_profiles.png", tag + ": cluster profiles")
    plot_cov_eigenvalues(gmm, data_eigs, fig_dir / "clustering_gmm_untuned_cov_eigenvalues.png",
                         f"{tag}: covariance eigenvalues\nThe data itself has {n_zero} zero eigenvalues "
                         f"(each one-hot group sums to a constant), so full covariances hit the reg_covar floor")

    print("\n========== UNTUNED BASELINE GMM ==========")
    for k, v in metrics.items():
        print(f"  {k:32s}: {v}")
    print("\nCovariance diagnostics:")
    print(cov_diag.to_string(index=False))
    print("\nCluster profiles:")
    print(profiles[["Cluster", "Count", "Share_Pct", "LeadTime_Mean", "TotalNights_Mean", "HasChildren_Pct",
                    "Top_MarketSegment", "PostHoc_ADR_Mean", "PostHoc_CancelRate_Pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
