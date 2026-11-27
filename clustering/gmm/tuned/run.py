"""Tuned (hyperparameter-optimised) Gaussian Mixture Model for Task C (clustering).

Search space
  feature space   : preprocessed PCA-90 features (8-D from stage5b_pca.py)
  n_components k  : 2 .. 12
  covariance_type : full, tied, diag, spherical
  reg_covar       : 1e-6, 1e-4, 1e-2, 1e-1
  initialisation  : k-means++

Selection (three steps, nothing uses IsCanceled / ADR)
1. Coarse grid, n_init=3. A fit is valid if it converged, uses all k components and its smallest
     cluster holds at least 2% of bookings.
2. BIC picks the covariance structure: for every k keep the valid fit with the lowest BIC.
     BIC penalises the number of free parameters directly, so it is not biased towards spherical clusters
     the way Silhouette / Davies-Bouldin / Calinski-Harabasz are.
3. BIC alone keeps falling as k grows on 52k rows, so it cannot choose k. Among the BIC-chosen models,
     the 6 with the best Silhouette (computed in the preprocessed feature space) are refitted with
     n_init=10 and tested for stability (Adjusted Rand Index against 5 refits with other seeds). The
     winner is the stable model (mean ARI >= 0.80) with the highest Silhouette; ties go to the lower
     Davies-Bouldin.
  The winner is refitted with n_init=20 and saved as a GMM pipeline.

Usage (from the repo root; run baseline first so the comparison table is filled):
  python clustering/gmm/tuned/run.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from clustering.gmm.baseline.run import (
    PALETTE, RANDOM_STATE, FLOOR_COL, base_arg_parser, build_cluster_profiles, cluster_metrics,
    covariance_diagnostics, data_zero_eigenvalues, load_data, make_dirs, n_free_parameters,
    plot_cluster_profiles, plot_cov_eigenvalues, plot_pca_2d, silhouette_sample_idx, summarize_model,
)

DEFAULT_FEATURES = "data/processed/h2_clustering_pca90.csv"
DEFAULT_RAW = "data/processed/h2_clustering_stage2.csv"

SPACES = {"preprocessed_pca90": None}
COV_TYPES = ["full", "tied", "diag", "spherical"]
REG_COVARS = [1e-6, 1e-4, 1e-2, 1e-1]
MIN_SHARE_PCT = 2.0
SHORTLIST = 6
STABILITY_SEEDS = [1, 2, 3, 4, 5]
STABILITY_MIN_ARI = 0.80
COV_STYLE = {"full": ("#2F5DA8", "o", "-"), "tied": ("#2E8B57", "s", "-"),
             "diag": ("#C8433B", "^", "--"), "spherical": ("#E08E2B", "D", "--")}
SPACE_STYLE = {"preprocessed_pca90": ("#2F5DA8", "s")}


def make_gmm(k, cov, reg, n_init, seed=RANDOM_STATE, max_iter=300):
    return GaussianMixture(n_components=k, covariance_type=cov, reg_covar=reg, n_init=n_init,
                           init_params="k-means++", max_iter=max_iter, random_state=seed)


def _fit_one(space, Z, X, sil_idx, k, cov, reg, n_init):
    """Fit one grid point and score it. Runs in a worker process."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        t0 = time.perf_counter()
        gmm = make_gmm(k, cov, reg, n_init).fit(Z)
        elapsed = time.perf_counter() - t0
    labels = gmm.predict(Z)
    row = {"feature_space": space, "model_dims": int(Z.shape[1]), "n_components": k, "covariance_type": cov,
           "reg_covar": reg, "converged": bool(gmm.converged_), "n_iter": int(gmm.n_iter_),
           "n_free_parameters": n_free_parameters(gmm), "bic_model_space": round(float(gmm.bic(Z)), 2),
           "aic_model_space": round(float(gmm.aic(Z)), 2), "fit_time_sec": round(elapsed, 3)}
    row.update(cluster_metrics(X, labels, sil_idx))
    row["valid"] = bool(row["converged"] and row["n_clusters_found"] == k and row["min_cluster_share_pct"] >= MIN_SHARE_PCT)
    return row


def _stability(Z, k, cov, reg, ref_labels):
    """Mean ARI between the reference labels and refits with other seeds."""
    aris = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for s in STABILITY_SEEDS:
            lab = make_gmm(k, cov, reg, n_init=3, seed=s).fit(Z).predict(Z)
            aris.append(adjusted_rand_score(ref_labels, lab))
    return float(np.mean(aris)), float(np.min(aris))


def _refit_shortlisted(space, Z, X, sil_idx, k, cov, reg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gmm = make_gmm(k, cov, reg, n_init=10).fit(Z)
    labels = gmm.predict(Z)
    row = {"feature_space": space, "model_dims": int(Z.shape[1]), "n_components": k, "covariance_type": cov,
           "reg_covar": reg, "n_init": 10, "converged": bool(gmm.converged_),
           "bic_model_space": round(float(gmm.bic(Z)), 2), "n_free_parameters": n_free_parameters(gmm)}
    row.update(cluster_metrics(X, labels, sil_idx))
    row["stability_ari_mean"], row["stability_ari_min"] = [round(v, 4) for v in _stability(Z, k, cov, reg, labels)]
    return row


# ---------------------------------------------------------------- plots
def plot_tuning_curves(search, chosen, out_path):
    """BIC, AIC and Silhouette vs k for every covariance type (best reg_covar by BIC)."""
    best = (search[search["valid"]].sort_values("bic_model_space")
            .groupby(["feature_space", "covariance_type", "n_components"], as_index=False).first())
    spaces = [s for s in SPACES if s in set(best["feature_space"])]
    fig, axes = plt.subplots(len(spaces), 3, figsize=(16, 4.3 * len(spaces)), squeeze=False)
    for r, space in enumerate(spaces):
        sub = best[best["feature_space"] == space]
        dims = int(sub["model_dims"].iloc[0])
        for c, (col, label) in enumerate([("bic_model_space", "BIC (lower = better)"),
                                          ("aic_model_space", "AIC (lower = better)"),
                                          ("silhouette", f"Silhouette in {dims}-D space (higher = better)")]):
            ax = axes[r, c]
            for cov, (color, marker, ls) in COV_STYLE.items():
                g = sub[sub["covariance_type"] == cov].sort_values("n_components")
                if not g.empty:
                    ax.plot(g["n_components"], g[col], color=color, marker=marker, ls=ls, lw=1.8, ms=5, label=cov)
            if chosen["feature_space"] == space:
                ax.axvline(chosen["n_components"], color="black", ls=":", lw=1.3, label="selected k")
            ax.set_title(f"{space} ({dims}-D): {label}", fontsize=10, fontweight="bold", loc="left")
            ax.set_xlabel("k (number of components)")
            ax.set_xticks(sorted(sub["n_components"].unique()))
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[r, 0].legend(fontsize=8, frameon=True, facecolor="white", edgecolor="#CCCCCC")
    fig.suptitle("GMM search: BIC / AIC (complexity-penalised fit) next to Silhouette across k and covariance types\n"
                 "(each point uses the reg_covar with the lowest BIC; only valid fits: converged, all k used, smallest cluster >= 2%)",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(bic_choice, shortlist, chosen, comp, out_path):
    """k comparison + shortlist stability + untuned vs tuned."""
    dims = int(chosen["model_dims"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
    for ax, (col, label) in zip(axes[0], [("silhouette", "Silhouette (higher = better)"),
                                          ("davies_bouldin", "Davies-Bouldin (lower = better)"),
                                          ("calinski_harabasz", "Calinski-Harabasz (higher = better)")]):
        for space, (color, marker) in SPACE_STYLE.items():
            g = bic_choice[bic_choice["feature_space"] == space].sort_values("n_components")
            if not g.empty:
                ax.plot(g["n_components"], g[col], color=color, marker=marker, lw=1.9, ms=6,
                        label=f"{space} ({int(g['model_dims'].iloc[0])}-D)")
        ax.scatter([chosen["n_components"]], [chosen[col]], s=260, facecolors="none", edgecolors="black",
                   linewidths=2, zorder=5, label="selected")
        ax.set_title(label, fontsize=10.5, fontweight="bold", loc="left")
        ax.set_xlabel("k (covariance type per point chosen by BIC)")
        ax.set_xticks(sorted(bic_choice["n_components"].unique()))
    axes[0, 0].legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#CCCCCC")

    # cluster balance
    ax = axes[1, 0]
    g = bic_choice[bic_choice["feature_space"] == chosen["feature_space"]].sort_values("n_components")
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["min_cluster_share_pct"], 0.4, color="#2B9EB3", edgecolor="black", lw=0.6, label="smallest cluster %")
    ax.bar(x + 0.2, g["max_cluster_share_pct"], 0.4, color="#2F5DA8", edgecolor="black", lw=0.6, label="largest cluster %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={int(k)}\n{c}" for k, c in zip(g["n_components"], g["covariance_type"])], fontsize=7.5)
    ax.set_title(f"Cluster size balance ({dims}-D preprocessed, BIC-chosen covariance)", fontsize=10.5,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#CCCCCC")

    # shortlist: silhouette and stability
    ax = axes[1, 1]
    names = [f"{r.covariance_type}\nk={r.n_components} reg={r.reg_covar:g}" for r in shortlist.itertuples()]
    x = np.arange(len(shortlist))
    is_sel = [(r.feature_space, r.n_components, r.covariance_type, r.reg_covar) ==
              (chosen["feature_space"], chosen["n_components"], chosen["covariance_type"], chosen["reg_covar"])
              for r in shortlist.itertuples()]
    b1 = ax.bar(x - 0.2, shortlist["silhouette"], 0.4, color="#2F5DA8", edgecolor="black", lw=0.6, label="Silhouette")
    b2 = ax.bar(x + 0.2, shortlist["stability_ari_mean"], 0.4, color="#E08E2B", edgecolor="black", lw=0.6,
                label="Stability (mean ARI, 5 seeds)")
    ax.axhline(STABILITY_MIN_ARI, color="#E08E2B", ls="--", lw=1.1)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([("★ " if s else "") + n for n, s in zip(names, is_sel)], fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.set_title("Shortlist refitted with n_init=10 (★ = selected)", fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="upper right")

    # untuned vs tuned, raw metric values side by side
    ax = axes[1, 2]
    rows = comp.set_index("configuration")
    cfgs = [c for c in ["untuned_baseline", "tuned"] if c in rows.index]
    mets = [("silhouette", "Silhouette ↑"), ("davies_bouldin", "Davies-Bouldin ↓")]
    x = np.arange(len(mets))
    for i, cfg in enumerate(cfgs):
        vals = [rows.loc[cfg, m] for m, _ in mets]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=["#C8433B", "#2E8B57"][i], edgecolor="black", lw=0.6,
                      label=f"{cfg} (k={int(rows.loc[cfg, 'n_components'])}, {rows.loc[cfg, 'covariance_type']})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in mets])
    txt = "\n".join(f"{cfg}: CH = {rows.loc[cfg, 'calinski_harabasz']:,.0f}, free params = {int(rows.loc[cfg, 'n_free_parameters'])}"
                    for cfg in cfgs)
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CCCCCC"))
    ax.set_ylim(0, max(rows.loc[cfgs, "davies_bouldin"].max(), 1) * 1.45)
    ax.set_title(f"Untuned vs tuned (metrics in the same {dims}-D space)", fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="upper right")

    for a in axes.flat:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("GMM: comparison across k, stability check, and untuned vs tuned",
                 fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = base_arg_parser("Hyperparameter-tuned GMM.")
    ap.set_defaults(features=DEFAULT_FEATURES, raw=DEFAULT_RAW)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--reuse-search", action="store_true",
                    help="reuse outputs/gmm/clustering_gmm_tuning_search.csv instead of refitting the grid")
    args = ap.parse_args()
    art_dir, fig_dir, tab_dir = make_dirs(args.artifacts_dir, args.figures_dir, args.tables_dir)

    X_df, X, raw_df, prof_df = load_data(args.features, args.raw, args.profile_cols)
    sil_idx = silhouette_sample_idx(len(X))
    ks = list(range(args.k_min, args.k_max + 1))

    # Input is already preprocessed (frequency-encoded + PCA-90); no secondary PCA is applied
    Zs = {s: X for s in SPACES}
    for s, Z in Zs.items():
        print(f"feature space {s:18s}: {Z.shape[1]:2d} dims (from {args.features})")

    # step 1: coarse grid
    grid = [(s, k, c, r) for s in SPACES for c in COV_TYPES for r in REG_COVARS for k in ks]
    search_csv = tab_dir / "clustering_gmm_tuning_search.csv"
    if args.reuse_search and search_csv.exists():
        search = pd.read_csv(search_csv)
        print(f"\nStep 1: reusing {len(search)} fits from {search_csv}")
    else:
        print(f"\nStep 1: {len(grid)} fits (n_init=3, k-means++) on {args.n_jobs} jobs ...")
        t0 = time.perf_counter()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(_fit_one)(s, Zs[s], X, sil_idx, k, c, r, 3) for s, k, c, r in grid)
        search = pd.DataFrame(rows)
        search.to_csv(search_csv, index=False)
        print(f"step 1 done in {time.perf_counter() - t0:.0f}s")
    print(f"valid fits: {int(search['valid'].sum())}/{len(search)}")

    # step 2: BIC chooses covariance type and reg_covar for each (space, k)
    bic_choice = (search[search["valid"]].sort_values("bic_model_space")
                  .groupby(["feature_space", "n_components"], as_index=False).first())
    bic_choice.to_csv(tab_dir / "clustering_gmm_tuning_bic_choice.csv", index=False)

    # step 3: shortlist by silhouette, refit with n_init=10, check stability
    top = bic_choice.sort_values("silhouette", ascending=False).head(SHORTLIST)
    print(f"\nStep 3: refitting {len(top)} shortlisted models (n_init=10) + stability over {len(STABILITY_SEEDS)} seeds ...")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_refit_shortlisted)(r.feature_space, Zs[r.feature_space], X, sil_idx, int(r.n_components),
                                        r.covariance_type, float(r.reg_covar)) for r in top.itertuples())
    shortlist = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    stable = shortlist[shortlist["stability_ari_mean"] >= STABILITY_MIN_ARI]
    pool = stable if not stable.empty else shortlist.sort_values("stability_ari_mean", ascending=False).head(1)
    chosen = pool.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0].to_dict()
    shortlist["selected"] = [(r.feature_space, r.n_components, r.covariance_type, r.reg_covar) ==
                             (chosen["feature_space"], chosen["n_components"], chosen["covariance_type"], chosen["reg_covar"])
                             for r in shortlist.itertuples()]
    shortlist.to_csv(tab_dir / "clustering_gmm_tuning_shortlist.csv", index=False)
    print(shortlist[["feature_space", "n_components", "covariance_type", "reg_covar", "silhouette", "davies_bouldin",
                     "calinski_harabasz", "stability_ari_mean", "min_cluster_share_pct", "selected"]].to_string(index=False))

    # final refit
    space, k, cov, reg = chosen["feature_space"], int(chosen["n_components"]), chosen["covariance_type"], float(chosen["reg_covar"])
    print(f"\nSelected: space={space}, k={k}, covariance_type={cov}, reg_covar={reg:g}. Final refit with n_init=20 ...")
    pipe = Pipeline([("reduce", "passthrough"), ("gmm", make_gmm(k, cov, reg, n_init=20, max_iter=500))])
    t0 = time.perf_counter()
    pipe.fit(X)
    fit_time = time.perf_counter() - t0

    metrics, labels, probs = summarize_model(pipe, X, sil_idx, fit_time, "tuned")
    metrics["feature_space"] = space
    metrics["stability_ari_mean"] = chosen["stability_ari_mean"]
    gmm = pipe.named_steps["gmm"]
    Z_final = Zs[space]
    n_zero, data_eigs = data_zero_eigenvalues(Z_final)
    cov_diag = covariance_diagnostics(gmm)
    metrics["data_zero_eigenvalues"] = n_zero
    metrics["max_condition_number"] = float(cov_diag["condition_number"].max())
    metrics["min_eig_at_floor_per_component"] = int(cov_diag[FLOOR_COL].min())
    profiles = build_cluster_profiles(raw_df, prof_df, labels, probs)

    joblib.dump(pipe, art_dir / "gmm_tuned_model.joblib")
    (art_dir / "gmm_tuned_metadata.json").write_text(json.dumps({
        "selected": {"feature_space": space, "model_dims": int(Z_final.shape[1]), "n_components": k,
                     "covariance_type": cov, "reg_covar": reg, "init_params": "k-means++", "n_init": 20, "max_iter": 500},
        "search_space": {"feature_space": space, "input_file": str(args.features),
                         "n_components": ks, "covariance_type": COV_TYPES, "reg_covar": REG_COVARS},
        "selection_rule": ("valid = converged, all k used, smallest cluster >= 2%; per k lowest BIC picks "
                           f"covariance_type and reg_covar; top 6 by silhouette ({Z_final.shape[1]}-D) refitted with n_init=10; "
                           f"winner = highest silhouette among models with mean seed ARI >= {STABILITY_MIN_ARI}"),
        "feature_names": list(X_df.columns),
        "metrics": metrics}, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(tab_dir / "clustering_gmm_tuned_metrics.csv", index=False)
    profiles.to_csv(tab_dir / "clustering_gmm_tuned_profiles.csv", index=False)
    cov_diag.to_csv(tab_dir / "clustering_gmm_tuned_covariance.csv", index=False)

    untuned_csv = tab_dir / "clustering_gmm_untuned_metrics.csv"
    parts = [pd.read_csv(untuned_csv)] if untuned_csv.exists() else []
    comp = pd.concat(parts + [pd.DataFrame([metrics])], ignore_index=True)
    keep = ["configuration", "feature_space", "model_dims", "n_components", "covariance_type", "reg_covar", "n_init",
            "n_free_parameters", "silhouette", "davies_bouldin", "calinski_harabasz", "bic_model_space",
            "mean_max_posterior", "mean_normalised_entropy", "max_condition_number", "min_cluster_share_pct",
            "max_cluster_share_pct", "training_time_sec"]
    comp = comp[[c for c in keep if c in comp.columns]]
    comp.to_csv(tab_dir / "clustering_gmm_comparison.csv", index=False)

    tag = f"GMM tuned ({space} {Z_final.shape[1]}-D, k={k}, {cov}, reg_covar={reg:g})"
    plot_tuning_curves(search, chosen, fig_dir / "clustering_gmm_tuning_curves.png")
    plot_dashboard(bic_choice, shortlist, chosen, comp, fig_dir / "clustering_gmm_k_comparison.png")
    plot_pca_2d(X, labels, pipe, fig_dir / "clustering_gmm_tuned_pca_2d.png", tag)
    plot_cluster_profiles(profiles, fig_dir / "clustering_gmm_tuned_profiles.png", tag + ": cluster profiles")
    plot_cov_eigenvalues(gmm, data_eigs, fig_dir / "clustering_gmm_tuned_cov_eigenvalues.png",
                         f"{tag}: covariance eigenvalues (in the model's own space)")

    print("\n========== TUNED GMM ==========")
    for key, v in metrics.items():
        print(f"  {key:32s}: {v}")
    print("\n========== UNTUNED vs TUNED ==========")
    print(comp.to_string(index=False))
    print("\nCovariance diagnostics:")
    print(cov_diag.to_string(index=False))
    print("\nCluster profiles:")
    print(profiles[["Cluster", "Count", "Share_Pct", "LeadTime_Mean", "TotalNights_Mean", "NoWeekendNight_Pct",
                    "HasChildren_Pct", "Top_MarketSegment", "Top_CountryGroup", "PostHoc_ADR_Mean",
                    "PostHoc_CancelRate_Pct"]].to_string(index=False))


if __name__ == "__main__":
    main()