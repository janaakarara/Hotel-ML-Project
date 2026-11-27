"""Encoding x dimensionality-reduction x hyper-parameter grid for the 4 clustering models (+ 2 extra methods).

Why: the tuned GMM (one-hot, PCA 90% -> 7-D, k=2) reports silhouette 0.3231, the encoding experiment reports 0.20 for
Frequency. Those numbers are NOT on the same scale (different metric AND different k), so here every combination is
scored with the same two rulers:
  fair   = Gower silhouette on the 11 ORIGINAL columns (3k rows) -> independent of encoding / PCA (main criterion)
  sil29  = Euclidean silhouette in the 29-D space of h2_clustering_stage5.csv (one-hot x0.5, 10k rows)
           -> the metric the tuned GMM reported (0.3231); it favours whatever separates in that particular space

Grid (every encoding x every reduction):
  encodings   One-hot x1.0, One-hot x0.5, Frequency, Label, Binary, FAMD, No categories (reference)
  reductions  none, PCA80, PCA90, PCA95, PCA90-whiten
  K-Means     k = 2..8                         (k-means++, n_init=10)
  GMM         k = 2..8, covariance full / diag (reg_covar=1e-6, n_init=3)
  Agglomerative k = 2..8                       (ward, 10k sample, rest by 5-NN)
  DBSCAN      min_samples 5/20/50 x eps = 90/95/98th pct of the k-distance (10k sample)
  extra: HDBSCAN (min_cluster_size 100/250/500), K-Prototypes (k=2..8, mixed data, no encoding),
         Gower* (Agglomerative-average / DBSCAN directly on the Gower matrix; biased -> never a winner)

Selection per model: valid = >=2 clusters, smallest cluster >= 2%, noise <= 30%; not trivial = ARI < 0.90 with every
single categorical / binary input column (the tuned GMM k=2 is exactly HasChildren -> a group-by, not a segmentation).
Top 5 by fair score are re-run
with 5 other seeds (K-Means / GMM / K-Prototypes) or 5 other 10k samples (Agglomerative / DBSCAN / HDBSCAN);
winner = highest fair score with mean ARI >= 0.80. Done twice: any k, and k >= 3 (k=2 is favoured by silhouette).

Run from the project root:  python src/04_encoding_comparison/run_grid.py
Outputs in src/04_encoding_comparison/outputs/grid/
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import grid_common as gc

OUT = HERE / "outputs" / "grid"
OUT.mkdir(parents=True, exist_ok=True)
N_JOBS = 14
MAIN = ["K-Means", "GMM", "Agglomerative", "DBSCAN"]
EXTRA = ["HDBSCAN (extra)", "K-Prototypes (extra)"]
STABLE_ARI, SHORTLIST = 0.80, 5
ENC_ORDER = [e for e in gc.ENCODINGS if e != "No categories"] + ["No categories"]
RED_ORDER = list(gc.REDUCTIONS)


def describe(r):
    p = []
    if pd.notna(r.get("k")):                p.append(f"k={int(r['k'])}")
    if pd.notna(r.get("cov")):              p.append(f"cov={r['cov']}")
    if pd.notna(r.get("min_samples")):      p.append(f"min_samples={int(r['min_samples'])}, eps=p{int(r['eps_pct'])} ({r['eps']:.3f})")
    if pd.notna(r.get("min_cluster_size")): p.append(f"min_cluster_size={int(r['min_cluster_size'])}")
    return ", ".join(p)


def main():
    t0 = time.time()
    # ------------------------------------------------------------------ 1. grid
    spaces = [(e, r) for e in gc.ENCODINGS for r in gc.REDUCTIONS if e != "No categories" or r == "none"]
    print(f"{len(spaces)} feature spaces x (K-Means, GMM, Agglomerative, DBSCAN, HDBSCAN) + Gower* + K-Prototypes "
          f"on {N_JOBS} workers ...", flush=True)
    jobs = [delayed(gc.run_space)(e, r) for e, r in spaces] + [delayed(gc.run_gower_methods)()] + \
           [delayed(gc.run_kprototypes)(k) for k in gc.K_RANGE]
    parts = Parallel(n_jobs=N_JOBS, verbose=5)(jobs)
    res = pd.DataFrame([row for part in parts for row in part])
    for c in ["k", "cov", "min_samples", "eps_pct", "eps", "min_cluster_size"]:
        if c not in res: res[c] = np.nan
    res["config"] = res.apply(describe, axis=1)
    res["eligible"] = (res.encoding != "No categories") & (res.encoding != "Gower*")
    res["trivial"] = res.trivial.astype(bool)
    res["ok"] = res.valid & res.eligible & ~res.trivial      # candidates for "winner" (valid, real encoding, not a group-by)
    print(f"grid done: {len(res)} runs in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ 2. shortlist + stability
    cand = []
    for model in MAIN + EXTRA:
        pool = res[(res.model == model) & res.ok].dropna(subset=["fair"])
        cand.append(pool.nlargest(SHORTLIST, "fair"))
        if model not in ("DBSCAN", "HDBSCAN (extra)"):
            cand.append(pool[pool.k >= 3].nlargest(SHORTLIST, "fair"))
    cand = pd.concat(cand)
    cand = cand[~cand.index.duplicated()]
    print(f"stability check on {len(cand)} shortlisted configurations ...", flush=True)
    stab = Parallel(n_jobs=N_JOBS)(delayed(gc.stability)(r) for _, r in cand.iterrows())
    res["ari_mean"], res["ari_min"] = np.nan, np.nan
    res.loc[cand.index, "ari_mean"] = [s[0] for s in stab]
    res.loc[cand.index, "ari_min"] = [s[1] for s in stab]
    res.to_csv(OUT / "grid_results.csv", index=False)

    def pick(model, min_k=None):
        p = res[(res.model == model) & res.ok & (res.ari_mean >= STABLE_ARI)]
        if min_k is not None:
            p = p[p.k >= min_k]
        return p.nlargest(1, "fair").iloc[0] if len(p) else None

    winners = {m: pick(m) for m in MAIN + EXTRA}
    winners_k3 = {m: pick(m, 3) for m in MAIN + EXTRA if m not in ("DBSCAN", "HDBSCAN (extra)")}

    # reference points
    def find(model, enc, red, **kw):
        q = res[(res.model == model) & (res.encoding == enc) & (res.reduction == red)]
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.iloc[0] if len(q) else None

    baselines = {"K-Means": find("K-Means", "One-hot x1.0", "none", k=3),
                 "GMM": find("GMM", "One-hot x1.0", "none", k=3, cov="full"),
                 "Agglomerative": find("Agglomerative", "One-hot x1.0", "none", k=4),
                 "DBSCAN": find("DBSCAN", "One-hot x1.0", "none", min_samples=5, eps_pct=90)}
    v1 = {"K-Means": find("K-Means", "Frequency", "none", k=3),
          "GMM": find("GMM", "Frequency", "none", k=3, cov="full"),
          "Agglomerative": find("Agglomerative", "Frequency", "none", k=4),
          "DBSCAN": find("DBSCAN", "One-hot x1.0", "none", min_samples=5, eps_pct=90)}
    # the tuned GMM was fitted on the current h2_clustering_stage5.csv = one-hot x0.5 (cat_weight 0.5) -> PCA90 = 7-D
    tuned_gmm = find("GMM", "One-hot x0.5", "PCA90", k=2, cov="full")

    # ------------------------------------------------------------------ 3. text report
    L = ["ENCODING x PCA x HYPER-PARAMETER GRID - BEST CONFIGURATION PER MODEL",
         "fair  = Gower silhouette on the original columns (main criterion, comparable across encodings/PCA)",
         "sil29 = Euclidean silhouette in the 29-D one-hot space (the metric of the tuned GMM, 0.3231)",
         f"valid = >=2 clusters, smallest cluster >= {gc.MIN_SHARE_PCT}%, noise <= {gc.MAX_NOISE_PCT}%; "
         f"stable = mean ARI >= {STABLE_ARI} over 5 seeds / samples", ""]
    if tuned_gmm is not None:
        L += ["Tuned GMM reproduced (One-hot x0.5 + PCA90, full, k=2): "
              f"sil29 {tuned_gmm.sil29:.4f} | fair {tuned_gmm.fair:.4f} | split {tuned_gmm.max_share_pct:.1f}/"
              f"{tuned_gmm.min_share_pct:.1f}% | = {tuned_gmm.dominant_col} (ARI {tuned_gmm.dominant_ari:.2f})"
              + ("  -> TRIVIAL split (one column)" if tuned_gmm.trivial else ""), ""]
    L += [f"trivial = partition has ARI >= {gc.TRIVIAL_ARI} with a single input column (a group-by, not a segmentation);",
          "trivial partitions are excluded from the winners but the best one per model is listed.", ""]

    def line(tag, r):
        if r is None:
            return f"  {tag:22s} none valid+stable"
        ari = f"ARI {r.ari_mean:.2f}" if pd.notna(r.ari_mean) else "ARI n/a"
        return (f"  {tag:22s} {r.encoding} + {r.reduction} ({int(r.dims)}-D), {r.config}  ->  fair {r.fair:.3f} "
                f"(+/-{r.fair_robust_std:.3f}) | sil29 {r.sil29:.3f} | {int(r.clusters)} clusters, smallest "
                f"{r.min_share_pct:.1f}%, noise {r.noise_pct:.0f}% | {ari} | closest column {r.dominant_col} "
                f"(ARI {r.dominant_ari:.2f}){' TRIVIAL' if r.trivial else ''}")

    for m in MAIN + EXTRA:
        L.append(m)
        if m in baselines:
            L.append(line("baseline (one-hot):", baselines[m]))
            L.append(line("encoding exp. v1:", v1[m]))
        L.append(line("GRID WINNER (any k):", winners[m]))
        if m in winners_k3:
            L.append(line("GRID WINNER (k>=3):", winners_k3[m]))
        tq = res[(res.model == m) & res.valid & res.eligible & res.trivial].dropna(subset=["fair"])
        if len(tq):
            t = tq.nlargest(1, "fair").iloc[0]
            L.append(f"  {'best TRIVIAL split:':22s} {t.encoding} + {t.reduction}, {t.config} -> fair {t.fair:.3f} | "
                     f"sil29 {t.sil29:.3f} | = {t.dominant_col} (ARI {t.dominant_ari:.2f})  [excluded]")
        sub = res[(res.model == m) & res.ok].dropna(subset=["fair"])
        if len(sub):
            best_red = sub.groupby("reduction").fair.max().sort_values(ascending=False)
            best_enc = sub.groupby("encoding").fair.max().sort_values(ascending=False)
            L.append("  best per encoding:  " + ", ".join(f"{e} {v:.3f}" for e, v in best_enc.items()))
            L.append("  best per reduction: " + ", ".join(f"{e} {v:.3f}" for e, v in best_red.items()))
        L.append("")
    g = res[res.encoding == "Gower*"].dropna(subset=["fair"])
    if len(g):
        b = g.nlargest(1, "fair").iloc[0]
        L += [f"Gower* (biased, not eligible): best {b.model} {b.config} fair {b.fair:.3f}", ""]
    L += [f"runtime {time.time() - t0:.0f}s, {len(res)} runs. Full table: grid_results.csv"]
    (OUT / "grid_winners.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L), flush=True)

    # ------------------------------------------------------------------ 4. figures
    # 4a. heatmaps: best valid fair score per encoding x reduction (max over hyper-parameters)
    models = MAIN + ["HDBSCAN (extra)"]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.6))
    for ax, m in zip(axes, models):
        sub = res[(res.model == m) & res.valid & ~res.trivial.astype(bool)].dropna(subset=["fair"])
        M = np.full((len(ENC_ORDER), len(RED_ORDER)), np.nan)
        ann = [["-"] * len(RED_ORDER) for _ in ENC_ORDER]
        for i, e in enumerate(ENC_ORDER):
            for j, r in enumerate(RED_ORDER):
                q = sub[(sub.encoding == e) & (sub.reduction == r)]
                if len(q):
                    b = q.loc[q.fair.idxmax()]
                    M[i, j] = b.fair
                    tag = f"k={int(b.k)}" if pd.notna(b.k) else f"{int(b.clusters)}cl"
                    ann[i][j] = f"{b.fair:.3f}\n{tag}"
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=np.nanmax(res[res.valid].fair), aspect="auto")
        for i in range(len(ENC_ORDER)):
            for j in range(len(RED_ORDER)):
                w = winners.get(m)
                bold = w is not None and w.encoding == ENC_ORDER[i] and w.reduction == RED_ORDER[j]
                ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=7.5,
                        color="red" if bold else ("white" if np.isnan(M[i, j]) or M[i, j] < 0.15 else "black"),
                        fontweight="bold" if bold else "normal")
        ax.set_xticks(range(len(RED_ORDER))); ax.set_xticklabels(RED_ORDER, rotation=30, fontsize=8)
        ax.set_yticks(range(len(ENC_ORDER))); ax.set_yticklabels(ENC_ORDER, fontsize=8)
        ax.set_title(m, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.8, label="fair score (best valid config)")
    fig.suptitle("Best fair score per encoding x reduction (max over k / params; red = stable winner)", fontsize=12)
    plt.savefig(OUT / "grid_heatmaps.png", dpi=140, bbox_inches="tight"); plt.close()

    # 4b. fair and sil29 vs k for the winning space of each k-based model
    kmodels = ["K-Means", "GMM", "Agglomerative", "K-Prototypes (extra)"]
    fig, axes = plt.subplots(1, len(kmodels), figsize=(5 * len(kmodels), 4.2))
    for ax, m in zip(axes, kmodels):
        w = winners.get(m)
        if w is None:
            continue
        q = res[(res.model == m) & (res.encoding == w.encoding) & (res.reduction == w.reduction)]
        if m == "GMM":
            q = q[q["cov"] == w["cov"]]
        q = q.sort_values("k")
        ax.plot(q.k, q.fair, "o-", label="fair (Gower)")
        ax.plot(q.k, q.sil29, "s--", label="sil29 (Euclid 29-D)")
        for _, r in q[~q.valid].iterrows():
            ax.plot(r.k, r.fair, "x", color="red", ms=10)
        for _, r in q[q.trivial].iterrows():
            ax.plot(r.k, r.fair, "o", mfc="none", mec="purple", ms=13)
        ax.set_title(f"{m}\n{w.encoding} + {w.reduction}" + (f", {w['cov']}" if m == "GMM" else ""), fontsize=9.5)
        ax.set_xlabel("k"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Score vs k in each model's winning space (red x = invalid: a cluster < 2%; purple ring = trivial split)")
    plt.tight_layout(); plt.savefig(OUT / "grid_k_curves.png", dpi=140); plt.close()

    # 4c. final comparison: baseline -> encoding exp v1 -> grid winner (k>=3) -> grid winner (any k)
    stages = [("baseline one-hot", baselines), ("encoding exp. v1", v1), ("grid best k>=3", winners_k3),
              ("grid best (any k)", winners)]
    colors = ["#bbbbbb", "#4f81bd", "#f0a04b", "#2e9e5b"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, metric, title in zip(axes, ["fair", "sil29"], ["fair score (Gower, main)", "sil29 (Euclid 29-D, team metric)"]):
        x = np.arange(len(MAIN + EXTRA))
        for i, (name, d) in enumerate(stages):
            vals = [d.get(m)[metric] if d.get(m) is not None else np.nan for m in MAIN + EXTRA]
            bars = ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, color=colors[i], label=name, edgecolor="black", lw=0.4)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.005, f"{v:.2f}", ha="center", fontsize=6.5,
                            rotation=90)
        if tuned_gmm is not None:
            ax.axhline(tuned_gmm[metric], color="red", ls=":", lw=1,
                       label=f"tuned GMM k=2 {tuned_gmm[metric]:.3f} (= {tuned_gmm.dominant_col} split)")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([m.replace(" (extra)", "\n(extra)") for m in MAIN + EXTRA], fontsize=9)
        ax.set_title(title); ax.legend(fontsize=7.5)
    fig.suptitle("Final comparison per model (higher = better)")
    plt.tight_layout(); plt.savefig(OUT / "grid_final_comparison.png", dpi=140); plt.close()

    # compact final table
    rows = []
    for m in MAIN + EXTRA:
        for tag, d in stages:
            r = d.get(m)
            if r is not None:
                rows.append(dict(model=m, stage=tag, encoding=r.encoding, reduction=r.reduction, dims=r.dims,
                                 config=r.config, clusters=r.clusters, min_share_pct=r.min_share_pct,
                                 noise_pct=r.noise_pct, fair=r.fair, fair_robust_std=r.fair_robust_std,
                                 sil29=r.sil29, ari_mean=r.ari_mean))
    pd.DataFrame(rows).to_csv(OUT / "grid_final_table.csv", index=False)
    print(f"saved to {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
