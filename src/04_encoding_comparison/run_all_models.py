"""Encoding experiment for ALL FOUR clustering models: K-Means, GMM, Agglomerative, DBSCAN.

Extends the first two-model version of this experiment (removed) (which only covered Agglomerative + DBSCAN) with the K-Means and
GMM baselines. Encodings and the "fair score" are identical to run.py, so the Agglomerative / DBSCAN numbers match.

Each model keeps the settings of its own untuned baseline in clustering/*/baseline/run.py; only the encoding of
the categorical columns changes:
  K-Means        k=3, k-means++, n_init=10, max_iter=300, all rows            (clustering/kmeans/baseline)
  GMM            k=3, covariance=full, reg_covar=1e-6, n_init=1, init=kmeans,
                 max_iter=100, all rows                                       (clustering/gmm/baseline)
  Agglomerative  k=4, ward, euclidean on a 10k sample, rest by 5-NN           (clustering/agglomerative/baseline)
  DBSCAN         min_samples=5 on the 10k sample, eps = 90th pct of 5-NN dist (same auto-eps rule as run.py)

Fair score = silhouette computed with the Gower distance on the 11 ORIGINAL columns (3,000-row sample), so it
does not depend on the encoding being tested -> scores are comparable across encodings (higher = better).
Robustness: the fair score is also recomputed on 5 other independent 3,000-row Gower samples (mean +/- std).

Gower* (clustering directly on the Gower matrix) only exists for Agglomerative / DBSCAN: K-Means and GMM need
feature vectors, so they get N/A there.

Run from the project root:  python src/04_encoding_comparison/run_all_models.py
Outputs (src/04_encoding_comparison/outputs/):
  all_models_results.csv, all_models_winners.txt, all_models_encoding_comparison.png
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
SEED, SAMPLE, GOWER_N = 42, 10_000, 3_000
K_AGGLO, K_KMEANS, K_GMM = 4, 3, 3           # each model keeps the k of its own baseline
ROBUST_SEEDS = [1, 2, 3, 4, 5]               # extra Gower samples for the robustness check
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

CATS = ["MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType"]
raw = pd.read_csv(ROOT / "data/processed/h2_clustering_stage4.csv")      # numbers already scaled, categories still text
NUMS = [c for c in raw.columns if c not in CATS + ["ArrivalDateMonth"]]
S = np.sort(np.random.default_rng(SEED).choice(len(raw), SAMPLE, replace=False))


# ---- the encodings (identical to run.py) ------------------------------------------------------
def base():                                   # numbers + month as sin/cos (same in every encoding)
    d = raw[NUMS].copy()
    a = 2 * np.pi * (raw["ArrivalDateMonth"] - 1) / 12
    d["Month_sin"], d["Month_cos"] = np.sin(a), np.cos(a)
    return d

z = lambda s: (s - s.mean()) / s.std()

def onehot(w):    return pd.concat([base(), pd.get_dummies(raw[CATS], dtype=float) * w], axis=1)
def frequency():  # category -> how common it is, then z-scored (1 column per category column)
    d = base()
    for c in CATS: d[c] = z(raw[c].map(raw[c].value_counts(normalize=True)))
    return d
def label():      # category -> 0,1,2.. then z-scored (order is arbitrary)
    d = base()
    for c in CATS: d[c] = z(raw[c].astype("category").cat.codes.astype(float))
    return d
def binary():     # category number written in bits
    d = base()
    for c in CATS:
        codes = raw[c].astype("category").cat.codes.to_numpy() + 1
        for b in range(int(codes.max()).bit_length()): d[f"{c}_bit{b}"] = (codes >> b) & 1
    return d
def famd():       # PCA on [scaled numbers + centred one-hot / sqrt(share)], components up to 80% variance
    num = base().apply(z)
    oh = pd.get_dummies(raw[CATS], dtype=float)
    oh = (oh - oh.mean()) / np.sqrt(oh.mean())
    return pd.DataFrame(PCA(n_components=0.8, random_state=SEED).fit_transform(pd.concat([num, oh], axis=1).to_numpy()))

ENCODINGS = {"One-hot x1.0": lambda: onehot(1.0), "One-hot block x0.5": lambda: onehot(0.5), "Frequency": frequency,
             "Label": label, "Binary": binary, "FAMD": famd, "Gower*": None, "No categories (reference)": base}


# ---- the four models, untuned ----------------------------------------------------------------
def kmeans(X):
    km = KMeans(n_clusters=K_KMEANS, init="k-means++", n_init=10, max_iter=300, random_state=SEED).fit(X)
    return km.labels_, None

def gmm(X):
    g = GaussianMixture(n_components=K_GMM, covariance_type="full", reg_covar=1e-6, n_init=1,
                        init_params="kmeans", max_iter=100, random_state=SEED).fit(X)
    return g.predict(X), None

def agglomerative(X):
    ls = AgglomerativeClustering(n_clusters=K_AGGLO).fit_predict(X[S])
    lab = KNeighborsClassifier(n_neighbors=5).fit(X[S], ls).predict(X)
    lab[S] = ls
    return lab, None

def dbscan(X):
    eps = max(float(np.percentile(NearestNeighbors(n_neighbors=5).fit(X[S]).kneighbors(X[S])[0][:, -1], 90)), 1e-3)
    lab = np.full(len(X), -3)                 # -3 = not clustered, -1 = noise
    lab[S] = DBSCAN(eps=eps, min_samples=5, n_jobs=-1).fit_predict(X[S])
    return lab, round(eps, 3)

ALGOS = {"K-Means": kmeans, "GMM": gmm, "Agglomerative": agglomerative, "DBSCAN": dbscan}


# ---- fair score: Gower silhouette on the 11 original columns ---------------------------------
def gower_matrix(idx):
    g = raw.iloc[idx]
    G = np.zeros((len(idx), len(idx)), dtype=np.float32)
    for c in NUMS:
        v = g[c].to_numpy(); G += np.abs(v[:, None] - v[None, :]) / (raw[c].max() - raw[c].min())
    m = g["ArrivalDateMonth"].to_numpy(); dm = np.abs(m[:, None] - m[None, :]); G += np.minimum(dm, 12 - dm) / 6
    for c in CATS:
        v = g[c].to_numpy(); G += (v[:, None] != v[None, :])
    return G / (len(NUMS) + 1 + len(CATS))

def gower_sil(lab, idx, G):
    l = lab[idx]; keep = l >= 0
    return silhouette_score(G[np.ix_(keep, keep)], l[keep], metric="precomputed") if len(set(l[keep])) >= 2 else np.nan

gidx = np.sort(np.random.default_rng(SEED).choice(S, GOWER_N, replace=False))   # same sample as run.py
G = gower_matrix(gidx)
# extra samples drawn from the 10k sample S, so DBSCAN / Agglomerative (sample-only labels) can be scored too
robust = [(ix, gower_matrix(ix)) for ix in
          (np.sort(np.random.default_rng(s).choice(S, GOWER_N, replace=False)) for s in ROBUST_SEEDS)]


def gower_run(algo):                          # Gower as a method (distance matrix, average linkage)
    lab = np.full(len(raw), -3)
    if algo == "Agglomerative":
        lab[gidx] = AgglomerativeClustering(n_clusters=K_AGGLO, metric="precomputed", linkage="average").fit_predict(G)
        return lab, None
    if algo == "DBSCAN":
        eps = max(float(np.percentile(np.sort(G, axis=1)[:, 4], 90)), 1e-3)
        lab[gidx] = DBSCAN(eps=eps, min_samples=5, metric="precomputed").fit_predict(G)
        return lab, round(eps, 3)
    return None, None                         # K-Means / GMM: not applicable


# ---- run --------------------------------------------------------------------------------------
rows = []
for enc, build in ENCODINGS.items():
    X = build().to_numpy(dtype=float) if build else None
    for algo, run in ALGOS.items():
        t0 = time.time()
        lab, eps = gower_run(algo) if build is None else run(X)
        rec = dict(algorithm=algo, encoding=enc, columns=X.shape[1] if X is not None else len(NUMS) + 1 + len(CATS),
                   can_win="reference" not in enc)
        if lab is None:
            rec.update(clusters=np.nan, noise_pct=np.nan, eps=None, fair_score=np.nan,
                       robust_mean=np.nan, robust_std=np.nan, min_cluster_pct=np.nan, seconds=0.0, note="N/A")
        else:
            used, keep = lab != -3, lab >= 0
            sizes = np.bincount(lab[keep])
            sizes = sizes[sizes > 0]
            rs = [gower_sil(lab, ix, Gm) if build is not None else np.nan for ix, Gm in robust]
            rec.update(clusters=len(sizes), noise_pct=round(100 * (1 - keep.sum() / used.sum()), 1), eps=eps,
                       fair_score=round(gower_sil(lab, gidx, G), 4),
                       robust_mean=round(np.nanmean(rs), 4) if not np.all(np.isnan(rs)) else np.nan,
                       robust_std=round(np.nanstd(rs), 4) if not np.all(np.isnan(rs)) else np.nan,
                       min_cluster_pct=round(100 * sizes.min() / sizes.sum(), 1),
                       seconds=round(time.time() - t0, 1), note="")
        rows.append(rec)
        print(f"{algo:14s} {enc:28s} clusters={rec['clusters']!s:>4} noise={rec['noise_pct']!s:>5}%  "
              f"score={rec['fair_score']}  robust={rec['robust_mean']}+/-{rec['robust_std']}  ({rec['seconds']}s)")
res = pd.DataFrame(rows)
res.to_csv(OUT / "all_models_results.csv", index=False)


# ---- winners ----------------------------------------------------------------------------------
lines, winners = ["BEST ENCODING PER MODEL (fair score = Gower silhouette on the original columns, higher = better)", ""], {}
for algo in ALGOS:
    r = res[(res.algorithm == algo) & res.can_win & (res.noise_pct <= 50)].dropna(subset=["fair_score"])
    r = r.sort_values("fair_score", ascending=False)
    winners[algo] = r.iloc[0].encoding
    ref = res[(res.algorithm == algo) & ~res.can_win].iloc[0]
    lines.append(f"{algo}  ->  {winners[algo]}  (score {r.iloc[0].fair_score:.3f}, "
                 f"robust {r.iloc[0].robust_mean:.3f} +/- {r.iloc[0].robust_std:.3f})")
    r2 = r[r.encoding != "Gower*"]
    if winners[algo] == "Gower*":
        lines.append(f"    best without Gower: {r2.iloc[0].encoding}  (score {r2.iloc[0].fair_score:.3f})")
    lines.append(f"    runner-up: {r2.iloc[1].encoding if r2.iloc[0].encoding == winners[algo] else r2.iloc[0].encoding}"
                 f"  |  no-categories reference: {ref.fair_score:.3f}")
    lines.append("    ranking: " + ", ".join(f"{e} {s:.3f}" for e, s in zip(r.encoding, r.fair_score)))
    lines.append("")
lines += ["* Gower is clustered with the same ruler that scores it (3,000 rows, average linkage), so its score is",
          "  biased upwards; it does not apply to K-Means / GMM (they need feature vectors).",
          "Robust = mean +/- std of the fair score over 5 other independent 3,000-row Gower samples.",
          "Use the same encoding for the tuned version of each model."]
(OUT / "all_models_winners.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n" + "\n".join(lines))


# ---- plot -------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
names = list(ENCODINGS)
lo = min(-0.05, np.nanmin(res.fair_score) * 1.1); hi = np.nanmax(res.fair_score) * 2.0
for ax, algo in zip(axes, ALGOS):
    ypos = np.arange(len(names))[::-1]
    for y, enc in zip(ypos, names):
        r = res[(res.algorithm == algo) & (res.encoding == enc)].iloc[0]
        if pd.isna(r.fair_score):
            ax.text(0.004, y, "N/A" if r.note == "N/A" else "undefined (<2 clusters)", va="center", fontsize=8, color="gray")
            continue
        color = "#2e9e5b" if enc == winners[algo] else ("#bbbbbb" if not r.can_win else "#4f81bd")
        ax.barh(y, r.fair_score, color=color,
                xerr=r.robust_std if not pd.isna(r.robust_std) else None, ecolor="black", capsize=2)
        ax.text(max(r.fair_score, 0) + 0.006, y, f"{r.fair_score:.3f}  ({int(r.clusters)} cl, {r.noise_pct:.0f}% noise)",
                va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(names)
    ax.set_title(f"{algo}  -  best: {winners[algo]}"); ax.set_xlabel("fair score (higher = better)")
    ax.set_xlim(lo, hi)
fig.suptitle("Which encoding is best for each model?  green = winner, blue = others, gray = reference without categories"
             "  (error bar = std over 5 other Gower samples)")
plt.tight_layout(); plt.savefig(OUT / "all_models_encoding_comparison.png", dpi=150); plt.close()
print("saved to", OUT)
