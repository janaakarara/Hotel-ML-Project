"""Encoding experiment for AGGLOMERATIVE and DBSCAN (one-hot, one-hot block, frequency, label, binary, FAMD, Gower).
Same baseline settings as clustering/*/baseline, only the encoding of the categorical columns changes.
Run from the project root:  python experiments/encoding_experiment/run.py
Outputs (experiments/encoding_experiment/outputs/): encoding_comparison.png, results.csv, winners.txt
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
SEED, K, SAMPLE = 42, 4, 10_000      # k=4 and the 10k sample are the same as your agglomerative baseline
DBSCAN_EPS = None                    # None = pick eps automatically per encoding; put 0.5 to use sklearn's default
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

CATS = ["MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType"]
raw = pd.read_csv(ROOT / "data/processed/h2_clustering_stage4.csv")      # numbers already scaled, categories still text
NUMS = [c for c in raw.columns if c not in CATS + ["ArrivalDateMonth"]]
S = np.sort(np.random.default_rng(SEED).choice(len(raw), SAMPLE, replace=False))


# ---- the encodings: only this part differs between runs ----------------------------------------
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
def binary():     # category number written in bits (6 levels -> 3 columns of 0/1)
    d = base()
    for c in CATS:
        codes = raw[c].astype("category").cat.codes.to_numpy() + 1
        for b in range(int(codes.max()).bit_length()): d[f"{c}_bit{b}"] = (codes >> b) & 1
    return d

def famd():       # FAMD = PCA on [scaled numbers + centred one-hot / sqrt(share)]; keep components up to 80% variance
    num = base().apply(z)
    oh = pd.get_dummies(raw[CATS], dtype=float)
    oh = (oh - oh.mean()) / np.sqrt(oh.mean())
    return pd.DataFrame(PCA(n_components=0.8, random_state=SEED).fit_transform(pd.concat([num, oh], axis=1).to_numpy()))

ENCODINGS = {"One-hot x1.0": lambda: onehot(1.0), "One-hot block x0.5": lambda: onehot(0.5), "Frequency": frequency,
             "Label": label, "Binary": binary, "FAMD": famd, "Gower*": None, "No categories (reference)": base}
             # Gower* is not an encoding of X: it clusters directly on the Gower distance matrix (see below)


# ---- your two algorithms, untuned -------------------------------------------------------------
def agglomerative(X):                         # ward, euclidean, k=4 on the 10k sample, rest by 5 nearest neighbours
    ls = AgglomerativeClustering(n_clusters=K).fit_predict(X[S])
    lab = KNeighborsClassifier(n_neighbors=5).fit(X[S], ls).predict(X)
    lab[S] = ls
    return lab, None

def dbscan(X):                                # min_samples=5; runs on the same 10k sample (all rows ran out of memory)
    eps = DBSCAN_EPS
    if eps is None:                           # same rule for every encoding: 90th percentile of distance to 5th neighbour
        eps = max(float(np.percentile(NearestNeighbors(n_neighbors=5).fit(X[S]).kneighbors(X[S])[0][:, -1], 90)), 1e-3)
    lab = np.full(len(X), -3)                 # -3 = not clustered, -1 = noise
    lab[S] = DBSCAN(eps=eps, min_samples=5, n_jobs=-1).fit_predict(X[S])
    return lab, round(eps, 3)

ALGOS = {"Agglomerative": agglomerative, "DBSCAN": dbscan}


# ---- one fair score for everybody: Gower distance on the 11 ORIGINAL columns (each column = 1 equal vote) ----
gidx = np.sort(np.random.default_rng(SEED).choice(S, 3000, replace=False))
g = raw.iloc[gidx]
G = np.zeros((len(gidx), len(gidx)), dtype=np.float32)
for c in NUMS:
    v = g[c].to_numpy(); G += np.abs(v[:, None] - v[None, :]) / (raw[c].max() - raw[c].min())
m = g["ArrivalDateMonth"].to_numpy(); dm = np.abs(m[:, None] - m[None, :]); G += np.minimum(dm, 12 - dm) / 6
for c in CATS:
    v = g[c].to_numpy(); G += (v[:, None] != v[None, :])
G /= len(NUMS) + 1 + len(CATS)

def fair_score(lab):
    l = lab[gidx]; keep = l >= 0
    return silhouette_score(G[np.ix_(keep, keep)], l[keep], metric="precomputed") if len(set(l[keep])) >= 2 else np.nan


# ---- Gower as a method: cluster directly on the distance matrix G (3000 rows; ward is impossible on a distance matrix -> average linkage)
def gower_run(algo):
    lab = np.full(len(raw), -3)
    if algo == "Agglomerative":
        try:    m = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average")
        except TypeError: m = AgglomerativeClustering(n_clusters=K, affinity="precomputed", linkage="average")
        lab[gidx] = m.fit_predict(G); return lab, None
    eps = DBSCAN_EPS if DBSCAN_EPS is not None else max(float(np.percentile(np.sort(G, axis=1)[:, 4], 90)), 1e-3)
    lab[gidx] = DBSCAN(eps=eps, min_samples=5, metric="precomputed").fit_predict(G)
    return lab, round(eps, 3)


# ---- run, pick winners, draw ------------------------------------------------------------------
rows = []
for enc, build in ENCODINGS.items():
    X = build().to_numpy(dtype=float) if build else np.zeros((1, len(NUMS) + 1 + len(CATS)))
    for algo, run in ALGOS.items():
        lab, eps = gower_run(algo) if build is None else run(X)
        used, keep = lab != -3, lab >= 0
        noise = 100 * (1 - keep.sum() / used.sum())
        rows.append(dict(algorithm=algo, encoding=enc, columns=X.shape[1], clusters=len(set(lab[keep])),
                         noise_pct=round(noise, 1), eps=eps, fair_score=round(fair_score(lab), 4),
                         can_win="reference" not in enc))
        print(f"{algo:14s} {enc:28s} clusters={rows[-1]['clusters']:3d} noise={noise:5.1f}%  score={rows[-1]['fair_score']}")
res = pd.DataFrame(rows); res.to_csv(OUT / "results.csv", index=False)

lines, winners = ["BEST ENCODING PER ALGORITHM (higher fair score = better)", ""], {}
for algo in ALGOS:
    r = res[(res.algorithm == algo) & res.can_win & (res.noise_pct <= 50)].dropna(subset=["fair_score"])
    winners[algo] = r.loc[r.fair_score.idxmax(), "encoding"]
    lines.append(f"{algo:14s} -> {winners[algo]}  (score {r.fair_score.max():.3f})")
    r2 = r[r.encoding != "Gower*"]
    lines.append(f"{'':14s}    best without Gower: {r2.loc[r2.fair_score.idxmax(), 'encoding']}  (score {r2.fair_score.max():.3f})")
lines += ["", "* Gower is clustered with the same ruler that scores it, and on only 3000 rows with average linkage,",
          "  so its score is biased upwards. Treat a Gower win with care.", "Use the same encoding for the tuned version."]
(OUT / "winners.txt").write_text("\n".join(lines), encoding="utf-8"); print("\n" + "\n".join(lines))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
names = list(ENCODINGS)
for ax, algo in zip(axes, ALGOS):
    ypos = np.arange(len(names))[::-1]
    for y, enc in zip(ypos, names):
        r = res[(res.algorithm == algo) & (res.encoding == enc)].iloc[0]
        color = "#2e9e5b" if enc == winners[algo] else ("#bbbbbb" if not r.can_win else "#4f81bd")
        ax.barh(y, r.fair_score, color=color)
        ax.text(max(r.fair_score, 0) + 0.004, y, f"{r.fair_score:.3f}  ({r.clusters} clusters, {r.noise_pct:.0f}% noise)", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(names)
    ax.set_title(f"{algo}  -  best: {winners[algo]}"); ax.set_xlabel("fair score (higher = better)")
    ax.set_xlim(min(-0.05, res.fair_score.min() * 1.1), res.fair_score.max() * 1.9)
fig.suptitle("Which encoding is best?  green = winner, blue = others, gray = reference without categories")
plt.tight_layout(); plt.savefig(OUT / "encoding_comparison.png", dpi=150); plt.close()
print("saved to", OUT)
