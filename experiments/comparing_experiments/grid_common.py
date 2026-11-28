"""Shared code for run_grid.py: encodings, dimensionality reduction, the clustering models and the scores.

Kept in its own module so the parallel workers (joblib / loky) can import it.
Encodings and the Gower "fair score" are identical to run.py / run_all_models.py.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # 1 thread per worker, parallelism is per space
    os.environ.setdefault(_v, "1")

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
SEED, SAMPLE, GOWER_N = 42, 10_000, 3_000
K_RANGE = list(range(2, 9))
GMM_COVS = ["full", "diag"]
DBSCAN_GRID = [(ms, pct) for ms in (5, 20, 50) for pct in (90, 95, 98)]   # (min_samples, eps = pct-th percentile of k-dist)
HDBSCAN_GRID = [100, 250, 500]                                             # min_cluster_size on the 10k sample
MIN_SHARE_PCT, MAX_NOISE_PCT = 2.0, 30.0                                   # validity rule (2% = same as the tuned GMM)

ROOT = Path(__file__).resolve().parents[2]
CATS = ["MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType"]
raw = pd.read_csv(ROOT / "data/processed/h2_clustering_stage4.csv")      # numbers already scaled, categories still text
NUMS = [c for c in raw.columns if c not in CATS + ["ArrivalDateMonth"]]
N = len(raw)
S = np.sort(np.random.default_rng(SEED).choice(N, SAMPLE, replace=False))  # 10k sample for Agglomerative / DBSCAN


# ---- encodings (identical to run.py, FAMD is left un-truncated so the PCA levels below apply to it) ----------------
def base():
    d = raw[NUMS].copy()
    a = 2 * np.pi * (raw["ArrivalDateMonth"] - 1) / 12
    d["Month_sin"], d["Month_cos"] = np.sin(a), np.cos(a)
    return d

z = lambda s: (s - s.mean()) / s.std()

def onehot(w):    return pd.concat([base(), pd.get_dummies(raw[CATS], dtype=float) * w], axis=1)
def frequency():
    d = base()
    for c in CATS: d[c] = z(raw[c].map(raw[c].value_counts(normalize=True)))
    return d
def label():
    d = base()
    for c in CATS: d[c] = z(raw[c].astype("category").cat.codes.astype(float))
    return d
def binary():
    d = base()
    for c in CATS:
        codes = raw[c].astype("category").cat.codes.to_numpy() + 1
        for b in range(int(codes.max()).bit_length()): d[f"{c}_bit{b}"] = (codes >> b) & 1
    return d
def famd():       # FAMD input: scaled numbers + centred one-hot / sqrt(share); "PCA80" on this = the FAMD of run.py
    num = base().apply(z)
    oh = pd.get_dummies(raw[CATS], dtype=float)
    return pd.concat([num, (oh - oh.mean()) / np.sqrt(oh.mean())], axis=1)

ENCODINGS = {"One-hot x1.0": lambda: onehot(1.0), "One-hot x0.5": lambda: onehot(0.5), "Frequency": frequency,
             "Label": label, "Binary": binary, "FAMD": famd, "No categories": base}
REDUCTIONS = {"none": None, "PCA80": dict(n_components=0.80), "PCA90": dict(n_components=0.90),
              "PCA95": dict(n_components=0.95), "PCA90-whiten": dict(n_components=0.90, whiten=True)}


def build_space(enc, red):
    X = ENCODINGS[enc]().to_numpy(dtype=float)
    cfg = REDUCTIONS[red]
    if cfg is not None:
        X = PCA(**cfg, svd_solver="full", random_state=SEED).fit_transform(X)
    return X


# ---- scores ------------------------------------------------------------------------------------------------------
def gower_matrix(idx):
    g = raw.iloc[idx]
    G = np.zeros((len(idx), len(idx)), dtype=np.float32)
    for c in NUMS:
        v = g[c].to_numpy(); G += np.abs(v[:, None] - v[None, :]) / (raw[c].max() - raw[c].min())
    m = g["ArrivalDateMonth"].to_numpy(); dm = np.abs(m[:, None] - m[None, :]); G += np.minimum(dm, 12 - dm) / 6
    for c in CATS:
        v = g[c].to_numpy(); G += (v[:, None] != v[None, :])
    return G / (len(NUMS) + 1 + len(CATS))

GIDX = np.sort(np.random.default_rng(SEED).choice(S, GOWER_N, replace=False))          # same as run.py
ROBUST_IDX = [np.sort(np.random.default_rng(s).choice(S, GOWER_N, replace=False)) for s in (1, 2, 3, 4, 5)]
_G = {}

def gower_mats():
    if not _G:
        _G["main"] = gower_matrix(GIDX)
        _G["robust"] = [gower_matrix(ix) for ix in ROBUST_IDX]
    return _G["main"], _G["robust"]

def _gsil(lab, idx, G):
    l = lab[idx]
    if (l == -3).mean() > 0.5:                                    # not labelled on this sample (Gower* runs)
        return np.nan
    keep = l >= 0
    return silhouette_score(G[np.ix_(keep, keep)], l[keep], metric="precomputed") if len(set(l[keep])) >= 2 else np.nan

# the team's metric: Euclidean silhouette in the 29-D stage-5 space (currently one-hot x0.5 = cat_weight 0.5),
# on the fixed 10k rows of gmm/baseline/run.py
X29 = pd.read_csv(ROOT / "data/processed/h2_clustering_stage5.csv").to_numpy(dtype=float)
SIL_IDX = np.sort(np.random.RandomState(SEED).choice(N, min(10_000, N), replace=False))


# trivial-split check: a partition that (almost) equals ONE input column is a group-by, not a segmentation
# (e.g. the tuned GMM k=2 = HasChildren exactly). ARI against each categorical / binary column.
SINGLE_COLS = {c: raw[c].astype("category").cat.codes.to_numpy() for c in CATS + ["HasChildren"]}
TRIVIAL_ARI = 0.90


def dominant_column(lab):
    m = lab >= 0
    aris = {c: adjusted_rand_score(v[m], lab[m]) for c, v in SINGLE_COLS.items()}
    c = max(aris, key=aris.get)
    return c, round(float(aris[c]), 3)


def score(lab):
    used, keep = lab != -3, lab >= 0
    dom_col, dom_ari = dominant_column(lab) if keep.any() else (None, np.nan)
    sizes = np.bincount(lab[keep]) if keep.any() else np.array([])
    sizes = sizes[sizes > 0]
    G0, robust = gower_mats()
    rs = np.array([_gsil(lab, ix, Gm) for ix, Gm in zip(ROBUST_IDX, robust)], dtype=float)
    idx = SIL_IDX if used.all() else np.flatnonzero(used)        # sample-only methods: their own labelled rows
    l = lab[idx]; k = l >= 0
    sil29 = silhouette_score(X29[idx][k], l[k]) if len(set(l[k])) >= 2 else np.nan
    noise = 100 * (1 - keep.sum() / used.sum())
    min_share = 100 * sizes.min() / sizes.sum() if len(sizes) else np.nan
    return dict(clusters=len(sizes), noise_pct=round(noise, 1), min_share_pct=round(min_share, 2),
                max_share_pct=round(100 * sizes.max() / sizes.sum(), 2) if len(sizes) else np.nan,
                fair=round(_gsil(lab, GIDX, G0), 4),
                fair_robust_mean=round(np.nanmean(rs), 4) if np.isfinite(rs).any() else np.nan,
                fair_robust_std=round(np.nanstd(rs), 4) if np.isfinite(rs).any() else np.nan,
                sil29=round(sil29, 4) if np.isfinite(sil29) else np.nan,
                dominant_col=dom_col, dominant_ari=dom_ari, trivial=bool(dom_ari >= TRIVIAL_ARI),
                valid=bool(len(sizes) >= 2 and noise <= MAX_NOISE_PCT and min_share >= MIN_SHARE_PCT))


# ---- models ------------------------------------------------------------------------------------------------------
def kmeans_labels(X, k, seed=SEED):
    return KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=seed).fit_predict(X)

def gmm_labels(X, k, cov, seed=SEED):
    return GaussianMixture(n_components=k, covariance_type=cov, reg_covar=1e-6, n_init=3, init_params="kmeans",
                           max_iter=200, random_state=seed).fit(X).predict(X)

def agglo_labels(X, ks, sample=S):
    """Ward on the sample (scipy ward = sklearn ward), cut at every k, other rows by majority of 5 nearest sampled rows."""
    Xs = X[sample]
    Z = linkage(Xs, method="ward")
    nn = NearestNeighbors(n_neighbors=5).fit(Xs).kneighbors(X, return_distance=False)
    out = {}
    for k in ks:
        ls = fcluster(Z, k, criterion="maxclust") - 1
        votes = ls[nn]
        counts = np.zeros((len(X), ls.max() + 1))
        for j in range(votes.shape[1]):
            np.add.at(counts, (np.arange(len(X)), votes[:, j]), 1)
        lab = counts.argmax(1)
        lab[sample] = ls
        out[k] = lab
    return out

def dbscan_labels(X, ms, pct, sample=S, kdist=None):
    Xs = X[sample]
    if kdist is None:
        kdist = NearestNeighbors(n_neighbors=ms).fit(Xs).kneighbors(Xs)[0]
    eps = max(float(np.percentile(kdist[:, ms - 1], pct)), 1e-3)
    lab = np.full(len(X), -3)                                     # -3 = not clustered, -1 = noise
    lab[sample] = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xs)
    return lab, eps

def hdbscan_labels(X, mcs, sample=S):
    lab = np.full(len(X), -3)
    lab[sample] = HDBSCAN(min_cluster_size=mcs).fit_predict(X[sample])
    return lab

def kprototypes_labels(k, seed=SEED, n_init=3, max_iter=50):
    """K-Prototypes (Huang 1997): squared Euclidean on numbers + gamma * mismatches on categories. No encoding needed."""
    Xn = base().to_numpy(dtype=float)
    Xc = np.column_stack([raw[c].astype("category").cat.codes.to_numpy() for c in CATS])
    gamma = 0.5 * Xn.std(0).mean()
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(n_init):
        idx = rng.choice(N, k, replace=False)
        C, M = Xn[idx].copy(), Xc[idx].copy()
        lab = None
        for _ in range(max_iter):
            cost = ((Xn[:, None, :] - C[None]) ** 2).sum(-1) + gamma * (Xc[:, None, :] != M[None]).sum(-1)
            new = cost.argmin(1)
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            for j in range(k):
                m = lab == j
                if not m.any():
                    r = rng.integers(N); C[j], M[j] = Xn[r], Xc[r]; continue
                C[j] = Xn[m].mean(0)
                M[j] = [np.bincount(Xc[m, c]).argmax() for c in range(Xc.shape[1])]
        total = cost[np.arange(N), lab].sum()
        if total < best_cost:
            best, best_cost = lab.copy(), total
    return best


# ---- one feature space = one parallel task -------------------------------------------------------------------------
def run_space(enc, red):
    X = build_space(enc, red)
    rows = []

    def add(model, lab, **params):
        rows.append(dict(model=model, encoding=enc, reduction=red, dims=X.shape[1], **params, **score(lab)))

    for k in K_RANGE:
        add("K-Means", kmeans_labels(X, k), k=k)
    for cov in GMM_COVS:
        for k in K_RANGE:
            add("GMM", gmm_labels(X, k, cov), k=k, cov=cov)
    for k, lab in agglo_labels(X, K_RANGE).items():
        add("Agglomerative", lab, k=k)
    kdist = NearestNeighbors(n_neighbors=max(ms for ms, _ in DBSCAN_GRID)).fit(X[S]).kneighbors(X[S])[0]
    for ms, pct in DBSCAN_GRID:
        lab, eps = dbscan_labels(X, ms, pct, kdist=kdist)
        add("DBSCAN", lab, min_samples=ms, eps_pct=pct, eps=round(eps, 4))
    for mcs in HDBSCAN_GRID:
        add("HDBSCAN (extra)", hdbscan_labels(X, mcs), min_cluster_size=mcs)
    return rows


def run_gower_methods():
    """Gower* = cluster directly on the Gower matrix (3k rows). Biased: scored with the same ruler -> never a winner."""
    G0, _ = gower_mats()
    rows = []
    for k in K_RANGE:
        lab = np.full(N, -3)
        lab[GIDX] = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(G0)
        rows.append(dict(model="Agglomerative", encoding="Gower*", reduction="-", dims=np.nan, k=k, **score(lab)))
    srt = np.sort(G0, axis=1)
    for ms, pct in DBSCAN_GRID:
        eps = max(float(np.percentile(srt[:, ms], pct)), 1e-3)
        lab = np.full(N, -3)
        lab[GIDX] = DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit_predict(G0)
        rows.append(dict(model="DBSCAN", encoding="Gower*", reduction="-", dims=np.nan, min_samples=ms, eps_pct=pct,
                         eps=round(eps, 4), **score(lab)))
    return rows


def run_kprototypes(k):
    return [dict(model="K-Prototypes (extra)", encoding="none (mixed)", reduction="-", dims=len(NUMS) + 1 + len(CATS),
                 k=k, **score(kprototypes_labels(k)))]


# ---- stability of one configuration (row of the results table) ----------------------------------------------------
SEEDED = {"K-Means", "GMM", "K-Prototypes (extra)"}

def labels_for(row, seed=SEED, sample=S, X=None):
    m = row["model"]
    if m == "K-Prototypes (extra)":
        return kprototypes_labels(int(row["k"]), seed)
    X = build_space(row["encoding"], row["reduction"]) if X is None else X
    if m == "K-Means":       return kmeans_labels(X, int(row["k"]), seed)
    if m == "GMM":           return gmm_labels(X, int(row["k"]), row["cov"], seed)
    if m == "Agglomerative": return agglo_labels(X, [int(row["k"])], sample)[int(row["k"])]
    if m == "DBSCAN":        return dbscan_labels(X, int(row["min_samples"]), row["eps_pct"], sample)[0]
    if m == "HDBSCAN (extra)": return hdbscan_labels(X, int(row["min_cluster_size"]), sample)
    raise ValueError(m)

def stability(row, n=5):
    """Mean / min ARI against the seed-42 run: other seeds (K-Means, GMM, K-Prototypes) or other 10k samples."""
    X = None if row["model"] == "K-Prototypes (extra)" else build_space(row["encoding"], row["reduction"])
    ref = labels_for(row, X=X)
    aris = []
    for s in range(n):
        if row["model"] in SEEDED:
            lab = labels_for(row, seed=s, X=X)
        else:
            alt = np.sort(np.random.default_rng(100 + s).choice(N, SAMPLE, replace=False))
            lab = labels_for(row, sample=alt, X=X)
        m = (ref != -3) & (lab != -3)
        aris.append(adjusted_rand_score(ref[m], lab[m]))
    return round(float(np.mean(aris)), 3), round(float(np.min(aris)), 3)
