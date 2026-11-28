"""Do the proposed fixes help? One file, no other script needed.

Run from the project root:   python experiments/rejected_ideas/fix_test.py
Quick crash test (1 min):    python experiments/rejected_ideas/fix_test.py --quick

Variants tested (everything else identical to the current frequency matrix):
  current         the matrix the models use now
  kids x0.5       HasChildren column multiplied by 0.5
  kids x0.25      HasChildren column multiplied by 0.25
  no kids         HasChildren removed from the fit (still checked afterwards)
  meal ordinal    Meal = SC < BB < HB/FB (z-scored) instead of its frequency
  kids0.5+meal    both fixes together

Fair ruler (same for every variant): Gower silhouette on the ORIGINAL columns, 3,000 fixed rows.
Valid only if: smallest cluster >= 2%, stability ARI >= 0.8, and not a copy of one column (NMI < 0.5).
No IsCanceled / ADR anywhere.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

QUICK = "--quick" in sys.argv
ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "rejected_ideas" / "fix_test_outputs"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
KS = [3, 5] if QUICK else [3, 4, 5, 6, 7]
AGG_N = 1500 if QUICK else 6000
GOWER_N = 800 if QUICK else 3000

s2 = pd.read_csv(P / "h2_clustering_stage2.csv")
s3 = pd.read_csv(P / "h2_clustering_stage3.csv")
s4 = pd.read_csv(P / "h2_clustering_stage4.csv")
ref = pd.read_csv(P / "h2_clustering_stage5_frequency.csv")
N = len(s4)
CATS = ["MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType"]


def freq_z(s):
    share = s.map(s.value_counts(normalize=True))
    return (share - share.mean()) / share.std()


def build_base():
    d = pd.DataFrame(index=s4.index)
    d["Adults"] = s4["Adults"]
    for c in CATS:
        d[c] = freq_z(s4[c])
    for c in ["LeadTime_yj", "StaysInWeekNights_yj", "StaysInWeekendNights_sqrt", "HasChildren"]:
        d[c] = s4[c]
    ang = 2 * np.pi * (s4["ArrivalDateMonth"] - 1) / 12
    d["ArrivalMonth_sin"], d["ArrivalMonth_cos"] = np.sin(ang), np.cos(ang)
    return d[list(ref.columns)]


BASE = build_base()
print("rebuilt matrix equals stage5_frequency.csv:", bool(np.allclose(BASE.to_numpy(), ref.to_numpy(), atol=1e-6)))

MEAL_ORD = {"SC": 0, "BB": 1, "HB/FB": 2}
def meal_ordinal(d):
    m = s4["Meal"].map(MEAL_ORD).astype(float)
    d["Meal"] = (m - m.mean()) / m.std()
    return d

def kids(w):
    def f(d):
        d["HasChildren"] = d["HasChildren"] * w
        return d
    return f

VARIANTS = {
    "current": lambda d: d,
    "kids x0.5": kids(0.5),
    "kids x0.25": kids(0.25),
    "no kids": lambda d: d.drop(columns="HasChildren"),
    "meal ordinal": meal_ordinal,
    "kids0.5+meal": lambda d: meal_ordinal(kids(0.5)(d)),
}

rng = np.random.default_rng(7)
G_IDX = np.sort(rng.choice(N, GOWER_N, replace=False))
def gower_matrix():
    n = len(G_IDX); D = np.zeros((n, n)); f = 0
    for c in ["LeadTime", "StaysInWeekNights", "StaysInWeekendNights", "Adults"]:
        lo, hi = np.percentile(s2[c], [1, 99]); r = (hi - lo) or 1.0
        v = np.clip(s2[c].to_numpy(float)[G_IDX], lo, hi)
        D += np.abs(v[:, None] - v[None, :]) / r; f += 1
    k = (s2["Children"].to_numpy()[G_IDX] > 0)
    D += (k[:, None] != k[None, :]); f += 1
    for c in CATS:
        a = pd.factorize(s3[c])[0][G_IDX]
        D += (a[:, None] != a[None, :]); f += 1
    m = s2["ArrivalDateMonth"].to_numpy(float)[G_IDX]
    dm = np.abs(m[:, None] - m[None, :]); D += np.minimum(dm, 12 - dm) / 6; f += 1
    return D / f
GD = gower_matrix()

BINS = {}
for c in ["LeadTime", "StaysInWeekNights", "StaysInWeekendNights", "Adults"]:
    BINS[c] = pd.qcut(s2[c], 10, labels=False, duplicates="drop").to_numpy()
for c in CATS:
    BINS[c] = pd.factorize(s3[c])[0]
BINS["Month"] = s2["ArrivalDateMonth"].to_numpy()
BINS["HasChildren"] = (s2["Children"].to_numpy() > 0).astype(int)

def max_nmi(lab):
    best, name = 0.0, ""
    for c, b in BINS.items():
        v = normalized_mutual_info_score(b, lab)
        if v > best: best, name = v, c
    return best, name

def km_all(X):
    main = {k: KMeans(k, n_init=10, random_state=SEED).fit_predict(X) for k in KS}
    others = [{k: KMeans(k, n_init=5, random_state=s).fit_predict(X) for k in KS} for s in (1, 2)]
    return main, others

def agg_labels(X, seed):
    idx = np.sort(np.random.default_rng(seed).choice(N, AGG_N, replace=False))
    Z = linkage(X[idx], method="ward")
    out = {}
    for k in KS:
        ls = fcluster(Z, k, "maxclust") - 1
        knn = KNeighborsClassifier(5, weights="distance").fit(X[idx], ls)
        lab = np.concatenate([knn.predict(X[i:i + 5000]) for i in range(0, N, 5000)])
        lab[idx] = ls
        out[k] = lab
    return out

def agg_all(X):
    return agg_labels(X, SEED), [agg_labels(X, 10), agg_labels(X, 20)]

rows = []
t0 = time.time()
for vname, fn in VARIANTS.items():
    X = fn(BASE.copy()).to_numpy(float)
    for aname, runner in [("kmeans", km_all), ("agglomerative", agg_all)]:
        main, others = runner(X)
        for k in KS:
            lab = main[k]
            share = np.bincount(lab, minlength=k) / N * 100
            stab = float(np.mean([adjusted_rand_score(lab, o[k]) for o in others]))
            nmi, col = max_nmi(lab)
            kid_nmi = normalized_mutual_info_score(BINS["HasChildren"], lab)
            sil = silhouette_score(GD, lab[G_IDX], metric="precomputed") if len(np.unique(lab[G_IDX])) > 1 else np.nan
            valid = bool(share.min() >= 2 and stab >= 0.8 and nmi < 0.5)
            rows.append(dict(variant=vname, algo=aname, k=k, gower_sil=round(sil, 4), min_share=round(share.min(), 1),
                             stability=round(stab, 3), max_nmi=round(nmi, 2), nmi_col=col, kids_nmi=round(kid_nmi, 2), valid=valid))
        print(f"done {vname:14s} {aname:14s} ({time.time() - t0:.0f}s)", flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / "fix_test_results.csv", index=False)

lines = ["BEST VALID CONFIGURATION PER VARIANT (fair score = Gower silhouette on original columns, higher = better)", ""]
best_tbl = {}
for aname in ["kmeans", "agglomerative"]:
    lines.append(f"--- {aname} ---")
    for vname in VARIANTS:
        s = R[(R.variant == vname) & (R.algo == aname)]
        ok = s[s.valid]
        if len(ok):
            b = ok.sort_values("gower_sil", ascending=False).iloc[0]
            best_tbl[(aname, vname)] = b.gower_sil
            lines.append(f"{vname:14s} k={int(b.k)}  gower {b.gower_sil:.4f} | smallest {b.min_share}% | stability {b.stability} | "
                         f"closest column {b.nmi_col} (NMI {b.max_nmi}) | HasChildren NMI {b.kids_nmi}")
        else:
            b = s.sort_values("gower_sil", ascending=False).iloc[0]
            best_tbl[(aname, vname)] = np.nan
            lines.append(f"{vname:14s} NO VALID k. best near-miss k={int(b.k)} gower {b.gower_sil:.4f} | smallest {b.min_share}% | "
                         f"stability {b.stability} | NMI {b.max_nmi} ({b.nmi_col})")
    lines.append("")
lines.append("Full table: fix_test_results.csv   (settings: k=%s, agglomerative sample=%d, gower rows=%d)" % (KS, AGG_N, GOWER_N))
txt = "\n".join(lines); print("\n" + txt)
(OUT / "fix_test_summary.txt").write_text(txt, encoding="utf-8")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, aname in zip(axs, ["kmeans", "agglomerative"]):
    vals = [best_tbl[(aname, v)] for v in VARIANTS]
    ax.bar(range(len(vals)), np.nan_to_num(vals), color=["#2a78d6"] + ["#eb6834"] * (len(vals) - 1))
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(list(VARIANTS), rotation=30, ha="right")
    ax.set_title(f"{aname}: best valid Gower silhouette"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "fix_test_chart.png", dpi=150)
print("saved to", OUT)
