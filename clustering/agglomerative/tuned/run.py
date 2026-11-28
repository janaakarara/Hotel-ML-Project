import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "clustering"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
import common as C
X, SEED, SAMPLE = C.X, C.SEED, 10_000
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
CONFIGS = [("ward","euclidean"),("average","euclidean"),("average","cosine"),("complete","euclidean"),("complete","cosine")]
idx = np.sort(np.random.default_rng(SEED).choice(len(X), SAMPLE, replace=False))

def fit_full(sub, k, link, metric):
    Xs = X[sub]
    ls = AgglomerativeClustering(n_clusters=k, linkage=link, metric=metric).fit_predict(Xs)
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric=metric).fit(Xs, ls)
    lab = np.concatenate([knn.predict(X[i:i+2000]) for i in range(0, len(X), 2000)])
    lab[sub] = ls
    return lab

def evaluate(k, link, metric):
    lab = fit_full(idx, k, link, metric)
    share = np.bincount(lab, minlength=k) / len(X) * 100
    dep = C.dependence(lab)
    return {"k": k, "linkage": link, "metric": metric, "sil": C.common_silhouette(lab),
            "min_share": share.min(), "ari_col": dep["ari"], "nmi_col": dep["nmi"], "col": dep["nmi_col"],
            "rejected": bool(share.min() < 2 or C.is_single_column(dep)), "labels": lab}

res = Parallel(n_jobs=2)(delayed(evaluate)(k, l, m) for k in range(2, 11) for l, m in CONFIGS)
pd.DataFrame([{a: b for a, b in r.items() if a != "labels"} for r in res]).to_csv(OUT / "tuning_grid.csv", index=False)
ok = sorted([r for r in res if not r["rejected"]], key=lambda r: -r["sil"])
if not ok: raise RuntimeError("every config was rejected (single-column or tiny cluster)")
for r in ok[:5]:
    r["stab"] = float(np.mean([adjusted_rand_score(r["labels"], fit_full(
        np.sort(np.random.default_rng(s).choice(len(X), SAMPLE, replace=False)), r["k"], r["linkage"], r["metric"])) for s in (10, 20, 30)]))
short = ok[:5]
stable = [r for r in short if r["stab"] >= C.STAB_MIN]
best = stable[0] if stable else max(short, key=lambda r: r["stab"])
extra = [f"stability ARI (3 subsamples): {best['stab']:.3f}"]
if not stable: extra.append("WARNING: no shortlisted model reached stability 0.8; most stable one reported")
lab = best["labels"]
C.write_results(OUT, "AGGLOMERATIVE TUNED (12-D)", f"linkage={best['linkage']}, metric={best['metric']}, k={best['k']}, sample={SAMPLE}, knn=5", lab, extra)
plt.figure(figsize=(10, 5)); dendrogram(linkage(X[idx], method=best["linkage"], metric=best["metric"]), truncate_mode="lastp", p=30, no_labels=True)
plt.title("Agglomerative tuned: dendrogram"); plt.tight_layout(); plt.savefig(OUT / "dendrogram.png", dpi=150); plt.close()
p = PCA(n_components=2, random_state=SEED).fit_transform(X[C.SIL_IDX])
plt.figure(figsize=(7, 6)); plt.scatter(p[:, 0], p[:, 1], s=3, c=lab[C.SIL_IDX], cmap="tab10")
plt.title("Agglomerative tuned: 2-D PCA (plot only)"); plt.tight_layout(); plt.savefig(OUT / "pca.png", dpi=150); plt.close()

