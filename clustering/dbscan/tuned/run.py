import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "clustering"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import common as C
X, SEED = C.X, C.SEED
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
EPS_GRID = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
MS_GRID = [15, 25, 50, 100, 150]

def ev(eps, ms):
    lab = DBSCAN(eps=eps, min_samples=ms, n_jobs=1).fit_predict(X)
    keep = lab != -1; noise = 100 * (~keep).mean()
    sizes = np.unique(lab[keep], return_counts=True)[1] if keep.any() else np.array([])
    row = {"eps": eps, "ms": ms, "clusters": len(sizes), "noise_pct": round(noise, 2), "sil": np.nan,
           "ari_col": np.nan, "nmi_col": np.nan, "rejected": True}
    if len(sizes) >= 2:
        dep = C.dependence(lab)
        row.update(sil=C.common_silhouette(lab), ari_col=dep["ari"], nmi_col=dep["nmi"])
        row["rejected"] = bool(noise > 25 or not (2 <= len(sizes) <= 10) or sizes.min() / len(X) * 100 < 2 or C.is_single_column(dep))
    return row, lab

res = Parallel(n_jobs=2)(delayed(ev)(e, m) for e in EPS_GRID for m in MS_GRID)
pd.DataFrame([r for r, _ in res]).to_csv(OUT / "search_table.csv", index=False)   # spec: DBSCAN search table
ok = [(r, l) for r, l in res if not r["rejected"]]
warn = []
if not ok:
    def _share(l):
        s = np.unique(l[l != -1], return_counts=True)[1]
        return s.min() / len(X) * 100 if len(s) else 0
    near = [(r, l) for r, l in res if 2 <= r["clusters"] <= 10 and r["noise_pct"] <= 25]
    if not near: raise RuntimeError("no DBSCAN config even near-valid; see search_table.csv")
    ok = [(dict(r, sil=-r["nmi_col"]), l) for r, l in near]
    warn = ["WARNING: no config passed the single-column guard (NMI < 0.5). Reported the near-miss with the lowest NMI; its clusters are largely driven by one input column. It may also contain clusters under 2% of rows (check cluster sizes). Do not present as a real segmentation."]
r, lab = max(ok, key=lambda t: t[0]["sil"])
extra = [f"best eps={r['eps']}, min_samples={r['ms']}"]
extra += warn
if r["eps"] == max(EPS_GRID): extra.append("NOTE: winner is at the edge of the eps grid")
C.write_results(OUT, "DBSCAN TUNED (12-D)", f"eps={r['eps']}, min_samples={r['ms']}, euclidean", lab, extra)
kd = np.sort(NearestNeighbors(n_neighbors=r["ms"], n_jobs=-1).fit(X).kneighbors(X)[0][:, -1])
plt.figure(figsize=(8, 5)); plt.plot(kd); plt.axhline(r["eps"], color="green", ls="--", label=f"eps={r['eps']}")
plt.legend(); plt.title("DBSCAN tuned: k-distance"); plt.tight_layout(); plt.savefig(OUT / "k_distance.png", dpi=150); plt.close()
p = PCA(n_components=2, random_state=SEED).fit_transform(X[C.SIL_IDX]); l = lab[C.SIL_IDX]; n = l == -1
plt.figure(figsize=(7, 6)); plt.scatter(p[n, 0], p[n, 1], s=3, c="lightgray"); plt.scatter(p[~n, 0], p[~n, 1], s=3, c=l[~n] % 10, cmap="tab10")
plt.title("DBSCAN tuned: 2-D PCA (plot only)"); plt.tight_layout(); plt.savefig(OUT / "pca.png", dpi=150); plt.close()


