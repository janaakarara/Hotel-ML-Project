from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ARI_MAX, NMI_MAX, STAB_MIN = 0.9, 0.5, 0.8
_df = pd.read_csv(ROOT / "data/processed/h2_clustering_stage5_frequency.csv")
COLS = list(_df.columns)
X = _df.to_numpy(dtype=float)
SIL_IDX = np.sort(np.random.default_rng(SEED).choice(len(X), min(10_000, len(X)), replace=False))
DEP_IDX = np.sort(np.random.default_rng(1).choice(len(X), min(20_000, len(X)), replace=False))

def _binned():
    out = {}
    for j, c in enumerate(COLS):
        v = X[:, j]
        out[c] = (pd.qcut(v, 10, labels=False, duplicates="drop") if len(np.unique(v)) > 12 else pd.factorize(v)[0])
    ang = np.arctan2(X[:, COLS.index("ArrivalMonth_sin")], X[:, COLS.index("ArrivalMonth_cos")])
    out["ArrivalMonth"] = np.round(ang / (2 * np.pi) * 12).astype(int) % 12
    return {c: np.asarray(b)[DEP_IDX] for c, b in out.items()}
_B = _binned()

def common_silhouette(labels):
    l = labels[SIL_IDX]          # noise (-1) counts as its own group
    return float("nan") if len(np.unique(l)) < 2 else float(silhouette_score(X[SIL_IDX], l))

def dependence(labels):
    l = labels[DEP_IDX]; r = {"ari": 0.0, "ari_col": "", "nmi": 0.0, "nmi_col": ""}
    for c, b in _B.items():
        a = adjusted_rand_score(b, l); n = normalized_mutual_info_score(b, l)
        if a > r["ari"]: r["ari"], r["ari_col"] = a, c
        if n > r["nmi"]: r["nmi"], r["nmi_col"] = n, c
    return r

def is_single_column(dep):
    return dep["ari"] >= ARI_MAX or dep["nmi"] >= NMI_MAX

def write_results(out, title, settings, labels, extra=()):
    keep = labels != -1; dep = dependence(labels)
    sizes = np.unique(labels, return_counts=True)[1].tolist()
    lines = [title, f"settings: {settings}", "input: 12-D frequency matrix (no PCA)",
             f"rows: {len(X)} | cluster sizes (noise incl. if any): {sizes} | noise %: {100*(~keep).mean():.2f}",
             f"silhouette (common ruler, 10k fixed sample, noise = own group): {common_silhouette(labels):.4f}"]
    if len(np.unique(labels[keep])) >= 2:
        lines.append(f"davies-bouldin (non-noise): {davies_bouldin_score(X[keep], labels[keep]):.4f} | "
                     f"calinski-harabasz (non-noise): {calinski_harabasz_score(X[keep], labels[keep]):.1f}")
    lines.append(f"closest single column: ARI {dep['ari']:.2f} ({dep['ari_col']}) | NMI {dep['nmi']:.2f} ({dep['nmi_col']})")
    lines += list(extra)
    txt = "\n".join(lines); print(txt)
    (Path(out) / "results.txt").write_text(txt, encoding="utf-8")
    np.save(Path(out) / "labels.npy", labels)
