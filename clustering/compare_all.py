from pathlib import Path
import numpy as np, pandas as pd
import common as C
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
rows = []
for p in sorted((C.ROOT / "clustering").glob("*/*/outputs/labels.npy")):
    lab = np.load(p)
    if len(lab) != len(C.X): print("skip (row count differs):", p); continue
    d = C.dependence(lab); keep = lab != -1
    ok = len(np.unique(lab[keep])) >= 2
    db = davies_bouldin_score(C.X[keep], lab[keep]) if ok else float("nan")
    ch = calinski_harabasz_score(C.X[keep], lab[keep]) if ok else float("nan")
    rows.append({"model": p.parts[-4], "variant": p.parts[-3], "clusters": len(np.unique(lab[keep])),
                 "noise_pct": round(100 * (~keep).mean(), 2), "silhouette_common": round(C.common_silhouette(lab), 4),
                 "davies_bouldin": round(db, 4), "calinski_harabasz": round(ch, 1),
                 "max_ARI_col": round(d["ari"], 2), "max_NMI_col": round(d["nmi"], 2), "NMI_col": d["nmi_col"]})
df = pd.DataFrame(rows); df.to_csv(C.ROOT / "clustering/comparison.csv", index=False); print(df.to_string(index=False))
