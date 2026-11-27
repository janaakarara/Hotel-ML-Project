from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
# You can switch input between h2_clustering_stage5_freq.csv or h2_clustering_stage5.csv
IN_PATH = REPO_ROOT / "data/processed/h2_clustering_stage5_freq.csv"
OUT_PATH = REPO_ROOT / "data/processed/h2_clustering_pca90.csv"
META_PATH = REPO_ROOT / "data/processed/pca90_metadata.json"

df = pd.read_csv(IN_PATH)
X = df.to_numpy(dtype=float)

# Fit full PCA first to inspect cumulative variance
pca_full = PCA(random_state=42).fit(X)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

for thresh in (0.85, 0.90, 0.95):
    n_comp = int(np.searchsorted(cum_var, thresh) + 1)
    print(f"Variance >= {thresh*100:.0f}% -> {n_comp} components (actual: {cum_var[n_comp-1]*100:.2f}%)")

# Transform keeping 90% variance
pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X)

cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
out_df = pd.DataFrame(X_pca, columns=cols)
out_df.to_csv(OUT_PATH, index=False)

meta = {
    "input_file": str(IN_PATH.relative_to(REPO_ROOT)),
    "input_shape": list(X.shape),
    "output_shape": list(X_pca.shape),
    "variance_retained": round(float(pca.explained_variance_ratio_.sum()), 6),
    "explained_variance_ratio": [round(float(v), 6) for v in pca.explained_variance_ratio_],
}
META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"\nSaved {out_df.shape[0]} rows x {out_df.shape[1]} PCs -> {OUT_PATH}")