# PCA is used ONLY for 2-D visualisation / reporting. Clustering fits on the 12-D frequency matrix.
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
REPO_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = REPO_ROOT / "data/processed/h2_clustering_stage5_frequency.csv"
OUT_PATH = REPO_ROOT / "data/processed/h2_clustering_pca90.csv"
META_PATH = REPO_ROOT / "data/processed/pca90_metadata.json"
df = pd.read_csv(IN_PATH)
assert "ArrivalMonth" not in df.columns, "raw month column found: use sin/cos"
X = df.to_numpy(dtype=float)
cum = np.cumsum(PCA(random_state=42).fit(X).explained_variance_ratio_)
for t in (0.85, 0.90, 0.95):
    n = int(np.searchsorted(cum, t) + 1); print(f"Variance >= {t*100:.0f}% -> {n} components")
pca = PCA(n_components=0.90, random_state=42); Z = pca.fit_transform(X)
pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(Z.shape[1])]).to_csv(OUT_PATH, index=False)
META_PATH.write_text(json.dumps({"input_file": str(IN_PATH.relative_to(REPO_ROOT)), "output_shape": list(Z.shape),
    "variance_retained": round(float(pca.explained_variance_ratio_.sum()), 6),
    "explained_variance_ratio": [round(float(v), 6) for v in pca.explained_variance_ratio_]}, indent=2))
print("PC1 variance:", round(float(pca.explained_variance_ratio_[0]), 4), "->", OUT_PATH)
