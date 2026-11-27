from pathlib import Path
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
stage4_path = REPO_ROOT / "data/processed/h2_clustering_stage4.csv"
df = pd.read_csv(stage4_path)

# Detect categorical columns (object/category/string) present prior to Stage 5 One-Hot Encoding
cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
print(f"Found {len(cat_cols)} categorical columns to frequency-encode: {cat_cols}")

df_freq = df.copy()
freq_maps = {}
for col in cat_cols:
    freqs = df_freq[col].value_counts(normalize=True)
    freq_maps[col] = {str(k): round(float(v), 6) for k, v in freqs.items()}
    df_freq[col] = df_freq[col].map(freqs).astype(float)

# Scale encoded frequency columns so their variance (~1.0) matches Stage 4 continuous features
if cat_cols:
    scaler = StandardScaler()
    df_freq[cat_cols] = scaler.fit_transform(df_freq[cat_cols])

out_csv = REPO_ROOT / "data/processed/h2_clustering_stage5_freq.csv"
df_freq.to_csv(out_csv, index=False)

out_meta = REPO_ROOT / "data/processed/stage5_freq_encoding.json"
out_meta.write_text(json.dumps({"encoded_columns": cat_cols, "frequencies": freq_maps}, indent=2), encoding="utf-8")

print(f"Saved {df_freq.shape[0]} rows x {df_freq.shape[1]} cols -> {out_csv}")