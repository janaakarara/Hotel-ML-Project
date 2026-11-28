from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "data/processed/h2_clustering_stage2.csv")
post = pd.read_csv(ROOT / "data/processed/h2_clustering_profile_cols.csv")
assert len(raw) == len(post)
raw["Nights"] = raw["StaysInWeekNights"] + raw["StaysInWeekendNights"]
raw["HasChildren"] = (raw["Children"] > 0).astype(int)
CATS = ["MarketSegment", "Country", "Meal", "CustomerType", "ReservedRoomType"]
for p in sorted((ROOT / "clustering").glob("*/*/outputs/labels.npy")):
    lab = np.load(p)
    if len(lab) != len(raw):
        print("skip (row count differs):", p); continue
    g = raw.assign(cluster=lab).groupby("cluster")
    ph = post.assign(cluster=lab).groupby("cluster")
    out = pd.DataFrame({"rows": g.size()})
    out["share_pct"] = out["rows"] / len(raw) * 100
    out["lead_time_mean"] = g["LeadTime"].mean()
    out["nights_mean"] = g["Nights"].mean()
    out["adults_mean"] = g["Adults"].mean()
    out["pct_children"] = g["HasChildren"].mean() * 100
    for c in CATS:
        out["top_" + c] = g[c].agg(lambda s: s.value_counts().index[0])
    out["posthoc_adr_mean"] = ph["ADR"].mean()
    out["posthoc_cancel_rate_pct"] = ph["IsCanceled"].mean() * 100
    out.round(2).to_csv(p.parent / "cluster_profiles.csv")
    print("wrote", p.parent / "cluster_profiles.csv")
