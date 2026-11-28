"""Test 4 ideas for better scores and profiles. One file. Run from the project root:
   python experiments/rejected_ideas/improve_test.py            (full run)
   python experiments/rejected_ideas/improve_test.py --quick    (crash test)
1 profiles : heatmap + extra columns for the current tuned K-Means labels (extra columns used AFTER clustering only)
2 leave-one-column-out : K-Means without each input column
3 PCA      : K-Means on PCA(90%) and PCA(90%, whitened)
4 derived  : TotalNights + WeekendShare instead of the two night columns; and + TotalGuests
Fair ruler = Gower silhouette on original columns (same as fix_test.py). A gain counts only if >= 0.01 and valid.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

QUICK = "--quick" in sys.argv
ROOT = Path(__file__).resolve().parents[2]; P = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "rejected_ideas" / "improve_test_outputs"; OUT.mkdir(parents=True, exist_ok=True)
KS = [3, 5] if QUICK else [3, 4, 5, 6, 7]; GOWER_N = 800 if QUICK else 3000
s1 = pd.read_csv(P / "h2_clustering_stage1.csv"); s2 = pd.read_csv(P / "h2_clustering_stage2.csv")
s3 = pd.read_csv(P / "h2_clustering_stage3.csv"); s4 = pd.read_csv(P / "h2_clustering_stage4.csv")
pc = pd.read_csv(P / "h2_clustering_profile_cols.csv"); ref = pd.read_csv(P / "h2_clustering_stage5_frequency.csv")
N = len(s4); CATS = ["MarketSegment", "Meal", "Country", "CustomerType", "ReservedRoomType"]

def freq_z(s):
    sh = s.map(s.value_counts(normalize=True)); return (sh - sh.mean()) / sh.std()
d = pd.DataFrame(index=s4.index); d["Adults"] = s4["Adults"]
for c in CATS: d[c] = freq_z(s4[c])
for c in ["LeadTime_yj", "StaysInWeekNights_yj", "StaysInWeekendNights_sqrt", "HasChildren"]: d[c] = s4[c]
ang = 2 * np.pi * (s4["ArrivalDateMonth"] - 1) / 12; d["ArrivalMonth_sin"], d["ArrivalMonth_cos"] = np.sin(ang), np.cos(ang)
BASE = d[list(ref.columns)]
print("matrix equals stage5_frequency.csv:", bool(np.allclose(BASE.to_numpy(), ref.to_numpy(), atol=1e-6)))

G_IDX = np.sort(np.random.default_rng(7).choice(N, GOWER_N, replace=False))
def gower():
    D = np.zeros((GOWER_N, GOWER_N)); f = 0
    for c in ["LeadTime", "StaysInWeekNights", "StaysInWeekendNights", "Adults"]:
        lo, hi = np.percentile(s2[c], [1, 99]); v = np.clip(s2[c].to_numpy(float)[G_IDX], lo, hi)
        D += np.abs(v[:, None] - v[None, :]) / ((hi - lo) or 1.0); f += 1
    k = s2["Children"].to_numpy()[G_IDX] > 0; D += (k[:, None] != k[None, :]); f += 1
    for c in CATS:
        a = pd.factorize(s3[c])[0][G_IDX]; D += (a[:, None] != a[None, :]); f += 1
    m = s2["ArrivalDateMonth"].to_numpy(float)[G_IDX]; dm = np.abs(m[:, None] - m[None, :]); D += np.minimum(dm, 12 - dm) / 6
    return D / (f + 1)
GD = gower()
BINS = {c: pd.qcut(s2[c], 10, labels=False, duplicates="drop").to_numpy() for c in ["LeadTime", "StaysInWeekNights", "StaysInWeekendNights", "Adults"]}
for c in CATS: BINS[c] = pd.factorize(s3[c])[0]
BINS["Month"] = s2["ArrivalDateMonth"].to_numpy(); BINS["HasChildren"] = (s2["Children"].to_numpy() > 0).astype(int)
def max_nmi(lab):
    b, n = 0.0, ""
    for c, x in BINS.items():
        v = normalized_mutual_info_score(x, lab)
        if v > b: b, n = v, c
    return b, n

GROUPS = {"Adults": ["Adults"], "MarketSegment": ["MarketSegment"], "Meal": ["Meal"], "Country": ["Country"],
          "CustomerType": ["CustomerType"], "ReservedRoomType": ["ReservedRoomType"], "LeadTime": ["LeadTime_yj"],
          "WeekNights": ["StaysInWeekNights_yj"], "WeekendNights": ["StaysInWeekendNights_sqrt"],
          "HasChildren": ["HasChildren"], "Month": ["ArrivalMonth_sin", "ArrivalMonth_cos"]}
def derived1(x):
    x = x.drop(columns=["StaysInWeekNights_yj", "StaysInWeekendNights_sqrt"])
    tn = (s2["StaysInWeekNights"] + s2["StaysInWeekendNights"]).to_frame()
    x["TotalNights_yj"] = PowerTransformer("yeo-johnson").fit_transform(tn)[:, 0]
    ws = s2["StaysInWeekendNights"] / tn.iloc[:, 0]; x["WeekendShare"] = (ws - ws.mean()) / ws.std()
    return x
def derived2(x):
    x = derived1(x); g = s2["Adults"] + s2["Children"]; x["TotalGuests"] = (g - g.mean()) / g.std(); return x
VARIANTS = {"current": lambda x: x}
for g, cols in GROUPS.items(): VARIANTS["drop " + g] = (lambda cols: lambda x: x.drop(columns=cols))(cols)
VARIANTS["PCA 90%"] = lambda x: PCA(0.90, random_state=42).fit_transform(x.to_numpy(float))
VARIANTS["PCA 90% whitened"] = lambda x: PCA(0.90, whiten=True, random_state=42).fit_transform(x.to_numpy(float))
VARIANTS["derived: nights"] = derived1
VARIANTS["derived: nights+guests"] = derived2

rows, t0 = [], time.time()
for vn, fn in VARIANTS.items():
    X = np.asarray(fn(BASE.copy()), dtype=float)
    main = {k: KMeans(k, n_init=10, random_state=42).fit_predict(X) for k in KS}
    oth = [{k: KMeans(k, n_init=5, random_state=s).fit_predict(X) for k in KS} for s in (1, 2)]
    for k in KS:
        lab = main[k]; sh = np.bincount(lab, minlength=k) / N * 100
        st = float(np.mean([adjusted_rand_score(lab, o[k]) for o in oth])); nmi, col = max_nmi(lab)
        sil = silhouette_score(GD, lab[G_IDX], metric="precomputed")
        rows.append(dict(variant=vn, k=k, gower_sil=round(sil, 4), min_share=round(sh.min(), 1), stability=round(st, 3),
                         max_nmi=round(nmi, 2), nmi_col=col, dims=X.shape[1], valid=bool(sh.min() >= 2 and st >= 0.8 and nmi < 0.5)))
    print(f"done {vn:24s} ({time.time() - t0:.0f}s)", flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "improve_test_results.csv", index=False)

best = {}
for vn in VARIANTS:
    s = R[(R.variant == vn) & R.valid]
    best[vn] = s.sort_values("gower_sil", ascending=False).iloc[0] if len(s) else None
cur = best["current"].gower_sil
lines = ["K-MEANS, BEST VALID PER VARIANT (Gower silhouette on original columns; current = %.4f)" % cur, ""]
for vn, b in sorted(best.items(), key=lambda t: -(t[1].gower_sil if t[1] is not None else -9)):
    if b is None: lines.append(f"{vn:24s} NO VALID k"); continue
    dl = b.gower_sil - cur
    tag = "BETTER" if dl >= 0.01 else ("worse" if dl <= -0.01 else "no real change")
    lines.append(f"{vn:24s} k={int(b.k)} gower {b.gower_sil:.4f} ({dl:+.4f}, {tag}) | smallest {b.min_share}% | stability {b.stability} | "
                 f"closest column {b.nmi_col} (NMI {b.max_nmi}) | dims {int(b.dims)}")
txt = "\n".join(lines); print("\n" + txt); (OUT / "improve_test_summary.txt").write_text(txt, encoding="utf-8")
names = [v for v in VARIANTS if best[v] is not None]; vals = [best[v].gower_sil for v in names]
plt.figure(figsize=(8, 6)); plt.barh(names, vals, color=["#2a78d6" if n == "current" else "#eb6834" for n in names])
plt.axvline(cur, color="gray", ls="--"); plt.title("Best valid Gower silhouette (dashed = current)"); plt.tight_layout()
plt.savefig(OUT / "improve_test_chart.png", dpi=150); plt.close()

lp = ROOT / "clustering" / "kmeans" / "tuned" / "outputs" / "labels.npy"
lab = np.load(lp) if lp.exists() else KMeans(5, n_init=10, random_state=42).fit_predict(BASE.to_numpy(float))
assert len(lab) == N
F = pd.DataFrame({
 "LeadTime": s2["LeadTime"], "WeekNights": s2["StaysInWeekNights"], "WeekendNights": s2["StaysInWeekendNights"], "Adults": s2["Adults"],
 "Has children": (s2["Children"] > 0).astype(int), "Summer arrival (Jun-Aug)": s2["ArrivalDateMonth"].isin([6, 7, 8]).astype(int),
 "Online TA": (s3["MarketSegment"] == "Online TA").astype(int), "Offline TA/TO": (s3["MarketSegment"] == "Offline TA/TO").astype(int),
 "Direct": (s3["MarketSegment"] == "Direct").astype(int), "Groups/Corporate": s3["MarketSegment"].isin(["Groups", "Corporate"]).astype(int),
 "Meal SC": (s3["Meal"] == "SC").astype(int), "Meal HB/FB": (s3["Meal"] == "HB/FB").astype(int),
 "Transient-Party/Group": (s3["CustomerType"] == "Transient-Party/Group").astype(int),
 "Portugal": (s3["Country"] == "Portugal").astype(int), "Latin Europe": (s3["Country"] == "Latin Europe").astype(int),
 "Rest of Europe": (s3["Country"] == "Rest of Europe").astype(int), "Outside Europe": (s3["Country"] == "Outside Europe").astype(int),
 "Room A": (s3["ReservedRoomType"] == "A").astype(int), "Room D": (s3["ReservedRoomType"] == "D").astype(int)})
POST = {"ADR*": pc["ADR"], "Cancel rate*": pc["IsCanceled"], "Special requests*": pc["TotalOfSpecialRequests"],
        "Parking*": (pc["RequiredCarParkingSpaces"] > 0).astype(int), "Booking changes>0*": (pc["BookingChanges"] > 0).astype(int),
        "Non-refund deposit*": (pc["DepositType"] == "Non Refund").astype(int), "Repeated guest*": s1["IsRepeatedGuest"],
        "Past cancellations>0*": (s1["PreviousCancellations"] > 0).astype(int)}
for k, v in POST.items(): F[k] = v.to_numpy()
M = F.groupby(lab).mean(); Z = (M - F.mean()) / F.std(); size = pd.Series(lab).value_counts(normalize=True).sort_index() * 100
M.assign(share_pct=size.round(1)).round(3).to_csv(OUT / "cluster_profiles_rich.csv")
plain = [c for c in F.columns if not c.endswith("*")]
print("\nCLUSTER NAMES SUGGESTED BY THE DATA (distinctive pre-clustering columns; * columns were not used to cluster)")
for c in M.index:
    top = Z.loc[c, plain].abs().sort_values(ascending=False).index[:3]
    print(f" cluster {c} ({size[c]:.1f}%): " + ", ".join(f"{t} {'high' if Z.loc[c, t] > 0 else 'low'}" for t in top) +
          f" | cancel {M.loc[c, 'Cancel rate*'] * 100:.0f}% | ADR {M.loc[c, 'ADR*']:.0f}")
fig, ax = plt.subplots(figsize=(9, 9)); im = ax.imshow(Z.T.to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(M))); ax.set_xticklabels([f"C{c}\n{size[c]:.0f}%" for c in M.index]); ax.set_yticks(range(len(F.columns))); ax.set_yticklabels(F.columns)
for i, f in enumerate(F.columns):
    for j, c in enumerate(M.index):
        v = M.loc[c, f]; ax.text(j, i, f"{v:.0f}" if f in ("ADR*", "LeadTime") else (f"{v:.1f}" if f in ("WeekNights", "WeekendNights", "Adults", "Special requests*") else f"{v * 100:.0f}%"), ha="center", va="center", fontsize=8)
plt.colorbar(im, label="above / below average (in std)"); ax.set_title("K-Means tuned profiles (* = used only after clustering)")
plt.tight_layout(); plt.savefig(OUT / "profile_heatmap.png", dpi=150); print("saved to", OUT)
