"""Stage 2 for clustering: near-constant columns and redundant (duplicate-information) columns.

Everything here is NUMERIC + GRAPHS. No column is dropped by this script: it only measures and flags.
You decide the drops by reading the printed tables and the figures.

Usage: python src/02_preprocessing/stage2a_redundancy_check.py --input data/processed/h2_clustering_stage1.csv
Writes: outputs/02_redundancy/stage2a_*.csv and outputs/02_redundancy/stage2a_*.png
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, entropy

# Columns that are categories even when stored as numbers (codes / flags / cyclical month).
CAT_AS_NUM = ["ArrivalDateMonth", "IsRepeatedGuest", "has_company"]
NEAR_CONST = 0.95   # one value covers >= 95% of rows -> near-constant
HIGH_CORR = 0.80    # |association| >= 0.80 -> redundant pair


def split_columns(df):
    cat = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c])) or c in CAT_AS_NUM]
    num = [c for c in df.columns if c not in cat]
    return num, cat


# ---------- measure 1: near-constant ----------
def constancy_table(df):
    rows = []
    for c in df.columns:
        vc = df[c].value_counts(normalize=True)
        k = len(vc)
        rows.append({"column": c, "n_unique": k,
                     "top_value": vc.index[0], "top_share": round(vc.iloc[0], 4),
                     "norm_entropy": round(entropy(vc) / np.log(k), 3) if k > 1 else 0.0})
    t = pd.DataFrame(rows).sort_values("top_share", ascending=False)
    t["flag_near_constant"] = t.top_share >= NEAR_CONST
    return t


# ---------- measure 2: association between pairs ----------
def cramers_v(x, y):
    """Bias-corrected Cramer's V (0 = independent, 1 = one column fully determines the other)."""
    ct = pd.crosstab(x, y)
    n = ct.values.sum()
    chi2 = chi2_contingency(ct, correction=False)[0]
    r, k = ct.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / d)) if d > 0 else 0.0


def eta(num, cat):
    """Correlation ratio: share of the numeric column's variance explained by the category (0..1)."""
    g = num.groupby(cat)
    ssb = (g.count() * (g.mean() - num.mean()) ** 2).sum()
    sst = ((num - num.mean()) ** 2).sum()
    return float(np.sqrt(ssb / sst)) if sst > 0 else 0.0


def heat(mat, title, path, vmin=0, vmax=1, cmap="viridis", annot=True):
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * mat.shape[1] + 3), max(6, 0.6 * mat.shape[0] + 2)))
    im = ax.imshow(mat.values.astype(float), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=60, ha="right")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    if annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.25 else "black")
    fig.colorbar(im, ax=ax); ax.set_title(title); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def flag_pairs(mat, kind, symmetric):
    out = []
    for i, a in enumerate(mat.index):
        for j, b in enumerate(mat.columns):
            if symmetric and j <= i:
                continue
            if a == b:
                continue
            v = mat.iloc[i, j]
            if abs(v) >= HIGH_CORR:
                out.append({"kind": kind, "col_a": a, "col_b": b, "value": round(float(v), 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/h2_clustering_stage1.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out); (out / "02_redundancy").mkdir(parents=True, exist_ok=True); (out / "02_redundancy").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.input, dtype={"Agent": str})
    num, cat = split_columns(df)
    print(f"rows={len(df)}  numeric={len(num)}  categorical={len(cat)}")

    # 1) near-constant
    ct = constancy_table(df)
    ct.to_csv(out / "02_redundancy/stage2a_constancy.csv", index=False)
    print("\n=== 1) NEAR-CONSTANT CHECK (top_share = share of the most common value) ===")
    print(ct.to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 7))
    s = ct.sort_values("top_share")
    ax.barh(s.column, s.top_share, color=np.where(s.top_share >= NEAR_CONST, "tab:red", "tab:blue"))
    ax.axvline(NEAR_CONST, ls="--", c="k"); ax.set_xlabel("share of most common value")
    ax.set_title("Near-constant check (red = flagged)"); fig.tight_layout()
    fig.savefig(out / "02_redundancy/stage2a_constancy.png", dpi=130); plt.close(fig)

    # 2a) numeric vs numeric: Spearman
    sp = df[num].corr(method="spearman")
    sp.round(3).to_csv(out / "02_redundancy/stage2a_spearman.csv")
    heat(sp, "Spearman correlation (numeric)", out / "02_redundancy/stage2a_spearman.png", -1, 1, "coolwarm")

    # 2b) categorical vs categorical: Cramer's V
    cv = pd.DataFrame(index=cat, columns=cat, dtype=float)
    for i, x in enumerate(cat):
        for y in cat[i:]:
            cv.loc[x, y] = cv.loc[y, x] = 1.0 if x == y else cramers_v(df[x], df[y])
    cv.round(3).to_csv(out / "02_redundancy/stage2a_cramers_v.csv")
    heat(cv, "Cramer's V (categorical)", out / "02_redundancy/stage2a_cramers_v.png")

    # 2c) numeric vs categorical: eta
    et = pd.DataFrame({c: {n: eta(df[n], df[c]) for n in num} for c in cat})
    et.round(3).to_csv(out / "02_redundancy/stage2a_eta.csv")
    heat(et, "Correlation ratio eta (numeric by category)", out / "02_redundancy/stage2a_eta.png")

    flags = pd.DataFrame(flag_pairs(sp, "spearman", True) + flag_pairs(cv, "cramers_v", True) + flag_pairs(et, "eta", False))
    flags.to_csv(out / "02_redundancy/stage2a_flagged_pairs.csv", index=False)
    print(f"\n=== 2) REDUNDANT PAIRS (association >= {HIGH_CORR}) ===")
    print(flags.sort_values("value", ascending=False).to_string(index=False) if len(flags) else "none")

    # also show the next-strongest pairs so you see what is just below the threshold
    near = []
    for m, k in [(sp, "spearman"), (cv, "cramers_v")]:
        for i, x in enumerate(m.index):
            for j, y in enumerate(m.columns):
                if j > i:
                    near.append((k, x, y, abs(m.iloc[i, j])))
    near = pd.DataFrame(near, columns=["kind", "col_a", "col_b", "abs_value"]).sort_values("abs_value", ascending=False).head(10)
    print("\n=== top 10 strongest pairs (numeric and categorical) ===")
    print(near.round(3).to_string(index=False))
    print("\nFigures + tables saved in", out)


if __name__ == "__main__":
    main()
