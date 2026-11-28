"""Audit: do the outputs match the scripts? Reads files only. Run from the project root."""
import re, json, os, sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

ROOT = Path(__file__).resolve().parents[2]; P = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "audit" / "audit_output"; OUT.mkdir(parents=True, exist_ok=True)
res = []
def chk(ok, name, detail=""): res.append(("PASS" if ok else "FAIL", name, detail))
def soft(ok, name, detail=""): res.append(("PASS" if ok else "WARN", name, detail))
def info(name, detail=""): res.append(("INFO", name, detail))
def git(*a):
    try:
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=90)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return "ERR " + str(e)

# ---- 1. data chain ----
files = {"clean": "h2_clean.csv", "stage1": "h2_clustering_stage1.csv", "stage2": "h2_clustering_stage2.csv",
         "stage3": "h2_clustering_stage3.csv", "stage4": "h2_clustering_stage4.csv", "stage5": "h2_clustering_stage5_frequency.csv"}
D = {}
for k, f in files.items():
    p = P / f
    if not p.exists(): chk(False, f"data file {f}", "missing"); continue
    D[k] = pd.read_csv(p); info(f"{f} shape", str(D[k].shape))
if "stage5" not in D: sys.exit("stage5_frequency.csv missing - cannot continue")
X = D["stage5"]; N = len(X); Xv = X.to_numpy(float)
for k in ("stage1", "stage2", "stage3", "stage4"):
    if k in D: chk(len(D[k]) == N, f"{k} rows equal stage5 rows", f"{len(D[k])} vs {N}")
chk(X.shape[1] == 12, "stage5 has 12 columns", str(X.shape[1]))
chk(int(X.isna().sum().sum()) == 0, "stage5 has no NaN")
bad = [c for c in X.columns if "ArrivalMonth" not in c and (abs(X[c].mean()) > 0.01 or abs(X[c].std() - 1) > 0.02)]
chk(not bad, "stage5 columns are z-scored (mean 0, std 1)", str(bad))
FORB = ["IsCanceled", "ADR", "ReservationStatus", "AssignedRoomType", "BookingChanges", "DepositType", "RequiredCarParking", "TotalOfSpecialRequests"]
leak = [c for c in X.columns if any(f.lower() in c.lower() for f in FORB)]
chk(not leak, "no leakage column names in the clustering matrix", str(leak))
chk((P / "cleaning_log.csv").exists(), "cleaning_log.csv exists")

# ---- 2. clustering outputs vs comparison.csv ----
cmp_path = ROOT / "clustering" / "comparison.csv"
chk(cmp_path.exists(), "clustering/comparison.csv exists")
cmp = pd.read_csv(cmp_path)
info("comparison.csv columns", ", ".join(cmp.columns))
for m in ["kmeans", "gmm", "agglomerative", "dbscan"]:
    for t in ["baseline", "tuned"]:
        tag = f"{m}/{t}"; base = ROOT / "clustering" / m / t; out = base / "outputs"
        chk((base / "run.py").exists(), f"{tag} run.py exists")
        lp, rp = out / "labels.npy", out / "results.txt"
        if not lp.exists(): chk(False, f"{tag} labels.npy", "missing"); continue
        lab = np.load(lp); chk(len(lab) == N, f"{tag} labels length", f"{len(lab)} vs {N}")
        row = cmp[(cmp.model == m) & (cmp.variant == t)]
        if row.empty: chk(False, f"{tag} row in comparison.csv", "missing"); continue
        r0 = row.iloc[0]; keep = lab != -1; k = len(np.unique(lab[keep])); noise = (~keep).mean() * 100
        chk(k == int(r0["clusters"]), f"{tag} cluster count matches table", f"labels {k} vs table {int(r0['clusters'])}")
        chk(abs(noise - float(r0["noise_pct"])) < 0.05, f"{tag} noise % matches table", f"labels {noise:.2f} vs table {r0['noise_pct']}")
        if k >= 2:
            db = davies_bouldin_score(Xv[keep], lab[keep]); ch = calinski_harabasz_score(Xv[keep], lab[keep])
            chk(abs(db - r0["davies_bouldin"]) <= 0.01 * abs(r0["davies_bouldin"]) + 1e-3, f"{tag} Davies-Bouldin matches labels", f"recomputed {db:.4f} vs table {r0['davies_bouldin']}")
            chk(abs(ch - r0["calinski_harabasz"]) <= 0.01 * abs(r0["calinski_harabasz"]) + 1, f"{tag} Calinski-Harabasz matches labels", f"recomputed {ch:.1f} vs table {r0['calinski_harabasz']}")
        if rp.exists():
            txt = rp.read_text(encoding="utf-8", errors="ignore")
            nums = [float(x) for x in re.findall(r"silhouette[^0-9\-]{0,40}(-?\d+\.\d+)", txt, flags=re.I)]
            soft(any(abs(x - r0["silhouette_common"]) <= 0.0006 for x in nums), f"{tag} table silhouette appears in results.txt", f"table {r0['silhouette_common']} | results.txt {nums[:6]}")
            mm = re.search(r"cluster sizes:\s*(\{.*\})", txt)
            if mm:
                try:
                    sz = {int(a): int(b) for a, b in json.loads(mm.group(1)).items()}
                    real = {int(a): int(b) for a, b in zip(*np.unique(lab, return_counts=True))}
                    chk(sz == real, f"{tag} cluster sizes in results.txt equal labels.npy")
                except Exception as e: soft(False, f"{tag} cluster sizes parse", str(e))
        else: chk(False, f"{tag} results.txt", "missing")
pp = ROOT / "clustering" / "kmeans" / "tuned" / "outputs" / "cluster_profiles.csv"
if pp.exists():
    pr = pd.read_csv(pp)
    if "share_pct" in pr.columns: chk(abs(pr["share_pct"].sum() - 100) < 0.3, "kmeans tuned profile shares sum to 100", f"{pr['share_pct'].sum():.2f}")
else: chk(False, "kmeans tuned cluster_profiles.csv", "missing")

# ---- 3. docs mention only files that exist ----
ALLN, ALLP = set(), set(); BIG = []; NB = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in (".git", ".venv", "node_modules", "__pycache__")]
    for f in fn:
        ALLN.add(f); ALLP.add(str(Path(dp, f).relative_to(ROOT)).replace("\\", "/"))
        if os.path.getsize(Path(dp, f)) > 50e6: BIG.append(f)
        if f.endswith(".ipynb"): NB.append(f)
for md in [ROOT / "README.md", ROOT / "clustering" / "README.md", P / "README.md", ROOT / "outputs" / "README.md"]:
    if not md.exists(): continue
    t = md.read_text(encoding="utf-8", errors="ignore"); miss = []
    for tok in sorted(set(re.findall(r"`([^`\n]+)`", t))):
        tok = tok.strip().strip(".,:;")
        if re.search(r"[*<>{}|=\s]", tok) or not re.search(r"\.(csv|py|png|json|md|txt|npy|yaml|pdf)$", tok): continue
        if tok.replace("\\", "/") in ALLP or (md.parent / tok).exists() or (P / tok).exists() or Path(tok).name in ALLN: continue
        miss.append(tok)
    soft(not miss, f"{md.relative_to(ROOT)} mentions only existing files", "missing: " + ", ".join(miss[:12]))
rd = (ROOT / "README.md").read_text(encoding="utf-8", errors="ignore") if (ROOT / "README.md").exists() else ""
soft("github.com" in rd.lower(), "README has the GitHub repository URL")
soft(bool(re.search(r"\b[0-9a-f]{7,40}\b", rd)) and "commit" in rd.lower(), "README has a commit SHA")

# ---- 4. hygiene ----
soft(not BIG, "no file over 50 MB (GitHub limit is 100 MB)", str(BIG))
soft(not NB, "no notebooks in the repo", str(NB))
req = (ROOT / "requirements.txt").read_text(encoding="utf-8", errors="ignore").lower() if (ROOT / "requirements.txt").exists() else ""
for lib in ["pandas", "numpy", "scikit-learn", "matplotlib", "scipy"]:
    soft(lib in req, f"requirements.txt lists {lib}")
for f in ["fix_test.py", "improve_test.py"]:
    info(f"experiments/rejected_ideas/{f} exists", str((ROOT / "experiments" / "rejected_ideas" / f).exists()))

# ---- 5. git ----
tracked = git("ls-files").splitlines()
chk(not [f for f in tracked if f.endswith((".joblib", ".pkl"))], "no model files are tracked by git", str([f for f in tracked if f.endswith((".joblib", ".pkl"))][:5]))
authors = git("shortlog", "-sne", "HEAD")
soft("fedo" in authors.lower(), "your own account has commits", authors.replace("\n", " | "))
G = ["", "GIT BRANCH: " + git("branch", "--show-current"), "GIT REMOTE:\n" + git("remote", "-v"),
     "GIT STATUS (uncommitted):\n" + (git("status", "--short") or "(clean)"),
     "GIT AHEAD/BEHIND upstream (behind<TAB>ahead): " + git("rev-list", "--left-right", "--count", "@{u}...HEAD"),
     "TRACKED FILES: %d | experiments tracked: %s" % (len(tracked), [f for f in tracked if f.startswith("experiments/")][:12]),
     "GIT AUTHORS:\n" + authors, "GIT LOG (date | author | message):\n" + git("log", "--format=%ad | %an | %s", "--date=short", "-n", "60")]
lines = [f"[{a}] {b}" + (f"  -> {c}" if c else "") for a, b, c in res]
cnt = {k: sum(1 for r in res if r[0] == k) for k in ("PASS", "FAIL", "WARN", "INFO")}
rep = "\n".join(lines) + "\n\nSUMMARY: " + ", ".join(f"{k} {v}" for k, v in cnt.items()) + "\n" + "\n".join(G)
(OUT / "audit_report.txt").write_text(rep, encoding="utf-8"); print(rep); print("\nsaved to", OUT / "audit_report.txt")
