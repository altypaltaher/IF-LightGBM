# Tuned IF-LightGBM settings (subsampling as tuned) with max_bin=4095: raw x vs x+nu+pi on the 3x5 CV splits
"""Mechanism test on CICAndMal2017: repeat the representation ablation with column and row subsampling switched off
(feature_fraction = bagging_fraction = 1), all other IF-LightGBM settings unchanged, same 3x5 CV splits."""
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
src = open("1_experiments/run_cic6.py").read(); src = src[:src.index("# ---------------------------------------------------------------- main")].replace("N_JOBS = 2", "N_JOBS = 1")
ns = {}; exec(src, ns)
X, y = ns["load"]("data/CICANDMAL2017-75000-malware-7500 Goodware.csv", "Label1", None, False)
P = "IF-LightGBM (proposed)"; t = json.load(open("results/cic6/IFE_results/CICAndMal2017_tuned_hyperparameters.json"))[P]
pr = dict(t["params"], max_bin=4095)
def aug(Atr, Ate, parts):
    sc = MinMaxScaler().fit(Atr); out = []
    for A in (Atr, Ate):
        Z = np.clip(sc.transform(A), 0, 1); b = [np.asarray(A, float)]
        if "nu" in parts: b.append((1 - Z) ** 2)
        if "pi" in parts: b.append(Z * (1 - Z))
        if "dup" in parts: b.append(np.asarray(A, float))
        out.append(np.hstack(b))
    return out
reps = {"x": [], "x+ν+π": ["nu", "pi"]}
import os
FP = "results/exp18/finebin2_folds.csv"
rows = pd.read_csv(FP).to_dict("records") if os.path.exists(FP) else []
done = {r["fold"] for r in rows}
for f, (a, v) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y)):
    if f in done: continue
    for lab, parts in reps.items():
        Fa, Fv = aug(X.iloc[a].to_numpy(), X.iloc[v].to_numpy(), parts)
        p = ns["make"](P, pr, t["n_estimators"]).fit(Fa, y.iloc[a]).predict_proba(Fv)[:, 1]
        rows.append({"fold": f, "rep": lab, "n_train": len(a), "n_test": len(v), **ns["metrics"](y.iloc[v].to_numpy(), p)})
    pd.DataFrame(rows).to_csv(FP, index=False); print("fold", f, flush=True)
d = pd.DataFrame(rows)
out = []; base = d[d.rep == "x"].sort_values("fold"); ratio = 0.25
for lab in list(reps)[1:]:
    h = d[d.rep == lab].sort_values("fold")
    for k in ["accuracy", "roc_auc"]:
        dd = h[k].to_numpy() - base[k].to_numpy(); tt = dd.mean() / np.sqrt((1 / 15 + ratio) * dd.var(ddof=1)) if dd.var(ddof=1) > 0 else 0
        out.append({"rep": lab, "metric": k, "mean": h[k].mean(), "base": base[k].mean(), "diff_pp": 100 * dd.mean(), "wins": int((dd > 0).sum()),
                    "losses": int((dd < 0).sum()), "p": 2 * stats.t.sf(abs(tt), 14) if tt else 1.0})
o = pd.DataFrame(out); o.to_csv("results/exp18/finebin2_summary.csv", index=False); print(o.round(4).to_string(index=False))
