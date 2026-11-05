"""Nested CV for the key contrast (IF-LightGBM vs LightGBM): hyperparameters re-tuned inside every outer training fold
(same Optuna budget: 20 TPE trials, inner 3-fold CV), outer 5-fold stratified CV. Usage: python nested18.py <dataset>"""
import sys, json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
CFG = {"CICAndMal2017": ("1_experiments/run_cic6.py", "data/CICANDMAL2017-75000-malware-7500 Goodware.csv", "Label1"),
       "CIC-AAGM2017": ("1_experiments/run_aagm.py", "data/aagm2017_sample20k.csv", "y")}
ds = sys.argv[1]; script, path, target = CFG[ds]
src = open(script).read(); src = src[:src.index("# ---------------------------------------------------------------- main")]
src = src.replace("N_JOBS = 2", "N_JOBS = 1")
ns = {}; exec(src, ns)
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = ns["load"](path, target, None, False)
import os
outp = f"results/exp18/nested_{ds}.csv"
rows = pd.read_csv(outp).to_dict("records") if os.path.exists(outp) else []
done = {(r["fold"], r["model"]) for r in rows}; t0 = time.time()
for f, (a, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    Xa, ya, Xv, yv = X.iloc[a].reset_index(drop=True), y.iloc[a].reset_index(drop=True), X.iloc[v], y.iloc[v].to_numpy()
    for n in ["LightGBM (raw)", "IF-LightGBM (proposed)"]:
        if (f, n) in done: continue
        params, n_est, inner = ns["tune"](n, Xa, ya)
        Fa, Fv = ns["build"](n, Xa, ya.to_numpy(), Xv)
        p = ns["make"](n, params, n_est).fit(Fa, ya).predict_proba(Fv)[:, 1]
        rows.append({"dataset": ds, "fold": f, "model": n, "n_train": len(a), "n_test": len(v), "inner_auc": inner,
                     "n_estimators": n_est, **{k: v_ for k, v_ in params.items()}, **ns["metrics"](yv, p)})
        pd.DataFrame(rows).to_csv(outp, index=False)
        print(ds, f, n, f"acc={rows[-1]['accuracy']:.4f} auc={rows[-1]['roc_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"results/exp18/nested_{ds}.csv", index=False)
