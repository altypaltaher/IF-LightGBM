"""Out-of-fold predicted probabilities for the first 5-fold repeat of the main 3x5 CV (folds 0-4), tuned settings, five models."""
import sys, os, json, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
ds = sys.argv[1]; MODELS = ["IF-LightGBM (proposed)", "LightGBM (raw)", "XGBoost (raw)", "GBM (raw)", "AdaBoost (raw)"]
if ds == "CIC-Darknet2020":
    sys.argv = ["confirm.py", "darknet", "none"]; src = open("1_experiments/confirm.py").read()
    src = src[:src.index("# ---------------------------------------------------------------- stages")]; ns = {}; exec(src, ns)
    X, y = ns["X"], ns["y"]; tuned = json.load(open("results/confirm_darknet/tuned_hyperparameters.json"))
    build = lambda n, A, ya, B: ns["build"](n, A, B)
else:
    sc, path, t, tj = {"CICAndMal2017": ("1_experiments/run_cic6.py", "data/CICANDMAL2017-75000-malware-7500 Goodware.csv", "Label1", "results/cic6/IFE_results/CICAndMal2017_tuned_hyperparameters.json"),
                       "CIC-AAGM2017": ("1_experiments/run_aagm.py", "data/aagm2017_sample20k.csv", "y", "results/aagm2/IFE_results/CIC-AAGM2017_tuned_hyperparameters.json")}[ds]
    src = open(sc).read(); src = src[:src.index("# ---------------------------------------------------------------- main")]; ns = {}; exec(src, ns)
    X, y = ns["load"](path, t, None, False); tuned = json.load(open(tj)); build = ns["build"]
K = int(y.nunique()); OUT = f"results/v20/oof/oof_{ds}.npz"
P = {m: np.full((len(y), K), np.nan) for m in MODELS}
if os.path.exists(OUT):
    z = np.load(OUT, allow_pickle=True); P = z["P"].item()
for f, (a, v) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y)):
    if f >= 5: break
    for m in MODELS:
        if not np.isnan(P[m][v]).any(): continue
        A, B = build(m, X.iloc[a], y.iloc[a].to_numpy(), X.iloc[v])
        P[m][v] = ns["make"](m, tuned[m]["params"], tuned[m]["n_estimators"]).fit(A, y.iloc[a]).predict_proba(B)
        np.savez(OUT, P=np.array(P, dtype=object), y=y.to_numpy())
    print(ds, "fold", f, flush=True)
print(ds, "done", flush=True)
