"""Uniform ablation with the tuned IF-LightGBM settings on the main 3x5 CV splits:
x, [x,nu], [x,pi] (255 bins) and x with max_bin=4095.  Usage: ablate20.py DATASET"""
import sys, os, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
ds = sys.argv[1]; OUT = f"results/v20/m/ablation_{ds}.csv"
if ds == "CIC-Darknet2020":
    s = pd.read_parquet("results/confirm_darknet/sample.parquet"); y = s.pop("y").astype(int); X = s.astype(float)
    t = json.load(open("results/confirm_darknet/tuned_hyperparameters.json"))["IF-LightGBM (proposed)"]
    reps = {"x": ([], 255), "x+ν": (["nu"], 255), "x+π": (["pi"], 255)}
elif ds == "CIC-AAGM2017":
    src = open("1_experiments/run_aagm.py").read(); src = src[:src.index("# ---------------------------------------------------------------- main")]
    ns = {}; exec(src, ns); X, y = ns["load"]("data/aagm2017_sample20k.csv", "y", None, False)
    t = json.load(open("results/aagm2/IFE_results/CIC-AAGM2017_tuned_hyperparameters.json"))["IF-LightGBM (proposed)"]
    reps = {"x (max_bin 4095)": ([], 4095)}
K = int(y.nunique()); multi = K > 2
def aug(A, B, parts):
    sc = MinMaxScaler().fit(A); out = []
    for M in (A, B):
        Z = np.clip(sc.transform(M), 0, 1); b = [M]
        if "nu" in parts: b.append((1 - Z) ** 2)
        if "pi" in parts: b.append(Z * (1 - Z))
        out.append(np.hstack(b))
    return out
rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
done = {(r["fold"], r["rep"]) for r in rows}
for f, (a, v) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y)):
    for lab, (parts, mb) in reps.items():
        if (f, lab) in done: continue
        A, B = aug(X.iloc[a].to_numpy(float), X.iloc[v].to_numpy(float), parts)
        m = lgb.LGBMClassifier(n_estimators=t["n_estimators"], bagging_freq=1, random_state=42, n_jobs=2, verbosity=-1, max_bin=mb, **t["params"])
        P = m.fit(A, y.iloc[a]).predict_proba(B); yv = y.iloc[v].to_numpy(); pr = P.argmax(1)
        rows.append(dict(dataset=ds, fold=f, rep=lab, n_train=len(a), n_test=len(v), accuracy=accuracy_score(yv, pr),
                         f1=f1_score(yv, pr, average="macro" if multi else "binary"), mcc=matthews_corrcoef(yv, pr),
                         roc_auc=roc_auc_score(yv, P, multi_class="ovr") if multi else roc_auc_score(yv, P[:, 1])))
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(ds, "fold", f, flush=True)
