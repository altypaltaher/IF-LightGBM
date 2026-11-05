"""v18 robustness experiments (all with the tuned settings of Table 3; same 3x5 CV splits as the main study):
 A) runtime, TPR at FPR = 1% and SHAP dependence of the top hesitation feature (hold-out set);
 B) representation ablation for LightGBM: x, x+nu, x+pi, x+nu+pi (IF-LightGBM settings for all);
 C) CICAndMal2017 without Source Port and Destination Port (LightGBM vs IF-LightGBM, XGBoost vs IF-XGBoost)."""
import json, time, platform, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve

CFG = {"CICAndMal2017": ("1_experiments/run_cic6.py", "data/CICANDMAL2017-75000-malware-7500 Goodware.csv", "Label1",
                         "results/cic6/IFE_results/CICAndMal2017_tuned_hyperparameters.json"),
       "CIC-AAGM2017": ("1_experiments/run_aagm.py", "data/aagm2017_sample20k.csv", "y",
                        "results/aagm2/IFE_results/CIC-AAGM2017_tuned_hyperparameters.json")}
OUT = "results/exp18/"; import os; os.makedirs(OUT, exist_ok=True)
P = "IF-LightGBM (proposed)"

def aug(Atr, Ate, parts):
    sc = MinMaxScaler().fit(Atr); out = []
    for A in (Atr, Ate):
        Z = np.clip(sc.transform(A), 0, 1); blocks = [np.asarray(A, float)]
        if "nu" in parts: blocks.append((1 - Z) ** 2)
        if "pi" in parts: blocks.append(Z * (1 - Z))
        out.append(np.hstack(blocks))
    return out

def corrected(d, ratio):
    d = np.asarray(d); v = d.var(ddof=1)
    if v == 0: return 1.0
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * v); return 2 * stats.t.sf(abs(t), len(d) - 1)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1)
    return adj

rt_rows, abl_rows, port_rows = [], [], []
for ds, (script, path, target, tj) in CFG.items():
    src = open(script).read(); src = src[:src.index("# ---------------------------------------------------------------- main")]
    ns = {}; exec(src, ns); tuned = json.load(open(tj))
    X, y = ns["load"](path, target, None, False)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ytr, yte = ytr.to_numpy(), yte.to_numpy()
    # ---- A) runtime + TPR@1%FPR
    for n in ns["MODELS"]:
        t0 = time.perf_counter(); Fa, Fb = ns["build"](n, Xtr, ytr, Xte); t_aug = time.perf_counter() - t0
        m = ns["make"](n, tuned[n]["params"], tuned[n]["n_estimators"])
        t0 = time.perf_counter(); m.fit(Fa, ytr); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict_proba(Fb)[:, 1]; t_pred = time.perf_counter() - t0
        f, t, _ = roc_curve(yte, p)
        rt_rows.append({"dataset": ds, "model": n, "n_features": Fa.shape[1], "train_s": t_fit + t_aug,
                        "infer_us_per_flow": 1e6 * t_pred / len(yte), "tpr_at_fpr1": t[f <= 0.01].max(),
                        "tpr_at_fpr5": t[f <= 0.05].max()})
        if n == P: pm, Fte_P, Ftr_P = m, Fb, Fa
        print(ds, n, "done", flush=True)
    # SHAP dependence of top hesitation feature
    import shap
    rng = np.random.default_rng(42); idx = rng.choice(len(Fte_P), 800, replace=False)
    ex = shap.TreeExplainer(pm, data=Ftr_P[rng.choice(len(Ftr_P), 100, replace=False)], feature_perturbation="interventional", model_output="probability")
    sv = np.asarray(ex(Fte_P[idx]).values); sv = sv[:, :, -1] if sv.ndim == 3 else sv
    d0 = X.shape[1]; imp = np.abs(sv[:, 2 * d0:]).mean(0); j = int(np.argmax(imp)); name = X.columns[j]
    raw = Xte.iloc[idx, j].to_numpy(); sc = MinMaxScaler().fit(Xtr.iloc[:, [j]]); mu = np.clip(sc.transform(Xte.iloc[idx, [j]]), 0, 1).ravel()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    s = ax.scatter(mu, sv[:, 2 * d0 + j], c=sv[:, j] + sv[:, d0 + j], cmap="coolwarm", s=10)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set(xlabel=f"μ (scaled {name})", ylabel=f"SHAP value of π({name})", title=f"{ds}: π({name})")
    plt.colorbar(s, label="SHAP of x + ν (same feature)")
    fig.tight_layout(); fig.savefig(OUT + f"{ds}_pi_dependence.png", dpi=400, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(ds, "SHAP dependence:", name, flush=True)
    # ---- B) representation ablation and C) port removal, on the main CV splits
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y))
    pr = tuned[P]["params"]; ne = tuned[P]["n_estimators"]
    reps = {"x": [], "x+ν": ["nu"], "x+π": ["pi"], "x+ν+π (IF-LightGBM)": ["nu", "pi"]}
    for f, (a, v) in enumerate(splits):
        for lab, parts in reps.items():
            Fa, Fv = aug(X.iloc[a].to_numpy(), X.iloc[v].to_numpy(), parts)
            pp = ns["make"](P, pr, ne).fit(Fa, y.iloc[a]).predict_proba(Fv)[:, 1]
            abl_rows.append({"dataset": ds, "fold": f, "rep": lab, "n_train": len(a), "n_test": len(v), **ns["metrics"](y.iloc[v].to_numpy(), pp)})
        print(ds, "ablation fold", f, flush=True)
    if ds == "CICAndMal2017":
        Xp = X.drop(columns=[c for c in X.columns if c in ("Source Port", "Destination Port")])
        for f, (a, v) in enumerate(splits):
            for n in ["LightGBM (raw)", P, "XGBoost (raw)", "IF-XGBoost"]:
                Fa, Fv = ns["build"](n, Xp.iloc[a], y.iloc[a].to_numpy(), Xp.iloc[v])
                pp = ns["make"](n, tuned[n]["params"], tuned[n]["n_estimators"]).fit(Fa, y.iloc[a]).predict_proba(Fv)[:, 1]
                port_rows.append({"fold": f, "model": n, "n_train": len(a), "n_test": len(v), **ns["metrics"](y.iloc[v].to_numpy(), pp)})
            print("ports fold", f, flush=True)

pd.DataFrame(rt_rows).to_csv(OUT + "runtime_tpr.csv", index=False)
ab = pd.DataFrame(abl_rows); ab.to_csv(OUT + "ablation_folds.csv", index=False)
po = pd.DataFrame(port_rows); po.to_csv(OUT + "noports_folds.csv", index=False)
# summaries
rows = []
for ds, g in ab.groupby("dataset"):
    ratio = g.n_test.iloc[0] / g.n_train.iloc[0]; base = g[g.rep == "x"].sort_values("fold")
    tmp = []
    for lab in ["x+ν", "x+π", "x+ν+π (IF-LightGBM)"]:
        h = g[g.rep == lab].sort_values("fold")
        for k in ["accuracy", "roc_auc"]:
            d = h[k].to_numpy() - base[k].to_numpy()
            tmp.append({"dataset": ds, "rep": lab, "metric": k, "mean": h[k].mean(), "base_mean": base[k].mean(), "diff_pp": 100 * d.mean(),
                        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": corrected(d, ratio)})
    t = pd.DataFrame(tmp); t["p_holm"] = holm(t.p); rows.append(t)
abs_ = pd.concat(rows); abs_.to_csv(OUT + "ablation_summary.csv", index=False)
tmp = []; ratio = po.n_test.iloc[0] / po.n_train.iloc[0]
for a_, b_ in [(P, "LightGBM (raw)"), ("IF-XGBoost", "XGBoost (raw)"), (P, "XGBoost (raw)")]:
    for k in ["accuracy", "roc_auc"]:
        A = po[po.model == a_].sort_values("fold")[k].to_numpy(); Bv = po[po.model == b_].sort_values("fold")[k].to_numpy(); d = A - Bv
        tmp.append({"contrast": f"{a_} vs {b_}", "metric": k, "mean_a": A.mean(), "mean_b": Bv.mean(), "diff_pp": 100 * d.mean(),
                    "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": corrected(d, ratio)})
pt = pd.DataFrame(tmp); pt["p_holm"] = holm(pt.p); pt.to_csv(OUT + "noports_summary.csv", index=False)
po.groupby("model")[["accuracy", "f1", "mcc", "roc_auc"]].agg(["mean", "std"]).to_csv(OUT + "noports_cv_summary.csv")
import lightgbm, xgboost, sklearn, optuna, shap as _s
json.dump({"python": platform.python_version(), "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
           "optuna": optuna.__version__, "shap": _s.__version__, "numpy": np.__version__, "pandas": pd.__version__,
           "cpu": platform.processor() or platform.machine()}, open(OUT + "versions.json", "w"), indent=1)
pd.set_option("display.width", 220)
print(pd.DataFrame(rt_rows).round(4).to_string(index=False)); print(abs_.round(4).to_string(index=False)); print(pt.round(4).to_string(index=False))
print(po.groupby("model")[["accuracy", "roc_auc"]].mean().round(4))
