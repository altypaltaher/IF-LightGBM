"""Confirmatory runs on CIC-AAGM2017 under the paper's protocol.
  task = multi : benign / adware / general malware, 30k flows (all unique GM flows kept), tuned on the full hold-out train set
  task = bin100k : binary, 100k balanced flows, tuning on a stratified 20k subset of the hold-out train set (same for every model)
Protocol (identical for every model): Optuna TPE, 20 trials, inner 3-fold CV, early stopping for LightGBM/XGBoost,
hold-out 80/20 (McNemar), 3x5 RepeatedStratifiedKFold seed 42 with the tuned settings, corrected resampled t-test,
Wilcoxon, Cliff's delta, Holm over all contrasts x metrics.
Extra control: raw LightGBM with max_bin = 4095 (tuned with the same budget) -> tests whether any gain is only binning.
Fully resumable: Optuna studies in SQLite, CV folds in CSV.  Usage: python confirm.py TASK [STAGE]   STAGE = tune|holdout|cv|stats|all"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import binomtest, wilcoxon
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, precision_score,
                             recall_score, log_loss)
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

TASK = sys.argv[1]; STAGE = sys.argv[2] if len(sys.argv) > 2 else "all"
SEED, N_TRIALS, N_JOBS, TEST_SIZE = 42, 20, 2, 0.20
CV_FOLDS, CV_REPEATS = 5, 3
OUT = Path(f"results/confirm_{TASK}"); OUT.mkdir(exist_ok=True)
MODELS = ["LightGBM (raw)", "IF-LightGBM (proposed)", "XGBoost (raw)", "AdaBoost (raw)", "GBM (raw)",
          "LightGBM (raw, 4095 bins)"]

# ---------------------------------------------------------------- data (same rules as build_sample.py)
def data():
    cache = OUT / "sample.parquet"
    if cache.exists():
        s = pd.read_parquet(cache); return s.drop(columns="y"), s["y"]
    if TASK == "darknet":   # CIC-Darknet2020, 8 application classes, <=5000 unique flows per class (same as screen)
        d = pd.read_csv("data/Darknet.CSV", low_memory=False); d.columns = d.columns.str.strip()
        y = d.pop("Label.1").str.strip().str.lower().str.replace("_", "-")
        d = d.drop(columns=["Label", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"])
        X = d.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        X["__y"] = y.values; X = X.drop_duplicates(); X = X[~X.drop(columns="__y").duplicated(keep=False)]
        cls = sorted(X.__y.unique())
        S = pd.concat([X[X.__y == c].sample(min((X.__y == c).sum(), 5000), random_state=SEED) for c in cls]).reset_index(drop=True)
        y = S.pop("__y").map({c: i for i, c in enumerate(cls)}).astype(int); S = S.loc[:, S.nunique() > 1].astype(float)
        json.dump(cls, open(OUT / "classes.json", "w"))
        pd.concat([S, y.rename("y")], axis=1).to_parquet(cache); return S, y
    d = pd.read_csv("data/TotalFeatures-ISCXFlowMeter.csv", low_memory=False); d.columns = d.columns.str.strip()
    d = d.drop_duplicates(); feat = [c for c in d.columns if c != "calss"]
    d = d[~d.duplicated(subset=feat, keep=False)]
    lab = d["calss"].str.strip().str.lower()
    if TASK == "multi":
        d["y"] = lab.map({"benign": 0, "asware": 1, "generalmalware": 2})
        gm = d[d.y == 2]; k = (30000 - len(gm)) // 2
        s = pd.concat([d[d.y == 0].sample(k, random_state=SEED), d[d.y == 1].sample(k, random_state=SEED), gm])
    else:
        d["y"] = (lab != "benign").astype(int)
        s = pd.concat([d[d.y == c].sample(50000, random_state=SEED) for c in (0, 1)])
    X = s[feat].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    X = X.loc[:, X.nunique() > 1].reset_index(drop=True); y = s["y"].astype(int).reset_index(drop=True)
    pd.concat([X, y.rename("y")], axis=1).to_parquet(cache); return X, y

X, y = data(); K = int(y.nunique()); MULTI = K > 2
print(f"[{TASK}] rows={len(X):,} features={X.shape[1]} classes={np.bincount(y).tolist()}", flush=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

def build(name, A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    if not name.startswith("IF-"):
        return A, B
    sc = MinMaxScaler().fit(A); out = []
    for M in (A, B):
        Z = np.clip(sc.transform(M), 0, 1); out.append(np.hstack([M, (1 - Z) ** 2, Z * (1 - Z)]))   # x, nu, pi
    return out

# ---------------------------------------------------------------- models (search spaces identical to run_aagm.py)
def suggest(trial, name):
    if name.startswith("GBM"):
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 400, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True)}
    if "AdaBoost" in name:
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 400, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 1, 4)}
    if "XGBoost" in name:
        return {"max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    return {"num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True)}

def make(name, params, n_est):
    p = dict(params); p.pop("n_estimators", None)
    if name.startswith("GBM"):
        return GradientBoostingClassifier(n_estimators=n_est, random_state=SEED, **p)
    if "AdaBoost" in name:
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=p.pop("max_depth"), random_state=SEED),
                                  n_estimators=n_est, learning_rate=p["learning_rate"], random_state=SEED)
    if "XGBoost" in name:
        obj = dict(objective="multi:softprob", eval_metric="mlogloss") if MULTI else dict(objective="binary:logistic", eval_metric="logloss")
        return XGBClassifier(n_estimators=n_est, random_state=SEED, n_jobs=N_JOBS, **obj, **p)
    extra = {"max_bin": 4095} if "4095" in name else {}
    return lgb.LGBMClassifier(n_estimators=n_est, bagging_freq=1, random_state=SEED, n_jobs=N_JOBS, verbosity=-1, **extra, **p)

def fit_es(name, params, Xa, ya, Xv, yv):
    if "AdaBoost" in name or name.startswith("GBM"):
        return make(name, params, params["n_estimators"]).fit(Xa, ya), params["n_estimators"]
    m = make(name, params, 2000)
    if "XGBoost" in name:
        m.set_params(early_stopping_rounds=50); m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        return m, int(m.best_iteration) + 1
    m.fit(Xa, ya, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, int(m.best_iteration_ or 2000)

def auc(yv, P):
    return roc_auc_score(yv, P, multi_class="ovr", average="macro") if MULTI else roc_auc_score(yv, P[:, 1])

def tune(name):
    Xt, yt = X_tr, y_tr
    if TASK == "bin100k":   # same 20k tuning subset for every model
        Xt, _, yt, _ = train_test_split(X_tr, y_tr, train_size=20000, random_state=SEED, stratify=y_tr)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xt, yt))
    cache = [tuple(build(name, Xt.iloc[a], Xt.iloc[v])) + (yt.iloc[a].to_numpy(), yt.iloc[v].to_numpy()) for a, v in folds]
    def objective(trial):
        p = suggest(trial, name); s, it = [], []
        for Fa, Fv, ya, yv in cache:
            m, n = fit_es(name, p, Fa, ya, Fv, yv); s.append(auc(yv, m.predict_proba(Fv))); it.append(n)
        trial.set_user_attr("n_estimators", int(np.mean(it))); return float(np.mean(s))
    st = optuna.create_study(study_name=name, storage=f"sqlite:///{OUT}/optuna.db", load_if_exists=True,
                             direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    done = len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if done < N_TRIALS:
        st.optimize(objective, n_trials=N_TRIALS - done)
    b = st.best_trial; return {"params": b.params, "n_estimators": b.user_attrs["n_estimators"], "inner_cv_auc": b.value}

def metrics(yv, P):
    pred = P.argmax(1); avg = "macro" if MULTI else "binary"
    return {"accuracy": accuracy_score(yv, pred), "f1": f1_score(yv, pred, average=avg),
            "precision": precision_score(yv, pred, average=avg, zero_division=0),
            "recall": recall_score(yv, pred, average=avg), "mcc": matthews_corrcoef(yv, pred),
            "roc_auc": auc(yv, P), "log_loss": log_loss(yv, P, labels=list(range(K)))}

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1)
    return adj

def corrected_t(d, n_tr, n_te):
    d = np.asarray(d); v = d.var(ddof=1)
    if v == 0: return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / len(d) + n_te / n_tr) * v); return t, 2 * stats.t.sf(abs(t), len(d) - 1)

def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (sum((x > b).sum() for x in a) - sum((x < b).sum() for x in a)) / (len(a) * len(b))

# ---------------------------------------------------------------- stages
tfile = OUT / "tuned_hyperparameters.json"
tuned = json.load(open(tfile)) if tfile.exists() else {}
if STAGE in ("tune", "all"):
    for name in MODELS:
        if name in tuned: continue
        t0 = time.time(); tuned[name] = tune(name); json.dump(tuned, open(tfile, "w"), indent=2)
        print(f"  tuned {name:28s} inner AUC={tuned[name]['inner_cv_auc']:.4f} n_est={tuned[name]['n_estimators']} ({time.time()-t0:.0f}s)", flush=True)

if STAGE in ("holdout", "all"):
    hf = OUT / "holdout_metrics.csv"
    if not hf.exists():
        preds, rows = {}, []
        for name in MODELS:
            A, B = build(name, X_tr, X_te)
            P = make(name, tuned[name]["params"], tuned[name]["n_estimators"]).fit(A, y_tr).predict_proba(B)
            preds[name] = P.argmax(1); rows.append({"model": name, **metrics(y_te.to_numpy(), P)})
            print("  holdout", name, {k: round(v, 4) for k, v in rows[-1].items() if k != "model"}, flush=True)
        pd.DataFrame(rows).to_csv(hf, index=False)
        yt = y_te.to_numpy(); ref = preds["IF-LightGBM (proposed)"]; tests = []
        for name in MODELS:
            if name.startswith("IF-"): continue
            ca, cb = ref == yt, preds[name] == yt; b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
            tests.append({"comparison": f"IF-LightGBM vs {name}", "IF_right_other_wrong": b, "IF_wrong_other_right": c,
                          "mcnemar_p": 1.0 if b + c == 0 else binomtest(min(b, c), b + c, .5).pvalue})
        t = pd.DataFrame(tests); t["mcnemar_p_holm"] = holm(t.mcnemar_p); t.to_csv(OUT / "holdout_mcnemar.csv", index=False)
        print(t.to_string(index=False), flush=True)

if STAGE in ("cv", "all"):
    cf = OUT / "cv_fold_scores.csv"
    rows = pd.read_csv(cf).to_dict("records") if cf.exists() else []
    done = {(r["fold"], r["model"]) for r in rows}; t0 = time.time()
    for f, (a, v) in enumerate(RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=SEED).split(X, y)):
        for name in MODELS:
            if (f, name) in done: continue
            A, B = build(name, X.iloc[a], X.iloc[v])
            P = make(name, tuned[name]["params"], tuned[name]["n_estimators"]).fit(A, y.iloc[a]).predict_proba(B)
            rows.append({"fold": f, "model": name, "n_train": len(a), "n_test": len(v), **metrics(y.iloc[v].to_numpy(), P)})
            pd.DataFrame(rows).to_csv(cf, index=False)
        print(f"  cv fold {f+1}/15 ({time.time()-t0:.0f}s)", flush=True)

if STAGE in ("stats", "all", "cv"):
    cv = pd.read_csv(OUT / "cv_fold_scores.csv")
    if cv.groupby("model").fold.nunique().min() == CV_FOLDS * CV_REPEATS:
        print(cv.groupby("model")[["accuracy", "f1", "mcc", "roc_auc", "log_loss"]].agg(["mean", "std"]).round(4).to_string())
        ct = []
        for b_ in [m for m in MODELS if m != "IF-LightGBM (proposed)"]:
            for k in ["accuracy", "f1", "mcc", "roc_auc"]:
                pa = cv[cv.model == "IF-LightGBM (proposed)"].sort_values("fold")[k].to_numpy()
                pb = cv[cv.model == b_].sort_values("fold")[k].to_numpy()
                _, p = corrected_t(pa - pb, cv.n_train.iloc[0], cv.n_test.iloc[0])
                try: wp = wilcoxon(pa, pb).pvalue
                except ValueError: wp = 1.0
                ct.append({"contrast": f"IF-LightGBM vs {b_}", "metric": k, "mean_diff_pts": 100 * (pa - pb).mean(),
                           "wins": int((pa > pb).sum()), "losses": int((pa < pb).sum()), "corrected_t_p": p,
                           "wilcoxon_p": wp, "cliffs_delta": cliffs(pa, pb)})
        ct = pd.DataFrame(ct); ct["corrected_t_p_holm"] = holm(ct.corrected_t_p); ct.to_csv(OUT / "cv_contrasts.csv", index=False)
        print(ct.to_string(index=False, float_format=lambda v: f"{v:.4g}"), flush=True)
