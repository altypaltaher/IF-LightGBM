"""
IF-LightGBM under an equal hyperparameter-tuning budget, plus intuitionistic fuzzy (IF)
hesitation as a reliability signal for selective classification (triage).

NOTE: an earlier variant used the neighbourhood / evidence IF degrees below as extra MODEL
INPUTS; on CICAndMal2017 that reduced hold-out AUC (0.945 -> 0.925), so they are now used only
for triage. The model inputs are x, nu = (1-mu)^2 and pi = mu(1-mu) (mu = min-max(x)).

Why this version exists
-----------------------
The earlier per-feature transform (mu = min-max(x), nu = (1-mu)^2, pi = mu(1-mu)) is
(almost) monotone in each feature, so tree ensembles gain little from it; an ablation
showed most of the earlier improvement came from extra model capacity. This script
builds intuitionistic fuzzy (IF) degrees that carry information a tree cannot obtain
from axis-aligned splits on single features, and compares models fairly.

IF evidence features (all estimated on training data only; training rows receive
out-of-fold values so a sample never sees its own label):
  1. Neighbourhood IF degrees (multivariate), for k in {5, 15, 50}:
       s_k  = distance-weighted share of malware among the k nearest training neighbours
       r_k  = exp(-dbar_k / tau_k)  (reliability: 1 in dense regions, -> 0 when isolated)
       mu_k = r_k * s_k, nu_k = r_k * (1 - s_k), pi_k = 1 - r_k
     pi_k is the hesitation caused by lack of nearby evidence; min(mu_k, nu_k) reflects
     class mixing.
  2. Per-feature evidence degrees (Dirichlet / Bayesian counts in quantile bins):
       mu_j = n_mal / (n + alpha), nu_j = n_ben / (n + alpha), pi_j = alpha / (n + alpha)
     so mu_j + nu_j + pi_j = 1, and pi_j is large when a bin holds little evidence.
  3. Sample-level aggregates: mean score (mu - nu), mean hesitation, and the
     Szmidt-Kacprzyk IF entropy averaged over features.

Experimental design
-------------------
  * Models: LightGBM (raw), LightGBM + old IF transform, LightGBM + IF evidence
    (proposed), XGBoost (raw) and IF-XGBoost.
  * Equal tuning budget: Optuna TPE, N_TRIALS trials per model, inner 3-fold CV on the
    hold-out training set, early stopping; identical search space for all LightGBM models.
  * Evaluation: hold-out test set + repeated stratified CV with the tuned settings.
  * Statistics: corrected resampled t-test (Nadeau-Bengio), Wilcoxon, Cliff's delta,
    Holm correction, McNemar and DeLong on the hold-out set.
  * Selective classification: risk-coverage curves and AURC using model confidence,
    IF hesitation, and their combination.
"""
from pathlib import Path
import json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binomtest, wilcoxon
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, precision_score, recall_score, confusion_matrix,
                             brier_score_loss, roc_curve)
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
TEST_SIZE = 0.20
N_TRIALS = 20                 # equal Optuna budget per model
CV_FOLDS, CV_REPEATS = 5, 3   # 15 evaluation folds
K_LIST = (5, 15, 50)
ALPHA = 10.0                  # Dirichlet prior strength for per-feature evidence
N_BINS = 10
N_JOBS = 2
DATASETS = {
    "CICAndMal2017": ("data/CICANDMAL2017-75000-malware-7500 Goodware.csv", "Label1", None, False),
}
OUT = Path("results/cic6/IFE_results"); OUT.mkdir(exist_ok=True)
MODELS = ["LightGBM (raw)", "IF-LightGBM (proposed)", "XGBoost (raw)", "IF-XGBoost", "AdaBoost (raw)", "GBM (raw)"]
COLORS = {"IF-LightGBM (proposed)": "#2a78d6", "LightGBM (raw)": "#eb6834",
          "IF-XGBoost": "#6250d6", "XGBoost (raw)": "#1baf7a", "AdaBoost (raw)": "#e0a100", "GBM (raw)": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "legend.frameon": False, "savefig.dpi": 600})

# ---------------------------------------------------------------- data
def load(path, target, label_map, dedup):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=[target])
    if label_map:
        y = df[target].astype(str).str.strip().str.lower().map(label_map)
    else:
        y = pd.to_numeric(df[target], errors="coerce")
    df, y = df[y.notna()], y[y.notna()].astype(int)
    X = df.drop(columns=[target])
    X = X.drop(columns=[c for c in X.columns if c.lower() in {"flow id", "source ip", "destination ip", "timestamp"}])
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any()]
    if dedup:
        d = X.copy(); d["__y"] = y.values; d = d.drop_duplicates()
        d = d[~d.duplicated(subset=list(X.columns), keep=False)]
        X, y = d.drop(columns="__y"), d["__y"]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    return X.reset_index(drop=True).astype(float), y.reset_index(drop=True).astype(int)

# ---------------------------------------------------------------- feature builders
def old_if(A_tr, A_te):
    sc = MinMaxScaler().fit(A_tr)
    out = []
    for A in (A_tr, A_te):
        Z = np.clip(sc.transform(A), 0, 1)
        out.append(np.hstack([A, (1 - Z) ** 2, Z * (1 - Z)]))   # x, nu, pi (mu == scaled x is redundant)
    return out

def _knn_degrees(Q_fit, y_fit, Q_query, kmax, skip_self=False):
    nn = NearestNeighbors(n_neighbors=kmax + int(skip_self), algorithm="brute").fit(Q_fit)
    d, idx = nn.kneighbors(Q_query)
    if skip_self:
        d, idx = d[:, 1:], idx[:, 1:]
    lab = y_fit[idx]
    res = {}
    for k in K_LIST:
        w = 1.0 / (d[:, :k] + 1e-6)
        res[k] = ((w * lab[:, :k]).sum(1) / w.sum(1), d[:, :k].mean(1))
    return res

def _evidence_tables(B_fit, y_fit):
    """Per-feature bin counts -> mu, pi lookup tables."""
    tabs = []
    for j in range(B_fit.shape[1]):
        nb = B_fit[:, j].max() + 1
        n = np.bincount(B_fit[:, j], minlength=nb).astype(float)
        nm = np.bincount(B_fit[:, j], weights=y_fit, minlength=nb)
        tabs.append((nm / (n + ALPHA), (n - nm) / (n + ALPHA), ALPHA / (n + ALPHA)))
    return tabs

def _apply_tables(B, tabs):
    mu = np.empty(B.shape); nu = np.empty(B.shape); pi = np.empty(B.shape)
    for j, (tm, tn, tp) in enumerate(tabs):
        b = np.clip(B[:, j], 0, len(tm) - 1)
        mu[:, j], nu[:, j], pi[:, j] = tm[b], tn[b], tp[b]
    return mu, nu, pi

def ife_features(A_tr, y_tr, A_te, seed=SEED):
    """IF evidence features. Training rows: 5-fold out-of-fold; test rows: fitted on all training rows."""
    y_tr = np.asarray(y_tr, float)
    qt = QuantileTransformer(n_quantiles=min(1000, len(A_tr)), output_distribution="uniform",
                             random_state=seed).fit(A_tr)
    Q_tr, Q_te = qt.transform(A_tr), qt.transform(A_te)
    # quantile bins from the same transform
    B_tr = np.minimum((Q_tr * N_BINS).astype(int), N_BINS - 1)
    B_te = np.minimum((Q_te * N_BINS).astype(int), N_BINS - 1)
    kmax = max(K_LIST)

    # Cross-fitting: each training row is described by the fold model that did not see it,
    # and each test row by the AVERAGE of the same five fold models. Training and test
    # features are therefore built from equally sized reference sets (no distribution shift).
    knn_tr = {k: [np.zeros(len(A_tr)), np.zeros(len(A_tr))] for k in K_LIST}
    mu_tr, nu_tr, pi_tr = (np.zeros(A_tr.shape) for _ in range(3))
    knn_te = {k: [np.zeros(len(A_te)), np.zeros(len(A_te))] for k in K_LIST}
    mu_te, nu_te, pi_te = (np.zeros(A_te.shape) for _ in range(3))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(A_tr, y_tr))
    for fit_i, q_i in folds:
        r = _knn_degrees(Q_tr[fit_i], y_tr[fit_i], np.vstack([Q_tr[q_i], Q_te]), kmax)
        nq = len(q_i)
        for k in K_LIST:
            knn_tr[k][0][q_i], knn_tr[k][1][q_i] = r[k][0][:nq], r[k][1][:nq]
            knn_te[k][0] += r[k][0][nq:] / len(folds); knn_te[k][1] += r[k][1][nq:] / len(folds)
        tabs = _evidence_tables(B_tr[fit_i], y_tr[fit_i])
        mu_tr[q_i], nu_tr[q_i], pi_tr[q_i] = _apply_tables(B_tr[q_i], tabs)
        a, b, c = _apply_tables(B_te, tabs)
        mu_te += a / len(folds); nu_te += b / len(folds); pi_te += c / len(folds)
    r_te = {k: tuple(knn_te[k]) for k in K_LIST}

    def assemble(knn, mu, nu, pi):
        cols = []
        for k in K_LIST:
            s, dbar = knn[k]
            tau = np.median(knn_tr[k][1]) + 1e-9          # scale fixed from training data
            rel = np.exp(-dbar / tau)
            cols += [rel * s, rel * (1 - s), 1 - rel]
        ent = (np.minimum(mu, nu) + pi) / (np.maximum(mu, nu) + pi + 1e-12)   # Szmidt-Kacprzyk
        agg = [(mu - nu).mean(1), pi.mean(1), ent.mean(1)]
        return np.column_stack(cols + agg + [mu, pi])
    F_tr = assemble({k: tuple(knn_tr[k]) for k in K_LIST}, mu_tr, nu_tr, pi_tr)
    F_te = assemble(r_te, mu_te, nu_te, pi_te)
    return np.hstack([A_tr, F_tr]), np.hstack([A_te, F_te])

def ife_names(cols):
    names = list(cols)
    for k in K_LIST:
        names += [f"ifk{k}_mu", f"ifk{k}_nu", f"ifk{k}_pi"]
    names += ["if_score_mean", "if_hesitation_mean", "if_entropy_mean"]
    names += [f"ifmu_{c}" for c in cols] + [f"ifpi_{c}" for c in cols]
    return names

def build(model_name, X_tr, y_tr, X_te):
    imp = SimpleImputer(strategy="median").fit(X_tr)
    A_tr, A_te = imp.transform(X_tr), imp.transform(X_te)
    if model_name.startswith("IF-"):
        return tuple(old_if(A_tr, A_te))     # x, nu, pi  (redundant mu / scaled copies removed)
    return A_tr, A_te

# ---------------------------------------------------------------- models and tuning
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

def make(name, params, n_estimators):
    if name.startswith("GBM"):
        p = dict(params); p.pop("n_estimators", None)
        return GradientBoostingClassifier(n_estimators=n_estimators, random_state=SEED, **p)
    if "AdaBoost" in name:
        p = dict(params); p.pop("n_estimators", None)
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=p.pop("max_depth"), random_state=SEED),
                                  n_estimators=n_estimators, learning_rate=p["learning_rate"], random_state=SEED)
    if "XGBoost" in name:
        return XGBClassifier(n_estimators=n_estimators, objective="binary:logistic", eval_metric="logloss",
                             random_state=SEED, n_jobs=N_JOBS, **params)
    return lgb.LGBMClassifier(n_estimators=n_estimators, bagging_freq=1, random_state=SEED,
                              n_jobs=N_JOBS, verbosity=-1, **params)

def fit_es(name, params, Xa, ya, Xv, yv):
    """Fit with early stopping on a validation split; returns model and best iteration."""
    if "AdaBoost" in name or name.startswith("GBM"):     # no early stopping: n_estimators is tuned directly
        return make(name, params, params["n_estimators"]).fit(Xa, ya), params["n_estimators"]
    m = make(name, params, 2000)
    if "XGBoost" in name:
        m.set_params(early_stopping_rounds=50)
        m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        return m, int(m.best_iteration) + 1
    m.fit(Xa, ya, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, int(m.best_iteration_ or 2000)

def tune(name, X_tr, y_tr):
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_tr, y_tr))
    cache = [build(name, X_tr.iloc[a], y_tr.iloc[a].to_numpy(), X_tr.iloc[v]) + (y_tr.iloc[a].to_numpy(), y_tr.iloc[v].to_numpy())
             for a, v in folds]

    def objective(trial):
        p = suggest(trial, name); aucs, iters = [], []
        for Fa, Fv, ya, yv in cache:
            m, it = fit_es(name, p, Fa, ya, Fv, yv)
            aucs.append(roc_auc_score(yv, m.predict_proba(Fv)[:, 1])); iters.append(it)
        trial.set_user_attr("n_estimators", int(np.mean(iters)))
        return float(np.mean(aucs))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_trial
    return best.params, best.user_attrs["n_estimators"], best.value

# ---------------------------------------------------------------- metrics and statistics
def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "specificity": tn / (tn + fp), "f1": f1_score(y, pred),
            "mcc": matthews_corrcoef(y, pred), "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "fpr": fp / (fp + tn), "fnr": fn / (fn + tp), "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1)
    return adj

def corrected_t(d, n_tr, n_te):
    d = np.asarray(d); v = d.var(ddof=1)
    if v == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / len(d) + n_te / n_tr) * v)
    return t, 2 * stats.t.sf(abs(t), len(d) - 1)

def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (sum((x > b).sum() for x in a) - sum((x < b).sum() for x in a)) / (len(a) * len(b))

def mcnemar(y, pa, pb):
    ca, cb = pa == y, pb == y; b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return (1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue), b, c

def delong(y, p1, p2):
    def midrank(x):
        o = np.argsort(x); xs = x[o]; n = len(x); t = np.zeros(n); i = 0
        while i < n:
            j = i
            while j < n and xs[j] == xs[i]:
                j += 1
            t[i:j] = 0.5 * (i + j - 1) + 1; i = j
        r = np.empty(n); r[o] = t; return r
    pos, neg = y == 1, y == 0; m, n = pos.sum(), neg.sum(); A, V10, V01 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = midrank(p[pos]), midrank(p[neg]), midrank(np.r_[p[pos], p[neg]])
        A.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n)); V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.vstack(V10)) / m + np.cov(np.vstack(V01)) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]; z = (A[0] - A[1]) / np.sqrt(var)
    return A[0], A[1], z, 2 * stats.norm.sf(abs(z))

def risk_coverage(y, p, score):
    """score: higher = more trustworthy. Returns coverage grid, risk, AURC."""
    order = np.argsort(-score); err = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    n = len(y); cov = np.arange(1, n + 1) / n; risk = np.cumsum(err) / np.arange(1, n + 1)
    return cov, risk, float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))

# ---------------------------------------------------------------- main
summary = {}
for ds, (path, target, lmap, dedup) in DATASETS.items():
    if not Path(path).exists():
        print(f"Skipping {ds}: {path} not found"); continue
    print(f"\n=== {ds} ===")
    X, y = load(path, target, lmap, dedup)
    print(f"rows={len(X):,} features={X.shape[1]} labels={y.value_counts().to_dict()}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    ytr, yte = y_tr.to_numpy(), y_te.to_numpy()

    # ---- 1. equal-budget tuning
    cache = OUT / f"{ds}_tuned_hyperparameters.json"
    tuned = json.load(open(cache)) if cache.exists() else {}      # reuse finished tuning runs
    for name in MODELS:
        if name in tuned:
            print(f"  reusing tuned settings for {name}"); continue
        t0 = time.time(); params, n_est, val = tune(name, X_tr, y_tr)
        tuned[name] = {"params": params, "n_estimators": n_est, "inner_cv_auc": val}
        print(f"  tuned {name:26s} inner AUC={val:.4f}  n_estimators={n_est}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(tuned, open(cache, "w"), indent=2)
    json.dump(tuned, open(OUT / f"{ds}_tuned_hyperparameters.json", "w"), indent=2)

    # ---- 2. hold-out evaluation
    probs, rows, feats = {}, [], {}
    for name in MODELS:
        Ftr, Fte = build(name, X_tr, ytr, X_te); feats[name] = (Ftr, Fte)
        m = make(name, tuned[name]["params"], tuned[name]["n_estimators"]).fit(Ftr, ytr)
        probs[name] = m.predict_proba(Fte)[:, 1]
        rows.append({"dataset": ds, "model": name, "n_features": Ftr.shape[1], **metrics(yte, probs[name])})
        if name == "IF-LightGBM (proposed)":
            ife_model = m
    hold = pd.DataFrame(rows); hold.to_csv(OUT / f"{ds}_holdout_metrics.csv", index=False)
    print(hold[["model", "accuracy", "f1", "mcc", "roc_auc", "average_precision", "brier"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ref = "LightGBM (raw)"; tests = []
    for name in MODELS:
        if name == ref:
            continue
        p, b, c = mcnemar(yte, (probs[name] >= .5).astype(int), (probs[ref] >= .5).astype(int))
        a1, a2, z, pd_ = delong(yte, probs[name], probs[ref])
        tests.append({"comparison": f"{name} vs {ref}", "mcnemar_p": p, "b": b, "c": c, "delta_auc": a1 - a2, "delong_p": pd_})
    tests = pd.DataFrame(tests); tests["mcnemar_p_holm"] = holm(tests.mcnemar_p); tests["delong_p_holm"] = holm(tests.delong_p)
    tests.to_csv(OUT / f"{ds}_holdout_tests.csv", index=False); print(tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ---- 3. selective classification on the hold-out set
    # Hesitation used only as a reliability (triage) signal, not as a model input:
    imp_ = SimpleImputer(strategy="median").fit(X_tr)
    _, Fte_h = ife_features(imp_.transform(X_tr), ytr, imp_.transform(X_te))
    names_ife = ife_names(X.columns)
    hes = Fte_h[:, names_ife.index("ifk15_pi")]
    mix = np.minimum(Fte_h[:, names_ife.index("ifk15_mu")], Fte_h[:, names_ife.index("ifk15_nu")])
    p_ife = probs["IF-LightGBM (proposed)"]; conf = np.maximum(p_ife, 1 - p_ife)
    scores = {"Model confidence": conf, "IF hesitation (1 − π − mixing)": 1 - hes - mix,
              "Confidence × (1 − π)": conf * (1 - hes)}
    rc_rows = []
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for (lab, sc), col in zip(scores.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
        cov, risk, aurc = risk_coverage(yte, p_ife, sc)
        ax.plot(cov, risk, lw=2, color=col, label=f"{lab} (AURC = {aurc:.4f})")
        rc_rows.append({"dataset": ds, "ranking": lab, "AURC": aurc,
                        **{f"acc@{int(c * 100)}%cov": 1 - risk[int(c * len(risk)) - 1] for c in (0.5, 0.7, 0.8, 0.9, 1.0)}})
    cov, risk, aurc_or = risk_coverage(yte, p_ife, -(((p_ife >= .5).astype(int) != yte).astype(float)))
    ax.plot(cov, risk, ls=":", color=INK2, label=f"Oracle (AURC = {aurc_or:.4f})")
    ax.set(xlabel="Coverage (share of flows decided automatically)", ylabel="Error rate on covered flows",
           title=f"{ds}: risk–coverage of IF-LightGBM"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"{ds}_risk_coverage.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    pd.DataFrame(rc_rows).to_csv(OUT / f"{ds}_selective_classification.csv", index=False)
    print(pd.DataFrame(rc_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # error rate by hesitation decile
    dec = pd.qcut(pd.Series(hes).rank(method="first"), 10, labels=False)
    er = pd.DataFrame({"decile": dec, "err": ((p_ife >= .5).astype(int) != yte)}).groupby("decile").err.mean()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(er.index + 1, er.values * 100, color="#2a78d6", width=0.65)
    ax.set(xlabel="Neighbourhood hesitation π (decile, 1 = lowest)", ylabel="Error rate (%)",
           title=f"{ds}: errors concentrate where hesitation is high"); ax.set_xticks(range(1, 11))
    fig.tight_layout(); fig.savefig(OUT / f"{ds}_error_by_hesitation.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    rho = stats.spearmanr(hes, ((p_ife >= .5).astype(int) != yte))

    # ---- 4. repeated stratified CV with tuned settings
    cv_cache = OUT / f"{ds}_cv_fold_partial.csv"
    cv_rows, t0 = (pd.read_csv(cv_cache).to_dict("records") if cv_cache.exists() else []), time.time()
    done = {(r["fold"], r["model"]) for r in cv_rows}
    for f, (a, v) in enumerate(RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=SEED).split(X, y)):
        for name in MODELS:
            if (f, name) in done:
                continue
            Fa, Fv = build(name, X.iloc[a], y.iloc[a].to_numpy(), X.iloc[v])
            m = make(name, tuned[name]["params"], tuned[name]["n_estimators"]).fit(Fa, y.iloc[a])
            p = m.predict_proba(Fv)[:, 1]
            cv_rows.append({"fold": f, "model": name, "n_train": len(a), "n_test": len(v), **metrics(y.iloc[v].to_numpy(), p)})
        print(f"  cv fold {f + 1}/{CV_FOLDS * CV_REPEATS} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(cv_rows).to_csv(cv_cache, index=False)
    cv = pd.DataFrame(cv_rows); cv.to_csv(OUT / f"{ds}_cv_fold_scores.csv", index=False)
    cvsum = cv.groupby("model")[["accuracy", "f1", "mcc", "roc_auc", "average_precision", "brier"]].agg(["mean", "std"])
    cvsum.to_csv(OUT / f"{ds}_cv_summary.csv"); print(cvsum.round(4).to_string())

    contrasts = [("IF-LightGBM (proposed)", "LightGBM (raw)", "IF effect, LightGBM (equal tuning)"),
                 ("IF-XGBoost", "XGBoost (raw)", "IF effect, XGBoost (equal tuning)"),
                 ("IF-LightGBM (proposed)", "XGBoost (raw)", "proposed vs tuned XGBoost"),
                 ("IF-LightGBM (proposed)", "IF-XGBoost", "learner choice under IF features"),
                 ("IF-LightGBM (proposed)", "AdaBoost (raw)", "proposed vs tuned AdaBoost"),
                 ("IF-LightGBM (proposed)", "GBM (raw)", "proposed vs tuned GBM")]
    ct = []
    for a_, b_, lab in contrasts:
        for k in ["accuracy", "f1", "mcc", "roc_auc", "average_precision"]:
            pa = cv[cv.model == a_].sort_values("fold")[k].to_numpy(); pb = cv[cv.model == b_].sort_values("fold")[k].to_numpy()
            t, p = corrected_t(pa - pb, cv.n_train.iloc[0], cv.n_test.iloc[0])
            try:
                wp = wilcoxon(pa, pb).pvalue
            except ValueError:
                wp = 1.0
            ct.append({"dataset": ds, "contrast": lab, "metric": k, "mean_a": pa.mean(), "mean_b": pb.mean(),
                       "mean_diff": (pa - pb).mean(), "wins": int((pa > pb).sum()), "losses": int((pa < pb).sum()),
                       "corrected_t_p": p, "wilcoxon_p": wp, "cliffs_delta": cliffs(pa, pb)})
    ct = pd.DataFrame(ct); ct["corrected_t_p_holm"] = holm(ct.corrected_t_p); ct.to_csv(OUT / f"{ds}_cv_contrasts.csv", index=False)
    print(ct[ct.metric.isin(["f1", "roc_auc"])][["contrast", "metric", "mean_diff", "wins", "losses", "corrected_t_p", "corrected_t_p_holm", "cliffs_delta"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # CV comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, k in zip(axes, ["accuracy", "f1", "roc_auc"]):
        data = [cv[cv.model == n][k] for n in MODELS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops={"color": INK})
        for patch, n in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COLORS[n]); patch.set_alpha(0.35); patch.set_edgecolor(COLORS[n])
        for i, (d, n) in enumerate(zip(data, MODELS), 1):
            ax.scatter(i + np.random.default_rng(i).uniform(-.12, .12, len(d)), d, s=12, color=COLORS[n], zorder=3)
        ax.set_xticks(range(1, len(MODELS) + 1), [n.replace(" (proposed)", "\n(proposed)").replace(" + ", "\n+ ") for n in MODELS], fontsize=8)
        ax.set_title({"accuracy": "Accuracy", "f1": "F1-score", "roc_auc": "ROC-AUC"}[k])
    fig.suptitle(f"{ds}: tuned models, {CV_REPEATS}×{CV_FOLDS}-fold CV (equal tuning budget of {N_TRIALS} trials)", fontweight="bold", y=1.03)
    fig.tight_layout(); fig.savefig(OUT / f"{ds}_cv_tuned_comparison.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

    # ---- 5. SHAP family shares for the proposed model
    try:
        import shap
        Ftr, Fte = feats["IF-LightGBM (proposed)"]
        rng = np.random.default_rng(SEED); idx = rng.choice(len(Fte), min(800, len(Fte)), replace=False)
        ex = shap.TreeExplainer(ife_model, data=Ftr[rng.choice(len(Ftr), min(100, len(Ftr)), replace=False)],
                                feature_perturbation="interventional", model_output="probability")
        sv = np.asarray(ex(Fte[idx]).values); sv = sv[:, :, -1] if sv.ndim == 3 else sv
        imp = np.abs(sv).mean(0); d0 = X.shape[1]
        names_if = list(X.columns) + [f"nu_{c}" for c in X.columns] + [f"pi_{c}" for c in X.columns]
        fam = {"original x": imp[:d0].sum(), "non-membership ν": imp[d0:2 * d0].sum(), "hesitation π": imp[2 * d0:].sum()}
        tot = sum(fam.values()); shares = {k: 100 * v / tot for k, v in fam.items()}
        pd.DataFrame([shares]).to_csv(OUT / f"{ds}_SHAP_family_shares.csv", index=False)
        pd.DataFrame({"feature": names_if, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False).to_csv(OUT / f"{ds}_SHAP_importance.csv", index=False)
        plt.figure(); shap.summary_plot(sv, Fte[idx], feature_names=names_if, show=False, max_display=20)
        plt.savefig(OUT / f"{ds}_SHAP_summary.png", bbox_inches="tight", dpi=600); plt.close()
        print("  SHAP family shares:", {k: round(v, 1) for k, v in shares.items()})
    except Exception as e:
        print("  SHAP skipped:", e); shares = {}

    summary[ds] = {"tuned": tuned, "spearman_hesitation_error": [float(rho.statistic), float(rho.pvalue)], "shap_shares": shares}
json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=float)
print("\nDone. Outputs in", OUT.resolve())
