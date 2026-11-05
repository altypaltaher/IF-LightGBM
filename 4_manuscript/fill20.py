"""Fill v20 table placeholders and text tokens from saved result files (no hand-copied table numbers)."""
import json, os, numpy as np, pandas as pd
DRAFT = os.environ.get("DRAFT") == "1"
from scipy import stats

TUNED = {"CICAndMal2017": "results/cic6/IFE_results/CICAndMal2017_tuned_hyperparameters.json",
         "CIC-AAGM2017": "results/aagm2/IFE_results/CIC-AAGM2017_tuned_hyperparameters.json",
         "CIC-Darknet2020": "results/confirm_darknet/tuned_hyperparameters.json"}
ORDER = ["IF-LightGBM (proposed)", "LightGBM (raw)", "XGBoost (raw)", "GBM (raw)", "AdaBoost (raw)"]
SHORT = {"IF-LightGBM (proposed)": "IF-LightGBM", "LightGBM (raw)": "LightGBM", "XGBoost (raw)": "XGBoost", "GBM (raw)": "GBM", "AdaBoost (raw)": "AdaBoost"}
M = "results/v20/m/"

def pfmt(p):
    if not np.isfinite(p) or p <= 0: return "n/a"
    if p >= 0.9995: return "1.00"
    if p >= 0.001: return f"{p:.3f}"
    e = int(np.floor(np.log10(p))); m = round(p / 10 ** e, 1)
    if m >= 10: m, e = m / 10, e + 1
    return f"{m:.1f} × 10^{{−{-e}}}"
sg = lambda v: f"{v:+.2f}".replace("-", "−")

T3, T4, T5, T6 = [], [], [], []
S = pd.read_csv(M + "cv_summary_v20.csv", index_col=0); C = pd.read_csv(M + "cv_contrasts_v20.csv")
PR = pd.read_csv(M + "cv_prob_summary_v20.csv"); PC = pd.read_csv(M + "cv_prob_v20.csv")
for ds, tj in TUNED.items():
    tuned = json.load(open(tj))
    for i, m in enumerate(ORDER):
        p = tuned[m]["params"]; n = tuned[m]["n_estimators"]
        if "LightGBM" in m:
            s = (f"{p['num_leaves']} leaves; learning rate {p['learning_rate']:.4f}; min. samples per leaf {p['min_child_samples']}; "
                 f"feature fraction {p['feature_fraction']:.3f}; bagging fraction {p['bagging_fraction']:.3f}; λ_{{2}} {p['lambda_l2']:.3f}; {n} trees")
        elif "XGBoost" in m:
            s = (f"depth {p['max_depth']}; learning rate {p['learning_rate']:.4f}; subsample {p['subsample']:.3f}; column sample {p['colsample_bytree']:.3f}; "
                 f"min. child weight {p['min_child_weight']:.2f}; λ {p['reg_lambda']:.3f}; {n} trees")
        elif "AdaBoost" in m:
            s = f"{n} trees of depth {p['max_depth']}; learning rate {p['learning_rate']:.4f}"
        else:
            s = (f"{n} trees; depth {p['max_depth']}; learning rate {p['learning_rate']:.4f}; subsample {p['subsample']:.3f}; "
                 f"min. samples per leaf {p['min_samples_leaf']}")
        T3.append([ds if i == 0 else "", SHORT[m], s, f"{tuned[m]['inner_cv_auc']:.4f}"])
    d = S[S.dataset == ds].loc[ORDER]
    for i, m in enumerate(ORDER):
        row = [ds if i == 0 else "", f"**{SHORT[m]}**" if i == 0 else SHORT[m]]
        for c in ["accuracy", "f1", "mcc", "roc_auc"]:
            txt = f"{d.loc[m, c]:.4f} ± {d.loc[m, c + '_sd']:.4f}"
            row.append(f"**{txt}**" if round(d.loc[m, c], 4) == round(d[c].max(), 4) else txt)
        pm = PR[(PR.dataset == ds)].set_index("model")
        txt = f"{pm.loc[m, 'mean']:.4f} ± {pm.loc[m, 'sd']:.4f}"
        row.append(f"**{txt}**" if round(pm.loc[m, 'mean'], 4) == round(pm['mean'].min(), 4) else txt)
        T4.append(row)
    for i, b in enumerate(ORDER[1:]):
        a = C[(C.dataset == ds) & (C.baseline == b) & (C.metric == "accuracy")].iloc[0]
        u = C[(C.dataset == ds) & (C.baseline == b) & (C.metric == "roc_auc")].iloc[0]
        q = PC[(PC.dataset == ds) & (PC.baseline == b)].iloc[0]
        T5.append([ds if i == 0 else "", SHORT[b], f"{sg(a.diff_pp)} ({int(a.wins)}/{int(a.losses)})", pfmt(a.p_holm),
                   f"{sg(u.diff_pp)} ({int(u.wins)}/{int(u.losses)})", pfmt(u.p_holm),
                   f"{q['diff']:+.4f} ({int(q.better)}/{int(q.worse)})".replace("-", "−"), pfmt(q.p_holm), f"{u.cliffs:.2f}"])

# ---------- ablation (IF-LightGBM settings; same folds) ----------
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(run, 1)
    return adj
def corrected(d, ratio):
    v = d.var(ddof=1) if len(d) > 1 else np.nan
    if not np.isfinite(v): return np.nan
    if v == 0: return 1.0
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * v); return 2 * stats.t.sf(abs(t), len(d) - 1)
E = "results/exp18/"
ab = pd.read_csv(E + "ablation_folds.csv"); fb = pd.read_csv(E + "finebin2_folds.csv")
frames = []
for ds in ["CICAndMal2017", "CIC-AAGM2017"]:
    g = ab[ab.dataset == ds][["fold", "rep", "accuracy", "roc_auc"]].copy()
    g["rep"] = g.rep.replace({"x+ν+π (IF-LightGBM)": "x+ν+π"}); g["dataset"] = ds; frames.append(g)
dk = pd.read_csv(M + "ablation_CIC-Darknet2020.csv")[["dataset", "fold", "rep", "accuracy", "roc_auc"]]; frames.append(dk)
cvd = pd.read_csv("results/confirm_darknet/cv_fold_scores.csv"); cvd = cvd[cvd.model == "IF-LightGBM (proposed)"][["fold", "accuracy", "roc_auc"]].copy()
cvd["rep"] = "x+ν+π"; cvd["dataset"] = "CIC-Darknet2020"; frames.append(cvd)
A = pd.concat(frames, ignore_index=True)
REPS = [("x+ν", "[x, ν]"), ("x+π", "[x, π]"), ("x+ν+π", "[x, ν, π] (IF-LightGBM)")]
ABL = {}
for ds in ["CICAndMal2017", "CIC-AAGM2017", "CIC-Darknet2020"]:
    g = A[A.dataset == ds]; base = g[g.rep == "x"].sort_values("fold")
    if DRAFT: fl = sorted(set(g[g.rep == "x+π"].fold)); base = base[base.fold.isin(fl)]; g = g[g.fold.isin(fl)]
    else: assert len(base) == 15, ds
    rows = []
    for r, lab in REPS:
        h = g[g.rep == r].sort_values("fold"); assert (DRAFT or len(h) == 15) and (h.fold.values == base.fold.values).all(), (ds, r)
        for k in ["accuracy", "roc_auc"]:
            dd = h[k].to_numpy() - base[k].to_numpy()
            rows.append(dict(rep=r, lab=lab, metric=k, mean=h[k].mean(), diff=100 * dd.mean(), w=int((dd > 0).sum()), l=int((dd < 0).sum()), p=corrected(dd, 0.25)))
    t = pd.DataFrame(rows); t["ph"] = holm(t.p); ABL[ds] = t
    T6.append([ds, "x (raw features)", f"{base.accuracy.mean():.4f}", "—", "—", "—", "—", "—", "—"])
    for r, lab in REPS:
        a = t[(t.rep == r) & (t.metric == "accuracy")].iloc[0]; u = t[(t.rep == r) & (t.metric == "roc_auc")].iloc[0]
        T6.append(["", lab, f"{a['mean']:.4f}", sg(a['diff']), f"{a.w}/{a.l}", pfmt(a.ph), sg(u['diff']), f"{u.w}/{u.l}", pfmt(u.ph)])
pd.concat([v.assign(dataset=k) for k, v in ABL.items()]).to_csv(M + "ablation_summary_v20.csv", index=False)
for ds, t in ABL.items():
    print(ds); print(t.round(4).to_string(index=False))

if __name__ == "__main__":
    src = open("content20.js").read()
    for k, v in {"T3_ROWS": T3, "T4_ROWS": T4, "T5_ROWS": T5, "T6_ROWS": T6}.items():
        tok = f'"{{{{{k}}}}}"'; assert tok in src, k
        src = src.replace(tok, json.dumps(v, ensure_ascii=False))
    texts = json.load(open("texts20.json"))
    for k, v in texts.items():
        tok = "{{" + k + "}}"; assert tok in src, k
        src = src.replace(tok, v.replace("\\", "\\\\").replace('"', '\\"'))
    left = [l for l in src.splitlines() if "{{" in l and "s ∈ S" not in l]
    assert not left, left[:3]
    open("content20_filled.js", "w").write(src); print("filled")
