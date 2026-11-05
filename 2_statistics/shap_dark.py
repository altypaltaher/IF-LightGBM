import json, numpy as np, pandas as pd, lightgbm as lgb, shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
s = pd.read_parquet("results/confirm_darknet/sample.parquet"); y = s.pop("y").astype(int); X = s.astype(float)
cls = json.load(open("results/confirm_darknet/classes.json"))
t = json.load(open("results/confirm_darknet/tuned_hyperparameters.json"))["IF-LightGBM (proposed)"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = MinMaxScaler().fit(Xtr)
def aug(M): Z = np.clip(sc.transform(M), 0, 1); return np.hstack([M.to_numpy(float), (1 - Z) ** 2, Z * (1 - Z)])
Ftr, Fte = aug(Xtr), aug(Xte)
m = lgb.LGBMClassifier(n_estimators=t["n_estimators"], bagging_freq=1, random_state=42, n_jobs=2, verbosity=-1, **t["params"]).fit(Ftr, ytr)
rng = np.random.default_rng(42); idx = rng.choice(len(Fte), 800, replace=False); bg = Ftr[rng.choice(len(Ftr), 100, replace=False)]
ex = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent"); space = "raw (log-odds), tree-path-dependent"
sv = np.asarray(ex.shap_values(Fte[idx]))
if sv.ndim == 3 and sv.shape[0] == len(cls): sv = np.moveaxis(sv, 0, -1)
print("space", space, "shape", sv.shape)
imp = np.abs(sv).mean(axis=0); imp = imp.sum(axis=-1) if imp.ndim == 2 else imp     # sum over classes
d = X.shape[1]; fam = {"x": imp[:d].sum(), "nu": imp[d:2*d].sum(), "pi": imp[2*d:].sum()}; tot = sum(fam.values())
print({k: round(100 * v / tot, 1) for k, v in fam.items()})
names = list(X.columns) + [f"ν({c})" for c in X.columns] + [f"π({c})" for c in X.columns]
top = pd.Series(imp, index=names).sort_values(ascending=False); print(top.head(12).round(5).to_string())
json.dump({"space": space, "shares": {k: 100 * v / tot for k, v in fam.items()}, "top": top.head(15).to_dict()}, open("shap_darknet.json", "w"), indent=1, ensure_ascii=False)
np.save("shap_darknet_sv.npy", sv); np.save("shap_darknet_X.npy", Fte[idx]); json.dump(names, open("shap_darknet_names.json", "w"), ensure_ascii=False)
