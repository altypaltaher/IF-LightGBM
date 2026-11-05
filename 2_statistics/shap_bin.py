import json, numpy as np, pandas as pd, shap
from sklearn.model_selection import train_test_split
CFG={"CICAndMal2017":("1_experiments/run_cic6.py","data/CICANDMAL2017-75000-malware-7500 Goodware.csv","Label1","results/cic6/IFE_results/CICAndMal2017_tuned_hyperparameters.json"),
     "CIC-AAGM2017":("1_experiments/run_aagm.py","data/aagm2017_sample20k.csv","y","results/aagm2/IFE_results/CIC-AAGM2017_tuned_hyperparameters.json")}
P="IF-LightGBM (proposed)"; out={}
for ds,(sc,path,t,tj) in CFG.items():
    src=open(sc).read(); src=src[:src.index("# ---------------------------------------------------------------- main")]; ns={}; exec(src,ns)
    X,y=ns["load"](path,t,None,False); tu=json.load(open(tj))[P]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    Ftr,Fte=ns["build"](P,Xtr,ytr.to_numpy(),Xte); m=ns["make"](P,tu["params"],tu["n_estimators"]).fit(Ftr,ytr)
    rng=np.random.default_rng(42); idx=rng.choice(len(Fte),800,replace=False)
    sv=np.asarray(shap.TreeExplainer(m,feature_perturbation="tree_path_dependent").shap_values(Fte[idx]))
    if sv.ndim==3: sv=sv[...,-1] if sv.shape[-1]==2 else sv[-1]
    imp=np.abs(sv).mean(0); d=X.shape[1]; f={"x":imp[:d].sum(),"nu":imp[d:2*d].sum(),"pi":imp[2*d:].sum()}; tot=sum(f.values())
    names=list(X.columns)+[f"ν({c})" for c in X.columns]+[f"π({c})" for c in X.columns]
    top=pd.Series(imp,index=names).sort_values(ascending=False)
    out[ds]={"shares":{k:100*v/tot for k,v in f.items()},"top":top.head(15).to_dict()}
    print(ds,{k:round(100*v/tot,1) for k,v in f.items()}); print(top.head(8).round(4).to_string())
json.dump(out,open("shap_bin.json","w"),indent=1,ensure_ascii=False)
