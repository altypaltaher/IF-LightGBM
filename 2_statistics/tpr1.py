import numpy as np, json
from sklearn.metrics import roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import label_binarize
M=["IF-LightGBM (proposed)","LightGBM (raw)","XGBoost (raw)","GBM (raw)","AdaBoost (raw)"]
def tpr_at(y,s,f0=0.01):
    f,t,_=roc_curve(y,s); return t[f<=f0].max()
def tpr_model(y,p,f0=0.01):
    K=p.shape[1]
    if K==2: return tpr_at(y,p[:,1],f0)
    Y=label_binarize(y,classes=range(K)); return np.mean([tpr_at(Y[:,k],p[:,k],f0) for k in range(K)])
out={}
for ds in ["CICAndMal2017","CIC-AAGM2017","CIC-Darknet2020"]:
    z=np.load(f"oof/oof_{ds}.npz",allow_pickle=True); P=z["P"].item(); y=z["y"]
    folds=[v for f,(a,v) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42).split(np.zeros(len(y)),y)) if f<5]
    out[ds]={}
    for f0 in (0.01,0.05):
        pooled={m:tpr_model(y,P[m],f0) for m in M}
        perfold={m:[tpr_model(y[v],P[m][v],f0) for v in folds] for m in M}
        wins={m:int(sum(a>b for a,b in zip(perfold[M[0]],perfold[m]))) for m in M[1:]}
        out[ds][str(f0)]={"pooled":pooled,"perfold_mean":{m:float(np.mean(v)) for m,v in perfold.items()},"wins_vs":wins}
        print(ds,f0,{m.split()[0]:round(100*v,2) for m,v in pooled.items()},"IF wins/5:",wins)
json.dump(out,open("m/tpr_at_fpr.json","w"),indent=1)
