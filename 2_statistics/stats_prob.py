import numpy as np, pandas as pd
from scipy import stats
SRC={"CICAndMal2017":("results/cic6/IFE_results/CICAndMal2017_cv_fold_scores.csv","brier"),
     "CIC-AAGM2017":("results/aagm2/IFE_results/CIC-AAGM2017_cv_fold_scores.csv","brier"),
     "CIC-Darknet2020":("results/confirm_darknet/cv_fold_scores.csv","log_loss")}
P="IF-LightGBM (proposed)"; BASE=["LightGBM (raw)","XGBoost (raw)","GBM (raw)","AdaBoost (raw)"]
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*p[i]); adj[i]=min(run,1)
    return adj
def ct(d,ntr,nte): v=d.var(ddof=1); t=d.mean()/np.sqrt((1/len(d)+nte/ntr)*v); return 2*stats.t.sf(abs(t),len(d)-1)
rows=[]; summ=[]
for ds,(f,k) in SRC.items():
    cv=pd.read_csv(f); g=cv.groupby("model")[k]
    for m in [P]+BASE: summ.append(dict(dataset=ds,model=m,metric=k,mean=g.mean()[m],sd=g.std()[m]))
    tmp=[]
    for b in BASE:
        a=cv[cv.model==P].sort_values("fold")[k].to_numpy(); c=cv[cv.model==b].sort_values("fold")[k].to_numpy(); d=a-c
        tmp.append(dict(dataset=ds,baseline=b,metric=k,diff=d.mean(),better=int((d<0).sum()),worse=int((d>0).sum()),p=ct(d,cv.n_train.iloc[0],cv.n_test.iloc[0])))
    t=pd.DataFrame(tmp); t["p_holm"]=holm(t.p); rows.append(t)
R=pd.concat(rows); R.to_csv("cv_prob_v20.csv",index=False); pd.DataFrame(summ).to_csv("cv_prob_summary_v20.csv",index=False)
print(pd.DataFrame(summ).round(4).to_string(index=False)); print(R.to_string(index=False,float_format=lambda v:f"{v:.4g}"))
