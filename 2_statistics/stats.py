import json, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
SRC={"CICAndMal2017":"results/cic6/IFE_results/CICAndMal2017_cv_fold_scores.csv",
     "CIC-AAGM2017":"results/aagm2/IFE_results/CIC-AAGM2017_cv_fold_scores.csv",
     "CIC-Darknet2020":"results/confirm_darknet/cv_fold_scores.csv"}
P="IF-LightGBM (proposed)"; BASE=["LightGBM (raw)","XGBoost (raw)","GBM (raw)","AdaBoost (raw)"]; MET=["accuracy","f1","mcc","roc_auc"]
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*p[i]); adj[i]=min(run,1)
    return adj
def ct(d,ntr,nte):
    v=d.var(ddof=1); 
    if v==0: return 1.0
    t=d.mean()/np.sqrt((1/len(d)+nte/ntr)*v); return 2*stats.t.sf(abs(t),len(d)-1)
def cliffs(a,b): return (sum((x>b).sum() for x in a)-sum((x<b).sum() for x in a))/(len(a)*len(b))
allsum, allct = [], []
for ds,f in SRC.items():
    cv=pd.read_csv(f); cv=cv[cv.model.isin([P]+BASE)]
    assert cv.groupby("model").fold.nunique().eq(15).all(), ds
    g=cv.groupby("model")[MET]; s=g.mean().join(g.std(),rsuffix="_sd").loc[[P]+BASE]; s.insert(0,"dataset",ds); allsum.append(s)
    rows=[]
    for b in BASE:
        for k in MET:
            a=cv[cv.model==P].sort_values("fold")[k].to_numpy(); c=cv[cv.model==b].sort_values("fold")[k].to_numpy(); d=a-c
            rows.append(dict(dataset=ds,baseline=b,metric=k,diff_pp=100*d.mean(),wins=int((d>0).sum()),losses=int((d<0).sum()),
                p=ct(d,cv.n_train.iloc[0],cv.n_test.iloc[0]),wilcoxon_p=wilcoxon(a,c).pvalue if np.any(d!=0) else 1.0,cliffs=cliffs(a,c)))
    t=pd.DataFrame(rows); t["p_holm"]=holm(t.p); allct.append(t)
S=pd.concat(allsum); C=pd.concat(allct)
S.to_csv("cv_summary_v20.csv"); C.to_csv("cv_contrasts_v20.csv",index=False)
pd.set_option("display.width",250)
print(S.round(4).to_string()); print(C.round(4).to_string(index=False))
