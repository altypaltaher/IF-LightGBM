import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
SRC={"CICAndMal2017":"results/cic6/IFE_results/CICAndMal2017_cv_fold_scores.csv",
     "CIC-AAGM2017":"results/aagm2/IFE_results/CIC-AAGM2017_cv_fold_scores.csv",
     "CIC-Darknet2020":"results/confirm_darknet/cv_fold_scores.csv"}
P="IF-LightGBM (proposed)"; B=[("LightGBM (raw)","LightGBM"),("XGBoost (raw)","XGBoost"),("GBM (raw)","GBM"),("AdaBoost (raw)","AdaBoost")]
COL={"CICAndMal2017":"#2a78d6","CIC-AAGM2017":"#eb6834","CIC-Darknet2020":"#1baf7a"}
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.6),sharey=True)
for ax,(k,lab) in zip(axes,[("accuracy","Accuracy"),("roc_auc","ROC-AUC")]):
    yt=[]; yl=[]; pos=0
    for bi,(b,bn) in enumerate(B):
        for di,(ds,f) in enumerate(SRC.items()):
            cv=pd.read_csv(f); a=cv[cv.model==P].sort_values("fold")[k].to_numpy(); c=cv[cv.model==b].sort_values("fold")[k].to_numpy()
            d=100*(a-c); r=cv.n_test.iloc[0]/cv.n_train.iloc[0]; se=np.sqrt((1/len(d)+r)*d.var(ddof=1)); h=stats.t.ppf(.975,len(d)-1)*se
            y=-(bi*4+di); ax.errorbar(d.mean(),y,xerr=h,fmt="o",color=COL[ds],ms=4.5,capsize=2.5,lw=1.3,label=ds if (bi==0) else None)
        yt.append(-(bi*4+1)); yl.append(bn)
    ax.axvline(0,color="#52514e",lw=0.9,ls="--"); ax.set_xscale("symlog",linthresh=1,linscale=1.2); tk=[-0.5,0,0.5,1,2,5,10]; ax.set_xticks(tk,[str(t) for t in tk]); ax.set_xlim(-0.6,13)
    ax.set_xlabel(f"IF-LightGBM − baseline, {lab} (percentage points)"); ax.set_yticks(yt,yl); ax.grid(axis="x",color="#e4e3df")
fig.legend(*axes[0].get_legend_handles_labels(),frameon=False,fontsize=8.5,loc="upper center",ncol=3,bbox_to_anchor=(0.5,1.04))
fig.tight_layout(); fig.savefig("fig/fig_effect_v20.png",dpi=400,bbox_inches="tight",facecolor="white"); print("ok")
