import pandas as pd, numpy as np, sys
from scipy import stats
for d in sys.argv[1:]:
    x=pd.read_csv(f'exp18/nested_{d}.csv')
    A=x[x.model=="IF-LightGBM (proposed)"].sort_values('fold'); B=x[x.model=="LightGBM (raw)"].sort_values('fold')
    print(d, "folds", len(A))
    for k in ['accuracy','roc_auc','f1','mcc']:
        dd=A[k].to_numpy()-B[k].to_numpy(); J=len(dd); r=0.25
        t=dd.mean()/np.sqrt((1/J+r)*dd.var(ddof=1)); p=2*stats.t.sf(abs(t),J-1)
        print(f" {k}: IF {A[k].mean():.4f}±{A[k].std():.4f}  raw {B[k].mean():.4f}±{B[k].std():.4f}  d={100*dd.mean():+.3f}pp wins {int((dd>0).sum())}/{J} p={p:.2e}")
    print(" inner auc IF", A.inner_auc.mean().round(4), "raw", B.inner_auc.mean().round(4))
