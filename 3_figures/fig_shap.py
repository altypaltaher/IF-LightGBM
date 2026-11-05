import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
M="results/v20/m/"; SH=json.load(open(M+"shap_bin.json")); SH["CIC-Darknet2020"]=json.load(open(M+"shap_darknet.json"))
COL={"x":"#8a8984","ν":"#2a78d6","π":"#eb6834"}
plt.rcParams.update({"font.size":8.5,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,3,figsize=(13,4.6))
for ax,(lab,ds) in zip(axes,[("(a)","CICAndMal2017"),("(b)","CIC-AAGM2017"),("(c)","CIC-Darknet2020")]):
    top=list(SH[ds]["top"].items())[:12][::-1]; tot=None
    names=[k for k,_ in top]; vals=[v for _,v in top]
    fam=["π" if k.startswith("π(") else "ν" if k.startswith("ν(") else "x" for k in names]
    ax.barh(range(len(vals)),vals,color=[COL[f] for f in fam])
    ax.set_yticks(range(len(vals)),names); ax.set_xlabel("Mean |SHAP value| (log-odds)"+(", summed over classes" if ds=="CIC-Darknet2020" else ""))
    sh=SH[ds]["shares"]; ax.set_title(f"{lab} {ds}\nx {sh['x']:.1f}%, ν {sh['nu']:.1f}%, π {sh['pi']:.1f}%",fontsize=9)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COL["x"],label="original feature x"),Patch(color=COL["ν"],label="non-membership ν"),Patch(color=COL["π"],label="hesitation π")],
           loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(0.5,1.04))
fig.tight_layout(); fig.savefig("fig/fig_shap_v20.png",dpi=400,bbox_inches="tight",facecolor="white"); print("ok")
