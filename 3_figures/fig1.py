import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"]="DejaVu Sans"
fig,ax=plt.subplots(figsize=(13,6.0),dpi=300); ax.set_xlim(0,130); ax.set_ylim(0,64); ax.axis("off")
C={"data":"#E8EEF7","prep":"#EAF4EA","if":"#FFF4E0","model":"#F3E8F7","eval":"#FDECEC","edge":"#3A4A5C"}
def box(x,y,w,h,title,lines,col):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=col,ec=C["edge"],lw=1.2))
    ax.text(x+w/2,y+h-2.6,title,ha="center",va="top",fontsize=10.5,fontweight="bold",color="#1d2733")
    for i,l in enumerate(lines): ax.text(x+w/2,y+h-7.4-i*3.5,l,ha="center",va="top",fontsize=8.6,color="#22303f")
def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,lw=1.4,color=C["edge"]))
box(1,31,24,32,"1. Datasets",["CICAndMal2017","14,997 flows, 68 features","CIC-AAGM2017","20,000 flows, 74 features","CIC-Darknet2020","36,120 flows, 62 features"],C["data"])
box(29,31,23,32,"2. Data preparation",["Remove duplicates and","label conflicts","Drop identifiers","Bounds fitted on","training data only"],C["prep"])
box(56,31,30,32,"3. Intuitionistic fuzzy",["augmentation (per feature j)","μ = clip((x−min)/(max−min), 0, 1)","ν = (1 − μ)²","π = 1 − μ − ν = μ(1 − μ)","z = [x, ν, π]  (3d features)"],C["if"])
box(90,31,39,32,"4. Equal-budget tuning",["Optuna TPE, 20 trials per model,","inner 3-fold CV, early stopping","","IF-LightGBM (proposed)","LightGBM, XGBoost, GBM, AdaBoost"],C["model"])
arr(25.6,47,28.6,47); arr(52.6,47,55.6,47); arr(86.6,47,89.6,47)
box(6,2,36,23,"5a. Repeated cross-validation",["3 × 5 stratified folds","Accuracy, F1, MCC, ROC-AUC"],C["eval"])
box(47,2,36,23,"5b. Statistical inference",["Corrected resampled t-test","Holm correction","Cliff's δ, win/loss counts"],C["eval"])
box(88,2,40,23,"5c. Ablation and SHAP",["x, ν and π blocks;","histogram resolution","SHAP attribution shares","of x, ν and π"],C["eval"])
ax.plot([109.5,109.5],[30.5,28.5],color=C["edge"],lw=1.4); ax.plot([24,109.5],[28.5,28.5],color=C["edge"],lw=1.4)
for xx in (24,65,108): arr(xx,28.5,xx,25.8)
plt.savefig("fig/fig1_framework_v20.png",bbox_inches="tight",facecolor="white"); print("ok")
