"""Generate v20 prose; every number is read from result files."""
import json, numpy as np, pandas as pd
M = "results/v20/m/"
S = pd.read_csv(M + "cv_summary_v20.csv", index_col=0); C = pd.read_csv(M + "cv_contrasts_v20.csv")
AB = pd.read_csv(M + "ablation_summary_v20.csv")
SH = json.load(open(M + "shap_bin.json")); SH["CIC-Darknet2020"] = json.load(open(M + "shap_darknet.json"))
V = json.load(open("results/exp18/versions.json"))
PC = pd.read_csv(M + "cv_prob_v20.csv"); PR = pd.read_csv(M + "cv_prob_summary_v20.csv")
TPJ = json.load(open(M + "tpr_at_fpr.json")); TP = {k: v["0.01"]["pooled"] for k, v in TPJ.items()}; W = {k: v["0.01"]["wins_vs"]["LightGBM (raw)"] for k, v in TPJ.items()}
P = "IF-LightGBM (proposed)"; DS = ["CICAndMal2017", "CIC-AAGM2017", "CIC-Darknet2020"]

def s(ds, m, k): return S[(S.dataset == ds)].loc[m, k]
def pct(v): return f"{100 * v:.2f}%"
def c(ds, b, k): return C[(C.dataset == ds) & (C.baseline == b) & (C.metric == k)].iloc[0]
def pf(p):
    if p >= 0.001: return f"{p:.3f}"
    e = int(np.floor(np.log10(p))); m = round(p / 10 ** e, 1)
    if m >= 10: m, e = m / 10, e + 1
    return f"{m:.1f} × 10^{{−{-e}}}"
def d(x): return f"{x:+.2f}".replace("+", "+")
def ab(ds, rep, k): return AB[(AB.dataset == ds) & (AB.rep == rep) & (AB.metric == k)].iloc[0]
def sh(ds, f): return SH[ds]["shares"][f]

acc = {ds: s(ds, P, "accuracy") for ds in DS}; auc = {ds: s(ds, P, "roc_auc") for ds in DS}
T = {}
T["PYV"], T["LGBV"], T["XGBV"], T["SKV"], T["OPV"], T["SHV"] = V["python"], V["lightgbm"], V["xgboost"], V["scikit-learn"], V["optuna"], V["shap"]
T["T1_ACC"] = " / ".join(f"{100 * acc[ds]:.2f}" for ds in DS) + " (CV)"


# ---------------- 4.1
L, X, G, A = "LightGBM (raw)", "XGBoost (raw)", "GBM (raw)", "AdaBoost (raw)"
cic, ag, dk = DS
T["R41"] = (
 f"Table 4 summarises the cross-validated performance and Table 5 the paired contrasts, which Figure 2 visualises. IF-LightGBM obtained the highest mean accuracy, F1-score, MCC and ROC-AUC on all three datasets, with accuracies of {pct(acc[cic])}, {pct(acc[ag])} and {pct(acc[dk])} and ROC-AUC values of {auc[cic]:.4f}, {auc[ag]:.4f} and {auc[dk]:.4f} on CICAndMal2017, CIC-AAGM2017 and CIC-Darknet2020, respectively. "
 f"On CICAndMal2017, IF-LightGBM outperformed every baseline in all 15 folds on all four metrics, and all 16 contrasts were significant after Holm correction (p_{{Holm}} ≤ {max(C[C.dataset == cic].p_holm):.3f}). Relative to identically tuned LightGBM, the strongest baseline, accuracy increased by {c(cic, L, 'accuracy').diff_pp:.2f} points and ROC-AUC by {c(cic, L, 'roc_auc').diff_pp:.2f} points (Cliff’s δ = {c(cic, L, 'roc_auc').cliffs:.2f}); the gains over XGBoost, GBM and AdaBoost were {c(cic, X, 'accuracy').diff_pp:.2f}, {c(cic, G, 'accuracy').diff_pp:.2f} and {c(cic, A, 'accuracy').diff_pp:.2f} accuracy points. "
 f"The same pattern held on CIC-Darknet2020: IF-LightGBM won all 15 folds against every baseline on all four metrics, and all 16 contrasts were significant (p_{{Holm}} ≤ {max(C[C.dataset == dk].p_holm):.3f}). It exceeded LightGBM by {c(dk, L, 'accuracy').diff_pp:.2f} points in accuracy and {c(dk, L, 'f1').diff_pp:.2f} points in macro F1 (p_{{Holm}} = {c(dk, L, 'accuracy').p_holm:.3f}) and by {c(dk, L, 'roc_auc').diff_pp:.2f} points in ROC-AUC (p_{{Holm}} = {pf(c(dk, L, 'roc_auc').p_holm)}), and XGBoost, GBM and AdaBoost by {c(dk, X, 'accuracy').diff_pp:.2f}, {c(dk, G, 'accuracy').diff_pp:.2f} and {c(dk, A, 'accuracy').diff_pp:.2f} accuracy points. "
 f"On CIC-AAGM2017, IF-LightGBM significantly outperformed GBM ({d(c(ag, G, 'accuracy').diff_pp)} accuracy and {d(c(ag, G, 'roc_auc').diff_pp)} AUC points) and AdaBoost ({d(c(ag, A, 'accuracy').diff_pp)} and {d(c(ag, A, 'roc_auc').diff_pp)} points) on all metrics, and XGBoost in ROC-AUC ({d(c(ag, X, 'roc_auc').diff_pp)} points, 15/15 folds, p_{{Holm}} = {pf(c(ag, X, 'roc_auc').p_holm)}). Its accuracy advantage over XGBoost ({d(c(ag, X, 'accuracy').diff_pp)} points, {int(c(ag, X, 'accuracy').wins)}/{int(c(ag, X, 'accuracy').losses)} folds) was consistent but fell just short of significance after Holm correction (p_{{Holm}} = {c(ag, X, 'accuracy').p_holm:.3f}). "
 f"On this dataset, IF-LightGBM and LightGBM performed comparably ({d(c(ag, L, 'accuracy').diff_pp)} accuracy and {d(c(ag, L, 'roc_auc').diff_pp)} AUC points; p_{{Holm}} = 1.00), indicating that the fuzzy features offer at most a marginal benefit when LightGBM already discriminates these flows well. "
 f"IF-LightGBM also produced the most accurate probability estimates, with the lowest mean Brier score or log-loss on all three datasets (Table 4): its Brier score was lower than that of every baseline in all 15 folds on CICAndMal2017 ({PR[(PR.dataset == cic) & (PR.model == P)]['mean'].iloc[0]:.4f} versus {PR[(PR.dataset == cic) & (PR.model == L)]['mean'].iloc[0]:.4f} for LightGBM; p_{{Holm}} ≤ {pf(PC[PC.dataset == cic].p_holm.max())}), and so was its log-loss on CIC-Darknet2020 ({PR[(PR.dataset == dk) & (PR.model == P)]['mean'].iloc[0]:.4f} versus {PR[(PR.dataset == dk) & (PR.model == L)]['mean'].iloc[0]:.4f}; p_{{Holm}} ≤ {pf(PC[PC.dataset == dk].p_holm.max())}). On CIC-AAGM2017, its Brier score was significantly lower than those of XGBoost, GBM and AdaBoost (15/15 folds; p_{{Holm}} ≤ {pf(PC[(PC.dataset == ag) & (PC.baseline != L)].p_holm.max())}) and similar to that of LightGBM (p_{{Holm}} = {PC[(PC.dataset == ag) & (PC.baseline == L)].p_holm.iloc[0]:.2f}). "
 f"At the operationally relevant false-positive rate of 1%, IF-LightGBM detected {100*TP[cic]['IF-LightGBM (proposed)']:.1f}% of malicious flows on CICAndMal2017, compared with {100*TP[cic]['LightGBM (raw)']:.1f}% for LightGBM, {100*TP[cic]['XGBoost (raw)']:.1f}% for XGBoost, {100*TP[cic]['GBM (raw)']:.1f}% for GBM and {100*TP[cic]['AdaBoost (raw)']:.1f}% for AdaBoost—about {100*(TP[cic]['IF-LightGBM (proposed)']/TP[cic]['LightGBM (raw)']-1):.0f}% more detections than LightGBM at the same false-alarm rate. "
 f"The corresponding values were {100*TP[ag]['IF-LightGBM (proposed)']:.1f}% versus {100*TP[ag]['LightGBM (raw)']:.1f}% for LightGBM on CIC-AAGM2017, and a macro-averaged true-positive rate of {100*TP[dk]['IF-LightGBM (proposed)']:.1f}% versus {100*TP[dk]['LightGBM (raw)']:.1f}% over the eight categories on CIC-Darknet2020. "
 f"IF-LightGBM exceeded LightGBM at this operating point in {W[cic]}, {W[ag]} and {W[dk]} of the five folds of the first repeat, respectively; because this operating point is estimated from a single repeat, these values are descriptive.")

# ---------------- 4.2 ablation
def abl_sentence(ds):
    a_n, u_n = ab(ds, "x+ν", "accuracy"), ab(ds, "x+ν", "roc_auc")
    a_p, u_p = ab(ds, "x+π", "accuracy"), ab(ds, "x+π", "roc_auc")
    a_f, u_f = ab(ds, "x+ν+π", "accuracy"), ab(ds, "x+ν+π", "roc_auc")
    return a_n, u_n, a_p, u_p, a_f, u_f
R = {ds: abl_sentence(ds) for ds in DS}
fb = json.load(open(M + "finebin_numbers.json"))
T["R42"] = (
 f"Table 6 trains LightGBM with the IF-LightGBM settings on the raw features x and on each augmented representation. "
 f"On CICAndMal2017 and CIC-Darknet2020, both the non-membership and the hesitation block improved on x: [x, ν] changed ROC-AUC by {d(R[cic][1]['diff'])} and {d(R[dk][1]['diff'])} points and [x, π] by {d(R[cic][3]['diff'])} and {d(R[dk][3]['diff'])} points, and the full augmentation gave {d(R[cic][4]['diff'])} and {d(R[dk][4]['diff'])} accuracy points ({R[cic][4].w}/{R[cic][4].l} and {R[dk][4].w}/{R[dk][4].l} folds). "
 f"On CIC-AAGM2017, the effects were small ([x, ν, π]: {d(R[ag][4]['diff'])} accuracy and {d(R[ag][5]['diff'])} AUC points). "
 f"Because ν is a monotone transform of x (Proposition 1), its benefit cannot come from additional split capacity; it must arise from how LightGBM searches for splits. LightGBM evaluates thresholds only at the boundaries of at most 255 histogram bins per feature, so heavy-tailed variables are discretised coarsely, and the separately binned ν and π copies supply additional candidate thresholds on the same variable. "
 f"Consistent with this explanation, increasing LightGBM’s histogram resolution to max_bin = 4095 produced a similar or larger effect without the fuzzy features. On CIC-Darknet2020, LightGBM with 4095 bins, tuned with the same budget, reached {pct(fb['dk_x4095_acc'])} accuracy and {fb['dk_x4095_auc']:.4f} ROC-AUC, statistically indistinguishable from IF-LightGBM (unadjusted p ≥ {fb['dk_min_p']:.2f}). On CICAndMal2017, raw-feature LightGBM with 4095 bins and the IF-LightGBM settings reached {pct(fb['cic_x4095_acc'])} accuracy and {fb['cic_x4095_auc']:.4f} ROC-AUC, exceeding IF-LightGBM with default binning by {fb['cic_x4095_minus_IF_accuracy']:.2f} accuracy and {fb['cic_x4095_minus_IF_roc_auc']:.2f} AUC points in all 15 folds. "
 f"The intuitionistic fuzzy augmentation therefore improves LightGBM mainly by refining the discretisation of continuous traffic features; it achieves this without changing the learner’s settings and additionally provides interpretable non-membership and hesitation features, whereas tuning the histogram resolution directly is an alternative route to a similar or larger gain.")

# ---------------- 4.3 SHAP
TOPFEAT = ("A hesitation feature ranked first on both Android malware datasets—π(Min Packet Length) on CICAndMal2017 and π(Init_Win_bytes_forward) on CIC-AAGM2017. They were followed on CICAndMal2017 by π(Source Port), the source and destination ports and the minimum inter-arrival times, and on CIC-AAGM2017 by the initial forward TCP window size, the FIN-flag count (flow_fin) and the fuzzy degrees of the mean flow inter-arrival time. "
 "On CIC-Darknet2020, the maximum idle time and its hesitation and non-membership degrees were the most important features, followed by π(FWD Init Win Bytes), the number of forward data packets and π(Fwd Seg Size Min).")
def top(ds, n):
    return [k for k in list(SH[ds]["top"].keys())[:n]]
T["R43"] = (
 f"Figure 3 shows the features with the largest mean absolute SHAP values. The hesitation features accounted for {sh(cic, 'pi'):.1f}%, {sh(ag, 'pi'):.1f}% and {sh(dk, 'pi'):.1f}% of the total attribution on CICAndMal2017, CIC-AAGM2017 and CIC-Darknet2020, and the non-membership features for {sh(cic, 'nu'):.1f}%, {sh(ag, 'nu'):.1f}% and {sh(dk, 'nu'):.1f}%, so the fuzzy features together carried {sh(cic, 'nu') + sh(cic, 'pi'):.1f}%, {sh(ag, 'nu') + sh(ag, 'pi'):.1f}% and {sh(dk, 'nu') + sh(dk, 'pi'):.1f}% of the explanation. "
 f"{TOPFEAT} Because ν is a monotone transform of x (Proposition 1), the x and ν attributions of a feature should be read jointly, whereas the π attributions reflect the interval-type partitions of Proposition 2.")
json.dump(T, open("texts20_partial.json", "w"), ensure_ascii=False, indent=1)
print("ok")

# ---------------- 5.3 comparison with previous studies
T["D53"] = (
 f"IF-LightGBM’s cross-validated accuracy of {pct(acc[cic])} on CICAndMal2017 exceeds the 79.81% of an explainable random forest [@palma24] and the 87.75% of conversation-level features with Extra Trees [@abuthawabeh20]. Higher figures have been reported with different units of analysis or feature sets, such as 98.24% with AdaBoost over random forests [@ferreira21] and above 99% recall with application-level flow graphs [@busch21], so absolute numbers are not directly comparable. "
 f"On CIC-AAGM2017, the dataset authors reported an average accuracy of 91.41% over five classifiers with nine traffic features [@lashkari17], and AdStop reported 98.02% with a reduced flow-feature set [@alani22]; on CIC-Darknet2020, Rust-Nguyen et al. [@rustnguyen23] reported 92.3% accuracy for application categories with a random forest that also used IP-address information. "
 f"Our results ({pct(acc[ag])} and {pct(acc[dk])}) were obtained on de-duplicated flows without addresses or ports on CIC-AAGM2017 and CIC-Darknet2020, which removes common sources of optimism, so they are best compared within our controlled protocol, where IF-LightGBM ranks first on all three datasets.")

# ---------------- 5.4 limitations
T["D54"] = (
 "Several limitations should be noted. First, hyperparameters were tuned on an 80% partition that overlaps with the cross-validation folds. For the key contrast between IF-LightGBM and LightGBM, nested cross-validation (both models re-tuned within each of five outer folds) confirmed the gain on CICAndMal2017 (+0.49 accuracy points, 5/5 outer folds, p = 0.034), whereas on CIC-AAGM2017 the two models again performed comparably (identical ROC-AUC of 0.9616); nested cross-validation was not repeated on CIC-Darknet2020 because of its computational cost. "
 "Second, histogram resolution was left at LightGBM’s default for all models; as Section 4.2 shows, a finer resolution can provide a similar or larger improvement, so the ranking of the default-binning models should not be generalised to other binning settings. "
 "Third, splits are random at the flow level, so flows of the same application can occur in both training and test folds. In an application-grouped evaluation of CICAndMal2017, the ROC-AUC of all models fell to 0.61–0.64 and IF-LightGBM performed on par with LightGBM, so the flow-level results describe discrimination within the captured application population rather than generalisation to unseen applications. "
 "Fourth, CICAndMal2017 includes port features; without Source Port and Destination Port, accuracy fell by about five points for every model and the advantage of IF-LightGBM over LightGBM became small (+0.10 accuracy and +0.17 AUC points). "
 "Finally, the datasets were captured in 2015–2017 and were class-balanced, CIC-AAGM2017 and CIC-Darknet2020 were down-sampled for computational reasons, CIC-Darknet2020 contains application rather than malware labels, and adversarial robustness was not evaluated [@pendlebury19,@pierazzi20].")

lo, hi = min(sh(x, "pi") for x in DS), max(sh(x, "pi") for x in DS)
# ---------------- 5.1, 5.2
T["D51"] = (
 "Under an identical tuning budget, IF-LightGBM was the best of the five models on all three datasets. Its advantage over XGBoost, GBM and AdaBoost was large and consistent, and its advantage over identically tuned LightGBM—the strongest baseline—was significant on CICAndMal2017 and CIC-Darknet2020 and modest on CIC-AAGM2017. "
 "The analytical results and the ablation explain this pattern. Under exact split search, only the hesitation degree can add split capacity to a tree (Propositions 1 and 2); in practice, however, the non-membership degree also helped, and a finer histogram resolution produced a similar or larger gain. The benefit therefore stems mainly from refining how LightGBM discretises heavy-tailed continuous features, which is most useful when informative features take many distinct values and the default 255 bins are coarse, and least useful when LightGBM already separates the classes well; binary features cannot benefit at all (Proposition 3). "
 "In practice, the augmentation is an inexpensive, deterministic preprocessing step that delivers this refinement without changing the learner’s settings and yields interpretable non-membership and hesitation features, whereas tuning the histogram resolution directly is an alternative way to obtain a similar or larger improvement.")
T["D52"] = (
 "On CICAndMal2017, inter-arrival times, packet sizes and TCP initial window sizes rank high, consistent with the automated, periodic communication of adware, scareware and command-and-control traffic [@lashkari18]. Ports also carry much of the signal; because client source ports are ephemeral values assigned by the operating system, their importance most likely reflects properties of the capture sessions rather than malicious behaviour, and TCP initial window sizes can similarly fingerprint the device network stack [@arp22,@liu22xai]. "
 "CIC-AAGM2017 contains no port or address features, and the model relies on initial TCP window sizes, connection-termination behaviour and inter-arrival times, which plausibly reflect the short, repetitive advertisement-fetching connections typical of adware [@lashkari17]. "
 "On CIC-Darknet2020, idle times and segment sizes dominate, which is consistent with differences between interactive applications such as chat and VoIP and bulk transfers such as P2P and file transfer. These explanations describe what the models learned from the data and are not necessarily causal.")
# ---------------- abstract, conclusion, caption
T["ABSTRACT"] = (
 "Network-traffic features allow Android malware to be detected without access to the device, but flow statistics are noisy, heavy-tailed and overlap strongly between classes. "
 "This paper proposes IF-LightGBM, which augments every flow feature with its intuitionistic fuzzy non-membership and hesitation degrees before training LightGBM, and shows analytically when such features can add split capacity to a tree ensemble. "
 "IF-LightGBM was compared with LightGBM, XGBoost, gradient boosting machines and AdaBoost, all tuned with an identical Optuna budget, on two Android malware traffic datasets (CICAndMal2017 and CIC-AAGM2017) and on the CIC-Darknet2020 encrypted-traffic dataset, using repeated cross-validation with corrected resampled t-tests and Holm correction. "
 f"IF-LightGBM achieved the highest mean accuracy, F1-score, Matthews correlation coefficient and ROC-AUC on all three datasets (accuracy {pct(acc[cic])}, {pct(acc[ag])} and {pct(acc[dk])}). "
 "It significantly outperformed all four baselines on CICAndMal2017 and CIC-Darknet2020 and also yielded the most accurate probability estimates (lowest Brier score or log-loss); on CIC-AAGM2017, it significantly outperformed GBM and AdaBoost, outperformed XGBoost in ROC-AUC and performed comparably to LightGBM. "
 "An ablation indicated that the gain arises mainly from a finer discretisation of continuous traffic features, which can also be obtained by increasing LightGBM’s histogram resolution. "
 f"SHAP attributed {lo:.0f}–{hi:.0f}% of the explanation to the hesitation features,.")
T["CONCL"] = (
 "This paper proposed IF-LightGBM, which augments network-flow features with intuitionistic fuzzy non-membership and hesitation degrees before gradient boosting, and evaluated it against four boosting baselines tuned with an identical budget on CICAndMal2017, CIC-AAGM2017 and CIC-Darknet2020. "
 f"IF-LightGBM achieved the best cross-validated performance on all three datasets—{pct(acc[cic])}, {pct(acc[ag])} and {pct(acc[dk])} accuracy and {auc[cic]:.4f}, {auc[ag]:.4f} and {auc[dk]:.4f} ROC-AUC—and significantly outperformed LightGBM, XGBoost, GBM and AdaBoost on CICAndMal2017 and CIC-Darknet2020. On CIC-AAGM2017, it significantly outperformed GBM and AdaBoost and performed comparably to LightGBM. "
 "Analytically, only the hesitation degree can add split capacity to an exact-split tree ensemble; empirically, the gain arises mainly from a finer discretisation of heavy-tailed traffic features, which increasing LightGBM’s histogram resolution can also provide. "
 f"SHAP showed that the fuzzy features carry a substantial share of the model’s decisions, with {lo:.0f}–{hi:.0f}% attributed to hesitation features. "
 "Intuitionistic fuzzy augmentation is therefore a simple, interpretable way to strengthen gradient boosting on network-traffic data, and claims of improvement from fuzzy features should be tested against equally tuned baselines with ablations and variance-aware statistics. "
 "Future work will evaluate application-grouped and temporal splits on recent traffic corpora, data-driven intuitionistic fuzzy generators combined with tuned histogram resolution, and robustness to adversarial manipulation.")
T["FIG3_CAP"] = ("**Figure 3.** Twelve most important features of the tuned IF-LightGBM by mean absolute SHAP value (tree-path-dependent TreeSHAP in log-odds space; 800 flows of the 20% partition not used for tuning): (a) CICAndMal2017; (b) CIC-AAGM2017; (c) CIC-Darknet2020, with SHAP values summed over the eight class outputs. Colours denote original (x), non-membership (ν) and hesitation (π) features; panel titles give the attribution shares of the three families.")
json.dump(T, open("texts20.json", "w"), ensure_ascii=False, indent=1)
print(len(T["ABSTRACT"].split()), "abstract words")
