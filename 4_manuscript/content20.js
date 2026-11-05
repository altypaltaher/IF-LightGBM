// Manuscript v20 — IF-LightGBM on CICAndMal2017, CIC-AAGM2017 and CIC-Darknet2020 (five tuned models, repeated CV).
// Markup: [@key] citations; «…» italic; **…** bold; ^{…} sup; _{…} sub.
const B = [];
const h1 = t => B.push({ t: "h1", text: t });
const h2 = t => B.push({ t: "h2", text: t });
const p = t => B.push({ t: "p", text: t });
const pn = t => B.push({ t: "pn", text: t });
const bl = t => B.push({ t: "bullet", text: t });
const eq = (id, n) => B.push({ t: "eq", id, n });
const tbl = o => B.push(Object.assign({ t: "table" }, o));
const fig = o => B.push(Object.assign({ t: "fig" }, o));

B.push({ t: "type", text: "Article" });
B.push({ t: "title", text: "Intuitionistic Fuzzy Feature Augmentation of LightGBM for Network-Traffic-Based Android Malware Detection: A Controlled and Explainable Evaluation" });
B.push({ t: "authors", text: "Altyeb Altaher Taha ^{1,}*, Ahmad AbdulQadir AlRababah ^{2}, Abubakar Elsafi ^{3}, Ahmed Hamza Osman ^{4}, Yakubu Suleiman Baguda ^{4}, Asif Hassan Syed ^{2} and Arda Yunianta ^{4}" });
B.push({ t: "affil", lines: [
  "^{1} Department of Information Technology, Faculty of Computing and Information Technology in Rabigh, King Abdulaziz University, Jeddah 21911, Saudi Arabia",
  "^{2} Department of Computer Science, Faculty of Computing and Information Technology in Rabigh, King Abdulaziz University, Jeddah 21911, Saudi Arabia",
  "^{3} Department of Software Engineering, College of Computer Science and Engineering, University of Jeddah, Jeddah 21959, Saudi Arabia",
  "^{4} Department of Information Systems, Faculty of Computing and Information Technology in Rabigh, King Abdulaziz University, Jeddah 21911, Saudi Arabia",
  "***** Correspondence: aaataha@kau.edu.sa",
] });
B.push({ t: "abstract", text: "{{ABSTRACT}}" });
B.push({ t: "keywords", text: "Android malware detection; network traffic; intuitionistic fuzzy sets; hesitation degree; feature augmentation; LightGBM; gradient boosting; equal-budget hyperparameter tuning; corrected resampled t-test; SHAP" });

// ---------------- 1 Introduction
h1("1. Introduction");
p("Smartphones have become a primary interface for banking and payments—registered mobile-money accounts exceeded two billion in 2024 [@gsma25]—as well as for healthcare, education and e-government services [@liu25]. Given Android’s dominant share of the mobile operating-system market [@statcounter,@prasad24], its security is a major concern: the openness of the platform allows malicious applications to commit advertising fraud, intercept SMS-based authentication codes, exfiltrate credentials and conduct extortion [@polam24,@kouliaridis21]. Android malware detection is commonly categorised as static, dynamic or hybrid [@kouliaridis21,@haq24]. Static analysis inspects permissions and API calls without executing the application, whereas network-traffic analysis observes what the application actually does on the network; it is robust to code obfuscation and can be deployed on gateways without access to the device, but the resulting flow statistics are noisy, heavy-tailed and overlap strongly between benign and malicious applications [@lashkari17,@wang19jnca,@arora14]. Boosted decision-tree ensembles—AdaBoost [@freund97], gradient boosting machines (GBM) [@friedman01], XGBoost [@chen16] and LightGBM [@ke17]—are among the most widely used learners for such tabular features [@islam23,@grinsztajn22].");
p("Fuzzy set theory [@zadeh65] offers a natural way to represent graded, uncertain evidence and has been used in malware detection through neuro-fuzzy classifiers [@altaher17,@altaher17b], fuzzy clustering combined with boosting [@tahamalebary21], fuzzy feature selection [@seyfari24] and fuzzy classifier fusion [@atacak23,@taha21math]. Intuitionistic fuzzy sets (IFSs) [@atanassov86] generalise fuzzy sets by pairing a membership degree μ with a non-membership degree ν and an explicit hesitation degree π = 1 − μ − ν, and have improved several classifier families [@rezvani19,@ren22,@wu22].");
p("Three gaps motivate this work. First, fuzzy-enhanced detectors are usually compared with baselines that use different—often default—hyperparameters, a single train–test split and no significance tests, so reported gains cannot be separated from differences in tuning effort or from chance [@arp22,@pendlebury19,@molina23]. Second, IFSs have been used inside classifiers or to fuse classifier outputs [@rezvani19,@ren22,@wu22], but not as an input representation for gradient-boosted trees, and it has not been established when membership, non-membership and hesitation degrees can add anything to an axis-aligned tree learner. Third, few studies isolate which component of a fuzzy representation produces an improvement or check whether the improvement holds beyond a single benchmark. As a result, it remains unclear whether intuitionistic fuzzy features genuinely improve a strong, well-tuned boosting model on network traffic.");
p("To address these gaps, this paper proposes IF-LightGBM, which augments every traffic feature with its intuitionistic fuzzy non-membership and hesitation degrees and trains LightGBM on the augmented representation, and evaluates it under a protocol in which all competing models receive the same tuning budget. The main contributions are as follows:");
bl("IF-LightGBM, a simple and reproducible intuitionistic fuzzy feature augmentation for gradient boosting, together with an analysis of when such features can add split capacity to a tree ensemble (Propositions 1–3).");
bl("A controlled comparison with four equally tuned boosting baselines—LightGBM, XGBoost, GBM and AdaBoost—on two public Android malware traffic datasets, CICAndMal2017 and CIC-AAGM2017, and on the CIC-Darknet2020 encrypted-traffic dataset, using 3 × 5-fold repeated cross-validation with the corrected resampled t-test, Holm correction and Cliff’s δ.");
bl("An ablation that isolates the contributions of the non-membership and hesitation degrees and relates the gain to LightGBM’s histogram discretisation, and a SHAP analysis of the features that drive the decisions.");
p("Section 2 reviews related work, Section 3 describes the method and evaluation protocol, Section 4 reports the results, Section 5 discusses them and Section 6 concludes.");

// ---------------- 2 Related work
h1("2. Related Work");
h2("2.1. Network-Traffic-Based Malware and Encrypted-Traffic Classification");
p("The Canadian Institute for Cybersecurity released two widely used Android traffic benchmarks captured on real smartphones. CIC-AAGM2017 [@lashkari17,@aagmurl] contains the traffic of 1900 applications (250 adware, 150 general malware and 1500 benign), and CICAndMal2017 [@lashkari18,@cicurl] extends the collection to adware, ransomware, scareware and SMS malware. On CICAndMal2017, reported flow-level accuracies range from 79.81% for an explainable random forest [@palma24] to 87.75% for conversation-level features with Extra Trees [@abuthawabeh20], while Ferreira et al. [@ferreira21] reported 98.24% with AdaBoost over random forests and Busch et al. [@busch21] above 99% recall by modelling the flows of an application as a graph. On CIC-AAGM2017, the dataset authors reported an average accuracy of 91.41% for five classifiers using nine traffic features [@lashkari17], and AdStop reached 98.02% with 13 selected flow features [@alani22]. Earlier traffic-based detectors used rule-based classifiers [@arora14], reduced traffic-feature sets [@arora17], multi-level traffic features with C4.5 [@wang19jnca], text semantics of HTTP flows [@wang18tifs] and deep models combining static and traffic features [@feng20]. Encrypted traffic poses a related flow-classification problem: CIC-Darknet2020 [@lashkari20,@darkurl] labels VPN and Tor flows with application categories, and Rust-Nguyen et al. [@rustnguyen23] found random forests to outperform earlier machine-learning approaches on it. Static detectors based on permissions, API calls and opcodes [@alazab20,@su16,@arp14,@yerima19], including ensemble and fuzzy-integral approaches by our group [@tahabarukab22,@taha21math], are complementary to the traffic-based setting studied here.");
h2("2.2. Fuzzy and Intuitionistic Fuzzy Approaches");
p("Fuzzy methods in malware detection include an evolving neuro-fuzzy classifier on permissions [@altaher17], ANFIS with fuzzy c-means clustering [@altaher17b], fuzzy clustering combined with gradient boosting [@tahamalebary21], Choquet fuzzy-integral fusion [@taha21math], fuzzy simulated-annealing feature selection [@seyfari24] and Mamdani-based ensemble fusion [@atacak23]. These are type-1 constructions in which non-membership is implicitly the complement of membership. IFSs [@atanassov86] relax this assumption and quantify hesitation explicitly; IF distances [@szmidt00], entropies [@burillo96], IF c-means clustering [@chaira11], IF twin support vector machines [@rezvani19] and IF decision-tree forests [@ren22] exploit this additional degree of freedom, and Wu and Song [@wu22] used IFSs to fuse deep malware classifiers. Using IF degrees as «input features» of a gradient-boosted ensemble has not been studied for traffic-based malware detection.");
h2("2.3. Evaluation Rigour in ML-Based Security");
p("Arp et al. [@arp22] and Pendlebury et al. [@pendlebury19] showed that data snooping, inappropriate baselines and temporal bias inflate reported performance, and Molina-Coronado et al. [@molina23] demonstrated that duplicated samples lead to optimistic evaluations of Android detectors; duplicates are pervasive in Android corpora [@irolla18,@zhao21]. Comparisons over resampled splits require variance-corrected tests because training sets overlap [@nadeau03,@dietterich98]. SHAP [@lundberg17,@lundberg20] is the standard tool for explaining tree ensembles and is increasingly used to audit what security models learn [@galli24,@liu22xai,@charmet22]. Table 1 summarises representative studies; none compares a fuzzy representation against equally tuned boosting baselines with significance tests.");
tbl({ caption: "**Table 1.** Representative related studies (Acc. = reported accuracy; “—” = not reported).",
  widths: [1350, 1750, 2900, 1050, 1250],
  header: ["Study", "Dataset", "Approach", "Acc. (%)", "Equal tuning / stat. test"],
  rows: [
    ["[@palma24]", "CICAndMal2017", "Random forest with feature normalisation", "79.81", "No / No"],
    ["[@abuthawabeh20]", "CICAndMal2017", "Conversation-level features + Extra Trees", "87.75", "No / No"],
    ["[@ferreira21]", "CICAndMal2017", "AdaBoost with random-forest base learners", "98.24", "No / No"],
    ["[@busch21]", "CICAndMal2017", "Network-flow graph neural network", "— (recall 99.42)", "No / No"],
    ["[@lashkari17]", "CIC-AAGM2017", "Nine traffic features, five classifiers (average)", "91.41", "No / No"],
    ["[@alani22]", "Adware flows", "Flow-based adware detection (AdStop)", "98.02", "No / No"],
    ["[@rustnguyen23]", "CIC-Darknet2020", "Random forest, application categories", "92.3", "No / No"],
    ["[@taha21math]", "Drebin-215", "Choquet fuzzy-integral classifier fusion", "95.08", "No / No"],
    ["[@wu22]", "Windows malware", "Atanassov IFS fusion of deep models", "99.92", "No / No"],
    ["This work", "CICAndMal2017, CIC-AAGM2017, CIC-Darknet2020", "IF-LightGBM vs four equally tuned boosting models", "{{T1_ACC}}", "Yes / Yes"],
  ] });

// ---------------- 3 Methods
h1("3. Materials and Methods");
h2("3.1. Overview");
p("Figure 1 summarises the framework. After removal of duplicates and identifiers, the IF degrees of every feature are computed from bounds estimated on training data only. IF-LightGBM and four baselines are tuned with the same budget, compared by repeated cross-validation with variance-corrected tests, and IF-LightGBM is analysed by an ablation and by SHAP.");
fig({ file: "results/v20/fig/fig1_framework_v20.png", w: 620, caption: "**Figure 1.** Overview of the IF-LightGBM framework and the evaluation protocol." });
h2("3.2. Datasets");
p("**CICAndMal2017** [@lashkari18,@cicurl] contains bidirectional flow statistics extracted with CICFlowMeter from traffic captured on real smartphones. We used a publicly shared, class-balanced flow-level subset of 14,998 flows; after removing one record with a missing label and the identifiers Flow ID, Source IP, Destination IP and Timestamp, 14,997 flows (7498 malware, 7499 benign) with 68 numeric features remained, with no exact duplicates. **CIC-AAGM2017** [@lashkari17,@aagmurl] contains 631,955 flows described by 79 flow features and no IP addresses or ports. We removed its exact duplicates (42.5% of rows) and 3355 rows whose feature vectors occur with conflicting labels, merged adware and general malware into the malware class and drew a seeded, class-balanced sample of 20,000 unique flows; five constant features were dropped, leaving 74. **CIC-Darknet2020** [@lashkari20,@darkurl] contains 158,616 CICFlowMeter flows of regular, VPN and Tor traffic labelled with eight application categories (audio streaming, browsing, chat, e-mail, file transfer, P2P, video streaming and VoIP); inconsistent spellings of the category labels were merged. Flow ID, IP addresses, ports and Timestamp were removed because they identify sessions and, in the case of ports, largely encode the application that defines the label. We then removed exact duplicates (33.6% of rows) and 5520 rows with conflicting labels, leaving 99,876 unique flows, and sampled at most 5000 flows per category (36,120 flows; all 4190 e-mail and 1930 VoIP flows were kept); 15 constant features were dropped, leaving 62. Table 2 summarises the datasets.");
tbl({ caption: "**Table 2.** Characteristics of the datasets after cleaning.",
  widths: [2300, 2300, 2300, 2300],
  header: ["Property", "CICAndMal2017", "CIC-AAGM2017", "CIC-Darknet2020"],
  rows: [
    ["Domain", "Android malware", "Android malware", "Regular, VPN and Tor application traffic"],
    ["Flow extractor", "CICFlowMeter", "ISCXFlowMeter", "CICFlowMeter"],
    ["Classes", "Malware (adware, ransomware, scareware, SMS) vs benign", "Malware (adware, general malware) vs benign", "Eight application categories"],
    ["Raw flows / exact duplicates", "14,998 / 0", "631,955 / 268,615 (42.5%)", "158,616 / 53,220 (33.6%)"],
    ["Flows used", "14,997 (7498 / 7499)", "20,000 (10,000 / 10,000)", "36,120 (≤5000 per class)"],
    ["Features (IF-augmented, 3d)", "68 (204)", "74 (222)", "62 (186)"],
  ] });
h2("3.3. Intuitionistic Fuzzy Sets");
p("An intuitionistic fuzzy set A in a universe X [@atanassov86] is");
eq("ifs", 1);
pn("with membership and non-membership functions μ_{A}, ν_{A}: X → [0, 1]. The hesitation (intuitionistic index) of x is");
eq("pi", 2);
pn("When π_{A} ≡ 0, A reduces to a classical fuzzy set. A non-membership function can be generated from a membership function by an intuitionistic fuzzy generator N with N(μ) ≤ 1 − μ, which guarantees π ≥ 0 [@chaira11,@burillo96].");
h2("3.4. IF-LightGBM: Intuitionistic Fuzzy Feature Augmentation");
p("Let x_{ij} be the value of feature j for flow i, and m_{j}, M_{j} its minimum and maximum on the training data. The membership degree of x_{ij} to the fuzzy concept ‘high value of feature j’ is the clipped min–max normalisation");
eq("mu", 3);
pn("and the non-membership and hesitation degrees are obtained with the quadratic generator N(μ) = (1 − μ)^{2} ≤ 1 − μ:");
eq("nu", 4);
eq("pij", 5);
pn("Each flow is represented by the augmented vector");
eq("aug3", 6);
pn("which is the input of LightGBM [@ke17]. Because μ_{ij} is an affine function of x_{ij} on the training range, it is not added separately (Proposition 1). Test values outside the training range are clipped, so no information from test data is used.");
h2("3.5. Analytical Properties for Tree Ensembles");
p("A decision tree partitions the data with axis-aligned thresholds. For a sample S and a feature f, let P(f, S) = {{s ∈ S : f(s) ≤ t} : t ∈ ℝ} denote the family of subsets obtainable by one threshold on f.");
p("**Proposition 1** «(monotone redundancy).» «If g is strictly monotone on the range of x_{j}, then P(g ∘ x_{j}, S) = P(x_{j}, S) up to complementation. Hence μ_{j} and ν_{j}, which are strictly increasing and strictly decreasing functions of x_{j} on the training range, cannot create a split that is unavailable on x_{j}.»");
p("«Proof.» For increasing g, x ≤ t ⇔ g(x) ≤ g(t); for decreasing g, x ≤ t ⇔ g(x) ≥ g(t), whose complement is again a single-threshold set. □");
p("**Proposition 2** «(hesitation as a two-sided interval).» «Since π = μ(1 − μ) = 1/4 − (μ − 1/2)^{2}, for any c ∈ (0, 1/4) the set {π > c} equals {μ ∈ (a, 1 − a)} with a = (1 − √(1 − 4c))/2. One split on π therefore isolates a symmetric interval of the scaled feature, a partition that requires two splits on x_{j}.»");
p("«Proof.» π > c ⇔ (μ − 1/2)^{2} < 1/4 − c ⇔ |μ − 1/2| < √(1 − 4c)/2. □");
p("**Proposition 3** «(vacuity for binary features).» «If x_{j} ∈ {0, 1}, then μ_{j} = x_{j}, ν_{j} = 1 − x_{j} and π_{j} ≡ 0, so the augmented features induce exactly the partitions induced by x_{j}.»");
p("«Proof.» Immediate from Equations (3)–(5) with m_{j} = 0 and M_{j} = 1, together with Proposition 1. □");
p("Under exact split search, any benefit of the augmentation must therefore come from the hesitation features of continuous variables, which dominate flow statistics such as inter-arrival times, packet sizes and byte counts. LightGBM, however, searches thresholds over feature histograms with a limited number of bins (255 by default), and each transformed copy of a feature is binned separately; ν and π can therefore also supply thresholds on x that the histogram of x itself does not offer. Section 4.2 tests both effects.");
h2("3.6. Baselines and Equal-Budget Tuning");
p("IF-LightGBM is compared with four baselines trained on the raw features: LightGBM [@ke17], which isolates the effect of the fuzzy features; XGBoost [@chen16]; AdaBoost [@freund97] with decision-tree base learners (SAMME); and GBM [@friedman01] as implemented in scikit-learn. Every model was tuned with the same procedure: Optuna [@akiba19] with the tree-structured Parzen estimator [@bergstra11] (seed 42), 20 trials, inner 3-fold stratified cross-validation on a stratified 80% partition of each dataset and mean ROC-AUC as the objective. For LightGBM and XGBoost, the number of trees was selected by early stopping (patience 50); for AdaBoost and GBM, it was tuned directly (50–400). The search spaces covered the number of leaves (15–255), learning rate (0.01–0.2), minimum samples per leaf, feature and bagging fractions and L2 regularisation for LightGBM; depth (3–12), learning rate, row and column subsampling, minimum child weight and L2 regularisation for XGBoost; learning rate (0.01–1) and base-tree depth (1–4) for AdaBoost; and learning rate (0.01–0.3), depth (2–6), subsampling and minimum samples per leaf for GBM. Because all tuning runs used the same TPE seed, IF-LightGBM and LightGBM selected the same settings on CICAndMal2017 except for the number of trees, so that this comparison differs essentially only in the input representation. Table 3 lists the selected settings. Experiments were run in Python {{PYV}} with LightGBM {{LGBV}}, XGBoost {{XGBV}}, scikit-learn {{SKV}}, Optuna {{OPV}} and SHAP {{SHV}} on a two-core cloud CPU instance.");
tbl({ caption: "**Table 3.** Hyperparameters selected by equal-budget tuning (inner-CV ROC-AUC in the last column).",
  widths: [1550, 1200, 4250, 1000], small: true,
  header: ["Dataset", "Model", "Selected hyperparameters", "Inner AUC"],
  rows: "{{T3_ROWS}}" });
h2("3.7. Evaluation Protocol and Statistical Inference");
p("All tuned models were evaluated by 3 × 5-fold repeated stratified cross-validation (J = 15 folds; seed 42), with the IF bounds recomputed inside each training fold. Performance was measured by accuracy, F1-score, the Matthews correlation coefficient (MCC) [@chicco20] and ROC-AUC:");
eq("accf1", 7);
eq("mcc", 8);
pn("For the eight-class CIC-Darknet2020 data, F1 is macro-averaged over classes, MCC is its multiclass generalisation and ROC-AUC is the macro-average of the one-versus-rest AUCs. The quality of the predicted probabilities was measured by the Brier score, the mean squared difference between the predicted malware probability and the true label, on the two binary datasets, and by the log-loss (cross-entropy) on CIC-Darknet2020; lower values are better for both. To characterise performance at a low false-alarm rate, the true-positive rate at a false-positive rate of 1% (TPR@1%FPR) was computed from the out-of-fold predictions of the first 5-fold repeat (macro-averaged one-versus-rest for CIC-Darknet2020). Because training sets overlap across folds, the ordinary paired t-test is anti-conservative; IF-LightGBM was compared with each baseline by the corrected resampled t-test [@nadeau03]");
eq("ct", 9);
pn("where d̄ and σ̂^{2} are the mean and variance of the paired fold differences and n_{te}/n_{tr} = 0.25. Within each dataset, the 16 p-values of the four main metrics (four baselines × four metrics) were adjusted by Holm’s procedure [@holm79], and the four p-values of the probability-quality metric were adjusted separately; we also report win/loss counts over the 15 folds and Cliff’s δ as an effect size.");
h2("3.8. Ablation and Explainability");
p("To identify the source of any gain, LightGBM with the tuned IF-LightGBM settings was trained on the same cross-validation folds with the raw features x and with [x, ν], [x, π] and the full augmentation [x, ν, π]. To test the role of histogram discretisation, raw-feature LightGBM was also trained with a finer histogram resolution (max_bin = 4095 instead of the default 255): on CICAndMal2017 with the IF-LightGBM settings, and on CIC-Darknet2020 tuned with the same budget as the other models. SHAP values [@lundberg17,@lundberg20] of the tuned IF-LightGBM, trained on the 80% tuning partition, were computed for 800 flows of the remaining 20% with the exact tree-path-dependent TreeSHAP algorithm in log-odds space and aggregated into the shares of the total mean absolute SHAP value contributed by the original, ν and π features; for the eight-class CIC-Darknet2020 model, absolute SHAP values were summed over the class outputs.");

// ---------------- 4 Results
h1("4. Results");
h2("4.1. Comparison with Equally Tuned Baselines");
p("{{R41}}");
tbl({ caption: "**Table 4.** Repeated cross-validation (3 × 5 folds) of the tuned models, mean ± standard deviation (best per dataset in bold; lower is better for Brier score and log-loss). For CIC-Darknet2020, F1 is macro-averaged, ROC-AUC is the macro one-versus-rest AUC and the last column is the log-loss; for the other datasets it is the Brier score.",
  widths: [1500, 1250, 1300, 1300, 1300, 1300, 1400], small: true,
  header: ["Dataset", "Model", "Accuracy", "F1", "MCC", "ROC-AUC", "Brier / log-loss"],
  rows: "{{T4_ROWS}}" });
tbl({ caption: "**Table 5.** Paired cross-validation contrasts of IF-LightGBM against each baseline (ΔAcc. and ΔAUC in percentage points; ΔBrier/LL: difference in Brier score, or in log-loss for CIC-Darknet2020, where negative values favour IF-LightGBM; folds won/lost out of 15 in parentheses; p_{Holm}: corrected resampled t-test, Holm-adjusted within each dataset over the 16 tests of accuracy, F1, MCC and ROC-AUC, and separately over the four probability-quality tests; δ: Cliff’s delta for ROC-AUC).",
  widths: [1500, 950, 1200, 850, 1150, 850, 1350, 850, 650], small: true,
  header: ["Dataset", "Baseline", "ΔAcc. (W/L)", "p_{Holm}", "ΔAUC (W/L)", "p_{Holm}", "ΔBrier/LL (W/L)", "p_{Holm}", "δ"],
  rows: "{{T5_ROWS}}" });
fig({ file: "results/v20/fig/fig_effect_v20.png", w: 620, caption: "**Figure 2.** IF-LightGBM minus each tuned baseline across the 15 cross-validation folds: mean paired difference and corrected 95% confidence interval (unadjusted for multiplicity) for accuracy and ROC-AUC, on a symmetric-log axis. Intervals entirely to the right of zero indicate a consistent advantage of IF-LightGBM." });
h2("4.2. Ablation of the Fuzzy Features");
p("{{R42}}");
tbl({ caption: "**Table 6.** Representation ablation for LightGBM with the tuned IF-LightGBM settings (3 × 5-fold CV; Δ relative to the raw features x with default binning, in percentage points; W/L out of 15 folds; p_{Holm}: corrected resampled t-test, Holm-adjusted over the six tests of each dataset).",
  widths: [1550, 2000, 1000, 900, 750, 1050, 900, 750, 1050], small: true,
  header: ["Dataset", "Representation", "Acc.", "ΔAcc.", "W/L", "p_{Holm}", "ΔAUC", "W/L", "p_{Holm}"],
  rows: "{{T6_ROWS}}" });
h2("4.3. Explainability");
p("{{R43}}");
fig({ file: "results/v20/fig/fig_shap_v20.png", w: 620, caption: "{{FIG3_CAP}}" });

// ---------------- 5 Discussion
h1("5. Discussion");
h2("5.1. Principal Findings");
p("{{D51}}");
h2("5.2. Security Interpretation");
p("{{D52}}");
h2("5.3. Comparison with Previous Studies");
p("{{D53}}");
h2("5.4. Limitations");
p("{{D54}}");

// ---------------- 6 Conclusions
h1("6. Conclusions");
p("{{CONCL}}");

B.push({ t: "back", items: [
  ["Author Contributions:", "Conceptualization, A.A.T. and A.H.O.; methodology, A.A.T.; software, A.A.A.; validation, A.A.T.; formal analysis, A.A.T.; investigation, A.H.O.; resources, Y.S.B.; data curation, A.H.S.; writing—original draft preparation, A.Y.; writing—review and editing, A.H.O.; visualization, A.E.; supervision, A.A.T.; project administration, Y.S.B.; funding acquisition, A.A.T. All authors have read and agreed to the published version of the manuscript."],
  ["Funding:", "This project was funded by the Deanship of Scientific Research (DSR) at King Abdulaziz University, Jeddah, Saudi Arabia, under grant no. (IPP: 308-830-2026)."],
  ["Institutional Review Board Statement:", "Not applicable."],
  ["Data Availability Statement:", "CICAndMal2017 [@cicurl], CIC-AAGM2017 [@aagmurl] and CIC-Darknet2020 [@darkurl] are publicly available. The code, sampling seeds, tuned hyperparameters and fold-level scores are available from the corresponding author upon reasonable request."],
  ["Use of Generative AI:", "During the preparation of this manuscript, the authors used Claude (Anthropic) to assist with code development, statistical analysis and drafting. The authors have reviewed and edited the output and take full responsibility for the content of this publication."],
  ["Acknowledgments:", "The authors gratefully acknowledge the Deanship of Scientific Research (DSR) at King Abdulaziz University, Jeddah, Saudi Arabia, for funding this project under grant no. (IPP: 308-830-2026) and for technical support."],
  ["Conflicts of Interest:", "The authors declare no conflicts of interest."],
] });
B.push({ t: "abbr", rows: [
  ["GBM", "Gradient boosting machine"], ["IF / IFS", "Intuitionistic fuzzy / intuitionistic fuzzy set"], ["MCC", "Matthews correlation coefficient"],
  ["ROC-AUC", "Area under the receiver operating characteristic curve"], ["SHAP", "SHapley Additive exPlanations"],
  ["TPE", "Tree-structured Parzen estimator"], ["VPN", "Virtual private network"],
] });
B.push({ t: "refs" });
module.exports = B;
