# IF-LightGBM

Code and result files for the paper *Intuitionistic Fuzzy Feature Augmentation of LightGBM for Network-Traffic-Based Android Malware Detection: A Controlled and Explainable Evaluation* (A.A. Taha).

IF-LightGBM augments every flow feature x with its intuitionistic fuzzy non-membership ν = (1 − μ)² and hesitation π = μ(1 − μ), where μ is the clipped min–max scaled feature, and trains LightGBM on [x, ν, π].

Python 3.13.16, LightGBM 4.7.0, XGBoost 3.4.1, scikit-learn 1.9.1, Optuna 5.0.0, SHAP 0.54.1.
All random seeds are 42. Install with `pip install -r requirements.txt`. **Run every script from the repository root**; all paths are relative to it.

## Folder layout
- `data/` — put the downloaded CSV files here (not included; see below)
- `results/` — result files produced by the scripts. The files used for the paper (tuned hyperparameters, per-fold scores, statistical contrasts, SHAP shares) are included so the tables can be checked without re-running the experiments.

## Datasets (not included)
Download from the Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/) and place in `data/`:
- `data/CICANDMAL2017-75000-malware-7500 Goodware.csv` — CICAndMal2017 flow subset (14,998 flows; target `Label1`)
- `data/TotalFeatures-ISCXFlowMeter.csv` — CIC-AAGM2017; `python 1_experiments/build_sample.py` creates `data/aagm2017_sample20k.csv` (de-duplicated, balanced, 20,000 flows)
- `data/Darknet.CSV` — CIC-Darknet2020; `python 1_experiments/confirm.py darknet all` builds the de-duplicated sample (≤5000 flows per category)

## 1_experiments
| Script | Purpose |
|---|---|
| `run_cic6.py`, `run_aagm.py` | Equal-budget Optuna tuning (20 TPE trials, inner 3-fold CV) and 3×5 repeated CV for CICAndMal2017 and CIC-AAGM2017 |
| `confirm.py` | Same protocol for CIC-Darknet2020 (`python confirm.py darknet all`); resumable (Optuna SQLite + fold CSV); also includes the LightGBM max_bin = 4095 control |
| `exp18.py` | Ablation x, [x,ν], [x,π], [x,ν,π] (CICAndMal2017, CIC-AAGM2017), port-removal check |
| `ablate20.py` | Same ablation for CIC-Darknet2020 |
| `finebin2.py` | CICAndMal2017 with max_bin = 4095 and IF-LightGBM settings |
| `nested18.py` | Nested CV for IF-LightGBM vs LightGBM |
| `oof.py` | Out-of-fold predictions of the first 5-fold repeat (used for TPR@1%FPR) |

## 2_statistics
| Script | Purpose |
|---|---|
| `stats.py` | Table 4/5: means, corrected resampled t-test (Nadeau–Bengio), Holm over 16 tests per dataset, Cliff's δ |
| `stats_prob.py` | Brier score / log-loss contrasts (Holm over 4 tests per dataset) |
| `tpr1.py` | TPR at 1% FPR from out-of-fold predictions |
| `shap_bin.py`, `shap_dark.py` | SHAP family shares (tree-path-dependent TreeSHAP, log-odds) |
| `nested_stats.py` | Statistics for the nested CV |

## 3_figures
`fig1.py` (framework), `fig_effect.py` (Figure 2, paired differences), `fig_shap.py` (Figure 3, SHAP).

## Run order
1. Experiments (1_experiments) → 2. `stats.py`, `stats_prob.py`, `tpr1.py`, SHAP scripts → 3. figure scripts

## Citation
If you use this code, please cite the paper (citation details will be added on publication).

## License
MIT — see `LICENSE`. The datasets are not redistributed and remain under the terms of the Canadian Institute for Cybersecurity.
