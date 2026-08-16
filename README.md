# NW Sichuan Grassland Resilience: Machine-Learning Analysis

This repository contains the machine-learning workflow used to analyze the environmental drivers of grassland ecosystem resilience in northwestern Sichuan, China.

The current workflow supports reproducible model comparison, out-of-fold model interpretation, nonlinear response analysis, sensitivity testing, and supplementary SHAP interaction analysis.

## 1. Machine-learning workflow

Seven ecological drivers are included:

- `SM` — soil moisture
- `PRE` — precipitation
- `TMN` — minimum temperature
- `TMX` — maximum temperature
- `SR` — solar radiation
- `VPD` — vapor pressure deficit
- `GI` — grazing intensity

`Year` is additionally included in the main model as a temporal assistance/control variable. It participates in model training and tuning but is excluded when the contributions of the seven ecological drivers are normalized to 100%.

Five candidate models are compared:

- OLS
- exact RBF-SVR
- Random Forest (RF)
- LightGBM
- XGBoost

The main validation framework is **5 repetitions × 5 folds nested grouped cross-validation**, using `patch_id1` as the grouping unit so that annual observations from the same spatial grid do not occur in both training and test folds.

The final interpretation model is selected automatically according to the highest mean outer-fold Test R² across the 25 outer tasks. The interpretation model is not preset.

## 2. Repository structure

```text
NW-Sichuan-Grassland-Resilience-Code/
│
├── README.md
├── requirements.txt
├── .gitignore
├── CITATION.cff
│
├── data/
│   └── README.md
│
├── machine_learning/
│   ├── 01_Main_ML_NestedCV_OOFSHAP_ALE.py
│   ├── 02_Sensitivity_Analyses.py
│   ├── 03_Redraw_Final_Figures.py
│   ├── 04_SHAP_Interaction_SR_SM.py
│   └── ml_common.py
│
└── output/                    # generated automatically; not tracked by Git
```

The public GitHub version uses repository-relative paths. Keep the five Python files in the same `machine_learning/` directory.

## 3. Input data

Place the modeling table at:

```text
data/modeling_data.csv
```

The input table must contain at least:

```text
RI
patch_id1
Block100
Year
SM
PRE
TMN
TMX
SR
VPD
GI
```

### Required identifiers

- `RI`: grassland ecosystem resilience index.
- `patch_id1`: 20 km spatial-grid identifier used for the main grouped cross-validation.
- `Block100`: 100 km spatial-block identifier used for spatial sensitivity analysis.
- `Year`: temporal assistance/control predictor in the main model.

Each `patch_id1-Year` combination must be unique.

Predictor missing values are allowed and are imputed by the median **within each training fold**. Missing values in `RI` or `Year` are not allowed.

## 4. Variable units

| Variable | Unit |
|---|---|
| SM | mm |
| PRE | mm |
| TMN | °C |
| TMX | °C |
| SR | kWh/m² |
| VPD | kPa |
| GI | SU/ha |
| Tmean | °C |
| DTR | °C |

`Tmean` and `DTR` are generated only for the temperature reparameterization sensitivity analysis:

```text
Tmean = (TMX + TMN) / 2
DTR   = TMX - TMN
```

## 5. Main analysis

Run:

```bash
cd machine_learning
python 01_Main_ML_NestedCV_OOFSHAP_ALE.py
```

The main script performs:

1. comparison of OLS, exact RBF-SVR, RF, LightGBM, and XGBoost;
2. 5 × 5 repeated nested grouped cross-validation;
3. Optuna-TPE hyperparameter optimization;
4. automatic selection of the best-performing interpretation model;
5. Top-2 repeated k-fold corrected t-test and supplementary Wilcoxon test;
6. repeated OOF-SHAP based on the 25 outer models;
7. mean |SHAP|, Kendall's W, and patch-level cluster-bootstrap 95% confidence intervals;
8. contribution normalization across the seven ecological drivers after excluding `Year`;
9. Top-6 accumulated local effects (ALE) analysis;
10. export of model results, figures, configurations, software versions, hashes, and run records.

The script supports checkpoint-based continuation. Completed runs are recorded using `latest_main_run.txt`.

## 6. Sensitivity analyses

After the main analysis is complete, run:

```bash
python 02_Sensitivity_Analyses.py
```

Three sensitivity analyses are performed for the model selected by the main analysis.

### A. No-Year

The seven ecological drivers are retained, while `Year` is removed. The same 25 `patch_id1` outer test splits are reused, and hyperparameters are retuned within each outer training set.

Outputs include:

- predictive-performance comparison;
- SHAP contribution and ranking comparison;
- Top-6 ALE comparison;
- ALE turning-point stability.

### B. 100 km spatial block cross-validation

The model uses the seven ecological drivers plus `Year`, but both outer and inner grouped cross-validation are based on `Block100`.

This analysis evaluates the robustness of predictive performance under stronger spatial independence.

### C. Temperature reparameterization

`TMX` and `TMN` are replaced by:

```text
Tmean = (TMX + TMN) / 2
DTR   = TMX - TMN
```

The same main-analysis `patch_id1` outer splits are reused. The analysis compares:

- predictive performance;
- total temperature-group contribution;
- non-temperature SHAP ranking stability;
- sensitivity-specific OOF-SHAP and ALE results.

Completed sensitivity runs are recorded using `latest_sensitivity_run.txt`.

## 7. Figure redrawing

To regenerate publication figures from completed analysis outputs without retraining models or recalculating SHAP/ALE, run:

```bash
python 03_Redraw_Final_Figures.py
```

This script:

- reads the latest completed main and sensitivity runs;
- does not rerun Nested CV;
- does not rerun Optuna;
- does not recalculate OOF-SHAP;
- does not recalculate ALE;
- recreates publication-ready figures in PNG, TIF, PDF, and SVG formats.

Figures use Times New Roman and 600 dpi for raster output.

## 8. SR × SM SHAP interaction analysis

After the main analysis is complete, run:

```bash
python 04_SHAP_Interaction_SR_SM.py
```

This supplementary analysis:

- reuses the selected tree-based model from the main analysis;
- reuses the same 25 outer `patch_id1` splits;
- reuses the outer-task hyperparameters from the main analysis;
- rebuilds each outer model using its corresponding training data;
- calculates TreeSHAP interaction values only for the outer test fold;
- summarizes all 21 pairwise interactions among the seven ecological drivers;
- reports the SR × SM interaction strength and patch-level cluster-bootstrap 95% confidence interval;
- generates the formal SR × SM supplementary interaction figure.

`Year` remains in the fitted model but is excluded from the 21 ecological-driver pairwise interaction ranking.

This script requires the selected interpretation model to be tree-based (`RF`, `LightGBM`, or `XGBoost`).

## 9. Recommended execution order

For a full reproduction:

```bash
cd machine_learning

python 01_Main_ML_NestedCV_OOFSHAP_ALE.py
python 02_Sensitivity_Analyses.py
python 03_Redraw_Final_Figures.py
python 04_SHAP_Interaction_SR_SM.py
```

Scripts `03` and `04` are independent after the required upstream outputs exist, so their relative order can be changed.

## 10. Software environment

- **Python 3.11** was used for the machine-learning analyses.
- **R 4.5.2** was used for resilience-index calculation and spatial analyses.

### Python dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- numpy
- pandas
- scipy
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- optuna
- shap
- joblib
- openpyxl

Python dependencies are listed in `requirements.txt`. Each completed machine-learning run also exports the actually installed software versions to its configuration/environment folder.

### R environment

Required R packages are specified in the corresponding scripts under `R_analysis/`. For reproducibility, the R environment can additionally be recorded using `sessionInfo()`.

## 11. Reproducibility design

Key safeguards include:

- grouped train/test separation by spatial unit;
- 5 repeated outer partitions;
- fold-specific preprocessing;
- nested hyperparameter optimization;
- checkpoint continuation;
- Optuna SQLite storage;
- saved split manifests;
- saved hyperparameter tables;
- saved software versions;
- SHA-256 file hashes;
- output manifests;
- repeated OOF-SHAP rather than in-sample SHAP;
- ALE instead of PDP for nonlinear interpretation under correlated predictors;
- sensitivity analyses for temporal covariate dependence, spatial independence, and temperature-variable parameterization.

## 12. Output files

Analysis outputs are written under:

```text
output/
```

The scripts automatically create timestamped run directories such as:

```text
Final_ML_Main_YYYYMMDD_HHMMSS/
Final_ML_Sensitivity_YYYYMMDD_HHMMSS/
Final_ML_Interaction_SR_SM_YYYYMMDD_HHMMSS/
Final_Publication_Figures/
```

Pointer files such as `latest_main_run.txt` and `latest_sensitivity_run.txt` are generated automatically and should not be committed to Git.

## 13. Data availability

The complete modeling dataset is not included in the repository unless redistribution is permitted by the source-data licenses and the study's data-sharing policy.

To reproduce the workflow with an authorized copy of the modeling table, place it at:

```text
data/modeling_data.csv
```

See `data/README.md` for the required schema.

## 14. Citation

If you use this workflow, please cite the associated article and the archived software release.

The repository can be archived through Zenodo so that each software release receives a persistent DOI. Update `CITATION.cff` with the final article bibliographic information once the paper is published.

## 15. License

Add the license that matches the intended reuse conditions of the code before final public release. A permissive license such as MIT is commonly used for research code, but the final choice should be consistent with institutional and project requirements.
