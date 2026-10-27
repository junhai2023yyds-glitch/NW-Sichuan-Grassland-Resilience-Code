# NW Sichuan Grassland Resilience: Reproducible Analysis Code

This repository contains the reproducible code used for grassland ecosystem resilience-index calculation, spatial-change analysis, and machine-learning modeling and interpretation in northwestern Sichuan, China.

The repository includes the R workflow for data preprocessing, resilience-index calculation, spatial analysis, and supplementary temporal-dependence and endpoint checks; the Python workflow for model comparison, interpretation, period-specific grazing analysis, and driver-grouping robustness; and records of the optimal hyperparameters across the 25 outer tasks of the nested grouped cross-validation.

## 1. Analysis workflow

The repository contains two main analysis components.

### 1.1 R-based resilience and spatial analyses

The `R_analysis/` directory contains the scripts used for:

- raster and vector data preprocessing;
- grassland ecosystem resilience-index calculation;
- pixel-level spatial and temporal change analyses of resilience;
- calculation of actual pixel areas;
- generation of the corresponding analysis outputs used in the study.

The R workflow includes:

```text
R_analysis/
├── 01_RI_calculation.R
├── 02_RI_spatial_analysis.R
├── 03_EdgeRule_Sensitivity_NullModel_Portable.R
├── 04_RI_SerialDependence_LinearSensitivity.R
└── 05_RI_ExploratoryTurning_Audit.R
```

All data preprocessing, resilience-index calculation, and spatial heterogeneity analyses were performed in **R 4.5.2**. Raster and vector data reading, cropping, masking, pixel-wise calculations, and actual pixel-area calculations were implemented using the `terra` package (**v1.8-93**). Linear regression, statistical procedures, and file handling used the base R packages `stats`, `tools`, and `utils`, respectively. Pettitt change-point detection and segmented regression were implemented using custom functions in the analysis scripts; the `trend` and `segmented` packages were not used directly.

### Supplementary R analyses and interpretation

- `03_EdgeRule_Sensitivity_NullModel_Portable.R`: evaluates endpoint-rule sensitivity using alternative 3-, 5-, and 7-year endpoint restrictions and a sampled null-model analysis. It assesses sensitivity of the early turning-point concentration, not pixel-level validation of every turning point.
- `04_RI_SerialDependence_LinearSensitivity.R`: diagnoses serial dependence and evaluates regional and pixel-level **single-linear-slope** sensitivity using heteroskedasticity-and-autocorrelation-consistent (HAC) inference and Benjamini–Hochberg (BH) / Benjamini–Yekutieli (BY) multiple-testing corrections.
- `05_RI_ExploratoryTurning_Audit.R`: audits the original spatial-classification outputs and summarizes them alongside the linear-sensitivity results. It does **not** re-estimate trends, refit turning points, or apply a validated autocorrelation/FDR correction to the original turning-point map.

The original linear/turning-pattern coverage (approximately 54.01%) is an **exploratory classification summary**, not the area of change confirmed after temporal-dependence and multiple-testing correction. Linear-slope sensitivity percentages and candidate-turning percentages have different meanings and must not be substituted for one another. Experimental turning-point correction procedures that failed validation are not part of this recommended workflow.

Reproducing the RI series also requires the actual source inputs and preprocessing settings, including the temporal and spatial domain used for coefficient normalization. Availability of an RI-calculation script alone does not verify the provenance of an existing RI raster series.

### 1.2 Machine-learning analysis

Seven ecological drivers are included:

- `SM` — soil moisture
- `PRE` — precipitation
- `TMN` — minimum temperature
- `TMX` — maximum temperature
- `SR` — solar radiation
- `VPD` — vapor pressure deficit
- `GI` — grazing intensity

`Year` is additionally included in the main model as a temporal covariate. It participates in model training and tuning but is excluded when the contributions of the seven ecological drivers are normalized to 100%.

Five candidate models are compared:

- OLS
- exact RBF-SVR
- Random Forest (RF)
- LightGBM
- XGBoost

The main validation framework is **repeated nested grouped cross-validation: 5 inner folds and 5 outer folds repeated 5 times (25 outer tasks)**, using `patch_id1` as the grouping unit so that annual observations from the same spatial grid do not occur in both training and test folds.

The final interpretation model is selected automatically according to the highest mean outer-fold Test R² across the 25 outer tasks. The interpretation model is not preset.

## 2. Repository structure

```text
NW-Sichuan-Grassland-Resilience-Code/
│
├── README.md
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── LICENSE
│
├── R_analysis/
│   ├── 01_RI_calculation.R
│   ├── 02_RI_spatial_analysis.R
│   ├── 03_EdgeRule_Sensitivity_NullModel_Portable.R
│   ├── 04_RI_SerialDependence_LinearSensitivity.R
│   └── 05_RI_ExploratoryTurning_Audit.R
│
├── machine_learning/
│   ├── 01_Main_ML_NestedCV_OOFSHAP_ALE.py
│   ├── 02_Sensitivity_Analyses.py
│   ├── 03_Redraw_Final_Figures.py
│   ├── 04_SHAP_Interaction_SR_SM.py
│   ├── 05_GI_2001_Break_Period_Sensitivity.py
│   ├── 06_Driver_Grouping_Robustness.py
│   └── ml_common.py
│
├── supplementary/
│   └── NestedGroupCV_Best_Hyperparameters_25_Outer_Tasks.xlsx
│
├── data/
│   └── README.md
│
└── output/                    # generated automatically; not tracked by Git
```

Keep all seven Python files, including `ml_common.py`, in the same `machine_learning/` directory. Check the configuration of each script before execution: the supplementary R scripts do not all use the same default input and output folders. Relative R paths resolve from the working directory; run the R commands below from the repository root.

## 3. Supplementary hyperparameter records

The `supplementary/` directory contains the complete hyperparameter records associated with the nested grouped cross-validation.

The file

```text
NestedGroupCV_Best_Hyperparameters_25_Outer_Tasks.xlsx
```

reports the optimal hyperparameters obtained for the tuned models across all **25 outer tasks** generated by the 5 repetitions × 5 folds validation design. OLS is retained as an untuned linear baseline.

The workbook includes both a compact cross-model overview and model-specific sheets containing only the hyperparameters relevant to each algorithm.

## 4. Input data

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
- `Year`: temporal covariate in the main model.

Each `patch_id1-Year` combination must be unique.

Predictor missing values are allowed and are imputed by the median **within each training fold**. Missing values in `RI` or `Year` are not allowed.

## 5. Variable units

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

## 6. Main machine-learning analysis

Run:

```bash
cd machine_learning
python 01_Main_ML_NestedCV_OOFSHAP_ALE.py
```

The main script performs:

1. comparison of OLS, exact RBF-SVR, RF, LightGBM, and XGBoost;
2. nested grouped cross-validation with 5 inner folds and 5 outer folds repeated 5 times;
3. Optuna-TPE hyperparameter optimization;
4. automatic selection of the best-performing interpretation model;
5. Top-2 repeated k-fold corrected t-test and supplementary Wilcoxon test;
6. repeated OOF-SHAP based on the 25 outer models;
7. mean |SHAP|, Kendall's W, and patch-level cluster-bootstrap 95% confidence intervals;
8. contribution normalization across the seven ecological drivers after excluding `Year`;
9. Top-6 accumulated local effects (ALE) analysis;
10. export of model results, figures, configurations, software versions, hashes, and run records.

The script supports checkpoint-based continuation. Completed runs are recorded using `latest_main_run.txt`.

## 7. Sensitivity analyses

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

### D. GI data-transition and period-specific sensitivity

Run the supplementary grazing-intensity analysis using its repository filename:

```bash
python 05_GI_2001_Break_Period_Sensitivity.py
```

This analysis examines GI changes around the 2000–2001 transition and compares period-specific models for 1985–2000 and 2001–2020. It requires the completed main run and the same modeling table, including `Block100`. It reuses the main model choice and the 25 outer grid splits, with separate inner 5-fold tuning in each period. It uses 100 km spatial-block resampling for the GI-transition diagnostic and compares GI ALE responses on a common support and reference. Period-specific models are fitted and tuned separately; they are not simply the full-period model evaluated on two subsets.

Input/run overrides are available through `NW_GI05_DATA_CSV`, `NW_GI05_OUTPUT_ROOT`, and `NW_GI05_MAIN_RUN_DIR`.

Differences between periods indicate period dependence of the fitted associations. They do not isolate a causal effect of changing GI-data resolution, establish a universal optimal grazing intensity, or replace a common-resolution experiment using the original GI rasters. Cross-model ALE percentile bands are empirical dispersion summaries, not independent-sample confidence intervals.

### E. Driver-grouping robustness

After completing the main analysis, run:

```bash
python 06_Driver_Grouping_Robustness.py
```

This analysis requires the main-run OOF-SHAP, split, score, hyperparameter, and input-hash records. It rebuilds the 25 outer models with their saved hyperparameters and seeds, without a new model-selection or Optuna tuning stage. Optional No-Year comparisons use compatible sensitivity outputs linked to the same main run. Input/run overrides are available through `NW_C16_DATA_CSV`, `NW_C16_OUTPUT_ROOT`, and `NW_C16_MAIN_RUN_DIR`.

It evaluates alternative group assignments, including those for SR and VPD, and distinguishes:

- sums of individual mean absolute OOF-SHAP values;
- joint grouped permutation importance, expressed as the decrease in held-out R²;
- group-level Shapley attribution, calculated on the configured sample.

These are different quantities. A decrease in R² is not a percentage contribution, and group-level Shapley attribution is not interchangeable with a sum of individual SHAP magnitudes. Report the sampling scheme and normalization denominator for each result. `Year` is held fixed during grouped permutation and treated as a separate group in group-level Shapley attribution; ecological-only percentages exclude it from their denominator.

Consistency across 25 outer tasks is descriptive robustness evidence, not a significance test based on 25 independent samples. Grouping sensitivity does not remove dependence among predictors or the sensitivity of ecological rankings to including `Year`.

## 8. Figure redrawing

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

## 9. SR × SM SHAP interaction analysis

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

## 10. Recommended execution order

Configure input locations before running any script. Do not rerun RI production solely to execute supplementary checks if the appropriate existing annual RI rasters are already available.

### R analysis

From the repository root, run the required production steps, followed by the supplementary analyses:

```bash
Rscript R_analysis/01_RI_calculation.R
Rscript R_analysis/02_RI_spatial_analysis.R
Rscript R_analysis/03_EdgeRule_Sensitivity_NullModel_Portable.R
Rscript R_analysis/04_RI_SerialDependence_LinearSensitivity.R
Rscript R_analysis/05_RI_ExploratoryTurning_Audit.R
```

Dependencies and path configuration:

- `01`: configure the source-data and preprocessing settings inside the script before RI production.
- `02`: requires the annual RI inputs and produces the original spatial-classification outputs. Check its input/output configuration rather than assuming another script's defaults.
- `03`: uses annual RI rasters and the grassland mask directly. Its editable defaults are `ri_dir = "data/ri"`, `grass_shp = "data/mask/grassland.shp"`, and `report_root = "results"`.
- `04`: uses `RI_ANNUAL_DIR` (default `data/RI_annual`), `RI_GRASS_MASK` (default `data/mask/grassland.shp`), and `RI_SENSITIVITY_OUT` (default `output/reviewer2_comment9/04_sensitivity`).
- `05`: requires **both** the original `02` outputs and the `04` sensitivity outputs. Configure `RI_SPATIAL_OUT` (default `output/RI_spatial`) and `RI_SENSITIVITY_OUT`; use `RI_ORIGINAL_RUN` and `RI_SENSITIVITY_RUN` to select explicit runs where needed. Its output root is controlled by `RI_EXPLORATORY_OUT` (default `output/reviewer2_comment9/05_exploratory_audit`).

The `03` and `04` default annual-raster folders differ. Point both scripts to the same intended RI series. `03` and `04` can be run independently once their inputs exist; `05` must follow the required `02` and `04` outputs. Preserve masks, valid-area definitions, years, and input versions consistently across runs.

### Machine-learning analysis

```bash
cd machine_learning

python 01_Main_ML_NestedCV_OOFSHAP_ALE.py
python 02_Sensitivity_Analyses.py
python 03_Redraw_Final_Figures.py
python 04_SHAP_Interaction_SR_SM.py
python 05_GI_2001_Break_Period_Sensitivity.py
python 06_Driver_Grouping_Robustness.py
```

The numbering is a convenient presentation order, not a claim that every script consumes the preceding script's output. `03` uses the completed main and sensitivity outputs. `04`, `05`, and `06` use the relevant main-run artifacts. The two supplementary Python scripts (`05` and `06`) do not depend on each other. Check each supplementary script's configuration and required files before execution.

Use the actual repository filenames in commands. Some GI-script internal labels and output names retain a legacy `06` prefix although its repository filename starts with `05`. Likewise, legacy `S12_*` filenames in the endpoint/null workflow are output identifiers, not authoritative table numbers in the revised manuscript. Match outputs to the final supplementary tables by analysis content, not by historical prefixes.

## 11. Software environment

- **Python 3.11** was used for the machine-learning analyses.
- **R 4.5.2** was used for data preprocessing, resilience-index calculation, and spatial heterogeneity analyses.

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

`statsmodels` is an optional additional dependency for the GI script's auxiliary annual segmented HAC regression. It is not currently listed in `requirements.txt`; that auxiliary step is skipped if it is unavailable. Install it separately if that output is required.

Python dependencies are listed in `requirements.txt`. Each completed machine-learning run also exports the actually installed software versions to its configuration/environment folder.

### R environment

- **R 4.5.2** was used for data preprocessing, resilience-index calculation, and spatial heterogeneity analyses.
- **terra 1.8-93** was used for raster and vector data reading, cropping, masking, pixel-wise calculations, and actual pixel-area calculations.
- The base R packages **stats**, **tools**, and **utils** were used for linear regression/statistical procedures and file handling.
- Pettitt change-point detection and segmented regression were implemented using custom functions in the analysis scripts; the `trend` and `segmented` packages were not used directly.

The supplementary R scripts require `terra`. `sandwich` is optional for the linear-sensitivity script's cross-check rather than a mandatory dependency.

The R scripts under `R_analysis/` contain the implementation details. The software environment can additionally be recorded using `sessionInfo()`.

## 12. Reproducibility design

Key safeguards include:

- grouped train/test separation by spatial unit;
- 5 repeated outer partitions;
- fold-specific preprocessing;
- nested hyperparameter optimization;
- checkpoint continuation;
- Optuna SQLite storage;
- saved split manifests;
- complete hyperparameter records for the 25 outer tasks;
- saved software versions;
- SHA-256 file hashes;
- output manifests;
- repeated OOF-SHAP rather than in-sample SHAP;
- ALE instead of PDP for nonlinear interpretation under correlated predictors;
- sensitivity analyses for temporal covariate dependence, spatial independence, and temperature-variable parameterization;
- supplementary endpoint/null checks, serial-dependence and linear HAC/FDR sensitivity, exploratory classification auditing, period-specific GI analysis, and driver-grouping robustness.

Saved code and run metadata facilitate reproduction, but do not by themselves establish that every possible workflow has been rerun successfully. Match the archived inputs, settings, and actual outputs to the analysis being reproduced.

## 13. Output files

Machine-learning outputs are written under:

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

Supplementary scripts create additional analysis-specific outputs in their configured roots. R03 defaults to `results/`, whereas R04 and R05 default to separate subdirectories under `output/reviewer2_comment9/`. Keep their run records with the corresponding results.

Pointer files such as `latest_main_run.txt` and `latest_sensitivity_run.txt` are generated automatically and should not be committed to Git.

## 14. Data availability

The complete modeling dataset is not included in the repository unless redistribution is permitted by the source-data licenses and the study's data-sharing policy.

To reproduce the machine-learning workflow with an authorized copy of the modeling table, place it at:

```text
data/modeling_data.csv
```

See `data/README.md` for the required modeling-table schema. R analyses additionally require the relevant annual RI rasters, grassland mask, and, for upstream RI calculation, the source input series. The modeling CSV alone cannot reproduce the raster-based R analyses.

## 15. Code availability

Code for resilience-index calculation, spatial analysis, and machine-learning modeling and interpretation, together with the complete optimal hyperparameter records for the 25 outer tasks of the tuned models, is publicly available at:

https://github.com/junhai2023yyds-glitch/NW-Sichuan-Grassland-Resilience-Code

## 16. Citation

If you use this repository, please cite the associated article and this code repository. Citation metadata are provided in `CITATION.cff`.

## 17. License

This repository is released under the **MIT License**. See `LICENSE` for details.
