# Modeling data

The public analysis scripts expect the modeling table at:

```text
data/modeling_data.csv
```

Required columns:

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

Definitions:

- `RI`: response variable.
- `patch_id1`: 20 km spatial-grid grouping identifier.
- `Block100`: 100 km spatial-block grouping identifier.
- `Year`: year.
- `SM`, `PRE`, `TMN`, `TMX`, `SR`, `VPD`, `GI`: ecological drivers.

Requirements:

- each `patch_id1-Year` combination must be unique;
- `RI` and `Year` must not contain missing values;
- predictor missing values may be present and are imputed within training folds;
- each `patch_id1` must belong to only one `Block100`.

The full modeling dataset should only be distributed when permitted by the source-data licenses and the study's data-sharing policy.
