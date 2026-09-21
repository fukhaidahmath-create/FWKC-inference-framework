# FWKC reviewer execution

The R script is portable and does not assume the author's local drive.

## Input data

Provide a file containing binary columns `D` and `T` (0/1) using either:

- `data/FWKC_input.xlsx` (also `.xls` or `.csv` is accepted), or
- environment variable `FWKC_INPUT_PATH`.

In an interactive R/RStudio session, if neither is supplied, the script opens a file picker.

## V25 reference artifacts

Optional. To reuse completed V25 stages, place them under:

`v25_reference/run_publication/stage_archive/`

with:

- `stage_01_validation_bayes.rds`
- `stage_02_real_data.rds`

Alternatively set `FWKC_V25_SOURCE_DIR`.

If these artifacts are absent, the script recomputes the same-prior HMC cross-check and the B=499 real-data grid. That fallback requires `cmdstanr`, `posterior`, and a working CmdStan installation.

## Run

Interactive R/RStudio:

```r
source("FWKC_inference_framework_v1.1_reviewer.R")
```

Command line:

```text
Rscript FWKC_inference_framework_v1.1_reviewer.R
```

Core packages: `readxl`, `openxlsx`, `randomForest`, `posterior`.

## Reviewer-priority robustness tier

The reviewer-priority preset uses:

- `robustness_M = 40`
- `robustness_B = 49`

All other reviewer-priority statistical settings are unchanged from the preceding portable v1.1 reviewer script.
