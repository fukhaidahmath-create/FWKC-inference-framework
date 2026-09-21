# FWKC Inference Framework

This repository provides a reproducible implementation of the FWKC
(Finite-Sample Weighted Kappa Coefficient) agreement inference framework
described in the manuscript:

**"A Finite-Sample Inferential Framework for the Weighted Kappa Coefficient (FWKC) in Binary Diagnostic Agreement Studies"**

## Current Version

The current repository version is the reviewer-priority implementation:

**v1.1.1 reviewer-priority release.**

Main script:

[`FWKC_inference_framework_v1.1_reviewer.R`](FWKC_inference_framework_v1.1_reviewer.R)

This implementation is derived from the validated V25 publication engine and
is configured for reviewer-response analyses under a limited computational
time budget.

It is not intended to replace the maximum-precision V25 publication preset.

## Methodological Components

The framework integrates:

- Beta-binomial uncertainty modeling for sensitivity and specificity
- Monte Carlo posterior-predictive pseudo-raters
- Hierarchical Bayesian pooling
- Random Forest aggregation
- Inverse-SE Bayesian/ML fusion
- Full-pipeline nonparametric bootstrap inference
- Delta-covariance uncertainty diagnostics
- Finite-sample calibration
- Independent data-generating-mechanism (DGM) coverage evaluation
- Robustness and sensitivity analyses
- External cross-domain diagnostic-data validation
- Scalability and computational diagnostics
- Programmatic reviewer-readiness quality-control checks

## Reviewer-Priority v1.1 Configuration

The default `reviewer_priority` profile preserves the complete primary
simulation grid:

- Sample sizes: `n = 30, 50, 100`
- Prevalence values: `p = 0.1, ..., 0.9`
- Loss/weight values: `c = 0.1, ..., 0.9`
- Total primary scenarios: `243`
- Independent DGM datasets per scenario: `M = 120`
- Primary inner bootstrap: `B = 29`
- Monte Carlo pseudo-raters: `n_mc = 150`
- Bayesian chains: `4`
- Random Forest trees: `300`
- Gibbs iterations: `400`
- Burn-in: `100`
- Thinning: `1`
- Robustness repetitions: `robustness_M = 40`
- Robustness bootstrap: `robustness_B = 49`
- Real-data bootstrap: `B_boot_real = 499`

Higher-resolution bootstrap and Monte Carlo settings are retained in
prespecified sensitivity analyses.

The reviewer-priority configuration is a deadline-oriented computational
profile and is not designated as the maximum-precision publication preset.

## Input Data

The manuscript input dataset is provided in:

`data/FWKC_input.xlsx`

The input file contains the binary variables:

- `D`: reference/gold-standard status
- `T`: diagnostic test status

Both variables are coded as `0/1`.

The reviewer script automatically searches for the input dataset within the
repository. An alternative input path can also be supplied through the
`FWKC_INPUT_PATH` environment variable.


## Software Requirements

The core R packages used by the reviewer-priority script are:

- `readxl`
- `openxlsx`
- `randomForest`
- `posterior`

For local recomputation of the same-prior HMC cross-check, the following are
also required:

- `cmdstanr`
- CmdStan

## Running the Analysis

Clone or download the repository and run the main script from R or RStudio:

```r
source("FWKC_inference_framework_v1.1_reviewer.R")

