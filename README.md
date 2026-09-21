# FWKC Inference Framework

This repository provides a reproducible implementation of the FWKC
(Finite-Sample Weighted Kappa Coefficient) agreement inference framework
described in the manuscript:

**"A Finite-Sample Inferential Framework for the Weighted Kappa Coefficient
(FWKC) in Binary Diagnostic Agreement Studies"**

## Current Version

The current repository version is the reviewer-priority update:

**v1.1 reviewer-priority**

Main script:

[`FWKC_inference_framework_v1.1_reviewer.R`](FWKC_inference_framework_v1.1_reviewer.R)

This version is derived from the validated V25 publication engine and is
configured for reviewer-response analyses under a limited computational
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

Higher-resolution bootstrap and Monte Carlo settings are retained in
prespecified sensitivity analyses.

The reviewer-priority configuration is a deadline-oriented computational
profile. It is not designated as the maximum-precision publication preset.

## Relationship to V25

The reviewer-priority v1.1 workflow is designed to reuse completed,
high-resolution V25 validation components rather than recomputing them.

In particular, it reuses:

- the completed same-prior Gibbs versus collapsed-CmdStan-HMC cross-check;
- the completed V25 real-data `c = 0.1, ..., 0.9` analysis.

The default configuration therefore expects a completed V25 publication
results directory.

By default, the source directory is constructed as:

`FWKC_FINAL_V25_OFFICIAL_EFFICIENT_RESULTS/run_publication`

under the configured `FWKC_OUTPUT_BASE`.

Users running the reviewer-priority profile on another system should set
`FWKC_OUTPUT_BASE` or modify `reviewer_priority_source_dir` accordingly.

## Repository Structure

The repository currently contains:

```text
FWKC-inference-framework/
├── FWKC_inference_framework_v1.1_reviewer.R
├── README.md
├── LICENSE
└── .gitignore
