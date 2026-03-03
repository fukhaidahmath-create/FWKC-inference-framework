 FWKC Inference Framework

This repository provides the full reproducible implementation of the FWKC (Fkhadah Weighted Kappa) agreement inference framework described in:

"Fkhadah Weighted Kappa Coefficient (FWKC): A finite-sample inferential framework for agreement analysis in binary diagnostic tests"

The framework integrates:
- Nonparametric bootstrap resampling
- Monte Carlo pseudo-rater simulation
- Bayesian hierarchical pooling
- Random Forest aggregation
- Delta-covariance uncertainty propagation
- Shrinkage calibration
- Dual-truth coverage validation
- Planning-aware predictive power and sample-size diagnostics

---

 Repository Structure

The repository is organized according to the methodological stages:

- stage0_realdata – Real-data contingency table extraction  
- stage1_bootstrap – Bootstrap resampling procedures  
- stage2_bayesian – Hierarchical Bayesian modeling (brms/cmdstanr)  
- stage3_delta – Delta-covariance uncertainty propagation  
- stage4_shrinkage – Shrinkage calibration functions  
- stage5_coverage – Dual-truth coverage simulations  
- stage6_planning – Predictive power and sample-size surfaces  

---
 Reproducibility

- Software: R (version 4.x.x)
- Key packages: brms, cmdstanr, randomForest, tidyverse
- Random seed: 12345
- Number of bootstrap replicates: 500
- Number of Monte Carlo pseudo-raters: 500

All simulations are fully reproducible given the fixed random seed.

---

Code Availability

This repository accompanies the manuscript and contains all scripts necessary to reproduce the simulation results and figures.

---

## License

MIT License.
