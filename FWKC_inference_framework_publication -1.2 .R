###############################################################################
# FWKC_FINAL_PUBLICATION_READY_v25_EXTERNAL_VALIDATION_QC_FIXED_COMPLETE.R
# Integrated reviewer-responsive inferential framework for binary diagnostic
# loss-weighted kappa, merging the strongest validated features of v10 and v12.
#
# DESIGN PRINCIPLES
# -----------------
# 1) Preserve the submitted binary diagnostic kappa(c) TARGET ESTIMAND and the
#    baseline Bayesian + ML fusion concept. The primary fusion rule remains
#    inverse-SE to preserve continuity with "Final 1-1-2026"; inverse-variance,
#    equal, Bayes-only and ML-only rules are sensitivity analyses.
# 2) Use a proper Beta-binomial posterior for Se/Sp (Jeffreys primary; Uniform
#    sensitivity; legacy Beta parameterization audit-only).
# 3) Keep Direct inference statistically independent of FWKC success at BOTH
#    bootstrap-replicate and independent-DGM-dataset levels. Direct bootstrap
#    replicates are never discarded because the FWKC layer failed.
# 4) Evaluate frequentist coverage across INDEPENDENT DGM datasets; bootstrap
#    replicates quantify within-dataset uncertainty only.
# 5) Preserve the submitted full p=0.1,...,0.9 by c=0.1,...,0.9 grid at
#    n=30,50,100 for continuity, using the real-data anchor only in this primary
#    continuity tier; evaluate generalizability in a SEPARATE prespecified
#    non-circular robustness tier with iid, clustered-beta and mixture DGMs.
# 6) Report bias, RMSE, MAE, coverage, interval width, method-specific failure
#    rates, Wilson coverage intervals, failure-as-noncoverage sensitivity,
#    runtime, scalability and practical benefit.
# 7) Use crossed Bayesian pooling with four chains + R-hat in publication mode,
#    and require a Gibbs-vs-collapsed-CmdStan-HMC cross-check on identical pseudo-rater data.
# 8) Fully specify RF; use symmetric leave-one-rater-out predictors, target
#    exclusion and OOB predictions; compare with cross-fitted natural splines.
# 9) Use full-pipeline nonparametric bootstrap for primary FWKC uncertainty.
# 10) Treat finite-sample calibration as SECONDARY. Explicitly distinguish the
#     variance-scale factor lambda_var=min(1,n/n0) from the equivalent SE-scale
#     factor lambda_se=sqrt(lambda_var), while also reporting stronger no-root
#     SE expansion and the submitted legacy contraction for audit only.
# 11) Save results incrementally, isolate run modes/checkpoints, record input
#     provenance + MD5, write progress to Console and CSV, and archive each main
#     stage independently so a long run can be inspected/recovered safely.
# 12) A hard publication QC gate writes PUBLICATION_READY.flag only when the
#     prespecified computational/statistical checks pass; otherwise it writes
#     RESULTS_NOT_REPORTABLE.flag. This is a programmatic QC gate, not a promise
#     of journal acceptance or a substitute for editorial/scientific review.
# 13) External validation is optional and NEVER fabricated. If no independent
#     verified real dataset is supplied, the code records that limitation.
# 14) The manuscript real-data anchor has total n=620 (Table 2). The primary
#     finite-sample continuity grid remains EXACTLY n=30,50,100 so previously
#     submitted 81-scenario-per-n comparisons remain interpretable. A separate
#     prespecified ANCHOR-SAMPLE-SIZE BRIDGE uses the observed input n (expected
#     to equal 620 in publication mode), the observed prevalence, and the full
#     c=0.1,...,0.9 grid. This directly links simulation behavior to the real
#     dataset without redefining or post-hoc expanding the primary small-sample
#     experiment. An optional full p-by-c anchor-size bridge is available but is
#     not the default because it adds large computational cost with little gain.
# 15) PARALLEL EXECUTION changes scheduling only, NOT the statistical design.
#     The same scenarios, M, B, n_mc, Bayesian chains/iterations, priors, fusion,
#     DGM seeds, sensitivity settings and QC thresholds are retained. Independent
#     scenarios are distributed across a persistent Windows-safe PSOCK cluster.
# 16) RNG safety is explicit: large namespace-separation seeds are normalized only
#     when required by R's integer seed limit; all already-valid v15/v14 seeds are
#     numerically unchanged. This prevents overflow in anchor/sensitivity scenarios.
# 17) Parallelism is single-level to prevent oversubscription. Worker logs are
#     independent, parent progress/checkpoint files are race-safe, and the Console
#     reports overall stages, PSOCK batches, scenario starts/completions and ETAs.
# 18) Optional network PSOCK execution is supported without changing inference. It
#     is disabled by default and requires the same R environment/packages plus a
#     shared output/checkpoint filesystem visible at the same path to every node.
# 19) The scalability timing benchmark remains sequential so timing comparisons are
#     interpretable and are not artificially improved by parallel scheduling.
# 20) The publication Gibbs-vs-collapsed-CmdStan-HMC cross-check is fail-fast and diagnostic-rich:
#     each c result is saved immediately, passed c values can be resumed safely,
#     and mean agreement, R-hat, divergent transitions and max-treedepth hits are
#     printed explicitly. Fail-fast changes only wasted compute after a failed gate;
#     the requirement that ALL prespecified c values pass is unchanged.
# 21) brms cross-check chains may use all available LOCAL cores up to the fixed
#     chain count. This changes wall-clock scheduling only; chains/iterations/warmup,
#     model, priors, pseudo-rater data and pass thresholds are unchanged.
# 22) Output persistence is hardened for long Windows runs: parent directories are
#     recreated/verified before writes, atomic files use collision-free temporary
#     names with retry/fallback copy, and abort-state persistence is best-effort so
#     an I/O problem can never hide the original statistical/computational error.
# 23) V25 SCIENTIFIC/PRACTICAL RECOVERY:
#     - performs an actual multi-directory write/commit self-test before expensive work;
#     - stages inferential files in system temp and only then commits them with retries;
#     - preserves an emergency copy when the configured output tree becomes unavailable;
#     - runs a prespecified HMC rescue (0.9999 / treedepth 18) only for sampler pathology,
#       with identical data/model/chains/iterations, and records every attempt.
# 27) V25 ACCELERATION IS INFERENCE-PRESERVING: the primary Gibbs engine uses an
#     exact blocked/collapsed conditional factorization with vectorized chains;
#     PSOCK uses all local cores and larger load-balanced batches; every completed
#     DGM replicate is journaled atomically, while nested simulation bootstraps do
#     not rewrite growing checkpoint objects. M, B, n_mc, chains, iterations, DGMs,
#     priors, fusion, estimands and QC thresholds are unchanged.
# 28) V25 POSTERIOR-API SAFETY: rank-normalized R-hat and ESS are computed
#     directly from the iterations-by-chains matrix with posterior::rhat(),
#     posterior::ess_bulk() and posterior::ess_tail(). The code never depends
#     on tibble column names returned by summarise_draws(), eliminating the
#     version-specific zero-length diagnostic error observed in v19. A cheap
#     diagnostics self-test runs before any expensive brms fit.
# 29) V25 CROSS-CHECK ACCELERATION/VALIDITY: the hard publication engine check
#     uses an independent custom CmdStan HMC implementation of the SAME crossed
#     Gaussian model AND the SAME normal/inverse-gamma priors as the Gibbs engine.
#     The MC-index random effect is integrated out analytically in the HMC
#     likelihood (exact marginalization), reducing HMC dimension from ~1000
#     latent MC effects to only mu + R rater effects + 3 variance components.
#     This is mathematically equivalent for the marginal posterior of mu, is
#     substantially faster, and avoids the prior mismatch of the former brms
#     implementation. The older brms implementation remains in the script for
#     optional audit use but is not the hard publication gate.
# 30) V25 LIVE DIAGNOSTICS: expensive steps print BEGIN/END records with elapsed
#     time, CmdStan prints chain progress every configured refresh interval, and
#     CURRENT_STEP.txt is updated before/after every cross-check substep.
# 31) V25 OFFICIAL-EFFICIENT PROFILE: primary reviewer-facing evidence is NOT
#     reduced (243 scenarios, M=500, B_DGM=199, n_mc=500, 4 chains, RF=300).
#     Runtime reductions affect only anchor/robustness/sensitivity/external
#     audit precision, covariance diagnostics, scalability repetitions, and
#     the first HMC audit attempt. Hard success, stability, convergence and
#     reportability gates remain active; strict_max_precision is retained as
#     a prespecified fallback if any efficient-profile gate fails.
###############################################################################

options(stringsAsFactors = FALSE)
options(warn = 1)

###############################################################################
# 1) DEPENDENCIES
###############################################################################

required_packages <- c("readxl", "openxlsx", "randomForest")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Missing required R package(s): ",
    paste(missing_packages, collapse = ", "),
    "\nInstall them before running this script."
  )
}

suppressPackageStartupMessages({
  library(readxl)
  library(openxlsx)
  library(randomForest)
})

###############################################################################
# 2) CONFIGURATION
###############################################################################

# Prefer a stable non-Documents location on the current Windows workstation when
# available, because the observed v17 failure occurred under C:/Users/.../Documents.
# Users can override this without editing the script by setting FWKC_OUTPUT_BASE.
fwkc_output_base <- Sys.getenv("FWKC_OUTPUT_BASE", unset = "")
if (!nzchar(fwkc_output_base)) {
  preferred_candidates <- c(
    "D:/phd/paper",
    getwd()
  )
  usable <- preferred_candidates[vapply(preferred_candidates, dir.exists, logical(1))]
  fwkc_output_base <- if (length(usable) > 0L) usable[1] else getwd()
}

config <- list(
  # Reproducibility / provenance
  script_version = "v25_official_efficient_publication_complete",
  seed = 20260912,

  # Execution
  run_analysis = TRUE,
  run_mode = "publication",  # test | verification | moderate | production | publication | custom
  # Publication profile:
  # "efficient_official" preserves the full 243-scenario primary design and M=500
  # coverage precision while reducing only computational integration/secondary
  # layers, with explicit stability/QC gates. "strict_max_precision" restores the
  # heavier v25 publication settings.
  publication_profile = "efficient_official",  # efficient_official | strict_max_precision
  input_path = NA_character_,
  output_root = file.path(fwkc_output_base, "FWKC_FINAL_V25_OFFICIAL_EFFICIENT_RESULTS"),
  output_dir = NA_character_,  # derived from run_mode below
  resume = TRUE,
  verbose = TRUE,
  console_progress = TRUE,
  write_progress_status = TRUE,
  progress_updates_per_stage = 50,
  console_detail_level = "detailed",       # quiet | normal | detailed
  write_live_step_file = TRUE,

  # V25 automatic two-run publication preflight:
  # first publication run proves I/O + estimand + Bayes cross-check, then exits cleanly;
  # rerunning the identical script reuses only signature-matched PASS checks and continues.
  publication_preflight_auto = TRUE,

  # Resilient output I/O for long Windows/network runs (execution-only).
  # V25 stages every payload in the process-local system temp directory first,
  # then commits it to the results tree with directory recreation + retries.
  # This prevents a transient disappearance/lock of Documents/validation from
  # destroying an already-computed Bayesian cross-check result.
  io_retry_attempts = 12L,
  io_retry_delay_seconds = 0.50,
  io_verify_writable = TRUE,
  io_self_test = TRUE,
  io_emergency_root = NA_character_,  # auto -> LOCALAPPDATA/FWKC_V25_EMERGENCY, else tempdir()

  # Parallel execution (COMPUTATIONAL ONLY; no statistical parameter changes).
  # Local mode is the default. A single persistent PSOCK cluster is reused across
  # real-data, primary simulation, anchor bridge, robustness and sensitivity stages.
  parallel_enabled = TRUE,
  parallel_backend = "PSOCK",
  parallel_mode = "local",             # local | network
  parallel_workers = 0L,                # 0 = auto in local mode
  parallel_reserve_cores = 0L,
  parallel_max_workers = 8L,
  parallel_batch_multiplier = 4L,
  parallel_scenario_batch_size = 0L,    # 0 = auto: workers * batch_multiplier
  parallel_stages = c(
    "real_data", "main_simulation", "anchor_bridge", "robustness", "sensitivity", "external"
  ),
  parallel_worker_progress = TRUE,
  parallel_worker_logs = TRUE,
  brms_parallel_chains = TRUE,
  brms_crosscheck_active = FALSE,       # internal execution flag
  brms_seed_override = NA_integer_,     # internal deterministic CmdStan seed override

  # OPTIONAL network/grid PSOCK mode. Leave parallel_mode="local" for one PC.
  # To use several machines, set parallel_mode="network", list hostnames/IPs in
  # parallel_hosts (repeat a host to request >1 worker on it if appropriate), and
  # set parallel_network_shared_fs=TRUE only after output_root is a shared path
  # mounted identically on every node. All nodes need compatible R/packages.
  parallel_hosts = character(0),
  parallel_network_shared_fs = FALSE,

  # Core pseudo-rater construction
  prior_type = "jeffreys",          # jeffreys | uniform | legacy
  uncertainty_generator = "beta_posterior", # beta_posterior | empirical_bootstrap
  pseudo_mode = "posterior_predictive",      # posterior_predictive | parameter_only
  n_raters = 3,
  n_mc = 1000,
  eps = 1e-8,

  # Hierarchical Bayesian pooling
  bayes_engine = "gibbs",          # gibbs | brms
  bayes_chains = 4,
  bayes_iter = 1000,
  bayes_burn = 300,
  bayes_thin = 2,
  # V25 exact blocked/collapsed Gibbs implementation. This changes only the
  # computational sampler, not the crossed Gaussian model, priors, chains,
  # iterations, burn-in, thinning, or inferential target. All chains are
  # vectorized in one iteration loop and the fixed/random effects are drawn
  # from their exact conditional block factorization to remove the slow
  # location-shift mixing seen in the centered single-site Gibbs sampler.
  gibbs_sampler = "blocked_collapsed_vectorized_v25",
  gibbs_rank_rhat_if_available = TRUE,
  bayes_rhat_max = 1.05,
  require_bayes_convergence = TRUE,
  allow_bayes_fallback = FALSE,
  bayes_mu0 = 0,
  bayes_mu_sd = 2,
  bayes_a_sigma = 2,
  bayes_b_sigma = 1,
  bayes_a_tau = 2,
  bayes_b_tau = 1,
  run_bayes_engine_crosscheck = TRUE,
  require_bayes_crosscheck_in_publication = TRUE,
  bayes_crosscheck_c = c(0.2, 0.5, 0.8),
  # V25 hard gate: independent same-prior collapsed CmdStan HMC.
  bayes_crosscheck_engine = "cmdstan_collapsed_same_prior",
  bayes_crosscheck_cmdstan_refresh = 100L,  # live CmdStan chain progress
  bayes_crosscheck_keep_cmdstan_csv = TRUE,
  run_brms_supplemental_crosscheck = FALSE, # optional audit only; NOT publication gate

  bayes_crosscheck_abs_floor = 0.02,
  bayes_crosscheck_se_multiplier = 2,
  bayes_crosscheck_brms_iter = 2000,
  bayes_crosscheck_brms_warmup = 1000,
  # Computational/diagnostic controls only; no reduction in chains or iterations.
  brms_adapt_delta = 0.99,
  brms_max_treedepth = 13L,
  # Cross-check-only HMC tuning: same model/data/chains/iterations/priors.
  # The hard gate uses the collapsed custom CmdStan model; legacy brms controls
  # are retained as compatible aliases and for optional audit runs.
  # V25 uses a prespecified two-step HMC rescue ONLY when sampler geometry fails:
  # Based on the observed v16/v18 sampler diagnostics on this manuscript data,
  # 0.999 repeatedly produced divergences. V25 therefore starts directly at the
  # stronger numerical control 0.9999/18 instead of spending ~15-20 minutes on a
  # known weak first attempt. This is sampler tuning only; model/data/chains/iters
  # are unchanged. A still stricter rescue is used only if sampler pathology remains.
  bayes_crosscheck_brms_adapt_delta = 0.9999,
  bayes_crosscheck_brms_max_treedepth = 18L,
  bayes_crosscheck_auto_rescue = TRUE,
  bayes_crosscheck_rescue_adapt_delta = 0.99999,
  bayes_crosscheck_rescue_max_treedepth = 20L,
  # V25 adaptive computation: the first HMC audit is shorter but must satisfy
  # R-hat, zero-divergence and minimum ESS gates. Only a diagnostic failure
  # triggers the longer rescue run; this changes compute time, not the model.
  bayes_crosscheck_min_ess_bulk = 200,
  bayes_crosscheck_min_ess_tail = 200,
  bayes_crosscheck_rescue_iter = 1600L,
  bayes_crosscheck_rescue_warmup = 800L,
  bayes_crosscheck_use_all_local_cores = TRUE,
  bayes_crosscheck_incremental_save = TRUE,
  bayes_crosscheck_resume_passed = TRUE,
  bayes_crosscheck_fail_fast = TRUE,
  bayes_crosscheck_require_zero_divergences = TRUE,
  bayes_crosscheck_require_zero_max_treedepth = FALSE, # recorded; not a hard gate unless TRUE

  # Transformation / ML / fusion
  transformation = "logit11",      # logit11 | fisher_z | identity
  ml_layer = "rf",                 # rf | spline | none
  rf_ntree = 300,
  rf_mtry = 3,
  rf_nodesize = 10,
  rf_maxnodes = 100,
  spline_folds = 5,
  fusion_rule = "inverse_se",      # PRIMARY continuity rule

  # Bootstrap inference
  alpha = 0.05,
  primary_ci = "percentile",
  min_valid_boot_fraction = 0.80,
  B_boot = 199,                     # generic fallback
  B_boot_real = 1000,               # real-data inference
  B_boot_dgm = 199,                 # nested independent-DGM bootstrap
  B_boot_sim = 199,                 # compatibility alias = B_boot_dgm
  B_boot_external = 1000,
  checkpoint_every_boot = 10,
  # Inner-bootstrap recovery policy (execution-only). The nested DGM analysis
  # already has an atomic per-DGM replicate journal, so persisting a growing
  # 199-row inner-bootstrap checkpoint inside every one of 121,500 DGM datasets
  # would create massive avoidable I/O. Simulation/robustness/anchor/sensitivity
  # therefore recompute at most the single in-flight DGM replicate after a crash.
  # Real/external data retain periodic inner-bootstrap recovery.
  checkpoint_inner_bootstrap_simulation = FALSE,
  checkpoint_every_boot_real = 10L,
  checkpoint_every_boot_external = 10L,

  # Secondary finite-sample calibration
  n0 = 100,
  calibration_mode = "sqrt_se_expand",
  use_calibrated_secondary_ci = TRUE,

  # Direct competing inference
  compute_bca = TRUE,
  bca_max_n = 1000,

  # Pseudo-rater covariance diagnostic
  covariance_boot_B = 1000,
  ridge_diag = 1e-10,

  # PRIMARY continuity simulation: complete submitted p x c grid.
  M_dgm = 500,
  nominal_coverage = 0.95,
  coverage_conf_level = 0.95,
  n_values = c(30, 50, 100),
  p_values = seq(0.1, 0.9, by = 0.1),
  c_values = seq(0.1, 0.9, by = 0.1),
  main_dgm_ids = c("anchor_iid"),
  scenario_design = "full_factorial",
  scenario_cap = NA_integer_,

  # Manuscript real-data anchor-size bridge.
  # Table 2 of the submitted manuscript totals n=620. Publication mode checks
  # that the selected real-data input has this sample size. The bridge itself
  # uses nrow(input) rather than hard-coding 620 into the simulation engine.
  manuscript_anchor_n = 620L,
  require_manuscript_anchor_n_in_publication = TRUE,
  run_anchor_sample_size_bridge = TRUE,
  anchor_bridge_design = "observed_prevalence_c_grid",
  # observed_prevalence_c_grid | full_pc_grid
  anchor_bridge_c_values = seq(0.1, 0.9, by = 0.1),
  anchor_bridge_p_values = seq(0.1, 0.9, by = 0.1),
  anchor_bridge_M = 500,
  anchor_bridge_B = 199,
  anchor_bridge_n_mc = 1000,

  # Separate non-circular robustness tier
  run_dgm_robustness = TRUE,
  robustness_dgm_ids = c(
    "moderate_iid", "high_iid",
    "sensitivity_dominant_iid", "specificity_dominant_iid",
    "moderate_clustered", "moderate_mixture"
  ),
  robustness_p_values = c(0.1, 0.5, 0.9),
  robustness_c_values = c(0.1, 0.5, 0.9),
  robustness_M = 200,
  robustness_B = 149,
  robustness_scenario_cap = NA_integer_,

  # Heterogeneous DGM settings
  cluster_size = 10,
  cluster_phi = 25,
  mixture_delta = 0.15,

  # Sensitivity analyses
  run_sensitivity = TRUE,
  sensitivity_M = 100,
  sensitivity_B = 199,
  sensitivity_n_mc = 500,
  sensitivity_scenarios = 6,

  # Scalability benchmark
  run_scalability = TRUE,
  scalability_n = c(100, 500, 1000, 2500, 5000),
  scalability_reps = 5,
  scalability_B = 39,
  scalability_n_mc = 300,

  # External real-data generalizability analysis.
  # Primary default uses a VERIFIED BUILT-IN copy of the published AuditC
  # diagnostic-accuracy 2x2 tables (14 primary studies), originating from
  # Kriston et al. (2008) and distributed in the CRAN package mada.
  # The built-in copy avoids a runtime package/network dependency and is checked
  # against fixed row/column totals before use. This is genuine independent,
  # CROSS-DOMAIN real diagnostic data; it is NOT same-disease coronary transport
  # validation. A user-supplied independent counts file can replace it.
  external_validation_source = "AuditC_builtin_verified",  # AuditC_builtin_verified | mada_AuditC | file | none
  external_counts_file = NA_character_,
  external_mada_dataset = "AuditC",
  external_max_studies = NA_integer_,           # publication: all available studies
  external_c_values = seq(0.1, 0.9, by = 0.1),
  external_validation_required_in_publication = TRUE,
  external_min_dataset_success_rate = 0.90,

  # Legacy diagnostics are audit-only / secondary
  retain_legacy_diagnostics = TRUE,
  target_power = 0.80,

  # Publication minimums used by the QC/readiness audit.
  # V25 efficient-official keeps the PRIMARY coverage design unchanged:
  # M_dgm=500 and B_boot_dgm=199. n_mc is reduced to 500 only because the
  # prespecified n_mc sensitivity tier (500/1000/2000) is retained and a hard
  # stability gate is added below. Secondary tiers use separate minimums.
  publication_min_M = 500,
  publication_min_B_dgm = 199,
  publication_min_B_real = 499,
  publication_min_n_mc = 500,
  publication_min_robustness_M = 100,
  publication_min_robustness_B = 79,
  publication_min_anchor_M = 150,
  publication_min_anchor_B = 99,
  publication_min_anchor_n_mc = 500,
  publication_min_sensitivity_M = 30,
  publication_min_sensitivity_B = 79,
  publication_min_sensitivity_n_mc = 500,
  publication_min_sensitivity_scenarios = 4,
  publication_min_B_external = 299,
  publication_min_covariance_B = 299,
  # Prespecified MC-integration stability audit for choosing n_mc=500.
  nmc_stability_max_abs_bias_diff = 0.03,
  nmc_stability_max_abs_rmse_diff = 0.03,
  nmc_stability_max_abs_mae_diff = 0.03,

  # Output / recovery
  # V25 durability + speed: every completed DGM replicate is saved as one
  # small atomic RDS journal item immediately. The growing full scenario RDS/CSV
  # is therefore checkpointed less frequently, avoiding O(M^2) disk rewriting
  # while preserving replicate-level crash recovery. No statistical setting changes.
  checkpoint_every_dgm = 50,
  dgm_rep_journal_enabled = TRUE,
  cleanup_dgm_rep_journal_on_complete = TRUE,
  make_plots = TRUE,
  make_pipeline_diagram = TRUE,
  write_reviewer_manifest = TRUE,
  write_session_info = TRUE,
  baseline_code_id = "Final 1-1-2026"
)

# ---------------- Run-mode presets ----------------
# TEST preset: end-to-end functional/inferential integration check. It exercises
# every major path, INCLUDING external real-data validation, but its numerical
# outputs are explicitly non-reportable. Crucially, every bootstrap tier is >=20
# so the hard min_valid rule can actually be satisfied.
if (config$run_mode == "test") {
  config$bayes_engine <- "gibbs"
  config$bayes_chains <- 4L
  config$n_mc <- 100L
  config$bayes_iter <- 400L
  config$bayes_burn <- 200L
  config$bayes_thin <- 2L
  config$rf_ntree <- 80L

  config$B_boot <- 29L
  config$B_boot_real <- 39L
  config$B_boot_dgm <- 29L
  config$B_boot_sim <- config$B_boot_dgm
  config$B_boot_external <- 39L
  config$checkpoint_every_boot <- 5L
  config$checkpoint_every_boot_real <- 5L
  config$checkpoint_every_boot_external <- 5L

  config$M_dgm <- 5L
  config$scenario_cap <- 6L
  config$checkpoint_every_dgm <- 1L

  config$run_bayes_engine_crosscheck <- TRUE
  config$require_bayes_crosscheck_in_publication <- FALSE
  config$bayes_crosscheck_c <- c(0.2, 0.8)
  config$bayes_crosscheck_brms_iter <- 400L
  config$bayes_crosscheck_brms_warmup <- 200L
  config$bayes_crosscheck_rescue_iter <- 800L
  config$bayes_crosscheck_rescue_warmup <- 400L
  config$bayes_crosscheck_min_ess_bulk <- 100
  config$bayes_crosscheck_min_ess_tail <- 100
  config$bayes_crosscheck_cmdstan_refresh <- 25L
  config$bayes_crosscheck_auto_rescue <- TRUE
  config$require_bayes_convergence <- FALSE

  # FIX 1: these were 19 in v21 test, which made min_valid>=20 impossible.
  config$anchor_bridge_M <- 3L
  config$anchor_bridge_B <- 29L
  config$anchor_bridge_n_mc <- 100L
  config$robustness_M <- 3L
  config$robustness_B <- 29L
  config$robustness_scenario_cap <- 6L
  config$run_sensitivity <- TRUE
  config$sensitivity_M <- 3L
  config$sensitivity_B <- 29L
  config$sensitivity_n_mc <- 100L
  config$sensitivity_scenarios <- 2L

  config$covariance_boot_B <- 30L
  config$compute_bca <- FALSE
  config$scalability_n <- c(100L, 300L)
  config$scalability_reps <- 1L
  config$scalability_B <- 29L
  config$scalability_n_mc <- 80L

  # Execute a small but genuine external-data path in test mode.
  config$external_validation_source <- "AuditC_builtin_verified"
  config$external_max_studies <- 4L
  config$external_validation_required_in_publication <- FALSE

} else if (config$run_mode == "verification") {
  config$bayes_engine <- "gibbs"
  config$bayes_chains <- 2
  config$n_mc <- 100
  config$bayes_iter <- 250
  config$bayes_burn <- 80
  config$rf_ntree <- 60
  config$B_boot <- 15
  config$B_boot_real <- 20
  config$B_boot_dgm <- 20
  config$B_boot_sim <- config$B_boot_dgm
  config$B_boot_external <- 20
  config$M_dgm <- 3
  config$scenario_cap <- 3
  config$robustness_M <- 2
  config$robustness_B <- 9
  config$robustness_scenario_cap <- 3
  config$anchor_bridge_M <- 2
  config$anchor_bridge_B <- 9
  config$anchor_bridge_n_mc <- 100
  config$run_sensitivity <- FALSE
  config$run_bayes_engine_crosscheck <- FALSE
  config$require_bayes_crosscheck_in_publication <- FALSE
  config$require_bayes_convergence <- FALSE
  config$covariance_boot_B <- 30
  config$scalability_n <- c(100, 300)
  config$scalability_reps <- 1
  config$scalability_B <- 15
  config$scalability_n_mc <- 80

} else if (config$run_mode == "moderate") {
  config$bayes_engine <- "gibbs"
  config$bayes_chains <- 3
  config$n_mc <- 500
  config$bayes_iter <- 600
  config$bayes_burn <- 200
  config$rf_ntree <- 200
  config$B_boot <- 79
  config$B_boot_real <- 200
  config$B_boot_dgm <- 79
  config$B_boot_sim <- config$B_boot_dgm
  config$B_boot_external <- 200
  config$M_dgm <- 100
  config$scenario_cap <- 81
  config$robustness_M <- 40
  config$robustness_B <- 49
  config$robustness_scenario_cap <- 36
  config$anchor_bridge_M <- 50
  config$anchor_bridge_B <- 79
  config$anchor_bridge_n_mc <- 500
  config$sensitivity_M <- 30
  config$sensitivity_B <- 79
  config$sensitivity_n_mc <- 250
  config$covariance_boot_B <- 200
  config$scalability_n <- c(100, 500, 1000)
  config$scalability_reps <- 2
  config$scalability_B <- 19
  config$scalability_n_mc <- 150
  config$require_bayes_crosscheck_in_publication <- FALSE

} else if (config$run_mode == "production") {
  config$bayes_engine <- "gibbs"
  config$bayes_chains <- 4
  config$n_mc <- 750
  config$bayes_iter <- 800
  config$bayes_burn <- 250
  config$rf_ntree <- 300
  config$B_boot <- 149
  config$B_boot_real <- 500
  config$B_boot_dgm <- 149
  config$B_boot_sim <- config$B_boot_dgm
  config$B_boot_external <- 500
  config$M_dgm <- 250
  config$scenario_cap <- NA_integer_
  config$robustness_M <- 100
  config$robustness_B <- 79
  config$anchor_bridge_M <- 250
  config$anchor_bridge_B <- 149
  config$anchor_bridge_n_mc <- 750
  config$sensitivity_M <- 60
  config$sensitivity_B <- 99
  config$sensitivity_n_mc <- 400
  config$covariance_boot_B <- 500
  config$require_bayes_crosscheck_in_publication <- FALSE

} else if (config$run_mode == "publication") {
  # V25 manuscript-grade publication preset.
  # PRIMARY evidence is intentionally NOT weakened:
  #   full 243-scenario p-by-c grid, M_dgm=500, B_dgm=199, 4 chains, RF=300.
  # The balanced profile reduces only MC integration and secondary layers,
  # and adds explicit stability/QC gates. The strict profile reproduces the
  # heavier v25 publication precision settings.
  config$bayes_engine <- "gibbs"
  config$fusion_rule <- "inverse_se"
  config$bayes_chains <- 4
  config$bayes_iter <- 1000
  config$bayes_burn <- 300
  config$rf_ntree <- 300
  config$B_boot <- 199
  config$B_boot_dgm <- 199
  config$B_boot_sim <- config$B_boot_dgm
  config$M_dgm <- 500
  config$scenario_cap <- NA_integer_
  config$external_validation_source <- "AuditC_builtin_verified"
  config$external_max_studies <- NA_integer_
  config$external_validation_required_in_publication <- TRUE
  config$robustness_scenario_cap <- NA_integer_
  config$require_bayes_convergence <- TRUE
  config$require_bayes_crosscheck_in_publication <- TRUE
  config$run_bayes_engine_crosscheck <- TRUE
  config$bayes_crosscheck_c <- c(0.2,0.5,0.8)

  if (identical(config$publication_profile, "strict_max_precision")) {
    config$n_mc <- 1000
    config$B_boot_real <- 1000
    config$B_boot_external <- 1000
    config$robustness_M <- 200
    config$robustness_B <- 149
    config$anchor_bridge_M <- 500
    config$anchor_bridge_B <- 199
    config$anchor_bridge_n_mc <- 1000
    config$sensitivity_M <- 100
    config$sensitivity_B <- 199
    config$sensitivity_n_mc <- 500
    config$covariance_boot_B <- 1000
    config$bayes_crosscheck_brms_iter <- 2000L
    config$bayes_crosscheck_brms_warmup <- 1000L
    config$bayes_crosscheck_rescue_iter <- 3000L
    config$bayes_crosscheck_rescue_warmup <- 1500L
    config$bayes_crosscheck_min_ess_bulk <- 200
    config$bayes_crosscheck_min_ess_tail <- 200

    config$publication_min_B_real <- 1000
    config$publication_min_n_mc <- 1000
    config$publication_min_robustness_M <- 150
    config$publication_min_robustness_B <- 149
    config$publication_min_anchor_M <- 500
    config$publication_min_anchor_B <- 199
    config$publication_min_anchor_n_mc <- 1000
    config$publication_min_sensitivity_M <- 100
    config$publication_min_sensitivity_B <- 199
    config$publication_min_sensitivity_n_mc <- 500
    config$publication_min_B_external <- 1000
    config$publication_min_covariance_B <- 1000
  } else if (identical(config$publication_profile, "efficient_official")) {
    # OFFICIAL EFFICIENT PROFILE:
    # The reviewer-facing PRIMARY experiment remains unchanged:
    # 243 scenarios, M_dgm=500, B_boot_dgm=199, n_mc=500, 4 chains, RF=300.
    # Savings are confined to secondary/robustness/audit layers.
    config$n_mc <- 500
    config$B_boot_real <- 499
    config$B_boot_external <- 299
    config$robustness_M <- 100
    config$robustness_B <- 79
    config$anchor_bridge_M <- 150
    config$anchor_bridge_B <- 99
    config$anchor_bridge_n_mc <- 500
    config$sensitivity_M <- 30
    config$sensitivity_B <- 79
    config$sensitivity_n_mc <- 500
    config$sensitivity_scenarios <- 4L
    config$covariance_boot_B <- 299
    config$scalability_reps <- 3L
    config$scalability_B <- 29L
    config$scalability_n_mc <- 200L

    # Same-model computational audit: shorter first attempt, strict diagnostics,
    # and automatic rescue if R-hat/ESS/geometry gates are not satisfied.
    config$bayes_crosscheck_brms_iter <- 600L
    config$bayes_crosscheck_brms_warmup <- 300L
    config$bayes_crosscheck_rescue_iter <- 1200L
    config$bayes_crosscheck_rescue_warmup <- 600L
    config$bayes_crosscheck_min_ess_bulk <- 200
    config$bayes_crosscheck_min_ess_tail <- 200
  } else {
    stop("Unknown publication_profile: ", config$publication_profile,
         ". Use 'efficient_official' or 'strict_max_precision'.")
  }
}

# Isolate all outputs/checkpoints by run mode. This prevents a smoke-test
# checkpoint from ever being resumed inside a publication run.
config$output_dir <- file.path(
  config$output_root,
  paste0("run_", config$run_mode)
)

validate_config <- function(cfg) {
  problems <- character(0)

  if (!is.finite(cfg$alpha) || cfg$alpha <= 0 || cfg$alpha >= 1) {
    problems <- c(problems, "alpha must lie in (0,1).")
  }
  if (!is.finite(cfg$n_mc) || cfg$n_mc < 50) {
    problems <- c(problems, "n_mc is too small for stable Monte Carlo integration.")
  }
  if (!is.finite(cfg$B_boot_real) || cfg$B_boot_real < 19 ||
      !is.finite(cfg$B_boot_dgm) || cfg$B_boot_dgm < 19) {
    problems <- c(problems, "Bootstrap replicate counts are invalid/too small.")
  }
  if (!is.finite(cfg$M_dgm) || cfg$M_dgm < 1) {
    problems <- c(problems, "M_dgm must be positive.")
  }

  if (!is.logical(cfg$publication_preflight_auto) ||
      length(cfg$publication_preflight_auto) != 1L) {
    problems <- c(problems, "publication_preflight_auto must be TRUE or FALSE.")
  }
  if (!is.logical(cfg$parallel_enabled) || length(cfg$parallel_enabled) != 1L) {
    problems <- c(problems, "parallel_enabled must be TRUE or FALSE.")
  }
  if (!identical(toupper(cfg$parallel_backend), "PSOCK")) {
    problems <- c(problems, "Only the PSOCK parallel backend is supported in this publication script.")
  }
  if (!cfg$parallel_mode %in% c("local", "network")) {
    problems <- c(problems, "parallel_mode must be 'local' or 'network'.")
  }
  if (!is.finite(cfg$parallel_workers) || cfg$parallel_workers < 0) {
    problems <- c(problems, "parallel_workers must be 0 (auto) or a positive integer.")
  }
  if (!is.finite(cfg$parallel_reserve_cores) || cfg$parallel_reserve_cores < 0 ||
      !is.finite(cfg$parallel_max_workers) || cfg$parallel_max_workers < 1 ||
      !is.finite(cfg$parallel_batch_multiplier) || cfg$parallel_batch_multiplier < 1 ||
      !is.finite(cfg$parallel_scenario_batch_size) || cfg$parallel_scenario_batch_size < 0) {
    problems <- c(problems, "Parallel core/batch controls are invalid.")
  }
  if (identical(cfg$parallel_mode, "network")) {
    if (length(cfg$parallel_hosts) < 1L || any(!nzchar(cfg$parallel_hosts))) {
      problems <- c(problems, "Network parallel mode requires at least one non-empty parallel_hosts entry.")
    }
    if (!isTRUE(cfg$parallel_network_shared_fs)) {
      problems <- c(problems, paste0(
        "Network parallel mode requires parallel_network_shared_fs=TRUE after confirming ",
        "that output_root/checkpoints are visible at the same path on every node."
      ))
    }
  }

  if (!is.finite(cfg$io_retry_attempts) || cfg$io_retry_attempts < 1 ||
      !is.finite(cfg$io_retry_delay_seconds) || cfg$io_retry_delay_seconds < 0 ||
      !is.logical(cfg$io_verify_writable) || length(cfg$io_verify_writable) != 1L) {
    problems <- c(problems, "Resilient I/O controls are invalid.")
  }

  if (!is.finite(cfg$brms_adapt_delta) || cfg$brms_adapt_delta <= 0 || cfg$brms_adapt_delta >= 1) {
    problems <- c(problems, "brms_adapt_delta must lie in (0,1).")
  }
  if (!is.finite(cfg$brms_max_treedepth) || cfg$brms_max_treedepth < 1) {
    problems <- c(problems, "brms_max_treedepth must be a positive integer.")
  }
  if (!is.finite(cfg$bayes_crosscheck_brms_adapt_delta) ||
      cfg$bayes_crosscheck_brms_adapt_delta <= 0 || cfg$bayes_crosscheck_brms_adapt_delta >= 1) {
    problems <- c(problems, "bayes_crosscheck_brms_adapt_delta must lie in (0,1).")
  }
  if (!is.finite(cfg$bayes_crosscheck_brms_max_treedepth) ||
      cfg$bayes_crosscheck_brms_max_treedepth < 1) {
    problems <- c(problems, "bayes_crosscheck_brms_max_treedepth must be a positive integer.")
  }
  if (!is.logical(cfg$bayes_crosscheck_auto_rescue) ||
      length(cfg$bayes_crosscheck_auto_rescue) != 1L) {
    problems <- c(problems, "bayes_crosscheck_auto_rescue must be TRUE or FALSE.")
  }
  if (!is.finite(cfg$bayes_crosscheck_rescue_adapt_delta) ||
      cfg$bayes_crosscheck_rescue_adapt_delta <= 0 ||
      cfg$bayes_crosscheck_rescue_adapt_delta >= 1) {
    problems <- c(problems, "bayes_crosscheck_rescue_adapt_delta must lie in (0,1).")
  }
  if (!is.finite(cfg$bayes_crosscheck_rescue_max_treedepth) ||
      cfg$bayes_crosscheck_rescue_max_treedepth < cfg$bayes_crosscheck_brms_max_treedepth) {
    problems <- c(problems,
      "bayes_crosscheck_rescue_max_treedepth must be >= the primary cross-check max_treedepth.")
  }
  if (!is.finite(cfg$bayes_crosscheck_min_ess_bulk) || cfg$bayes_crosscheck_min_ess_bulk <= 0 ||
      !is.finite(cfg$bayes_crosscheck_min_ess_tail) || cfg$bayes_crosscheck_min_ess_tail <= 0) {
    problems <- c(problems, "Bayes cross-check minimum ESS thresholds must be positive.")
  }
  if (!is.finite(cfg$bayes_crosscheck_rescue_iter) ||
      cfg$bayes_crosscheck_rescue_iter < cfg$bayes_crosscheck_brms_iter ||
      !is.finite(cfg$bayes_crosscheck_rescue_warmup) ||
      cfg$bayes_crosscheck_rescue_warmup < cfg$bayes_crosscheck_brms_warmup ||
      cfg$bayes_crosscheck_rescue_warmup >= cfg$bayes_crosscheck_rescue_iter) {
    problems <- c(problems,
      "Bayes cross-check rescue iterations/warmup must be valid and >= the first attempt.")
  }
  for (nm in c("bayes_crosscheck_use_all_local_cores", "bayes_crosscheck_incremental_save",
               "bayes_crosscheck_resume_passed", "bayes_crosscheck_fail_fast",
               "bayes_crosscheck_require_zero_divergences",
               "bayes_crosscheck_require_zero_max_treedepth")) {
    if (!is.logical(cfg[[nm]]) || length(cfg[[nm]]) != 1L) {
      problems <- c(problems, paste0(nm, " must be TRUE or FALSE."))
    }
  }

  if (!cfg$gibbs_sampler %in% c("blocked_collapsed_vectorized_v25")) {
    problems <- c(problems, "gibbs_sampler must be blocked_collapsed_vectorized_v25 in this script.")
  }
  if (!is.logical(cfg$gibbs_rank_rhat_if_available) || length(cfg$gibbs_rank_rhat_if_available) != 1L) {
    problems <- c(problems, "gibbs_rank_rhat_if_available must be TRUE or FALSE.")
  }
  if (!is.finite(cfg$checkpoint_every_dgm) || cfg$checkpoint_every_dgm < 1) {
    problems <- c(problems, "checkpoint_every_dgm must be >=1.")
  }
  for (nm in c("dgm_rep_journal_enabled", "cleanup_dgm_rep_journal_on_complete")) {
    if (!is.logical(cfg[[nm]]) || length(cfg[[nm]]) != 1L) {
      problems <- c(problems, paste0(nm, " must be TRUE or FALSE."))
    }
  }

  if (!is.logical(cfg$checkpoint_inner_bootstrap_simulation) ||
      length(cfg$checkpoint_inner_bootstrap_simulation) != 1L) {
    problems <- c(problems, "checkpoint_inner_bootstrap_simulation must be TRUE or FALSE.")
  }
  if (!is.finite(cfg$checkpoint_every_boot_real) || cfg$checkpoint_every_boot_real < 1 ||
      !is.finite(cfg$checkpoint_every_boot_external) || cfg$checkpoint_every_boot_external < 1) {
    problems <- c(problems, "Real/external bootstrap checkpoint intervals must be >=1.")
  }

  if (!cfg$bayes_crosscheck_engine %in% c("cmdstan_collapsed_same_prior", "brms_legacy_audit")) {
    problems <- c(problems, "bayes_crosscheck_engine must be cmdstan_collapsed_same_prior or brms_legacy_audit.")
  }
  if (!is.finite(cfg$bayes_crosscheck_cmdstan_refresh) || cfg$bayes_crosscheck_cmdstan_refresh < 0) {
    problems <- c(problems, "bayes_crosscheck_cmdstan_refresh must be >=0.")
  }
  if (!cfg$console_detail_level %in% c("quiet", "normal", "detailed")) {
    problems <- c(problems, "console_detail_level must be quiet, normal, or detailed.")
  }
  for (nm in c("write_live_step_file", "bayes_crosscheck_keep_cmdstan_csv", "run_brms_supplemental_crosscheck")) {
    if (!is.logical(cfg[[nm]]) || length(cfg[[nm]]) != 1L) {
      problems <- c(problems, paste0(nm, " must be TRUE or FALSE."))
    }
  }

  if (identical(cfg$run_mode, "publication")) {
    required <- c(
      prior_type = identical(cfg$prior_type, "jeffreys"),
      uncertainty_generator = identical(cfg$uncertainty_generator, "beta_posterior"),
      pseudo_mode = identical(cfg$pseudo_mode, "posterior_predictive"),
      fusion_rule = identical(cfg$fusion_rule, "inverse_se"),
      gibbs_sampler = identical(cfg$gibbs_sampler, "blocked_collapsed_vectorized_v25"),
      bayes_chains = cfg$bayes_chains >= 4,
      bayes_iter = cfg$bayes_iter >= 1000,
      bayes_burn = cfg$bayes_burn >= 300,
      n_mc = cfg$n_mc >= cfg$publication_min_n_mc,
      rf_ntree = cfg$rf_ntree >= 300,
      B_boot_real = cfg$B_boot_real >= cfg$publication_min_B_real,
      B_boot_dgm = cfg$B_boot_dgm >= cfg$publication_min_B_dgm,
      M_dgm = cfg$M_dgm >= cfg$publication_min_M,
      full_grid_p = identical(as.numeric(cfg$p_values), as.numeric(seq(0.1,0.9,0.1))),
      full_grid_c = identical(as.numeric(cfg$c_values), as.numeric(seq(0.1,0.9,0.1))),
      anchor_primary = identical(cfg$main_dgm_ids, c("anchor_iid")),
      publication_profile = cfg$publication_profile %in% c("efficient_official","strict_max_precision"),
      anchor_size_bridge = isTRUE(cfg$run_anchor_sample_size_bridge) &&
        identical(cfg$anchor_bridge_design, "observed_prevalence_c_grid") &&
        cfg$anchor_bridge_M >= cfg$publication_min_anchor_M &&
        cfg$anchor_bridge_B >= cfg$publication_min_anchor_B &&
        cfg$anchor_bridge_n_mc >= cfg$publication_min_anchor_n_mc,
      manuscript_anchor_n_defined = is.finite(cfg$manuscript_anchor_n) &&
        cfg$manuscript_anchor_n > 0,
      robustness = isTRUE(cfg$run_dgm_robustness) &&
        cfg$robustness_M >= cfg$publication_min_robustness_M &&
        cfg$robustness_B >= cfg$publication_min_robustness_B,
      min_valid_boot_fraction = cfg$min_valid_boot_fraction >= 0.80,
      require_bayes_convergence = isTRUE(cfg$require_bayes_convergence),
      bayes_crosscheck = isTRUE(cfg$run_bayes_engine_crosscheck) &&
        isTRUE(cfg$require_bayes_crosscheck_in_publication) &&
        identical(cfg$bayes_crosscheck_engine, "cmdstan_collapsed_same_prior") &&
        cfg$bayes_crosscheck_brms_iter >= 600 &&
        cfg$bayes_crosscheck_brms_warmup >= 300,
      sensitivity = isTRUE(cfg$run_sensitivity),
      sensitivity_M = cfg$sensitivity_M >= cfg$publication_min_sensitivity_M,
      sensitivity_B = cfg$sensitivity_B >= cfg$publication_min_sensitivity_B,
      sensitivity_n_mc = cfg$sensitivity_n_mc >= cfg$publication_min_sensitivity_n_mc,
      sensitivity_scenarios = cfg$sensitivity_scenarios >= cfg$publication_min_sensitivity_scenarios,
      external_bootstrap = cfg$B_boot_external >= cfg$publication_min_B_external,
      covariance_bootstrap = cfg$covariance_boot_B >= cfg$publication_min_covariance_B,
      scalability = isTRUE(cfg$run_scalability)
    )
    bad <- names(required)[!required]
    if (length(bad) > 0) {
      problems <- c(
        problems,
        paste0("Publication configuration gate failed for: ", paste(bad, collapse = ", "), ".")
      )
    }

    mcse_nominal <- sqrt(
      cfg$nominal_coverage * (1 - cfg$nominal_coverage) / cfg$M_dgm
    )
    if (!is.finite(mcse_nominal) || mcse_nominal > 0.01) {
      problems <- c(
        problems,
        paste0(
          "M_dgm does not achieve nominal coverage MCSE <=0.01; current MCSE=",
          signif(mcse_nominal, 4), "."
        )
      )
    }
  }

  if (length(problems) > 0) {
    stop("Configuration validation failed:\n- ", paste(problems, collapse = "\n- "))
  }
  invisible(TRUE)
}

validate_config(config)

# Reproducible RNG stream. L'Ecuyer-CMRG is also suitable if the analysis is
# later parallelized without changing the statistical specification.
RNGkind("L'Ecuyer-CMRG")
base::set.seed(config$seed)



###############################################################################
# 3) OUTPUT + LOGGING
###############################################################################

dir.create(config$output_dir, showWarnings = FALSE, recursive = TRUE)
config$output_dir <- normalizePath(
  config$output_dir,
  winslash = "/",
  mustWork = TRUE
)

for (subdir in c(
  "validation", "real_data", "real_data/bootstrap_checkpoints",
  "real_data/bootstrap_partial", "simulation", "simulation/detail",
  "checkpoints", "robustness",
  "anchor_bridge", "anchor_bridge/simulation",
  "anchor_bridge/simulation/detail", "anchor_bridge/checkpoints",
  "sensitivity", "external",
  "external/bootstrap_checkpoints", "external/bootstrap_partial",
  "plots", "stage_archive", "parallel", "parallel_worker_logs"
)) {
  dir.create(
    file.path(config$output_dir, subdir),
    showWarnings = FALSE,
    recursive = TRUE
  )
}

log_file <- file.path(config$output_dir, "run_log.txt")

.fwkc_io_state <- new.env(parent = emptyenv())
.fwkc_io_state$counter <- 0L
.fwkc_io_state$degraded <- FALSE
.fwkc_io_state$emergency_files <- character(0)

io_attempts <- function() {
  x <- suppressWarnings(as.integer(config$io_retry_attempts))
  if (!is.finite(x) || x < 1L) 6L else x
}

io_delay <- function() {
  x <- suppressWarnings(as.numeric(config$io_retry_delay_seconds))
  if (!is.finite(x) || x < 0) 0.50 else x
}

io_emergency_root <- function() {
  configured <- tryCatch(as.character(config$io_emergency_root)[1], error = function(e) NA_character_)
  if (is.finite(nchar(configured)) && !is.na(configured) && nzchar(configured)) {
    root <- configured
  } else {
    local_app <- Sys.getenv("LOCALAPPDATA", unset = "")
    if (nzchar(local_app)) {
      root <- file.path(local_app, "FWKC_V25_EMERGENCY")
    } else {
      root <- file.path(tempdir(), "FWKC_V25_EMERGENCY")
    }
  }
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  normalizePath(root, winslash = "/", mustWork = FALSE)
}

io_spool_root <- function() {
  root <- file.path(tempdir(), paste0("FWKC_v25_spool_", Sys.getpid()))
  if (!dir.exists(root)) dir.create(root, recursive = TRUE, showWarnings = FALSE)
  root
}

next_io_token <- function() {
  .fwkc_io_state$counter <- .fwkc_io_state$counter + 1L
  paste0(Sys.getpid(), "_", sprintf("%08d", .fwkc_io_state$counter))
}

actual_write_probe <- function(directory) {
  probe <- tempfile(pattern = ".fwkc_v25_probe_", tmpdir = directory)
  ok <- FALSE
  con <- NULL
  tryCatch({
    con <- file(probe, open = "wb")
    writeBin(charToRaw("FWKC_V25_WRITE_TEST"), con)
    close(con)
    con <- NULL
    ok <- file.exists(probe) && file.info(probe)$size > 0
  }, error = function(e) {
    ok <<- FALSE
  }, finally = {
    if (!is.null(con)) try(close(con), silent = TRUE)
    if (file.exists(probe)) try(unlink(probe, force = TRUE), silent = TRUE)
  })
  isTRUE(ok)
}

ensure_parent_writable <- function(path, attempts = io_attempts(), delay = io_delay()) {
  d <- dirname(path)
  last_msg <- NULL
  for (a in seq_len(max(1L, as.integer(attempts)))) {
    ok <- tryCatch({
      if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
      if (!dir.exists(d)) stop("directory does not exist after dir.create")
      if (isTRUE(config$io_verify_writable) && !actual_write_probe(d)) {
        stop("actual write probe failed")
      }
      TRUE
    }, error = function(e) {
      last_msg <<- conditionMessage(e)
      FALSE
    })
    if (isTRUE(ok)) return(invisible(d))
    if (a < attempts && delay > 0) Sys.sleep(delay * min(a, 4L))
  }
  stop(
    "Output directory is unavailable/not writable after retries: ", d,
    if (!is.null(last_msg)) paste0(" | last_error=", last_msg) else ""
  )
}

stage_payload <- function(writer, suffix = ".bin") {
  spool <- io_spool_root()
  token <- next_io_token()
  tmp <- file.path(spool, paste0("payload_", token, suffix))
  writer(tmp)
  if (!file.exists(tmp)) stop("I/O staging failed: temporary payload was not created: ", tmp)
  tmp
}

save_emergency_copy <- function(staged, intended_path, label = "payload") {
  root <- io_emergency_root()
  rel <- gsub("[:\\\\/]+", "_", intended_path)
  emergency <- file.path(
    root,
    paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_", Sys.getpid(), "_", label, "_", rel)
  )
  ok <- tryCatch(file.copy(staged, emergency, overwrite = TRUE), error = function(e) FALSE)
  if (isTRUE(ok) && file.exists(emergency)) {
    .fwkc_io_state$degraded <- TRUE
    .fwkc_io_state$emergency_files <- unique(c(.fwkc_io_state$emergency_files, emergency))
    return(normalizePath(emergency, winslash = "/", mustWork = FALSE))
  }
  NA_character_
}

commit_staged_file <- function(staged, path, attempts = io_attempts(), delay = io_delay()) {
  last_msg <- NULL
  for (a in seq_len(max(1L, as.integer(attempts)))) {
    ok <- tryCatch({
      ensure_parent_writable(path, attempts = 1L, delay = 0)
      parent <- dirname(path)
      dest_tmp <- file.path(
        parent,
        paste0(".", basename(path), ".partial_", next_io_token())
      )
      on.exit(if (file.exists(dest_tmp)) try(unlink(dest_tmp, force = TRUE), silent = TRUE),
              add = TRUE)

      if (!file.copy(staged, dest_tmp, overwrite = TRUE)) {
        stop("copy from system-temp spool to destination staging file failed")
      }

      if (file.exists(path)) try(unlink(path, force = TRUE), silent = TRUE)
      moved <- suppressWarnings(file.rename(dest_tmp, path))
      if (!isTRUE(moved)) {
        moved <- suppressWarnings(file.copy(dest_tmp, path, overwrite = TRUE))
        if (isTRUE(moved)) try(unlink(dest_tmp, force = TRUE), silent = TRUE)
      }

      if (!isTRUE(moved) || !file.exists(path)) {
        stop("destination finalize failed")
      }

      src_size <- suppressWarnings(file.info(staged)$size)
      dst_size <- suppressWarnings(file.info(path)$size)
      if (is.finite(src_size) && is.finite(dst_size) && src_size != dst_size) {
        stop("destination size verification failed")
      }
      TRUE
    }, error = function(e) {
      last_msg <<- conditionMessage(e)
      FALSE
    })

    if (isTRUE(ok)) {
      try(unlink(staged, force = TRUE), silent = TRUE)
      return(invisible(path))
    }

    if (a < attempts && delay > 0) Sys.sleep(delay * min(a, 4L))
  }

  emergency <- save_emergency_copy(staged, path, label = "UNCOMMITTED")
  msg <- paste0(
    "Primary output commit failed after ", attempts, " attempts: ", path,
    if (!is.null(last_msg)) paste0(" | last_error=", last_msg) else "",
    if (!is.na(emergency)) paste0(" | COMPUTED PAYLOAD PRESERVED AT: ", emergency) else
      " | emergency preservation also failed"
  )
  stop(msg)
}

# Compatibility wrappers retained from v17 so no downstream call or reviewer-facing
# reproducibility script loses an established helper name.
next_atomic_tmp <- function(path) {
  file.path(io_spool_root(), paste0(basename(path), ".tmp_", next_io_token()))
}

finalize_atomic_file <- function(tmp, path, attempts = io_attempts(), delay = io_delay()) {
  commit_staged_file(tmp, path, attempts = attempts, delay = delay)
}

safe_write_lines <- function(text, path) {
  staged <- stage_payload(
    function(tmp) writeLines(text, con = tmp, useBytes = TRUE),
    suffix = ".txt"
  )
  commit_staged_file(staged, path)
  invisible(path)
}


write_live_step <- function(stage, detail = "", cfg = config, force = FALSE) {
  if (identical(cfg$console_detail_level, "quiet") && !isTRUE(force)) {
    return(invisible(NULL))
  }
  line <- paste0(
    "LIVE STEP | ", stage,
    if (nzchar(detail)) paste0(" | ", detail) else ""
  )
  log_message(line)
  if (isTRUE(cfg$write_live_step_file)) {
    payload <- c(
      paste0("timestamp=", format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")),
      paste0("stage=", stage),
      paste0("detail=", detail)
    )
    try(safe_write_lines(payload, file.path(cfg$output_dir, "CURRENT_STEP.txt")), silent = TRUE)
  }
  invisible(NULL)
}

save_csv_safe <- function(x, path) {
  staged <- stage_payload(
    function(tmp) write.csv(x, tmp, row.names = FALSE),
    suffix = ".csv"
  )
  commit_staged_file(staged, path)
  invisible(path)
}

safe_save_rds <- function(x, path) {
  staged <- stage_payload(
    function(tmp) saveRDS(x, tmp),
    suffix = ".rds"
  )
  commit_staged_file(staged, path)
  invisible(path)
}

safe_append_csv_row <- function(x, path) {
  # Append-only telemetry is less critical than inferential outputs, but it is
  # still protected by directory recreation/retries. A failed history append
  # never masks the statistical computation.
  last_msg <- NULL
  for (a in seq_len(io_attempts())) {
    ok <- tryCatch({
      ensure_parent_writable(path, attempts = 1L, delay = 0)
      write.table(
        x,
        file = path,
        sep = ",",
        row.names = FALSE,
        col.names = !file.exists(path),
        append = file.exists(path),
        qmethod = "double"
      )
      TRUE
    }, error = function(e) {
      last_msg <<- conditionMessage(e)
      FALSE
    })
    if (isTRUE(ok)) return(invisible(path))
    if (a < io_attempts()) Sys.sleep(io_delay() * min(a, 4L))
  }
  warning(
    "Progress-history append failed; computation continues. ",
    if (!is.null(last_msg)) last_msg else "",
    call. = FALSE
  )
  invisible(NULL)
}

log_message <- function(...) {
  txt <- paste0(...)
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", txt)
  cat(line, "\n")
  flush.console()

  # Do not let a log-file problem erase an inferential result.
  ok <- FALSE
  last_msg <- NULL
  for (a in seq_len(min(io_attempts(), 4L))) {
    ok <- tryCatch({
      ensure_parent_writable(log_file, attempts = 1L, delay = 0)
      cat(line, "\n", file = log_file, append = TRUE)
      TRUE
    }, error = function(e) {
      last_msg <<- conditionMessage(e)
      FALSE
    })
    if (isTRUE(ok)) break
    Sys.sleep(io_delay())
  }
  if (!isTRUE(ok)) {
    emergency_log <- file.path(io_emergency_root(), paste0("run_log_", Sys.getpid(), ".txt"))
    try(cat(line, "\n", file = emergency_log, append = TRUE), silent = TRUE)
    warning(
      "run_log primary write failed; console + emergency log preserved. ",
      if (!is.null(last_msg)) last_msg else "",
      call. = FALSE
    )
  }
  invisible(line)
}

run_io_self_test <- function(cfg) {
  if (!isTRUE(cfg$io_self_test)) return(invisible(NULL))
  dirs <- c(
    cfg$output_dir,
    file.path(cfg$output_dir, "validation"),
    file.path(cfg$output_dir, "stage_archive"),
    file.path(cfg$output_dir, "real_data"),
    file.path(cfg$output_dir, "simulation"),
    file.path(cfg$output_dir, "checkpoints"),
    file.path(cfg$output_dir, "anchor_bridge"),
    file.path(cfg$output_dir, "sensitivity")
  )

  rows <- lapply(seq_along(dirs), function(i) {
    d <- dirs[i]
    test_path <- file.path(d, paste0(".FWKC_V25_IO_TEST_", Sys.getpid(), "_", i, ".txt"))
    t0 <- proc.time()[["elapsed"]]
    ok <- TRUE
    note <- "pass"
    tryCatch({
      ensure_parent_writable(test_path)
      staged <- stage_payload(
        function(tmp) writeLines(c("FWKC_V25_IO_SELF_TEST", d), tmp, useBytes = TRUE),
        suffix = ".txt"
      )
      commit_staged_file(staged, test_path)
      if (!file.exists(test_path)) stop("test file not visible after commit")
      unlink(test_path, force = TRUE)
    }, error = function(e) {
      ok <<- FALSE
      note <<- conditionMessage(e)
    })
    data.frame(
      directory = d,
      pass = ok,
      elapsed_seconds = proc.time()[["elapsed"]] - t0,
      note = note,
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, rows)
  # Save to emergency first so the self-test result itself survives a primary FS problem.
  emergency_report <- file.path(io_emergency_root(), paste0("IO_SELF_TEST_", Sys.getpid(), ".csv"))
  try(write.csv(out, emergency_report, row.names = FALSE), silent = TRUE)

  if (all(out$pass)) {
    save_csv_safe(out, file.path(cfg$output_dir, "validation", "io_self_test.csv"))
    log_message(
      "I/O SELF-TEST PASSED | tested_directories=", nrow(out),
      " | emergency_root=", io_emergency_root()
    )
  } else {
    print(out)
    stop(
      "I/O SELF-TEST FAILED before expensive analysis. ",
      "No Bayesian cross-check was started. Emergency report: ", emergency_report
    )
  }
  invisible(out)
}

format_duration <- function(seconds) {
  if (!is.finite(seconds) || seconds < 0) return("NA")
  seconds <- as.numeric(seconds)
  hh <- floor(seconds / 3600)
  mm <- floor((seconds %% 3600) / 60)
  ss <- floor(seconds %% 60)
  sprintf("%02d:%02d:%02d", hh, mm, ss)
}

progress_interval <- function(total, updates = 100L) {
  total <- max(1L, as.integer(total))
  updates <- max(1L, as.integer(updates))
  max(1L, floor(total / updates))
}

progress_message <- function(
  stage,
  current,
  total,
  cfg,
  stage_start = NULL,
  detail = "",
  force = FALSE
) {
  if (!isTRUE(cfg$console_progress) && !isTRUE(force)) {
    return(invisible(NULL))
  }

  current <- as.integer(current)
  total <- max(1L, as.integer(total))
  current <- min(max(current, 0L), total)
  pct <- 100 * current / total

  elapsed <- NA_real_
  eta <- NA_real_
  if (!is.null(stage_start) && is.finite(stage_start)) {
    elapsed <- proc.time()[["elapsed"]] - stage_start
    if (current > 0 && current < total && is.finite(elapsed)) {
      eta <- elapsed * (total - current) / current
    } else if (current >= total) {
      eta <- 0
    }
  }

  txt <- paste0(
    "PROGRESS | ", stage,
    " | ", current, "/", total,
    " (", sprintf("%.1f", pct), "%)",
    if (is.finite(elapsed)) paste0(" | elapsed=", format_duration(elapsed)) else "",
    if (is.finite(eta)) paste0(" | ETA=", format_duration(eta)) else "",
    if (nzchar(detail)) paste0(" | ", detail) else ""
  )

  log_message(txt)

  if (isTRUE(cfg$write_progress_status)) {
    status <- data.frame(
      timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      stage = stage,
      current = current,
      total = total,
      percent = pct,
      elapsed_seconds = elapsed,
      eta_seconds = eta,
      detail = detail,
      stringsAsFactors = FALSE
    )

    save_csv_safe(
      status,
      file.path(cfg$output_dir, "PROGRESS_STATUS.csv")
    )

    history_path <- file.path(cfg$output_dir, "PROGRESS_HISTORY.csv")
    safe_append_csv_row(status, history_path)
  }

  invisible(NULL)
}

should_report_progress <- function(current, total, cfg) {
  step <- progress_interval(total, cfg$progress_updates_per_stage)
  current == 1L || current == total || current %% step == 0L
}

rbind_fill <- function(lst) {
  lst <- lst[!vapply(lst, is.null, logical(1))]
  if (length(lst) == 0) return(data.frame())

  all_names <- unique(unlist(lapply(lst, names)))

  lst2 <- lapply(lst, function(x) {
    missing_names <- setdiff(all_names, names(x))
    if (length(missing_names) > 0) {
      for (nm in missing_names) {
        x[[nm]] <- NA
      }
    }
    x[, all_names, drop = FALSE]
  })

  rownames_out <- NULL
  out <- do.call(rbind, lst2)
  rownames(out) <- rownames_out
  out
}

elapsed_seconds <- function(t0) {
  proc.time()[["elapsed"]] - t0
}

###############################################################################
# 3B) PERSISTENT PSOCK PARALLEL EXECUTION + RNG-SAFE HELPERS
###############################################################################

.fwkc_parallel_state <- new.env(parent = emptyenv())

# R's set.seed() requires an integer-range seed. Large prespecified IDs are used
# only for namespace separation. Valid legacy seeds stay exactly unchanged.
normalize_fwkc_seed <- function(seed) {
  x <- suppressWarnings(as.numeric(seed)[1])
  if (!is.finite(x)) stop("Invalid RNG seed encountered.")
  max_seed <- as.numeric(.Machine$integer.max - 1L)
  if (x >= 1 && x <= max_seed && abs(x - round(x)) < 1e-7) {
    return(as.integer(round(x)))
  }
  y <- x %% max_seed
  if (!is.finite(y) || y <= 0) y <- y + max_seed
  as.integer(floor(y))
}

set_fwkc_seed <- function(seed) {
  seed_int <- normalize_fwkc_seed(seed)
  base::set.seed(seed_int)
  invisible(seed_int)
}

resolve_local_worker_capacity <- function(cfg, n_tasks = Inf) {
  detected <- suppressWarnings(parallel::detectCores(logical = TRUE))
  if (length(detected) != 1L || !is.finite(detected) || detected < 1L) detected <- 1L
  requested <- suppressWarnings(as.integer(cfg$parallel_workers))
  if (!is.finite(requested) || requested <= 0L) {
    requested <- max(1L, as.integer(detected) - as.integer(cfg$parallel_reserve_cores))
  }
  cap <- max(1L, as.integer(cfg$parallel_max_workers))
  workers <- min(requested, cap, as.integer(detected))
  if (is.finite(n_tasks)) workers <- min(workers, max(1L, as.integer(n_tasks)))
  max(1L, as.integer(workers))
}

resolve_parallel_workers <- function(cfg, n_tasks = Inf) {
  if (!isTRUE(cfg$parallel_enabled)) return(1L)
  if (identical(cfg$parallel_mode, "network")) {
    workers <- length(cfg$parallel_hosts)
    if (workers < 1L) return(1L)
    if (is.finite(n_tasks)) workers <- min(workers, max(1L, as.integer(n_tasks)))
    return(max(1L, as.integer(workers)))
  }
  resolve_local_worker_capacity(cfg, n_tasks)
}

resolve_brms_cores <- function(cfg) {
  # brms/cmdstan chains run on the master computer; remote PSOCK hosts are never
  # counted as local CmdStan cores. During the one-time publication cross-check,
  # all available local logical cores may be used up to the FIXED chain count.
  if (!isTRUE(cfg$brms_parallel_chains)) return(1L)
  crosscheck_active <- isTRUE(cfg$brms_crosscheck_active)
  if (crosscheck_active && isTRUE(cfg$bayes_crosscheck_use_all_local_cores)) {
    detected <- suppressWarnings(parallel::detectCores(logical = TRUE))
    if (length(detected) != 1L || !is.finite(detected) || detected < 1L) detected <- 1L
    return(max(1L, min(as.integer(cfg$bayes_chains), as.integer(detected))))
  }
  local_cap <- resolve_local_worker_capacity(cfg, cfg$bayes_chains)
  max(1L, min(as.integer(cfg$bayes_chains), local_cap))
}

resolve_brms_crosscheck_cores <- function(cfg) {
  cfg2 <- cfg
  cfg2$brms_crosscheck_active <- TRUE
  resolve_brms_cores(cfg2)
}

parallel_stage_enabled <- function(cfg, stage_key, n_tasks) {
  isTRUE(cfg$parallel_enabled) &&
    identical(cfg$bayes_engine, "gibbs") &&
    stage_key %in% cfg$parallel_stages &&
    resolve_parallel_workers(cfg, n_tasks) > 1L
}

parallel_runtime_plan <- function(cfg) {
  physical <- suppressWarnings(parallel::detectCores(logical = FALSE))
  logical <- suppressWarnings(parallel::detectCores(logical = TRUE))
  workers <- resolve_parallel_workers(cfg)
  hosts_txt <- if (length(cfg$parallel_hosts) > 0) paste(cfg$parallel_hosts, collapse = ";") else ""
  data.frame(
    parallel_enabled = isTRUE(cfg$parallel_enabled),
    backend = cfg$parallel_backend,
    mode = cfg$parallel_mode,
    physical_cores_detected_master = if(length(physical)) physical else NA_integer_,
    logical_cores_detected_master = if(length(logical)) logical else NA_integer_,
    reserve_cores = cfg$parallel_reserve_cores,
    max_local_workers = cfg$parallel_max_workers,
    configured_workers = cfg$parallel_workers,
    effective_scenario_workers = workers,
    network_hosts = hosts_txt,
    network_shared_fs_confirmed = isTRUE(cfg$parallel_network_shared_fs),
    brms_local_cores = resolve_brms_cores(cfg),
    brms_crosscheck_cores = resolve_brms_crosscheck_cores(cfg),
    statistical_M_dgm = cfg$M_dgm,
    statistical_B_boot_dgm = cfg$B_boot_dgm,
    statistical_B_boot_real = cfg$B_boot_real,
    statistical_n_mc = cfg$n_mc,
    statistical_design_changed = FALSE,
    scalability_benchmark_parallelized = FALSE,
    note = paste(
      "Execution-only parallelism; estimand/M/B/n_mc/chains/iterations/priors/",
      "fusion/DGM grids/sensitivity/QC remain unchanged."
    ),
    stringsAsFactors = FALSE
  )
}

parallel_export_names <- function() {
  nm <- ls(envir = .GlobalEnv, all.names = TRUE)
  funs <- nm[vapply(
    nm,
    function(z) tryCatch(is.function(get(z, envir = .GlobalEnv, inherits = FALSE)),
                         error = function(e) FALSE),
    logical(1)
  )]
  unique(c(funs, intersect(c("log_file", "config", ".fwkc_io_state"), nm)))
}

parallel_cluster_spec <- function(cfg) {
  if (identical(cfg$parallel_mode, "network")) {
    return(as.character(cfg$parallel_hosts))
  }
  as.integer(resolve_parallel_workers(cfg))
}

start_parallel_cluster <- function(cfg) {
  workers <- resolve_parallel_workers(cfg)
  if (!isTRUE(cfg$parallel_enabled) || workers <= 1L) return(NULL)

  if (exists("cluster", envir = .fwkc_parallel_state, inherits = FALSE)) {
    cl0 <- get("cluster", envir = .fwkc_parallel_state, inherits = FALSE)
    if (!is.null(cl0)) return(cl0)
  }

  if (!identical(toupper(cfg$parallel_backend), "PSOCK")) {
    stop("Only PSOCK is supported in the v16 publication script.")
  }
  if (identical(cfg$parallel_mode, "network") && !isTRUE(cfg$parallel_network_shared_fs)) {
    stop("Network PSOCK requires a shared output/checkpoint filesystem visible at the same path on every node.")
  }

  spec <- parallel_cluster_spec(cfg)
  cluster_stdout <- if (isTRUE(cfg$parallel_worker_progress)) {
    ""
  } else {
    file.path(cfg$output_dir, "parallel_worker_logs", "cluster_stdout.log")
  }

  log_message(
    "PARALLEL CLUSTER START | backend=PSOCK | mode=", cfg$parallel_mode,
    " | workers=", workers,
    if (identical(cfg$parallel_mode, "network")) paste0(" | hosts=", paste(cfg$parallel_hosts, collapse=",")) else "",
    " | statistical settings unchanged"
  )

  cl <- parallel::makePSOCKcluster(spec, outfile = cluster_stdout)

  export_names <- parallel_export_names()
  parallel::clusterExport(cl, varlist = export_names, envir = .GlobalEnv)

  parallel::clusterEvalQ(cl, {
    options(stringsAsFactors = FALSE)
    options(warn = 1)
    suppressPackageStartupMessages(library(randomForest))
    RNGkind("L'Ecuyer-CMRG")
    # Prevent nested native BLAS/OpenMP oversubscription inside each PSOCK worker.
    Sys.setenv(
      OMP_NUM_THREADS = "1",
      OPENBLAS_NUM_THREADS = "1",
      MKL_NUM_THREADS = "1",
      VECLIB_MAXIMUM_THREADS = "1",
      NUMEXPR_NUM_THREADS = "1"
    )
    NULL
  })

  # Independent worker logs; stdout may simultaneously appear in RStudio Console.
  parallel::clusterCall(
    cl,
    function(root, enable_logs) {
      host <- Sys.info()[["nodename"]]
      if (is.null(host) || is.na(host) || !nzchar(host)) host <- "unknownhost"
      safe_host <- gsub("[^A-Za-z0-9_.-]", "_", host)
      if (isTRUE(enable_logs)) {
        d <- file.path(root, "parallel_worker_logs")
        dir.create(d, recursive = TRUE, showWarnings = FALSE)
        log_file <<- file.path(d, paste0("worker_", safe_host, "_", Sys.getpid(), ".log"))
      } else {
        log_file <<- tempfile(pattern = "fwkc_worker_", fileext = ".log")
      }
      NULL
    },
    cfg$output_dir,
    isTRUE(cfg$parallel_worker_logs)
  )

  inventory <- parallel::clusterCall(cl, function() {
    data.frame(
      host = as.character(Sys.info()[["nodename"]]),
      pid = Sys.getpid(),
      R_version = as.character(getRversion()),
      platform = R.version$platform,
      stringsAsFactors = FALSE
    )
  })
  inventory <- rbind_fill(inventory)
  save_csv_safe(inventory, file.path(cfg$output_dir, "parallel", "WORKER_INVENTORY.csv"))

  assign("cluster", cl, envir = .fwkc_parallel_state)
  assign("workers", workers, envir = .fwkc_parallel_state)
  log_message("PARALLEL CLUSTER READY | workers=", workers)
  cl
}

get_parallel_cluster <- function(cfg) {
  if (!isTRUE(cfg$parallel_enabled) || resolve_parallel_workers(cfg) <= 1L) return(NULL)
  if (!exists("cluster", envir = .fwkc_parallel_state, inherits = FALSE)) {
    return(start_parallel_cluster(cfg))
  }
  get("cluster", envir = .fwkc_parallel_state, inherits = FALSE)
}

stop_parallel_cluster <- function() {
  if (exists("cluster", envir = .fwkc_parallel_state, inherits = FALSE)) {
    cl <- get("cluster", envir = .fwkc_parallel_state, inherits = FALSE)
    if (!is.null(cl)) try(parallel::stopCluster(cl), silent = TRUE)
    rm(list = "cluster", envir = .fwkc_parallel_state)
    if (exists("workers", envir = .fwkc_parallel_state, inherits = FALSE)) {
      rm(list = "workers", envir = .fwkc_parallel_state)
    }
  }
  invisible(NULL)
}

fwkc_real_data_task <- function(i, c_grid, df, cfg_real, show_boot_progress = FALSE) {
  tryCatch({
    cc <- c_grid[i]
    t0 <- proc.time()[["elapsed"]]
    log_message("WORKER REAL-DATA START | c=", cc, " | index=", i)
    fit <- pipeline_bootstrap_inference(
      df_in = df,
      c_val = cc,
      cfg = cfg_real,
      keep_boot = FALSE,
      seed_offset = 1000 + i,
      show_boot_progress = show_boot_progress,
      progress_label = paste0("Real-data bootstrap c=", format(cc, trim = TRUE))
    )

    if (is.null(fit) || isTRUE(fit$failed)) {
      log_message("WORKER REAL-DATA END | c=", cc, " | fit_failed | elapsed=",
                  format_duration(proc.time()[["elapsed"]] - t0))
      return(list(
        ok = TRUE,
        index = as.integer(i),
        row = data.frame(c_val=cc,n=nrow(df),Direct_failed=TRUE,FWKC_failed=TRUE,
                         stringsAsFactors=FALSE),
        test = NULL,
        error = NA_character_,
        pid = Sys.getpid()
      ))
    }

    row <- data.frame(
      c_val = cc, n = nrow(df), prevalence = fit$prevalence, Se = fit$Se, Sp = fit$Sp,
      Kappa_Direct = fit$direct_point,
      Direct_Boot_Lower = fit$direct_boot_lower, Direct_Boot_Upper = fit$direct_boot_upper,
      Direct_Delta_Lower = fit$direct_delta_lower, Direct_Delta_Upper = fit$direct_delta_upper,
      Direct_Logit_Lower = fit$direct_logit_lower, Direct_Logit_Upper = fit$direct_logit_upper,
      Direct_Basic_Lower = fit$direct_basic_lower, Direct_Basic_Upper = fit$direct_basic_upper,
      Direct_BCa_Lower = fit$direct_bca_lower, Direct_BCa_Upper = fit$direct_bca_upper,
      Kappa_FWKC = fit$fwkc_point, FWKC_Lower = fit$fwkc_lower, FWKC_Upper = fit$fwkc_upper,
      FWKC_Calibrated_Lower = fit$fwkc_cal_lower, FWKC_Calibrated_Upper = fit$fwkc_cal_upper,
      Calibration_Lambda = fit$calibration_lambda,
      FWKC_Cal_Sqrt_Lower = fit$fwkc_cal_sqrt_lower, FWKC_Cal_Sqrt_Upper = fit$fwkc_cal_sqrt_upper,
      FWKC_Cal_LinearSE_Lower = fit$fwkc_cal_linear_lower, FWKC_Cal_LinearSE_Upper = fit$fwkc_cal_linear_upper,
      FWKC_Cal_Adaptive_Lower = fit$fwkc_cal_adaptive_lower, FWKC_Cal_Adaptive_Upper = fit$fwkc_cal_adaptive_upper,
      FWKC_Legacy_LinearContract_Lower = fit$fwkc_legacy_linear_contract_lower,
      FWKC_Legacy_LinearContract_Upper = fit$fwkc_legacy_linear_contract_upper,
      Lambda_SE_Sqrt = fit$calibration_lambda_se_sqrt,
      Lambda_Variance_Linear = fit$calibration_lambda_variance_linear,
      Calibration_Identity_Diff = fit$calibration_identity_diff,
      Bayes_Mean = fit$bayes_mean, Bayes_SE = fit$bayes_se, Bayes_Rhat = fit$bayes_rhat,
      Bayes_Convergence_OK = fit$bayes_convergence_ok,
      ML_Mean = fit$ml_mean, W_Bayes = fit$w_bayes, W_ML = fit$w_ml, RF_OOB_MSE = fit$rf_oob_mse,
      Pseudo_SE_Empirical = fit$pseudo_se_empirical, Pseudo_SE_Bootstrap = fit$pseudo_se_bootstrap,
      Component_Boot_Cov = fit$component_boot_cov, Component_Boot_Cor = fit$component_boot_cor,
      PABAK = fit$PABAK, Gwet_AC1 = fit$Gwet_AC1,
      Direct_Boot_Seconds = fit$direct_boot_seconds, FWKC_Core_Seconds = fit$core_seconds,
      FWKC_Boot_Seconds = fit$fwkc_boot_seconds,
      Valid_Direct_Boot = fit$valid_direct_boot, Valid_FWKC_Boot = fit$valid_boot,
      Valid_Boot = fit$valid_boot, Direct_failed = fit$direct_failed, FWKC_failed = fit$fwkc_failed,
      InternalRef_Inclusion = fit$internal_reference_inclusion,
      Legacy_Estimated_n = fit$legacy_estimated_n, Legacy_Predictive_Power = fit$legacy_predictive_power,
      stringsAsFactors = FALSE
    )
    pt <- fit$paired_tests
    if (!is.null(pt)) pt$c_val <- cc
    log_message("WORKER REAL-DATA END | c=", cc,
                " | direct=", signif(fit$direct_point,5),
                " | FWKC=", signif(fit$fwkc_point,5),
                " | elapsed=", format_duration(proc.time()[["elapsed"]] - t0))
    list(ok=TRUE,index=as.integer(i),row=row,test=pt,error=NA_character_,pid=Sys.getpid())
  }, error = function(e) {
    list(ok=FALSE,index=as.integer(i),row=NULL,test=NULL,error=conditionMessage(e),pid=Sys.getpid())
  })
}

parallel_scenario_worker <- function(s, cfg_worker, stage_label) {
  sid <- s$scenario_id[1]
  t0 <- proc.time()[["elapsed"]]
  tryCatch({
    log_message(
      "WORKER SCENARIO START | stage=", stage_label,
      " | scenario=", sid, " | dgm=", s$dgm_id[1],
      " | n=", s$n[1], " | p=", s$p_true[1], " | c=", s$c_val[1]
    )
    ans <- evaluate_scenario(s, cfg_worker)
    log_message(
      "WORKER SCENARIO END | stage=", stage_label,
      " | scenario=", sid, " | elapsed=", format_duration(proc.time()[["elapsed"]] - t0)
    )
    list(ok=TRUE,result=ans,scenario_id=sid,pid=Sys.getpid(),error=NA_character_)
  }, error = function(e) {
    log_message("WORKER SCENARIO ERROR | stage=", stage_label,
                " | scenario=", sid, " | ", conditionMessage(e))
    list(ok=FALSE,result=NULL,scenario_id=sid,pid=Sys.getpid(),error=conditionMessage(e))
  })
}

run_scenario_grid_compute <- function(
  scenarios, cfg, stage_key, stage_label, checkpoint_csv, extra_columns = NULL
) {
  n_tasks <- nrow(scenarios)
  if (n_tasks <= 0) return(list())
  results <- vector("list", n_tasks)
  stage_start <- proc.time()[["elapsed"]]

  decorate <- function(z) {
    if (!is.null(z) && !is.null(extra_columns) && length(extra_columns) > 0) {
      for (nm in names(extra_columns)) z[[nm]] <- extra_columns[[nm]]
    }
    z
  }
  write_checkpoint <- function() {
    done <- results[!vapply(results, is.null, logical(1))]
    if (length(done) > 0) save_csv_safe(rbind_fill(done), checkpoint_csv)
    invisible(NULL)
  }

  if (!parallel_stage_enabled(cfg, stage_key, n_tasks)) {
    for (i in seq_len(n_tasks)) {
      sc <- scenarios[i, , drop = FALSE]
      log_message(stage_label, " scenario ", i, "/", n_tasks,
                  " | ", sc$dgm_id, " | n=", sc$n, " | p=", signif(sc$p_true,5), " | c=", sc$c_val)
      results[[i]] <- decorate(evaluate_scenario(sc, cfg))
      write_checkpoint()
      progress_message(stage=stage_label,current=i,total=n_tasks,cfg=cfg,stage_start=stage_start,
                       detail=paste0("scenario=",sc$scenario_id,"; sequential"))
    }
    return(results)
  }

  cl <- get_parallel_cluster(cfg)
  workers <- resolve_parallel_workers(cfg, n_tasks)
  cfg_worker <- cfg
  cfg_worker$parallel_enabled <- FALSE  # strict single-level parallelism
  cfg_worker$parallel_workers <- 1L
  cfg_worker$write_progress_status <- FALSE
  cfg_worker$verbose <- isTRUE(cfg$parallel_worker_progress)
  cfg_worker$console_progress <- isTRUE(cfg$parallel_worker_progress)

  manual_batch <- suppressWarnings(as.integer(cfg$parallel_scenario_batch_size))
  if (is.finite(manual_batch) && manual_batch > 0L) {
    batch_size <- manual_batch
  } else {
    batch_size <- workers * max(1L, as.integer(cfg$parallel_batch_multiplier))
  }
  batch_size <- max(1L, min(batch_size, n_tasks))
  batches <- split(seq_len(n_tasks), ceiling(seq_len(n_tasks) / batch_size))

  log_message("PARALLEL STAGE START | ",stage_label," | workers=",workers,
              " | tasks=",n_tasks," | batches=",length(batches)," | mode=",cfg$parallel_mode)

  for (b in seq_along(batches)) {
    ids <- batches[[b]]
    tasks <- lapply(ids, function(i) scenarios[i, , drop=FALSE])
    log_message(
      "PARALLEL BATCH START | ", stage_label,
      " | batch=", b, "/", length(batches),
      " | workers=", workers,
      " | scenarios=", paste(scenarios$scenario_id[ids], collapse=",")
    )
    raw <- parallel::parLapplyLB(cl, tasks, parallel_scenario_worker,
                                 cfg_worker=cfg_worker, stage_label=stage_label)

    errors <- character(0)
    for (j in seq_along(ids)) {
      item <- raw[[j]]
      ii <- ids[j]
      if (isTRUE(item$ok)) {
        results[[ii]] <- decorate(item$result)
        log_message("SCENARIO COMPLETE | ",stage_label," | scenario=",item$scenario_id,
                    " | worker_pid=",item$pid)
      } else {
        errors <- c(errors,paste0("scenario ",item$scenario_id,": ",item$error))
      }
    }
    write_checkpoint()
    completed <- sum(!vapply(results,is.null,logical(1)))
    progress_message(
      stage=paste0(stage_label," [parallel]"),current=completed,total=n_tasks,cfg=cfg,
      stage_start=stage_start,
      detail=paste0("batch=",b,"/",length(batches),"; workers=",workers,
                    "; completed=",completed,"/",n_tasks)
    )
    log_message("PARALLEL BATCH END | ",stage_label," | batch=",b,"/",length(batches),
                " | completed=",completed,"/",n_tasks)
    if (length(errors)>0) {
      stop("Parallel stage failed after saving completed scenario outputs:\n- ",
           paste(errors,collapse="\n- "))
    }
  }
  log_message("PARALLEL STAGE COMPLETE | ",stage_label," | workers=",workers,
              " | elapsed=",format_duration(proc.time()[["elapsed"]]-stage_start))
  results
}

write_parallel_plan <- function(cfg) {
  plan <- parallel_runtime_plan(cfg)
  save_csv_safe(plan, file.path(cfg$output_dir, "parallel", "PARALLEL_RUNTIME_PLAN.csv"))
  # Root-level copy is convenient when inspecting a run directory quickly.
  save_csv_safe(plan, file.path(cfg$output_dir, "PARALLEL_RUNTIME_PLAN.csv"))
  plan
}

# Backward-compatible computational wrappers retained from both v15 branches.
# They are not used by the v16 main pipeline, but keep earlier helper interfaces
# available without changing any inferential definition.
fwkc_worker_function_names <- function() {
  parallel_export_names()
}

make_fwkc_psock_cluster <- function(cfg, stage_key = "compat", n_tasks = Inf) {
  list(
    cluster = get_parallel_cluster(cfg),
    workers = resolve_parallel_workers(cfg, n_tasks),
    stage_key = stage_key,
    persistent = TRUE
  )
}

fwkc_parallel_worker_eval_scenario <- function(i, scenarios, cfg) {
  tryCatch(
    list(
      ok = TRUE,
      index = as.integer(i),
      value = evaluate_scenario(scenarios[i, , drop = FALSE], cfg),
      error = NA_character_
    ),
    error = function(e) list(
      ok = FALSE,
      index = as.integer(i),
      value = NULL,
      error = conditionMessage(e)
    )
  )
}

run_scenarios_parallel_safe <- function(
  scenarios, cfg, stage_label, checkpoint_csv = NULL, stage_key = "main_simulation"
) {
  if (is.null(checkpoint_csv)) {
    checkpoint_csv <- file.path(
      cfg$output_dir, "simulation",
      paste0(gsub("[^A-Za-z0-9_-]", "_", stage_label), "_checkpoint.csv")
    )
  }
  run_scenario_grid_compute(
    scenarios = scenarios,
    cfg = cfg,
    stage_key = stage_key,
    stage_label = stage_label,
    checkpoint_csv = checkpoint_csv
  )
}

###############################################################################
# 4) INPUT DATA
###############################################################################

load_input_data <- function(cfg) {
  input_path <- cfg$input_path
  if (is.na(input_path) || !nzchar(input_path)) {
    input_path <- file.choose()
  }

  x <- read_excel(input_path)

  if (!all(c("D", "T") %in% names(x))) {
    stop("Input file must contain columns D (gold standard) and T (test).")
  }

  D_raw <- suppressWarnings(as.integer(as.character(x$D)))
  T_raw <- suppressWarnings(as.integer(as.character(x$T)))

  if (any(is.na(D_raw)) || any(is.na(T_raw)) ||
      any(!D_raw %in% c(0, 1)) || any(!T_raw %in% c(0, 1))) {
    stop("Columns D and T must contain only 0/1 values without missing data.")
  }

  out <- data.frame(
    D = factor(D_raw, levels = c(0, 1)),
    T = factor(T_raw, levels = c(0, 1))
  )

  input_resolved <- normalizePath(input_path, winslash = "/", mustWork = TRUE)
  input_md5 <- unname(tools::md5sum(input_resolved))
  cc <- table(out$D, out$T)
  input_meta <- data.frame(
    input_path = input_resolved,
    input_md5 = input_md5,
    n = nrow(out),
    TN = as.integer(cc[1, 1]),
    FP = as.integer(cc[1, 2]),
    FN = as.integer(cc[2, 1]),
    TP = as.integer(cc[2, 2]),
    loaded_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    stringsAsFactors = FALSE
  )
  save_csv_safe(
    input_meta,
    file.path(cfg$output_dir, "validation", "input_provenance.csv")
  )
  attr(out, "input_provenance") <- input_meta

  log_message(
    "Loaded input file: ", input_resolved,
    " | n=", nrow(out),
    " | MD5=", input_md5
  )
  out
}

# Publication-facing consistency check between the selected real-data file and
# the manuscript's Table 2 total sample size. Cell labels are deliberately NOT
# hard-coded here because the raw input remains the authoritative source for the
# D/T orientation; input_provenance.csv records the actual 2x2 counts and MD5.
validate_manuscript_anchor_sample <- function(df, cfg) {
  observed_n <- nrow(df)
  expected_n <- as.integer(cfg$manuscript_anchor_n)
  match_n <- is.finite(observed_n) && is.finite(expected_n) && observed_n == expected_n

  out <- data.frame(
    manuscript_table = "Table 2 - Weiner et al. coronary artery disease anchor",
    expected_n = expected_n,
    observed_input_n = observed_n,
    exact_match = match_n,
    bridge_design = cfg$anchor_bridge_design,
    bridge_uses_observed_input_n = TRUE,
    stringsAsFactors = FALSE
  )

  save_csv_safe(
    out,
    file.path(cfg$output_dir, "validation", "manuscript_anchor_sample_size_check.csv")
  )

  if (!match_n) {
    msg <- paste0(
      "Selected real-data input has n=", observed_n,
      "; manuscript Table 2 anchor expects n=", expected_n, "."
    )
    if (identical(cfg$run_mode, "publication") &&
        isTRUE(cfg$require_manuscript_anchor_n_in_publication)) {
      stop(msg, " Publication run stopped to prevent mismatch with the manuscript.")
    }
    log_message("WARNING | ", msg)
  } else {
    log_message("Manuscript anchor sample-size check passed: n=", observed_n)
  }

  out
}

###############################################################################
# 4B) SAFE EXPLICIT TARGET-PREVALENCE RESAMPLING (BASELINE INNOVATION B)
###############################################################################

# The submitted code globally overrode base::sample() to force prevalence p.
# That behavior is preserved here as an explicit helper rather than a global
# side effect. Primary frequentist coverage uses explicit DGM generation, not
# this empirical stress-resampling helper.
resample_target_prevalence <- function(df_in, n, p_target) {
  if (!all(c("D", "T") %in% names(df_in))) {
    stop("df_in must contain D and T.")
  }
  if (!is.finite(n) || n < 2 || !is.finite(p_target) ||
      p_target <= 0 || p_target >= 1) {
    stop("n must be >=2 and p_target must lie strictly between 0 and 1.")
  }

  D01 <- as.integer(as.character(df_in$D))
  idx1 <- which(D01 == 1)
  idx0 <- which(D01 == 0)
  if (length(idx1) == 0 || length(idx0) == 0) {
    stop("Both disease strata must be present for target-prevalence resampling.")
  }

  n1 <- max(1L, min(as.integer(n) - 1L, as.integer(round(n * p_target))))
  n0 <- as.integer(n) - n1
  id <- c(
    sample(idx1, n1, replace = TRUE),
    sample(idx0, n0, replace = TRUE)
  )
  id <- sample(id, length(id), replace = FALSE)
  df_in[id, , drop = FALSE]
}

###############################################################################
# 5) WEIGHTED-KAPPA ESTIMAND
###############################################################################

clamp <- function(x, lo, hi) {
  pmin(pmax(x, lo), hi)
}

weighted_kappa_counts <- function(TP, FN, FP, TN, c_val) {
  if (!is.finite(c_val) || c_val <= 0 || c_val >= 1) {
    return(NA_real_)
  }

  num <- TP * TN - FN * FP

  den <- c_val * (TP + FN) * (FN + TN) +
    (1 - c_val) * (FP + TN) * (TP + FP)

  if (!is.finite(den) || abs(den) < 1e-14) {
    return(NA_real_)
  }

  clamp(as.numeric(num / den), -1, 1)
}

weighted_kappa_loss <- function(TP, FN, FP, TN, c_val) {
  n <- TP + FN + FP + TN

  if (!is.finite(n) || n <= 0 ||
      !is.finite(c_val) || c_val <= 0 || c_val >= 1) {
    return(NA_real_)
  }

  # c = L_FN / (L_FN + L_FP)
  observed_loss <-
    c_val * FN / n +
    (1 - c_val) * FP / n

  random_loss <-
    c_val * (TP + FN) * (FN + TN) / n^2 +
    (1 - c_val) * (FP + TN) * (TP + FP) / n^2

  if (!is.finite(random_loss) || random_loss <= 0) {
    return(NA_real_)
  }

  clamp(1 - observed_loss / random_loss, -1, 1)
}

# Explicit conventional weighted-agreement derivation.
# Rows are gold-standard status D=(0,1); columns are test status T=(0,1).
# With normalized disagreement losses L_FN=c and L_FP=1-c, agreement weights
# are W = 1-L, hence W00=W11=1, W01=c, W10=1-c.
# Then kappa_w = (A_o^w-A_e^w)/(1-A_e^w) = 1-L_o/L_e.
weighted_kappa_weight_matrix <- function(TP, FN, FP, TN, c_val) {
  n <- TP + FN + FP + TN

  if (!is.finite(n) || n <= 0 ||
      !is.finite(c_val) || c_val <= 0 || c_val >= 1) {
    return(NA_real_)
  }

  P <- matrix(
    c(TN, FP, FN, TP),
    nrow = 2,
    byrow = TRUE
  ) / n

  W <- matrix(
    c(1, c_val, 1 - c_val, 1),
    nrow = 2,
    byrow = TRUE
  )

  row_marg <- rowSums(P)
  col_marg <- colSums(P)
  P_e <- outer(row_marg, col_marg)

  A_o <- sum(W * P)
  A_e <- sum(W * P_e)

  if (!is.finite(A_e) || abs(1 - A_e) < 1e-14) {
    return(NA_real_)
  }

  clamp((A_o - A_e) / (1 - A_e), -1, 1)
}

weight_matrix_documentation <- function() {
  c_grid <- seq(0.1, 0.9, by = 0.1)

  data.frame(
    c_val = c_grid,
    L_FN = c_grid,
    L_FP = 1 - c_grid,
    W_D0_T0 = 1,
    W_D0_T1 = c_grid,
    W_D1_T0 = 1 - c_grid,
    W_D1_T1 = 1,
    interpretation = paste0(
      "c=L_FN/(L_FN+L_FP); c>0.5 emphasizes false-negative loss"
    ),
    stringsAsFactors = FALSE
  )
}


cohen_kappa_counts <- function(TP, FN, FP, TN) {
  n <- TP + FN + FP + TN
  if (!is.finite(n) || n <= 0) return(NA_real_)

  po <- (TP + TN) / n

  p_gold_pos <- (TP + FN) / n
  p_gold_neg <- (FP + TN) / n
  p_test_pos <- (TP + FP) / n
  p_test_neg <- (FN + TN) / n

  pe <- p_gold_pos * p_test_pos +
    p_gold_neg * p_test_neg

  if (!is.finite(pe) || abs(1 - pe) < 1e-14) {
    return(NA_real_)
  }

  (po - pe) / (1 - pe)
}

weighted_kappa_population <- function(p, Se, Sp, c_val) {
  if (any(!is.finite(c(p, Se, Sp, c_val))) ||
      p <= 0 || p >= 1 ||
      Se < 0 || Se > 1 ||
      Sp < 0 || Sp > 1 ||
      c_val <= 0 || c_val >= 1) {
    return(NA_real_)
  }

  Q <- p * Se + (1 - p) * (1 - Sp)

  num <- p * (1 - p) * (Se + Sp - 1)
  den <- p * (1 - Q) * c_val +
    (1 - p) * Q * (1 - c_val)

  if (!is.finite(den) || abs(den) < 1e-14) {
    return(NA_real_)
  }

  clamp(num / den, -1, 1)
}

true_kappa_from_params <- function(p, Se, Sp, c_val) {
  weighted_kappa_population(p, Se, Sp, c_val)
}


# Preliminary extension of the AGREEMENT ESTIMAND only.
# This does not claim that the binary FWKC pseudo-rater/inference architecture
# has already been validated for multiclass outcomes.
weighted_kappa_multiclass <- function(
  table_matrix,
  weights = c("quadratic", "linear", "unweighted")
) {
  X <- as.matrix(table_matrix)

  if (nrow(X) != ncol(X) || nrow(X) < 2) {
    stop("table_matrix must be a square K x K contingency table.")
  }

  if (any(!is.finite(X)) || any(X < 0)) {
    stop("table_matrix must contain finite non-negative counts.")
  }

  n <- sum(X)
  if (n <= 0) return(NA_real_)

  K <- nrow(X)

  if (is.matrix(weights)) {
    W <- weights
    if (!all(dim(W) == c(K, K))) {
      stop("Custom weight matrix must have the same dimension as table_matrix.")
    }
  } else {
    weights <- match.arg(weights)
    d <- abs(outer(seq_len(K), seq_len(K), "-"))

    if (weights == "unweighted") {
      W <- 1 * (d == 0)
    } else if (weights == "linear") {
      W <- 1 - d / (K - 1)
    } else {
      W <- 1 - (d / (K - 1))^2
    }
  }

  P <- X / n
  row_marg <- rowSums(P)
  col_marg <- colSums(P)
  P_e <- outer(row_marg, col_marg)

  A_o <- sum(W * P)
  A_e <- sum(W * P_e)

  if (!is.finite(A_e) || abs(1 - A_e) < 1e-14) {
    return(NA_real_)
  }

  (A_o - A_e) / (1 - A_e)
}

###############################################################################
# 6) COUNTS + DIAGNOSTIC / AGREEMENT METRICS
###############################################################################

compute_counts <- function(D, T) {
  D01 <- as.integer(as.character(D))
  T01 <- as.integer(as.character(T))

  TP <- sum(D01 == 1 & T01 == 1)
  FN <- sum(D01 == 1 & T01 == 0)
  FP <- sum(D01 == 0 & T01 == 1)
  TN <- sum(D01 == 0 & T01 == 0)

  list(
    TP = TP,
    FN = FN,
    FP = FP,
    TN = TN,
    n = length(D01)
  )
}

compute_pabak <- function(TP, FN, FP, TN) {
  n <- TP + FN + FP + TN
  if (n <= 0) return(NA_real_)
  po <- (TP + TN) / n
  2 * po - 1
}

compute_gwet_ac1 <- function(TP, FN, FP, TN) {
  n <- TP + FN + FP + TN
  if (n <= 0) return(NA_real_)

  po <- (TP + TN) / n
  p_gold_pos <- (TP + FN) / n
  p_test_pos <- (TP + FP) / n
  pi_pos <- (p_gold_pos + p_test_pos) / 2

  pe <- 2 * pi_pos * (1 - pi_pos)

  if (!is.finite(pe) || abs(1 - pe) < 1e-14) {
    return(NA_real_)
  }

  (po - pe) / (1 - pe)
}

compute_metrics <- function(D, T, c_val) {
  cc <- compute_counts(D, T)

  with(cc, {
    Se <- if ((TP + FN) > 0) TP / (TP + FN) else NA_real_
    Sp <- if ((TN + FP) > 0) TN / (TN + FP) else NA_real_
    PPV <- if ((TP + FP) > 0) TP / (TP + FP) else NA_real_
    NPV <- if ((TN + FN) > 0) TN / (TN + FN) else NA_real_

    prevalence <- if (n > 0) (TP + FN) / n else NA_real_
    test_positive_rate <- if (n > 0) (TP + FP) / n else NA_real_

    list(
      TP = TP,
      FN = FN,
      FP = FP,
      TN = TN,
      n = n,
      Se = Se,
      Sp = Sp,
      PPV = PPV,
      NPV = NPV,
      prevalence = prevalence,
      test_positive_rate = test_positive_rate,
      Kappa = weighted_kappa_counts(TP, FN, FP, TN, c_val),
      PABAK = compute_pabak(TP, FN, FP, TN),
      Gwet_AC1 = compute_gwet_ac1(TP, FN, FP, TN)
    )
  })
}

###############################################################################
# 7) ESTIMAND VALIDATION
###############################################################################

validate_estimand <- function(cfg) {
  set_fwkc_seed(cfg$seed + 101)

  checks <- vector("list", 250)

  for (i in seq_len(250)) {
    TP <- sample(1:100, 1)
    FN <- sample(1:60, 1)
    FP <- sample(1:60, 1)
    TN <- sample(1:100, 1)
    c_val <- runif(1, 0.05, 0.95)

    n <- TP + FN + FP + TN
    p <- (TP + FN) / n
    Se <- TP / (TP + FN)
    Sp <- TN / (TN + FP)

    k_counts <- weighted_kappa_counts(TP, FN, FP, TN, c_val)
    k_loss <- weighted_kappa_loss(TP, FN, FP, TN, c_val)
    k_matrix <- weighted_kappa_weight_matrix(TP, FN, FP, TN, c_val)
    k_pop <- weighted_kappa_population(p, Se, Sp, c_val)

    k_half <- weighted_kappa_counts(TP, FN, FP, TN, 0.5)
    k_cohen <- cohen_kappa_counts(TP, FN, FP, TN)

    checks[[i]] <- data.frame(
      TP = TP, FN = FN, FP = FP, TN = TN,
      c_val = c_val,
      k_counts = k_counts,
      k_loss = k_loss,
      k_weight_matrix = k_matrix,
      k_population = k_pop,
      k_c_half = k_half,
      k_cohen = k_cohen,
      diff_counts_loss = abs(k_counts - k_loss),
      diff_counts_weight_matrix = abs(k_counts - k_matrix),
      diff_counts_population = abs(k_counts - k_pop),
      diff_half_cohen = abs(k_half - k_cohen)
    )
  }

  checks <- do.call(rbind, checks)

  max_diff <- max(
    checks$diff_counts_loss,
    checks$diff_counts_weight_matrix,
    checks$diff_counts_population,
    checks$diff_half_cohen,
    na.rm = TRUE
  )

  if (!is.finite(max_diff) || max_diff > 1e-10) {
    stop("Estimand equivalence check failed. max difference = ", max_diff)
  }

  save_csv_safe(
    checks,
    file.path(cfg$output_dir, "validation", "estimand_equivalence.csv")
  )

  # Transparent check against the printed Weiner/Nofuentes table.
  # This is intentionally NOT a pass/fail unit test because the printed table
  # values are not exactly reproduced by the displayed formula + printed counts.
  published <- data.frame(
    c_val = seq(0.1, 0.9, by = 0.1),
    published_kappa = c(
      0.527, 0.462, 0.412, 0.371, 0.338,
      0.310, 0.287, 0.267, 0.249
    )
  )

  published$computed_from_printed_counts <- vapply(
    published$c_val,
    function(cc) weighted_kappa_counts(473, 81, 22, 44, cc),
    numeric(1)
  )

  published$absolute_difference <- abs(
    published$computed_from_printed_counts -
      published$published_kappa
  )

  published$note <- paste(
    "Reference discrepancy only; not used as a unit-test target.",
    "The paper's displayed formula and printed 2x2 table are the",
    "primary algebraic basis of the estimand validation."
  )

  save_csv_safe(
    published,
    file.path(
      cfg$output_dir,
      "validation",
      "weiner_nofuentes_reference_discrepancy.csv"
    )
  )

  weights_doc <- weight_matrix_documentation()
  save_csv_safe(
    weights_doc,
    file.path(
      cfg$output_dir,
      "validation",
      "weight_matrix_derivation.csv"
    )
  )

  shrinkage_check <- data.frame(
    n = c(30, 50, 100),
    n0 = cfg$n0,
    lambda_variance_no_root = pmin(1, c(30, 50, 100) / cfg$n0),
    lambda_se_square_root = sqrt(pmin(1, c(30, 50, 100) / cfg$n0)),
    stringsAsFactors = FALSE
  )
  shrinkage_check$identity_difference <- abs(
    sqrt(shrinkage_check$lambda_variance_no_root) -
      shrinkage_check$lambda_se_square_root
  )
  shrinkage_check$se_expansion_multiplier <-
    1 / shrinkage_check$lambda_se_square_root
  shrinkage_check$variance_expansion_multiplier <-
    1 / shrinkage_check$lambda_variance_no_root
  save_csv_safe(
    shrinkage_check,
    file.path(cfg$output_dir, "validation", "shrinkage_dual_scale_check.csv")
  )

  log_message(
    "Estimand validation passed; max algebraic numerical difference = ",
    format(max_diff, scientific = TRUE)
  )
  log_message(
    "Shrinkage variance/SE identity max difference = ",
    format(max(shrinkage_check$identity_difference), scientific = TRUE)
  )

  list(
    equivalence = checks,
    reference_discrepancy = published,
    weight_matrix = weights_doc,
    shrinkage_check = shrinkage_check,
    max_algebraic_difference = max_diff
  )
}

###############################################################################
# 8) KAPPA TRANSFORMATIONS
###############################################################################

transform_kappa <- function(k, method) {
  e <- 1e-8
  k <- clamp(k, -1 + e, 1 - e)

  if (method == "identity") return(k)
  if (method == "fisher_z") return(atanh(k))
  if (method == "logit11") return(log((1 + k) / (1 - k)))

  stop("Unknown transformation: ", method)
}

inverse_transform_kappa <- function(z, method) {
  if (method == "identity") return(clamp(z, -1, 1))
  if (method == "fisher_z") return(tanh(z))
  if (method == "logit11") return(tanh(z / 2))

  stop("Unknown transformation: ", method)
}

###############################################################################
# 9) PROPER BETA POSTERIOR FOR Se / Sp
###############################################################################

prior_ab <- function(prior_type) {
  if (prior_type == "jeffreys") {
    return(c(a = 0.5, b = 0.5))
  }

  if (prior_type == "uniform") {
    return(c(a = 1, b = 1))
  }

  if (prior_type == "legacy") {
    return(c(a = NA_real_, b = NA_real_))
  }

  stop("Unknown prior_type: ", prior_type)
}

draw_sesp <- function(metrics, n_draw, prior_type, eps = 1e-8) {
  if (prior_type == "legacy") {
    # Submitted parameterization retained ONLY for sensitivity analysis.
    if (!is.finite(metrics$Se) || !is.finite(metrics$Sp)) {
      return(list(
        Se = rep(NA_real_, n_draw),
        Sp = rep(NA_real_, n_draw)
      ))
    }

    Se_a <- max(metrics$Se * metrics$n, eps)
    Se_b <- max((1 - metrics$Se) * metrics$n, eps)
    Sp_a <- max(metrics$Sp * metrics$n, eps)
    Sp_b <- max((1 - metrics$Sp) * metrics$n, eps)

  } else {
    pr <- prior_ab(prior_type)

    # Proper conditional posteriors:
    # Se | data ~ Beta(TP+a, FN+b)
    # Sp | data ~ Beta(TN+a, FP+b)
    Se_a <- metrics$TP + pr["a"]
    Se_b <- metrics$FN + pr["b"]
    Sp_a <- metrics$TN + pr["a"]
    Sp_b <- metrics$FP + pr["b"]
  }

  list(
    Se = clamp(rbeta(n_draw, Se_a, Se_b), 1e-6, 1 - 1e-6),
    Sp = clamp(rbeta(n_draw, Sp_a, Sp_b), 1e-6, 1 - 1e-6)
  )
}

draw_sesp_empirical_bootstrap <- function(metrics, n_draw) {
  n_diseased <- metrics$TP + metrics$FN
  n_nondiseased <- metrics$TN + metrics$FP

  if (n_diseased <= 0 || n_nondiseased <= 0 ||
      !is.finite(metrics$Se) || !is.finite(metrics$Sp)) {
    return(list(
      Se = rep(NA_real_, n_draw),
      Sp = rep(NA_real_, n_draw)
    ))
  }

  # For Bernoulli outcomes, resampling within each disease stratum is
  # distributionally equivalent to Binomial(n_stratum, empirical proportion).
  Se_draw <- rbinom(
    n_draw,
    size = n_diseased,
    prob = metrics$Se
  ) / n_diseased

  Sp_draw <- rbinom(
    n_draw,
    size = n_nondiseased,
    prob = metrics$Sp
  ) / n_nondiseased

  list(
    Se = clamp(Se_draw, 1e-6, 1 - 1e-6),
    Sp = clamp(Sp_draw, 1e-6, 1 - 1e-6)
  )
}

draw_uncertainty_parameters <- function(metrics, n_draw, cfg) {
  if (cfg$uncertainty_generator == "beta_posterior") {
    return(
      draw_sesp(
        metrics = metrics,
        n_draw = n_draw,
        prior_type = cfg$prior_type,
        eps = cfg$eps
      )
    )
  }

  if (cfg$uncertainty_generator == "empirical_bootstrap") {
    return(
      draw_sesp_empirical_bootstrap(
        metrics = metrics,
        n_draw = n_draw
      )
    )
  }

  stop(
    "Unknown uncertainty_generator: ",
    cfg$uncertainty_generator
  )
}

###############################################################################
# 10) PSEUDO-RATER GENERATION
###############################################################################

metrics_from_probability_draws <- function(
  p_emp, Se_draw, Sp_draw, c_val
) {
  TPp <- p_emp * Se_draw
  FNp <- p_emp * (1 - Se_draw)
  FPp <- (1 - p_emp) * (1 - Sp_draw)
  TNp <- (1 - p_emp) * Sp_draw

  den <-
    c_val * (TPp + FNp) * (FNp + TNp) +
    (1 - c_val) * (FPp + TNp) * (TPp + FPp)

  Kappa <- ifelse(
    abs(den) < 1e-14,
    NA_real_,
    (TPp * TNp - FNp * FPp) / den
  )
  Kappa <- clamp(Kappa, -1, 1)

  PPV <- ifelse((TPp + FPp) > 0, TPp / (TPp + FPp), NA_real_)
  NPV <- ifelse((TNp + FNp) > 0, TNp / (TNp + FNp), NA_real_)

  pos_rate <- TPp + FPp
  po <- TPp + TNp
  PABAK <- 2 * po - 1

  pi_pos <- (p_emp + pos_rate) / 2
  pe_ac1 <- 2 * pi_pos * (1 - pi_pos)
  Gwet <- ifelse(
    abs(1 - pe_ac1) < 1e-14,
    NA_real_,
    (po - pe_ac1) / (1 - pe_ac1)
  )

  list(
    Kappa = Kappa,
    Se = Se_draw,
    Sp = Sp_draw,
    PPV = PPV,
    NPV = NPV,
    PositiveRate = pos_rate,
    PABAK = PABAK,
    Gwet_AC1 = Gwet
  )
}

metrics_from_posterior_predictive <- function(
  D01, Se_draw, Sp_draw, c_val
) {
  n <- length(D01)
  id1 <- which(D01 == 1)
  id0 <- which(D01 == 0)
  n1 <- length(id1)
  n0 <- length(id0)
  m <- length(Se_draw)

  # EXACT computational acceleration:
  # Conditional on a given Se_j, the number of true positives among n1
  # independent diseased subjects is Binomial(n1, Se_j). Likewise, conditional
  # on Sp_j, false positives are Binomial(n0, 1-Sp_j). Drawing the sufficient
  # counts directly is distributionally identical to allocating an n-by-m
  # Bernoulli matrix and taking column sums, but avoids O(n*m) memory.
  if (n1 > 0) {
    TP <- rbinom(m, size = n1, prob = Se_draw)
  } else {
    TP <- rep(0, m)
  }

  if (n0 > 0) {
    FP <- rbinom(m, size = n0, prob = 1 - Sp_draw)
  } else {
    FP <- rep(0, m)
  }

  FN <- n1 - TP
  TN <- n0 - FP

  den <-
    c_val * (TP + FN) * (FN + TN) +
    (1 - c_val) * (FP + TN) * (TP + FP)

  Kappa <- ifelse(
    abs(den) < 1e-14,
    NA_real_,
    (TP * TN - FN * FP) / den
  )
  Kappa <- clamp(Kappa, -1, 1)

  PPV <- ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_)
  NPV <- ifelse((TN + FN) > 0, TN / (TN + FN), NA_real_)
  pos_rate <- (TP + FP) / n
  po <- (TP + TN) / n
  PABAK <- 2 * po - 1

  p_gold <- (TP + FN) / n
  pi_pos <- (p_gold + pos_rate) / 2
  pe_ac1 <- 2 * pi_pos * (1 - pi_pos)
  Gwet <- ifelse(
    abs(1 - pe_ac1) < 1e-14,
    NA_real_,
    (po - pe_ac1) / (1 - pe_ac1)
  )

  list(
    Kappa = Kappa,
    Se = Se_draw,
    Sp = Sp_draw,
    PPV = PPV,
    NPV = NPV,
    PositiveRate = pos_rate,
    PABAK = PABAK,
    Gwet_AC1 = Gwet
  )
}

generate_pseudo_raters <- function(df_in, c_val, cfg) {
  metrics <- compute_metrics(df_in$D, df_in$T, c_val)
  D01 <- as.integer(as.character(df_in$D))

  if (!is.finite(metrics$prevalence)) {
    return(NULL)
  }

  # IMPORTANT:
  # One latent Se/Sp draw is shared by all pseudo-raters at MC index j.
  # Pseudo-raters then receive conditionally independent outcome realizations.
  # This creates a matched structure across raters instead of arbitrarily
  # aligning unrelated Monte Carlo draws.
  common_draws <- draw_uncertainty_parameters(
    metrics = metrics,
    n_draw = cfg$n_mc,
    cfg = cfg
  )

  if (all(!is.finite(common_draws$Se)) ||
      all(!is.finite(common_draws$Sp))) {
    return(NULL)
  }

  raters <- vector("list", cfg$n_raters)

  for (r in seq_len(cfg$n_raters)) {
    if (cfg$pseudo_mode == "posterior_predictive") {
      rr <- metrics_from_posterior_predictive(
        D01 = D01,
        Se_draw = common_draws$Se,
        Sp_draw = common_draws$Sp,
        c_val = c_val
      )

    } else if (cfg$pseudo_mode == "parameter_only") {
      rr <- metrics_from_probability_draws(
        p_emp = metrics$prevalence,
        Se_draw = common_draws$Se,
        Sp_draw = common_draws$Sp,
        c_val = c_val
      )

    } else {
      stop("Unknown pseudo_mode: ", cfg$pseudo_mode)
    }

    rr$MC_index <- seq_len(cfg$n_mc)
    raters[[r]] <- rr
  }

  list(
    metrics = metrics,
    raters = raters,
    common_draws = common_draws
  )
}

###############################################################################
# 11) HIERARCHICAL BAYESIAN POOLING: CONJUGATE GIBBS
###############################################################################

rinvgamma1 <- function(shape, rate) {
  1 / rgamma(1, shape = shape, rate = rate)
}

# V25 compatibility-safe posterior diagnostics for ONE scalar parameter.
# posterior::rhat/ess_bulk/ess_tail accept an iterations x chains matrix
# directly. This avoids version-dependent summarise_draws() column naming.
safe_scalar_numeric <- function(x) {
  y <- suppressWarnings(tryCatch(as.numeric(x), error = function(e) numeric(0)))
  if (length(y) < 1L || !is.finite(y[1])) return(NA_real_)
  as.numeric(y[1])
}

posterior_chain_diagnostics <- function(draw_matrix) {
  out <- list(rhat = NA_real_, ess_bulk = NA_real_, ess_tail = NA_real_, ok = FALSE)
  x <- tryCatch(as.matrix(draw_matrix), error = function(e) NULL)
  if (is.null(x) || nrow(x) < 4L || ncol(x) < 2L || any(!is.finite(x))) return(out)
  if (!requireNamespace("posterior", quietly = TRUE)) return(out)

  out$rhat <- safe_scalar_numeric(
    suppressWarnings(tryCatch(posterior::rhat(x), error = function(e) NA_real_))
  )
  out$ess_bulk <- safe_scalar_numeric(
    suppressWarnings(tryCatch(posterior::ess_bulk(x), error = function(e) NA_real_))
  )
  out$ess_tail <- safe_scalar_numeric(
    suppressWarnings(tryCatch(posterior::ess_tail(x), error = function(e) NA_real_))
  )
  out$ok <- all(is.finite(c(out$rhat, out$ess_bulk, out$ess_tail)))
  out
}

run_posterior_diagnostics_selftest <- function(cfg) {
  validation_dir <- file.path(cfg$output_dir, "validation")
  if (!requireNamespace("posterior", quietly = TRUE)) {
    out <- data.frame(
      check = "posterior_matrix_diagnostics",
      rhat = NA_real_, ess_bulk = NA_real_, ess_tail = NA_real_,
      pass = FALSE, reason = "posterior_package_unavailable",
      stringsAsFactors = FALSE
    )
    save_csv_safe(out, file.path(validation_dir, "posterior_diagnostics_selftest.csv"))
    if (identical(cfg$run_mode, "publication")) {
      stop("V25 preflight failed: posterior package is unavailable for Bayesian diagnostics.")
    }
    return(out)
  }

  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  base::set.seed(20262001L)
  test_draws <- matrix(stats::rnorm(400L * 4L), nrow = 400L, ncol = 4L)
  dg <- posterior_chain_diagnostics(test_draws)
  pass <- isTRUE(dg$ok) && dg$rhat < 1.10 && dg$ess_bulk > 0 && dg$ess_tail > 0
  out <- data.frame(
    check = "posterior_matrix_diagnostics",
    rhat = dg$rhat, ess_bulk = dg$ess_bulk, ess_tail = dg$ess_tail,
    pass = pass,
    reason = if (pass) "pass" else "matrix_diagnostic_API_failed_or_nonfinite",
    stringsAsFactors = FALSE
  )
  save_csv_safe(out, file.path(validation_dir, "posterior_diagnostics_selftest.csv"))
  log_message(
    "POSTERIOR DIAGNOSTICS SELF-TEST | pass=", pass,
    " | rhat=", signif(dg$rhat, 6),
    " | ESS_bulk/tail=", signif(dg$ess_bulk, 6), "/", signif(dg$ess_tail, 6)
  )
  if (identical(cfg$run_mode, "publication") && !isTRUE(pass)) {
    stop(
      "V25 preflight failed before brms: posterior matrix diagnostics are not functioning. ",
      "Inspect validation/posterior_diagnostics_selftest.csv."
    )
  }
  out
}

hierarchical_gibbs_pool <- function(raters, cfg) {
  K <- do.call(cbind, lapply(raters, function(rr) rr$Kappa))

  empirical_fallback <- function(engine_label) {
    all_k <- as.numeric(K)
    all_k <- all_k[is.finite(all_k)]
    list(
      mean = mean(all_k, na.rm = TRUE),
      se = sd(all_k, na.rm = TRUE),
      draws = all_k,
      engine = engine_label,
      sampler = cfg$gibbs_sampler,
      rhat_mu = NA_real_,
      rhat_mu_classic = NA_real_,
      rhat_mu_rank = NA_real_,
      ess_bulk_mu = NA_real_,
      ess_tail_mu = NA_real_,
      convergence_ok = FALSE
    )
  }

  if (ncol(K) < 2 || nrow(K) < 5) {
    return(empirical_fallback("empirical_fallback"))
  }

  # Pseudo-raters share the same latent Se/Sp draw at MC index j, so only
  # complete matched MC rows are retained in the crossed model.
  keep <- complete.cases(K)
  K <- K[keep, , drop = FALSE]
  if (nrow(K) < 5) {
    return(empirical_fallback("empirical_fallback"))
  }

  Z <- apply(K, 2, transform_kappa, method = cfg$transformation)
  Z <- as.matrix(Z)
  M <- nrow(Z)  # MC-index levels
  R <- ncol(Z)  # pseudo-rater levels
  N <- M * R
  prior_mu_var <- cfg$bayes_mu_sd^2

  n_keep <- floor(
    max(cfg$bayes_iter - cfg$bayes_burn, 1) /
      max(cfg$bayes_thin, 1)
  )
  if (n_keep < 20) {
    return(empirical_fallback("gibbs_insufficient_postburn"))
  }

  # -------------------------------------------------------------------------
  # Exact blocked/collapsed Gibbs for the SAME crossed Gaussian model:
  #   Z_mr = mu + u_r + v_m + e_mr
  #   u_r ~ N(0,tau_r^2), v_m ~ N(0,tau_mc^2), e_mr ~ N(0,sigma^2)
  # with the same normal prior on mu and inverse-gamma variance priors used in
  # the submitted engine.  The old centered single-site Gibbs updated mu,u,v
  # conditionally one after another; with only three rater levels that creates
  # strong location-shift autocorrelation and can yield very large R-hat.
  #
  # V25 draws the Gaussian block exactly via the factorization
  #   p(mu,u,v | variances,Z)
  #     = p(mu | variances,Z)
  #       p(u | mu,variances,Z)
  #       p(v | u,mu,variances,Z),
  # where u is sampled after analytically integrating out v.  Balanced crossed
  # structure makes all required covariance operations closed-form.  This is an
  # exact re-expression of the conditional posterior, not an approximation and
  # not a change of prior/model/estimand.
  # -------------------------------------------------------------------------

  n_chains <- max(2L, as.integer(cfg$bayes_chains))
  row_sums <- rowSums(Z)
  col_sums <- colSums(Z)
  grand_sum <- sum(row_sums)
  grand_sq <- sum(Z * Z)
  col_centered <- col_sums - mean(col_sums)

  base_var <- max(var(as.numeric(Z)), 1e-6)
  u0 <- colMeans(Z) - mean(Z)
  v0_effect <- rowMeans(Z) - mean(Z)
  base_tau_r <- max(if (length(u0) > 1L) var(u0) else 0, 1e-6)
  base_tau_mc <- max(if (length(v0_effect) > 1L) var(v0_effect) else 0, 1e-6)

  # Overdispersed variance-component starts; the Gaussian block itself is drawn
  # exactly each iteration, so no arbitrary starting mu/u/v values are needed.
  sigma2 <- base_var * runif(n_chains, 0.5, 1.5)
  tau_r2 <- base_tau_r * runif(n_chains, 0.5, 1.5)
  tau_mc2 <- base_tau_mc * runif(n_chains, 0.5, 1.5)

  draw_mat <- matrix(NA_real_, nrow = n_keep, ncol = n_chains)
  keep_idx <- 0L

  for (it in seq_len(cfg$bayes_iter)) {
    # 1) mu | variance components, Z, with BOTH random-effect blocks integrated.
    # The all-ones vector is an eigenvector of the balanced crossed covariance
    # with eigenvalue sigma2 + M*tau_r2 + R*tau_mc2.
    lambda_mu <- sigma2 + M * tau_r2 + R * tau_mc2
    lambda_mu <- pmax(lambda_mu, 1e-12)
    var_mu <- 1 / (1 / prior_mu_var + N / lambda_mu)
    mean_mu <- var_mu * (
      cfg$bayes_mu0 / prior_mu_var + grand_sum / lambda_mu
    )
    mu <- rnorm(n_chains, mean = mean_mu, sd = sqrt(var_mu))

    # 2) u | mu, variances, Z, after integrating out the MC-index effects v.
    # Compound-symmetry yields one parallel and R-1 orthogonal eigenvalues.
    sigma2_safe <- pmax(sigma2, 1e-12)
    tau_r2_safe <- pmax(tau_r2, 1e-12)
    tau_mc2_safe <- pmax(tau_mc2, 1e-12)

    p_orth <- 1 / tau_r2_safe + M / sigma2_safe
    p_par <- 1 / tau_r2_safe + M / (sigma2_safe + R * tau_mc2_safe)

    mean_s <- mean(col_sums) - M * mu
    centered_scale <- 1 / (sigma2_safe * p_orth)
    parallel_loc <- (
      mean_s / (sigma2_safe + R * tau_mc2_safe)
    ) / p_par

    u_mean <- outer(col_centered, centered_scale, "*") +
      matrix(rep(parallel_loc, each = R), nrow = R, ncol = n_chains)

    z_u <- matrix(rnorm(R * n_chains), nrow = R, ncol = n_chains)
    z_u_centered <- sweep(z_u, 2, colMeans(z_u), "-")
    u_noise_orth <- sweep(z_u_centered, 2, sqrt(p_orth), "/")
    u_noise_parallel <- matrix(
      rep(rnorm(n_chains, 0, sqrt(1 / (R * p_par))), each = R),
      nrow = R,
      ncol = n_chains
    )
    u <- u_mean + u_noise_orth + u_noise_parallel
    sum_u <- colSums(u)

    # 3) v | u, mu, variances, Z.  Conditional independence across MC indices
    # permits a vectorized draw for all M levels and all chains.
    var_v <- 1 / (R / sigma2_safe + 1 / tau_mc2_safe)
    v_num <- outer(row_sums, rep(1, n_chains)) -
      matrix(rep(R * mu + sum_u, each = M), nrow = M, ncol = n_chains)
    v_mean <- sweep(v_num, 2, var_v / sigma2_safe, "*")
    v <- v_mean + sweep(
      matrix(rnorm(M * n_chains), nrow = M, ncol = n_chains),
      2,
      sqrt(var_v),
      "*"
    )
    sum_v <- colSums(v)

    # Residual SSE from sufficient statistics; avoids constructing M x R
    # residual matrices separately for each chain/iteration.
    sum_u2 <- colSums(u * u)
    sum_v2 <- colSums(v * v)
    cross_u <- as.numeric(crossprod(col_sums, u))
    cross_v <- as.numeric(crossprod(row_sums, v))

    sse <- grand_sq -
      2 * mu * grand_sum -
      2 * cross_u -
      2 * cross_v +
      N * mu^2 +
      M * sum_u2 +
      R * sum_v2 +
      2 * mu * M * sum_u +
      2 * mu * R * sum_v +
      2 * sum_u * sum_v
    sse <- pmax(sse, 1e-12)

    sigma2 <- 1 / rgamma(
      n_chains,
      shape = cfg$bayes_a_sigma + N / 2,
      rate = cfg$bayes_b_sigma + sse / 2
    )
    tau_r2 <- 1 / rgamma(
      n_chains,
      shape = cfg$bayes_a_tau + R / 2,
      rate = cfg$bayes_b_tau + sum_u2 / 2
    )
    tau_mc2 <- 1 / rgamma(
      n_chains,
      shape = cfg$bayes_a_tau + M / 2,
      rate = cfg$bayes_b_tau + sum_v2 / 2
    )

    if (
      it > cfg$bayes_burn &&
      ((it - cfg$bayes_burn) %% cfg$bayes_thin == 0)
    ) {
      keep_idx <- keep_idx + 1L
      if (keep_idx <= n_keep) draw_mat[keep_idx, ] <- mu
    }
  }

  draw_mat <- draw_mat[seq_len(min(keep_idx, n_keep)), , drop = FALSE]
  if (nrow(draw_mat) < 20L || any(!is.finite(draw_mat))) {
    return(empirical_fallback("gibbs_blocked_failed_fallback"))
  }

  # Classical Gelman-Rubin R-hat retained for continuity with the submitted code.
  n_common <- nrow(draw_mat)
  chain_means <- colMeans(draw_mat)
  chain_vars <- apply(draw_mat, 2, var)
  W <- mean(chain_vars)
  B <- n_common * var(chain_means)
  var_hat <- ((n_common - 1) / n_common) * W + B / n_common
  rhat_classic <- if (is.finite(W) && W > 0) sqrt(var_hat / W) else NA_real_

  # Modern rank-normalized R-hat/ESS. V25 calls posterior diagnostics directly
  # on the iterations-by-chains matrix; it never indexes summarise_draws() tibble
  # columns, which vary across posterior package versions.
  rhat_rank <- NA_real_
  ess_bulk_mu <- NA_real_
  ess_tail_mu <- NA_real_
  if (isTRUE(cfg$gibbs_rank_rhat_if_available) &&
      requireNamespace("posterior", quietly = TRUE)) {
    post_diag <- posterior_chain_diagnostics(draw_mat)
    rhat_rank <- safe_scalar_numeric(post_diag$rhat)
    ess_bulk_mu <- safe_scalar_numeric(post_diag$ess_bulk)
    ess_tail_mu <- safe_scalar_numeric(post_diag$ess_tail)
  }

  rhat_for_gate <- safe_scalar_numeric(rhat_classic)
  if (length(rhat_rank) == 1L && is.finite(rhat_rank)) {
    finite_rhats <- c(rhat_for_gate, rhat_rank)
    finite_rhats <- finite_rhats[is.finite(finite_rhats)]
    rhat_for_gate <- if (length(finite_rhats) > 0L) max(finite_rhats) else NA_real_
  }
  convergence_ok <- is.finite(rhat_for_gate) && rhat_for_gate <= cfg$bayes_rhat_max

  mu_draws <- as.numeric(draw_mat)
  k_draws <- inverse_transform_kappa(mu_draws, cfg$transformation)

  list(
    mean = mean(k_draws, na.rm = TRUE),
    se = sd(k_draws, na.rm = TRUE),
    draws = k_draws,
    engine = "gibbs_crossed_blocked_collapsed_vectorized",
    sampler = cfg$gibbs_sampler,
    rhat_mu = rhat_for_gate,
    rhat_mu_classic = rhat_classic,
    rhat_mu_rank = rhat_rank,
    ess_bulk_mu = ess_bulk_mu,
    ess_tail_mu = ess_tail_mu,
    convergence_ok = convergence_ok
  )
}

###############################################################################
# 11B) OPTIONAL BRMS ENGINE: REVISED CROSSED HIERARCHICAL MODEL
###############################################################################

.brms_templates <- new.env(parent = emptyenv())

extract_cmdstan_sampling_diagnostics <- function(brms_fit, cfg) {
  out <- list(
    divergent_transitions = NA_integer_,
    total_postwarmup_transitions = NA_integer_,
    divergence_rate = NA_real_,
    max_treedepth_hits = NA_integer_,
    min_ebfmi = NA_real_,
    diagnostics_source = "unavailable"
  )

  update_from_draws_array <- function(sd, source_label) {
    if (is.null(sd)) return(FALSE)
    sd <- tryCatch(posterior::as_draws_array(sd), error = function(e) sd)
    if (is.null(sd) || length(dim(sd)) < 3L) return(FALSE)
    vars <- dimnames(sd)[[3]]
    if (is.null(vars)) return(FALSE)

    touched <- FALSE
    if ("divergent__" %in% vars) {
      dv <- sd[, , "divergent__", drop = TRUE]
      out$divergent_transitions <<- as.integer(sum(dv > 0, na.rm = TRUE))
      out$total_postwarmup_transitions <<- as.integer(sum(is.finite(dv)))
      if (out$total_postwarmup_transitions > 0L) {
        out$divergence_rate <<- out$divergent_transitions / out$total_postwarmup_transitions
      }
      touched <- TRUE
    }
    if ("treedepth__" %in% vars) {
      td <- sd[, , "treedepth__", drop = TRUE]
      max_td <- suppressWarnings(as.integer(cfg$brms_max_treedepth))
      if (!is.finite(max_td) || max_td < 1L) max_td <- 13L
      out$max_treedepth_hits <<- as.integer(sum(td >= max_td, na.rm = TRUE))
      touched <- TRUE
    }
    if ("energy__" %in% vars) {
      en <- sd[, , "energy__", drop = FALSE]
      vals <- numeric(0)
      for (ch in seq_len(dim(en)[2])) {
        e <- as.numeric(en[, ch, 1])
        e <- e[is.finite(e)]
        if (length(e) >= 3L && is.finite(var(e)) && var(e) > 0) {
          vals <- c(vals, mean(diff(e)^2) / var(e))
        }
      }
      if (length(vals) > 0L) out$min_ebfmi <<- min(vals, na.rm = TRUE)
      touched <- TRUE
    }
    if (touched) out$diagnostics_source <<- source_label
    touched
  }

  # Preferred path for cmdstanr-backed brms fits.
  csfit <- tryCatch(brms_fit$fit, error = function(e) NULL)
  if (!is.null(csfit)) {
    sd1 <- tryCatch(
      csfit$sampler_diagnostics(format = "draws_array"),
      error = function(e) NULL
    )
    ok <- update_from_draws_array(sd1, "cmdstanr::sampler_diagnostics(draws_array)")

    if (!isTRUE(ok)) {
      sd2 <- tryCatch(csfit$sampler_diagnostics(), error = function(e) NULL)
      ok <- update_from_draws_array(sd2, "cmdstanr::sampler_diagnostics(default)")
    }
  }

  # Backend-agnostic brms fallback.  This is important on systems where brms
  # exposes sampler diagnostics through nuts_params even when the underlying
  # CmdStan object method differs across cmdstanr/brms versions.
  if (!is.finite(out$divergent_transitions) ||
      !is.finite(out$total_postwarmup_transitions) ||
      !is.finite(out$max_treedepth_hits)) {
    np <- tryCatch(brms::nuts_params(brms_fit), error = function(e) NULL)
    if (!is.null(np)) {
      np <- tryCatch(as.data.frame(np), error = function(e) NULL)
    }
    if (!is.null(np) && nrow(np) > 0L) {
      nms <- tolower(names(np))
      pcol <- which(nms %in% c("parameter", "param"))[1]
      vcol <- which(nms %in% c("value", "val"))[1]
      ccol <- which(nms %in% c("chain", ".chain"))[1]
      if (is.finite(pcol) && is.finite(vcol)) {
        par <- as.character(np[[pcol]])
        val <- suppressWarnings(as.numeric(np[[vcol]]))

        idiv <- grepl("diverg", par, ignore.case = TRUE)
        if (any(idiv)) {
          dv <- val[idiv]
          out$divergent_transitions <- as.integer(sum(dv > 0, na.rm = TRUE))
          out$total_postwarmup_transitions <- as.integer(sum(is.finite(dv)))
          if (out$total_postwarmup_transitions > 0L) {
            out$divergence_rate <- out$divergent_transitions / out$total_postwarmup_transitions
          }
        }

        itd <- grepl("treedepth", par, ignore.case = TRUE)
        if (any(itd)) {
          td <- val[itd]
          max_td <- suppressWarnings(as.integer(cfg$brms_max_treedepth))
          if (!is.finite(max_td) || max_td < 1L) max_td <- 13L
          out$max_treedepth_hits <- as.integer(sum(td >= max_td, na.rm = TRUE))
        }

        iene <- grepl("energy", par, ignore.case = TRUE)
        if (any(iene)) {
          vals <- numeric(0)
          if (is.finite(ccol)) {
            chains <- np[[ccol]][iene]
            ener <- val[iene]
            for (ch in unique(chains)) {
              e <- ener[chains == ch]
              e <- e[is.finite(e)]
              if (length(e) >= 3L && is.finite(var(e)) && var(e) > 0) {
                vals <- c(vals, mean(diff(e)^2) / var(e))
              }
            }
          } else {
            e <- val[iene]
            e <- e[is.finite(e)]
            if (length(e) >= 3L && is.finite(var(e)) && var(e) > 0) {
              vals <- mean(diff(e)^2) / var(e)
            }
          }
          if (length(vals) > 0L) out$min_ebfmi <- min(vals, na.rm = TRUE)
        }
        out$diagnostics_source <- if (out$diagnostics_source == "unavailable") {
          "brms::nuts_params"
        } else {
          paste(out$diagnostics_source, "+ brms::nuts_params")
        }
      }
    }
  }

  out
}

brms_hierarchical_pool <- function(raters, cfg) {
  needed <- c("brms", "cmdstanr", "posterior")
  missing <- needed[
    !vapply(needed, requireNamespace, logical(1), quietly = TRUE)
  ]

  if (length(missing) > 0) {
    stop(
      "bayes_engine='brms' requires: ",
      paste(missing, collapse = ", ")
    )
  }

  brms_failure <- function(label) {
    if (isTRUE(cfg$allow_bayes_fallback)) {
      return(hierarchical_gibbs_pool(raters, cfg))
    }
    list(
      mean = NA_real_,
      se = NA_real_,
      draws = numeric(0),
      engine = label,
      rhat_mu = NA_real_,
      rhat_mu_classic = NA_real_,
      rhat_mu_rank = NA_real_,
      convergence_ok = FALSE,
      divergent_transitions = NA_integer_,
      total_postwarmup_transitions = NA_integer_,
      divergence_rate = NA_real_,
      max_treedepth_hits = NA_integer_,
      min_ebfmi = NA_real_,
      ess_bulk_intercept = NA_real_,
      ess_tail_intercept = NA_real_,
      diagnostics_source = "unavailable"
    )
  }

  cmdstan_ver <- tryCatch(
    cmdstanr::cmdstan_version(error_on_NA = FALSE),
    error = function(e) NA_character_
  )
  if (length(cmdstan_ver) == 0 || any(is.na(cmdstan_ver))) {
    stop(
      "CmdStan is not installed/configured. ",
      "Run cmdstanr::install_cmdstan() before publication analysis."
    )
  }

  z_list <- lapply(
    raters,
    function(rr) transform_kappa(rr$Kappa, cfg$transformation)
  )

  dat <- data.frame(
    z = unlist(z_list),
    Rater = factor(rep(
      seq_along(z_list),
      times = vapply(z_list, length, integer(1))
    )),
    MC_index = factor(
      unlist(
        lapply(
          raters,
          function(rr) rr$MC_index
        )
      )
    )
  )

  dat <- dat[
    is.finite(dat$z),
    ,
    drop = FALSE
  ]

  if (nrow(dat) < 30 || length(unique(dat$Rater)) < 2) {
    return(brms_failure("brms_insufficient_data"))
  }

  # Fixed weakly informative priors are used across sample sizes so that the
  # n0 threshold is not silently used twice (Bayesian prior + CI calibration).
  # MC_index is a crossed random effect because raters share the same latent
  # Se/Sp draw at each Monte Carlo index.
  # Cross-engine sensitivity check:
  # match the Gibbs prior for the population mean exactly. brms scale priors remain
  # weakly regularizing because brms parameterizes SDs directly whereas the conjugate
  # Gibbs engine places inverse-gamma priors on variances. Therefore the cross-check
  # is deliberately a robustness/agreement check, not a claim of identical posteriors.
  priors <- c(
    brms::set_prior(
      sprintf("normal(%s,%s)", format(cfg$bayes_mu0, scientific = FALSE),
              format(cfg$bayes_mu_sd, scientific = FALSE)),
      class = "Intercept"
    ),
    brms::set_prior("exponential(1)", class = "sd"),
    brms::set_prior("exponential(1)", class = "sigma")
  )

  # Compiled Stan code depends on the model/priors/transformation, not on
  # adapt_delta, max_treedepth, iter or warmup. Keeping those out of the cache key
  # lets a V25 rescue attempt reuse the already compiled model safely.
  key <- paste(
    "crossed_gaussian_v20",
    cfg$transformation,
    "matched_mu_prior_exp1_sd_sigma",
    sep = "_"
  )

  brms_seed <- if (!is.null(cfg$brms_seed_override) &&
                   length(cfg$brms_seed_override) == 1L &&
                   is.finite(cfg$brms_seed_override)) {
    normalize_fwkc_seed(cfg$brms_seed_override)
  } else {
    normalize_fwkc_seed(cfg$seed + 771337)
  }

  fit <- NULL

  if (!exists(key, envir = .brms_templates, inherits = FALSE)) {
    fit <- tryCatch(
      brms::brm(
        z ~ 1 + (1 | Rater) + (1 | MC_index),
        data = dat,
        prior = priors,
        chains = cfg$bayes_chains,
        cores = resolve_brms_cores(cfg),
        iter = cfg$bayes_iter,
        warmup = cfg$bayes_burn,
        backend = "cmdstanr",
        seed = brms_seed,
        control = list(
          adapt_delta = cfg$brms_adapt_delta,
          max_treedepth = as.integer(cfg$brms_max_treedepth)
        ),
        refresh = 0,
        save_pars = brms::save_pars(all = TRUE),
        silent = 2
      ),
      error = function(e) NULL
    )

    if (!is.null(fit)) {
      assign(key, fit, envir = .brms_templates)
    }

  } else {
    template <- get(key, envir = .brms_templates)

    fit <- tryCatch(
      update(
        template,
        newdata = dat,
        recompile = FALSE,
        chains = cfg$bayes_chains,
        cores = resolve_brms_cores(cfg),
        iter = cfg$bayes_iter,
        warmup = cfg$bayes_burn,
        backend = "cmdstanr",
        seed = brms_seed,
        control = list(
          adapt_delta = cfg$brms_adapt_delta,
          max_treedepth = as.integer(cfg$brms_max_treedepth)
        ),
        refresh = 0,
        silent = 2
      ),
      error = function(e) NULL
    )
  }

  if (is.null(fit)) {
    return(brms_failure("brms_fit_failed"))
  }

  dr <- posterior::as_draws_df(fit)

  if (!"b_Intercept" %in% names(dr)) {
    return(brms_failure("brms_missing_intercept"))
  }

  k_draws <- inverse_transform_kappa(
    dr$b_Intercept,
    cfg$transformation
  )

  # V25 computes all chain diagnostics from a plain iterations x chains matrix.
  # This is stable across posterior/tibble versions and supplies modern rank-Rhat
  # and ESS directly without relying on summary-column names.
  rhat_mu_classic <- NA_real_
  rhat_mu_rank <- NA_real_
  rhat_mu <- NA_real_
  ess_bulk_intercept <- NA_real_
  ess_tail_intercept <- NA_real_
  if (".chain" %in% names(dr)) {
    ch <- split(dr$b_Intercept, dr$.chain)
    if (length(ch) >= 2) {
      n_common <- min(vapply(ch, length, integer(1)))
      if (n_common >= 20) {
        ch_mat <- do.call(cbind, lapply(ch, function(x) tail(x, n_common)))
        W <- mean(apply(ch_mat, 2, var))
        B <- n_common * var(colMeans(ch_mat))
        var_hat <- ((n_common - 1) / n_common) * W + B / n_common
        if (is.finite(W) && W > 0) rhat_mu_classic <- sqrt(var_hat / W)

        pd <- posterior_chain_diagnostics(ch_mat)
        rhat_mu_rank <- safe_scalar_numeric(pd$rhat)
        ess_bulk_intercept <- safe_scalar_numeric(pd$ess_bulk)
        ess_tail_intercept <- safe_scalar_numeric(pd$ess_tail)

        rvals <- c(rhat_mu_classic, rhat_mu_rank)
        rvals <- rvals[is.finite(rvals)]
        if (length(rvals) > 0L) rhat_mu <- max(rvals)
      }
    }
  }

  sampler_diag <- extract_cmdstan_sampling_diagnostics(fit, cfg)

  list(
    mean = mean(k_draws, na.rm = TRUE),
    se = sd(k_draws, na.rm = TRUE),
    draws = k_draws,
    engine = "brms",
    rhat_mu = rhat_mu,
    rhat_mu_classic = rhat_mu_classic,
    rhat_mu_rank = rhat_mu_rank,
    convergence_ok = is.finite(rhat_mu) && rhat_mu <= cfg$bayes_rhat_max,
    divergent_transitions = sampler_diag$divergent_transitions,
    total_postwarmup_transitions = sampler_diag$total_postwarmup_transitions,
    divergence_rate = sampler_diag$divergence_rate,
    max_treedepth_hits = sampler_diag$max_treedepth_hits,
    min_ebfmi = sampler_diag$min_ebfmi,
    ess_bulk_intercept = ess_bulk_intercept,
    ess_tail_intercept = ess_tail_intercept,
    diagnostics_source = sampler_diag$diagnostics_source
  )
}

bayesian_pool <- function(raters, cfg) {
  if (cfg$bayes_engine == "gibbs") {
    return(hierarchical_gibbs_pool(raters, cfg))
  }

  if (cfg$bayes_engine == "brms") {
    return(brms_hierarchical_pool(raters, cfg))
  }

  stop("Unknown bayes_engine: ", cfg$bayes_engine)
}

###############################################################################
# 12) RANDOM FOREST / SPLINE AGGREGATION
###############################################################################

build_aligned_ml_data <- function(raters, common_draws, cfg) {
  K <- do.call(cbind, lapply(raters, `[[`, "Kappa"))
  colnames(K) <- paste0("K_Rater", seq_len(ncol(K)))

  out <- data.frame(
    MC_index = seq_len(nrow(K)),
    Se = common_draws$Se,
    Sp = common_draws$Sp,
    K,
    check.names = FALSE
  )

  out
}

fit_rf_layer <- function(ml_data, cfg) {
  rater_cols <- grep("^K_Rater", names(ml_data), value = TRUE)

  if (length(rater_cols) < 2 || nrow(ml_data) < 40) {
    return(NULL)
  }

  all_pred <- list()
  mse_vec <- rep(NA_real_, length(rater_cols))

  for (r in seq_along(rater_cols)) {
    target <- rater_cols[r]
    others <- setdiff(rater_cols, target)

    # The target pseudo-rater is NEVER included in its own predictors.
    use_cols <- c("Se", "Sp", others)
    dat <- ml_data[, c(target, use_cols), drop = FALSE]
    dat <- dat[complete.cases(dat), , drop = FALSE]

    if (nrow(dat) < 30) next

    y <- transform_kappa(
      dat[[target]],
      cfg$transformation
    )

    # In very sparse 2x2 samples the pseudo-rater response can collapse to
    # only a handful of distinct kappa values. randomForest warns that
    # regression is inappropriate in this case. Rather than force an unstable
    # ML fit, skip that target; if all targets are skipped the fusion layer
    # automatically falls back to the Bayesian component.
    y_finite <- y[is.finite(y)]
    if (length(y_finite) < 20) next

    n_unique_y <- length(unique(y_finite))
    if (n_unique_y <= 5) next

    x <- dat[, use_cols, drop = FALSE]

    # Transform other kappa predictors, but not Se/Sp.
    for (nm in others) {
      x[[nm]] <- transform_kappa(
        x[[nm]],
        cfg$transformation
      )
    }

    mtry_use <- max(
      1,
      min(cfg$rf_mtry, ncol(x))
    )

    # maxnodes must respect the amount of training information available.
    # This prevents randomForest's "maxnodes exceeds its max value" warning
    # in small/sparse verification scenarios.
    maxnodes_use <- NULL

    if (
      !is.null(cfg$rf_maxnodes) &&
        length(cfg$rf_maxnodes) == 1 &&
        is.finite(cfg$rf_maxnodes)
    ) {
      maxnodes_by_data <- max(
        2L,
        floor(
          nrow(x) /
            max(1L, as.integer(cfg$rf_nodesize))
        )
      )

      maxnodes_use <- min(
        as.integer(cfg$rf_maxnodes),
        as.integer(maxnodes_by_data)
      )
    }

    fit <- tryCatch(
      randomForest(
        x = x,
        y = y,
        ntree = cfg$rf_ntree,
        mtry = mtry_use,
        nodesize = cfg$rf_nodesize,
        maxnodes = maxnodes_use,
        importance = FALSE,
        keep.forest = TRUE,
        na.action = na.omit
      ),
      error = function(e) NULL
    )

    if (is.null(fit)) next

    # OOB prediction prevents in-sample fitted-value optimism.
    pred_z <- fit$predicted
    pred_z <- pred_z[is.finite(pred_z)]

    if (length(pred_z) < 20) next

    all_pred[[length(all_pred) + 1L]] <-
      inverse_transform_kappa(
        pred_z,
        cfg$transformation
      )

    mse_vec[r] <- tail(fit$mse, 1)
  }

  if (length(all_pred) == 0) return(NULL)

  pred_k <- unlist(all_pred)
  pred_k <- pred_k[is.finite(pred_k)]

  if (length(pred_k) < 20) return(NULL)

  list(
    mean = mean(pred_k),
    se = sd(pred_k),
    draws = pred_k,
    oob_mse = mean(mse_vec, na.rm = TRUE),
    type = "rf_leave_one_rater_out"
  )
}

fit_spline_layer <- function(ml_data, cfg) {
  rater_cols <- grep("^K_Rater", names(ml_data), value = TRUE)

  if (length(rater_cols) < 2 || nrow(ml_data) < 60) {
    return(NULL)
  }

  all_pred <- list()

  for (r in seq_along(rater_cols)) {
    target <- rater_cols[r]
    others <- setdiff(rater_cols, target)

    dat <- ml_data[
      ,
      c(target, "Se", "Sp", others),
      drop = FALSE
    ]
    dat <- dat[complete.cases(dat), , drop = FALSE]

    if (nrow(dat) < 40) next

    work <- data.frame(
      y = transform_kappa(
        dat[[target]],
        cfg$transformation
      ),
      Se = dat$Se,
      Sp = dat$Sp
    )

    for (nm in others) {
      work[[nm]] <- transform_kappa(
        dat[[nm]],
        cfg$transformation
      )
    }

    Kfold <- max(
      2,
      min(
        cfg$spline_folds,
        floor(nrow(work) / 20)
      )
    )

    folds <- sample(
      rep(seq_len(Kfold), length.out = nrow(work))
    )
    pred_z <- rep(NA_real_, nrow(work))

    other_term <- if (length(others) > 0) {
      paste(others, collapse = " + ")
    } else {
      "1"
    }

    form <- as.formula(
      paste0(
        "y ~ splines::ns(Se, df=3) + ",
        "splines::ns(Sp, df=3) + ",
        other_term
      )
    )

    for (k in seq_len(Kfold)) {
      train <- work[folds != k, , drop = FALSE]
      test <- work[folds == k, , drop = FALSE]

      fit <- tryCatch(
        lm(form, data = train),
        error = function(e) NULL
      )

      if (!is.null(fit)) {
        pred_z[folds == k] <- tryCatch(
          predict(fit, newdata = test),
          error = function(e) {
            rep(NA_real_, nrow(test))
          }
        )
      }
    }

    pred_z <- pred_z[is.finite(pred_z)]

    if (length(pred_z) >= 20) {
      all_pred[[length(all_pred) + 1L]] <-
        inverse_transform_kappa(
          pred_z,
          cfg$transformation
        )
    }
  }

  if (length(all_pred) == 0) return(NULL)

  pred_k <- unlist(all_pred)
  pred_k <- pred_k[is.finite(pred_k)]

  if (length(pred_k) < 20) return(NULL)

  list(
    mean = mean(pred_k),
    se = sd(pred_k),
    draws = pred_k,
    oob_mse = NA_real_,
    type = "spline_cross_fitted_leave_one_rater_out"
  )
}

fit_ml_layer <- function(raters, common_draws, cfg) {
  if (cfg$ml_layer == "none") {
    return(NULL)
  }

  dat <- build_aligned_ml_data(
    raters = raters,
    common_draws = common_draws,
    cfg = cfg
  )

  if (cfg$ml_layer == "rf") {
    return(fit_rf_layer(dat, cfg))
  }

  if (cfg$ml_layer == "spline") {
    return(fit_spline_layer(dat, cfg))
  }

  stop("Unknown ml_layer: ", cfg$ml_layer)
}

###############################################################################
# 13) FUSION
###############################################################################

fuse_components <- function(bayes, ml, rule) {
  b <- bayes$mean
  sb <- bayes$se

  if (!is.finite(b)) {
    return(list(
      mean = NA_real_,
      w_bayes = NA_real_,
      w_ml = NA_real_
    ))
  }

  if (is.null(ml) || !is.finite(ml$mean) || rule == "bayes_only") {
    return(list(
      mean = b,
      w_bayes = 1,
      w_ml = 0
    ))
  }

  m <- ml$mean
  sm <- ml$se

  if (!is.finite(sb) || sb <= 0) sb <- 1e6
  if (!is.finite(sm) || sm <= 0) sm <- 1e6

  if (rule == "ml_only") {
    wb <- 0
    wm <- 1

  } else if (rule == "equal") {
    wb <- 0.5
    wm <- 0.5

  } else if (rule == "inverse_se") {
    a <- 1 / sb
    d <- 1 / sm
    wb <- a / (a + d)
    wm <- 1 - wb

  } else if (rule == "inverse_variance") {
    a <- 1 / (sb^2)
    d <- 1 / (sm^2)
    wb <- a / (a + d)
    wm <- 1 - wb

  } else {
    stop("Unknown fusion_rule: ", rule)
  }

  list(
    mean = clamp(wb * b + wm * m, -1, 1),
    w_bayes = wb,
    w_ml = wm
  )
}

###############################################################################
# 14) PSEUDO-RATER COVARIANCE DIAGNOSTICS
###############################################################################

pseudo_rater_covariance_diagnostics <- function(raters, cfg) {
  K <- do.call(cbind, lapply(raters, `[[`, "Kappa"))
  K <- K[complete.cases(K), , drop = FALSE]

  if (nrow(K) < 10) {
    return(list(
      se_equal_empirical = NA_real_,
      se_equal_bootstrap = NA_real_,
      mean_pairwise_cov = NA_real_
    ))
  }

  R <- ncol(K)
  w <- rep(1 / R, R)

  S <- cov(K)
  S <- S + diag(cfg$ridge_diag, R)

  se_emp <- as.numeric(
    sqrt(max(drop(t(w) %*% S %*% w), 0))
  )

  offdiag <- S[row(S) != col(S)]
  mean_cov <- if (length(offdiag) > 0) {
    mean(offdiag, na.rm = TRUE)
  } else {
    NA_real_
  }

  se_boot <- replicate(cfg$covariance_boot_B, {
    id <- sample.int(nrow(K), nrow(K), replace = TRUE)
    Sb <- cov(K[id, , drop = FALSE])
    Sb <- Sb + diag(cfg$ridge_diag, R)
    sqrt(max(drop(t(w) %*% Sb %*% w), 0))
  })

  list(
    se_equal_empirical = se_emp,
    se_equal_bootstrap = median(se_boot, na.rm = TRUE),
    mean_pairwise_cov = mean_cov
  )
}

###############################################################################
# 15) ONE CORE FWKC FIT ON ONE DATASET (NO OUTER BOOTSTRAP)
###############################################################################

fwkc_core <- function(df_in, c_val, cfg) {
  gen <- generate_pseudo_raters(df_in, c_val, cfg)

  if (is.null(gen)) {
    return(NULL)
  }

  metrics <- gen$metrics
  raters <- gen$raters

  bayes <- bayesian_pool(raters, cfg)
  if (isTRUE(cfg$require_bayes_convergence) &&
      !isTRUE(bayes$convergence_ok)) {
    return(NULL)
  }
  ml <- fit_ml_layer(raters, gen$common_draws, cfg)
  fused <- fuse_components(bayes, ml, cfg$fusion_rule)

  cov_diag <- pseudo_rater_covariance_diagnostics(
    raters,
    cfg
  )

  list(
    point = fused$mean,
    bayes_mean = bayes$mean,
    bayes_se = bayes$se,
    bayes_rhat = if (!is.null(bayes$rhat_mu)) bayes$rhat_mu else NA_real_,
    bayes_convergence_ok = if (!is.null(bayes$convergence_ok)) bayes$convergence_ok else NA,
    ml_mean = if (is.null(ml)) NA_real_ else ml$mean,
    ml_se = if (is.null(ml)) NA_real_ else ml$se,
    rf_oob_mse = if (is.null(ml)) NA_real_ else ml$oob_mse,
    w_bayes = fused$w_bayes,
    w_ml = fused$w_ml,
    pseudo_se_empirical = cov_diag$se_equal_empirical,
    pseudo_se_bootstrap = cov_diag$se_equal_bootstrap,
    pseudo_mean_cov = cov_diag$mean_pairwise_cov,
    metrics = metrics
  )
}

###############################################################################
# 16) DIRECT ASYMPTOTIC DELTA CI (GENUINE COMPETING INFERENCE)
###############################################################################

kappa_from_prob3 <- function(q3, c_val) {
  if (length(q3) != 3) return(NA_real_)

  p4 <- 1 - sum(q3)
  if (any(q3 < 0) || p4 < 0) return(NA_real_)

  weighted_kappa_counts(
    TP = q3[1],
    FN = q3[2],
    FP = q3[3],
    TN = p4,
    c_val = c_val
  )
}

numeric_gradient_prob3 <- function(q3, c_val, h = 1e-6) {
  g <- rep(NA_real_, 3)

  for (j in seq_len(3)) {
    step <- min(
      h,
      max(q3[j] / 4, h / 10),
      max((1 - sum(q3)) / 4, h / 10)
    )

    qp <- q3
    qm <- q3
    qp[j] <- qp[j] + step
    qm[j] <- qm[j] - step

    fp <- kappa_from_prob3(qp, c_val)
    fm <- kappa_from_prob3(qm, c_val)

    if (is.finite(fp) && is.finite(fm)) {
      g[j] <- (fp - fm) / (2 * step)
    } else {
      f0 <- kappa_from_prob3(q3, c_val)
      if (is.finite(fp) && is.finite(f0)) {
        g[j] <- (fp - f0) / step
      } else if (is.finite(fm) && is.finite(f0)) {
        g[j] <- (f0 - fm) / step
      }
    }
  }

  g
}

direct_delta_ci <- function(df_in, c_val, alpha) {
  met <- compute_metrics(df_in$D, df_in$T, c_val)
  n <- met$n

  q <- c(
    met$TP,
    met$FN,
    met$FP,
    met$TN
  ) / n

  q3 <- q[1:3]
  g <- numeric_gradient_prob3(q3, c_val)

  if (any(!is.finite(g))) {
    return(list(
      point = met$Kappa,
      se = NA_real_,
      lower = NA_real_,
      upper = NA_real_
    ))
  }

  Sigma3 <- (
    diag(q3) -
      tcrossprod(q3, q3)
  ) / n

  var_k <- drop(t(g) %*% Sigma3 %*% g)

  if (!is.finite(var_k) || var_k < 0) {
    return(list(
      point = met$Kappa,
      se = NA_real_,
      lower = NA_real_,
      upper = NA_real_
    ))
  }

  se <- sqrt(var_k)
  z <- qnorm(1 - alpha / 2)

  list(
    point = met$Kappa,
    se = se,
    lower = clamp(met$Kappa - z * se, -1, 1),
    upper = clamp(met$Kappa + z * se, -1, 1)
  )
}


transform_derivative_kappa <- function(k, method) {
  e <- 1e-8
  kk <- clamp(k, -1 + e, 1 - e)

  if (method == "identity") return(1)
  if (method == "fisher_z") return(1 / (1 - kk^2))
  if (method == "logit11") return(2 / (1 - kk^2))

  stop("Unknown transformation: ", method)
}

direct_transformed_delta_ci <- function(
  df_in,
  c_val,
  alpha,
  transformation = "logit11"
) {
  base <- direct_delta_ci(
    df_in = df_in,
    c_val = c_val,
    alpha = alpha
  )

  if (!is.finite(base$point) || !is.finite(base$se)) {
    return(list(
      point = base$point,
      se = NA_real_,
      lower = NA_real_,
      upper = NA_real_
    ))
  }

  z0 <- transform_kappa(
    base$point,
    transformation
  )

  dz <- abs(
    transform_derivative_kappa(
      base$point,
      transformation
    )
  )

  se_z <- dz * base$se
  crit <- qnorm(1 - alpha / 2)

  list(
    point = base$point,
    se = base$se,
    lower = inverse_transform_kappa(
      z0 - crit * se_z,
      transformation
    ),
    upper = inverse_transform_kappa(
      z0 + crit * se_z,
      transformation
    )
  )
}

###############################################################################
# 17) CORRECTED FINITE-SAMPLE CALIBRATION SENSITIVITY
###############################################################################

calibration_factors <- function(
  n,
  n0,
  prevalence = 0.5
) {
  if (!is.finite(n) || n <= 0 || !is.finite(n0) || n0 <= 0) {
    return(list(
      ratio_n_n0 = NA_real_,
      lambda_variance_linear = NA_real_,
      lambda_se_sqrt = NA_real_,
      n_eff = NA_real_,
      lambda_adaptive_variance = NA_real_,
      lambda_adaptive_se_sqrt = NA_real_
    ))
  }

  ratio <- min(1, n / n0)
  lambda_var <- ratio
  lambda_se <- sqrt(lambda_var)

  p <- clamp(prevalence, 1e-6, 1 - 1e-6)
  n_eff <- n * 2 * min(p, 1 - p)
  lambda_adapt_var <- min(1, n_eff / n0)
  lambda_adapt_se <- sqrt(lambda_adapt_var)

  list(
    ratio_n_n0 = ratio,
    lambda_variance_linear = lambda_var,
    lambda_se_sqrt = lambda_se,
    n_eff = n_eff,
    lambda_adaptive_variance = lambda_adapt_var,
    lambda_adaptive_se_sqrt = lambda_adapt_se
  )
}

calibration_lambda <- function(
  n,
  n0,
  prevalence = 0.5,
  mode = "sqrt_se_expand"
) {
  f <- calibration_factors(n, n0, prevalence)

  if (mode == "none") return(1)

  # Backward-compatible aliases are accepted deliberately.
  if (mode %in% c("fixed_sqrt", "sqrt_se_expand")) {
    return(f$lambda_se_sqrt)
  }

  if (mode %in% c("fixed_linear", "linear_se_expand", "legacy_linear_contract")) {
    return(f$lambda_variance_linear)
  }

  if (mode %in% c("adaptive_min_stratum", "adaptive_sqrt_expand")) {
    return(f$lambda_adaptive_se_sqrt)
  }

  stop("Unknown calibration_mode: ", mode)
}

calibrated_ci_from_boot <- function(
  point,
  boot_values,
  alpha,
  transformation,
  n,
  n0,
  prevalence,
  mode
) {
  x <- boot_values[is.finite(boot_values)]

  if (length(x) < 10 || !is.finite(point)) {
    return(list(
      lower = NA_real_, upper = NA_real_,
      se_raw_transformed = NA_real_, se_cal_transformed = NA_real_,
      lambda = NA_real_, lambda_variance = NA_real_, lambda_se = NA_real_,
      mode = mode
    ))
  }

  z0 <- transform_kappa(point, transformation)
  zb <- transform_kappa(x, transformation)
  se_z <- sd(zb, na.rm = TRUE)
  f <- calibration_factors(n, n0, prevalence)
  lam <- calibration_lambda(n, n0, prevalence, mode)

  if (!is.finite(se_z) || !is.finite(lam) || lam <= 0) {
    return(list(
      lower = NA_real_, upper = NA_real_,
      se_raw_transformed = se_z, se_cal_transformed = NA_real_,
      lambda = lam,
      lambda_variance = f$lambda_variance_linear,
      lambda_se = f$lambda_se_sqrt,
      mode = mode
    ))
  }

  se_cal <- switch(
    mode,
    none = se_z,
    fixed_sqrt = se_z / f$lambda_se_sqrt,
    sqrt_se_expand = se_z / f$lambda_se_sqrt,
    fixed_linear = se_z / f$lambda_variance_linear,
    linear_se_expand = se_z / f$lambda_variance_linear,
    adaptive_min_stratum = se_z / f$lambda_adaptive_se_sqrt,
    adaptive_sqrt_expand = se_z / f$lambda_adaptive_se_sqrt,
    legacy_linear_contract = se_z * f$lambda_variance_linear,
    stop("Unknown calibration_mode: ", mode)
  )

  crit <- qnorm(1 - alpha / 2)

  list(
    lower = inverse_transform_kappa(
      z0 - crit * se_cal,
      transformation
    ),
    upper = inverse_transform_kappa(
      z0 + crit * se_cal,
      transformation
    ),
    se_raw_transformed = se_z,
    se_cal_transformed = se_cal,
    lambda = lam,
    lambda_variance = f$lambda_variance_linear,
    lambda_se = f$lambda_se_sqrt,
    lambda_adaptive_se = f$lambda_adaptive_se_sqrt,
    n_eff = f$n_eff,
    mode = mode
  )
}

###############################################################################
# 17B) PRESERVED BASELINE DIAGNOSTICS -- SECONDARY ONLY
###############################################################################

legacy_analytic_n_estimate <- function(
  se_current,
  delta,
  alpha = 0.05,
  power = 0.80
) {
  if (!is.finite(se_current) || !is.finite(delta) ||
      se_current <= 0 || delta <= 0) {
    return(NA_real_)
  }
  z_alpha <- qnorm(1 - alpha / 2)
  z_beta <- qnorm(power)
  out <- ((z_alpha + z_beta)^2 * se_current^2) / (delta^2)
  as.numeric(max(min(out, 1e6), 1))
}

legacy_predictive_power <- function(point, se, alpha = 0.05) {
  if (!is.finite(point) || !is.finite(se) || se <= 0) return(NA_real_)
  zcrit <- qnorm(1 - alpha / 2)
  1 - pnorm(zcrit - abs(point) / se)
}

# This is NOT frequentist coverage. It is retained only to preserve the
# baseline code's internal-reference diagnostic in a correctly labelled form.
internal_reference_inclusion <- function(reference, lower, upper) {
  if (!all(is.finite(c(reference, lower, upper)))) return(NA_integer_)
  as.integer(reference >= lower && reference <= upper)
}

###############################################################################
# 18) FULL PIPELINE BOOTSTRAP INFERENCE
###############################################################################

bootstrap_ci_from_vector <- function(
  point,
  boot_values,
  alpha,
  method,
  transformation
) {
  x <- boot_values[is.finite(boot_values)]

  if (length(x) < 10 || !is.finite(point)) {
    return(c(lower = NA_real_, upper = NA_real_, se = NA_real_))
  }

  if (method == "percentile") {
    q <- quantile(
      x,
      probs = c(alpha / 2, 1 - alpha / 2),
      na.rm = TRUE,
      names = FALSE
    )

    return(c(
      lower = clamp(q[1], -1, 1),
      upper = clamp(q[2], -1, 1),
      se = sd(x)
    ))
  }

  if (method == "basic") {
    q <- quantile(
      x,
      probs = c(alpha / 2, 1 - alpha / 2),
      na.rm = TRUE,
      names = FALSE
    )

    lo <- 2 * point - q[2]
    hi <- 2 * point - q[1]

    return(c(
      lower = clamp(lo, -1, 1),
      upper = clamp(hi, -1, 1),
      se = sd(x)
    ))
  }

  if (method == "normal_transformed") {
    z0 <- transform_kappa(point, transformation)
    zb <- transform_kappa(x, transformation)
    sez <- sd(zb)

    crit <- qnorm(1 - alpha / 2)

    return(c(
      lower = inverse_transform_kappa(
        z0 - crit * sez,
        transformation
      ),
      upper = inverse_transform_kappa(
        z0 + crit * sez,
        transformation
      ),
      se = sd(x)
    ))
  }

  stop("Unknown bootstrap CI method: ", method)
}


jackknife_direct_kappa <- function(df_in, c_val) {
  n <- nrow(df_in)

  if (n < 5) {
    return(rep(NA_real_, n))
  }

  vapply(
    seq_len(n),
    function(i) {
      compute_metrics(
        df_in$D[-i],
        df_in$T[-i],
        c_val
      )$Kappa
    },
    numeric(1)
  )
}

bca_ci <- function(
  point,
  boot_values,
  jack_values,
  alpha
) {
  b <- boot_values[is.finite(boot_values)]
  j <- jack_values[is.finite(jack_values)]

  if (length(b) < 20 || length(j) < 5 || !is.finite(point)) {
    return(c(lower = NA_real_, upper = NA_real_))
  }

  prop_less <- mean(b < point)
  prop_less <- clamp(
    prop_less,
    1 / (2 * length(b)),
    1 - 1 / (2 * length(b))
  )

  z0 <- qnorm(prop_less)

  jbar <- mean(j)
  num <- sum((jbar - j)^3)
  den <- 6 * (sum((jbar - j)^2)^(3 / 2))

  acc <- if (is.finite(den) && den > 0) {
    num / den
  } else {
    0
  }

  z_alpha <- qnorm(
    c(alpha / 2, 1 - alpha / 2)
  )

  adj <- pnorm(
    z0 +
      (z0 + z_alpha) /
      (1 - acc * (z0 + z_alpha))
  )

  adj <- clamp(adj, 0, 1)

  q <- quantile(
    b,
    probs = adj,
    na.rm = TRUE,
    names = FALSE,
    type = 6
  )

  c(
    lower = clamp(q[1], -1, 1),
    upper = clamp(q[2], -1, 1)
  )
}

paired_bootstrap_comparison <- function(
  direct_point,
  fwkc_point,
  direct_boot,
  fwkc_boot,
  alpha
) {
  ok <- is.finite(direct_boot) & is.finite(fwkc_boot)

  d_boot <- fwkc_boot[ok] - direct_boot[ok]
  d0 <- fwkc_point - direct_point

  if (length(d_boot) < 10 || !is.finite(d0)) {
    return(data.frame(
      n_pairs = length(d_boot),
      point_difference = d0,
      bootstrap_se_difference = NA_real_,
      z_normal = NA_real_,
      p_normal = NA_real_,
      diff_percentile_lower = NA_real_,
      diff_percentile_upper = NA_real_,
      p_centered_bootstrap = NA_real_,
      bootstrap_prob_FWKC_gt_Direct = NA_real_
    ))
  }

  # IMPORTANT:
  # sd(d_boot) estimates the sampling SE of the paired estimator difference.
  # It must NOT be divided by sqrt(B), because B controls Monte Carlo precision
  # of the bootstrap approximation, not the clinical sampling variance.
  se_diff <- sd(d_boot)

  if (is.finite(se_diff) && se_diff > 0) {
    z <- d0 / se_diff
    p_norm <- 2 * (1 - pnorm(abs(z)))
  } else {
    z <- NA_real_
    p_norm <- NA_real_
  }

  q <- quantile(
    d_boot,
    probs = c(alpha / 2, 1 - alpha / 2),
    na.rm = TRUE,
    names = FALSE
  )

  # Centered bootstrap test under H0: difference = 0.
  centered <- d_boot - d0
  p_boot <- (
    1 + sum(abs(centered) >= abs(d0), na.rm = TRUE)
  ) / (
    length(centered) + 1
  )

  data.frame(
    n_pairs = length(d_boot),
    point_difference = d0,
    bootstrap_se_difference = se_diff,
    z_normal = z,
    p_normal = p_norm,
    diff_percentile_lower = q[1],
    diff_percentile_upper = q[2],
    p_centered_bootstrap = p_boot,
    bootstrap_prob_FWKC_gt_Direct =
      mean(d_boot > 0, na.rm = TRUE)
  )
}

pipeline_bootstrap_inference <- function(
  df_in,
  c_val,
  cfg,
  keep_boot = FALSE,
  seed_offset = 0,
  show_boot_progress = FALSE,
  progress_label = "Inner bootstrap"
) {
  n <- nrow(df_in)

  # Compute the direct estimator FIRST. Direct inference must remain available
  # even when FWKC fails; this removes conditioning of the benchmark on success
  # of the proposed method.
  direct0 <- compute_metrics(df_in$D, df_in$T, c_val)
  delta0 <- direct_delta_ci(df_in, c_val, cfg$alpha)
  logit_delta0 <- direct_transformed_delta_ci(
    df_in = df_in,
    c_val = c_val,
    alpha = cfg$alpha,
    transformation = "logit11"
  )

  set_fwkc_seed(cfg$seed + seed_offset)

  t_core <- proc.time()[["elapsed"]]
  core0 <- tryCatch(
    fwkc_core(df_in, c_val, cfg),
    error = function(e) {
      log_message(
        "FWKC core failed at c=", c_val,
        " | seed_offset=", seed_offset,
        " | ", conditionMessage(e)
      )
      NULL
    }
  )
  core_seconds <- elapsed_seconds(t_core)
  core_ok <- !is.null(core0) && is.finite(core0$point)

  # Inner-bootstrap checkpoint. The key is reproducible for real-data,
  # independent-DGM, sensitivity, and external-validation calls because each
  # call supplies a deterministic seed_offset and has its own output tree.
  cp_boot_path <- file.path(
    cfg$output_dir,
    "checkpoints",
    sprintf(
      "inner_boot_seed%010d_c%03d.rds",
      as.integer(seed_offset),
      as.integer(round(c_val * 100))
    )
  )
  dir.create(dirname(cp_boot_path), recursive = TRUE, showWarnings = FALSE)

  boot_rows <- vector("list", cfg$B_boot)
  start_b <- 1L
  last_completed <- 0L
  direct_boot_seconds <- 0
  fwkc_boot_seconds <- 0
  boot_progress_start <- proc.time()[["elapsed"]]

  if (isTRUE(cfg$resume) && file.exists(cp_boot_path)) {
    cp <- tryCatch(readRDS(cp_boot_path), error = function(e) NULL)
    if (!is.null(cp) &&
        identical(as.integer(cp$B_boot), as.integer(cfg$B_boot)) &&
        isTRUE(all.equal(cp$c_val, c_val)) &&
        !is.null(cp$boot_rows)) {
      completed <- min(length(cp$boot_rows), cfg$B_boot)
      if (completed > 0) {
        for (j in seq_len(completed)) boot_rows[[j]] <- cp$boot_rows[[j]]
        start_b <- completed + 1L
        last_completed <- completed
        if (!is.null(cp$rng_state)) {
          assign(".Random.seed", cp$rng_state, envir = .GlobalEnv)
        }
        if (!is.null(cp$direct_boot_seconds)) {
          direct_boot_seconds <- cp$direct_boot_seconds
        }
        if (!is.null(cp$fwkc_boot_seconds)) {
          fwkc_boot_seconds <- cp$fwkc_boot_seconds
        }
        log_message(
          "Resuming inner bootstrap at replicate ", start_b,
          " / ", cfg$B_boot,
          " | c=", c_val,
          " | seed_offset=", seed_offset
        )
      }
    }
  }

  save_inner_checkpoint <- function(last_b) {
    if (last_b < 1) return(invisible(NULL))
    safe_save_rds(
      list(
        B_boot = cfg$B_boot,
        c_val = c_val,
        seed_offset = seed_offset,
        boot_rows = boot_rows[seq_len(last_b)],
        rng_state = .Random.seed,
        direct_boot_seconds = direct_boot_seconds,
        fwkc_boot_seconds = fwkc_boot_seconds
      ),
      cp_boot_path
    )
    invisible(NULL)
  }

  if (start_b <= cfg$B_boot) {
    tryCatch(
      {
        for (b in start_b:cfg$B_boot) {
          idx <- sample.int(n, n, replace = TRUE)
          db <- df_in[idx, , drop = FALSE]

          td <- proc.time()[["elapsed"]]
          direct_b <- compute_metrics(db$D, db$T, c_val)$Kappa
          direct_boot_seconds <- direct_boot_seconds + elapsed_seconds(td)

          boot_rows[[b]] <- data.frame(
            bootstrap = b,
            Kappa_Direct = direct_b,
            Bayes = NA_real_,
            ML = NA_real_,
            FWKC = NA_real_,
            Bayes_Rhat = NA_real_,
            Bayes_Convergence_OK = NA,
            W_Bayes = NA_real_,
            W_ML = NA_real_,
            Pseudo_SE_Empirical = NA_real_,
            Pseudo_SE_Bootstrap = NA_real_,
            stringsAsFactors = FALSE
          )

          # If the observed-data FWKC core itself is invalid, there is no reason
          # to run the expensive FWKC layer on bootstrap samples; Direct still
          # receives its full independent bootstrap distribution.
          if (core_ok) {
            tf <- proc.time()[["elapsed"]]
            fw_b <- tryCatch(
              fwkc_core(db, c_val, cfg),
              error = function(e) NULL
            )
            fwkc_boot_seconds <- fwkc_boot_seconds + elapsed_seconds(tf)

            if (!is.null(fw_b) && is.finite(fw_b$point)) {
              boot_rows[[b]]$Bayes <- fw_b$bayes_mean
              boot_rows[[b]]$ML <- fw_b$ml_mean
              boot_rows[[b]]$FWKC <- fw_b$point
              boot_rows[[b]]$Bayes_Rhat <- fw_b$bayes_rhat
              boot_rows[[b]]$Bayes_Convergence_OK <- fw_b$bayes_convergence_ok
              boot_rows[[b]]$W_Bayes <- fw_b$w_bayes
              boot_rows[[b]]$W_ML <- fw_b$w_ml
              boot_rows[[b]]$Pseudo_SE_Empirical <- fw_b$pseudo_se_empirical
              boot_rows[[b]]$Pseudo_SE_Bootstrap <- fw_b$pseudo_se_bootstrap
            }
          }

          last_completed <- b

          if (
            cfg$checkpoint_every_boot > 0 &&
              (b %% cfg$checkpoint_every_boot == 0 || b == cfg$B_boot)
          ) {
            save_inner_checkpoint(b)
          }

          if (
            isTRUE(show_boot_progress) &&
              should_report_progress(b, cfg$B_boot, cfg)
          ) {
            progress_message(
              stage = progress_label,
              current = b,
              total = cfg$B_boot,
              cfg = cfg,
              stage_start = boot_progress_start,
              detail = paste0(
                "c=", format(c_val, trim = TRUE),
                "; valid direct=",
                sum(vapply(boot_rows[seq_len(b)], function(z) {
                  !is.null(z) && is.finite(z$Kappa_Direct[1])
                }, logical(1))),
                "; valid FWKC=",
                sum(vapply(boot_rows[seq_len(b)], function(z) {
                  !is.null(z) && is.finite(z$FWKC[1])
                }, logical(1)))
              )
            )
          }
        }
      },
      interrupt = function(e) {
        # Save only fully completed bootstrap iterations. If interruption occurs
        # inside the FWKC core, the partially written current row is rerun.
        save_inner_checkpoint(last_completed)
        stop(e)
      }
    )
  }

  boot_df <- rbind_fill(boot_rows)
  if (file.exists(cp_boot_path)) try(unlink(cp_boot_path), silent = TRUE)

  valid_direct <- sum(is.finite(boot_df$Kappa_Direct))
  valid_fwkc <- sum(is.finite(boot_df$FWKC))
  min_valid <- max(20L, ceiling(cfg$B_boot * cfg$min_valid_boot_fraction))

  direct_success <- is.finite(direct0$Kappa) && valid_direct >= min_valid
  fwkc_success <- core_ok && valid_fwkc >= min_valid

  # ---------------- Direct inference: ALWAYS independent of FWKC success -----
  if (direct_success) {
    direct_ci <- bootstrap_ci_from_vector(
      point = direct0$Kappa,
      boot_values = boot_df$Kappa_Direct,
      alpha = cfg$alpha,
      method = "percentile",
      transformation = cfg$transformation
    )
    direct_basic_ci <- bootstrap_ci_from_vector(
      point = direct0$Kappa,
      boot_values = boot_df$Kappa_Direct,
      alpha = cfg$alpha,
      method = "basic",
      transformation = cfg$transformation
    )

    if (isTRUE(cfg$compute_bca) && n <= cfg$bca_max_n) {
      direct_jack <- jackknife_direct_kappa(df_in, c_val)
      direct_bca_ci <- bca_ci(
        point = direct0$Kappa,
        boot_values = boot_df$Kappa_Direct,
        jack_values = direct_jack,
        alpha = cfg$alpha
      )
    } else {
      direct_bca_ci <- c(lower = NA_real_, upper = NA_real_)
    }
  } else {
    direct_ci <- c(lower = NA_real_, upper = NA_real_, se = NA_real_)
    direct_basic_ci <- c(lower = NA_real_, upper = NA_real_, se = NA_real_)
    direct_bca_ci <- c(lower = NA_real_, upper = NA_real_)
  }

  # ---------------- FWKC full-pipeline inference ------------------------------
  if (fwkc_success) {
    fwkc_ci <- bootstrap_ci_from_vector(
      point = core0$point,
      boot_values = boot_df$FWKC,
      alpha = cfg$alpha,
      method = cfg$primary_ci,
      transformation = cfg$transformation
    )

    cal_selected <- calibrated_ci_from_boot(
      point = core0$point, boot_values = boot_df$FWKC,
      alpha = cfg$alpha, transformation = cfg$transformation,
      n = n, n0 = cfg$n0, prevalence = direct0$prevalence,
      mode = cfg$calibration_mode
    )
    cal_sqrt <- calibrated_ci_from_boot(
      point = core0$point, boot_values = boot_df$FWKC,
      alpha = cfg$alpha, transformation = cfg$transformation,
      n = n, n0 = cfg$n0, prevalence = direct0$prevalence,
      mode = "sqrt_se_expand"
    )
    cal_linear <- calibrated_ci_from_boot(
      point = core0$point, boot_values = boot_df$FWKC,
      alpha = cfg$alpha, transformation = cfg$transformation,
      n = n, n0 = cfg$n0, prevalence = direct0$prevalence,
      mode = "linear_se_expand"
    )
    cal_adaptive <- calibrated_ci_from_boot(
      point = core0$point, boot_values = boot_df$FWKC,
      alpha = cfg$alpha, transformation = cfg$transformation,
      n = n, n0 = cfg$n0, prevalence = direct0$prevalence,
      mode = "adaptive_sqrt_expand"
    )
    cal_legacy <- calibrated_ci_from_boot(
      point = core0$point, boot_values = boot_df$FWKC,
      alpha = cfg$alpha, transformation = cfg$transformation,
      n = n, n0 = cfg$n0, prevalence = direct0$prevalence,
      mode = "legacy_linear_contract"
    )
    lam <- cal_selected$lambda
    cal_lower <- cal_selected$lower
    cal_upper <- cal_selected$upper
  } else {
    fwkc_ci <- c(lower = NA_real_, upper = NA_real_, se = NA_real_)
    empty_cal <- list(
      lower = NA_real_, upper = NA_real_, lambda = NA_real_,
      lambda_variance = NA_real_, lambda_se = NA_real_,
      lambda_adaptive_se = NA_real_, n_eff = NA_real_
    )
    cal_selected <- cal_sqrt <- cal_linear <- cal_adaptive <- cal_legacy <- empty_cal
    lam <- calibration_lambda(n, cfg$n0, direct0$prevalence, cfg$calibration_mode)
    cal_lower <- cal_upper <- NA_real_
  }

  calibration_identity_diff <- if (fwkc_success) {
    abs(sqrt(cal_sqrt$lambda_variance) - cal_sqrt$lambda_se)
  } else NA_real_

  # Component dependence is descriptive and uses only matched valid replicates.
  ok_comp <- is.finite(boot_df$Bayes) & is.finite(boot_df$ML)
  component_cov <- if (sum(ok_comp) >= 20) {
    cov(boot_df$Bayes[ok_comp], boot_df$ML[ok_comp])
  } else NA_real_
  component_cor <- if (sum(ok_comp) >= 20) {
    cor(boot_df$Bayes[ok_comp], boot_df$ML[ok_comp])
  } else NA_real_

  pt <- paired_bootstrap_comparison(
    direct_point = direct0$Kappa,
    fwkc_point = if (core_ok) core0$point else NA_real_,
    direct_boot = boot_df$Kappa_Direct,
    fwkc_boot = boot_df$FWKC,
    alpha = cfg$alpha
  )

  fwkc_point <- if (core_ok) core0$point else NA_real_
  fwkc_se <- unname(fwkc_ci["se"])

  legacy_n <- NA_real_
  legacy_power <- NA_real_
  internal_inclusion <- NA_integer_
  if (isTRUE(cfg$retain_legacy_diagnostics) && fwkc_success) {
    delta_legacy <- abs(fwkc_point - direct0$Kappa)
    legacy_n <- legacy_analytic_n_estimate(
      se_current = fwkc_se,
      delta = delta_legacy,
      alpha = cfg$alpha,
      power = cfg$target_power
    )
    legacy_power <- legacy_predictive_power(
      point = fwkc_point,
      se = fwkc_se,
      alpha = cfg$alpha
    )
    internal_inclusion <- internal_reference_inclusion(
      reference = direct0$Kappa,
      lower = unname(fwkc_ci["lower"]),
      upper = unname(fwkc_ci["upper"])
    )
  }

  list(
    # Direct estimator and competing inference
    direct_point = direct0$Kappa,
    direct_boot_lower = unname(direct_ci["lower"]),
    direct_boot_upper = unname(direct_ci["upper"]),
    direct_boot_se = unname(direct_ci["se"]),
    direct_delta_lower = delta0$lower,
    direct_delta_upper = delta0$upper,
    direct_delta_se = delta0$se,
    direct_logit_lower = logit_delta0$lower,
    direct_logit_upper = logit_delta0$upper,
    direct_basic_lower = unname(direct_basic_ci["lower"]),
    direct_basic_upper = unname(direct_basic_ci["upper"]),
    direct_bca_lower = unname(direct_bca_ci["lower"]),
    direct_bca_upper = unname(direct_bca_ci["upper"]),

    # FWKC estimator and primary/secondary inference
    fwkc_point = fwkc_point,
    fwkc_regularized_point = fwkc_point,
    fwkc_lower = unname(fwkc_ci["lower"]),
    fwkc_upper = unname(fwkc_ci["upper"]),
    fwkc_boot_se = fwkc_se,
    fwkc_cal_lower = cal_lower,
    fwkc_cal_upper = cal_upper,
    calibration_lambda = lam,
    calibration_mode_selected = cfg$calibration_mode,

    fwkc_cal_sqrt_lower = cal_sqrt$lower,
    fwkc_cal_sqrt_upper = cal_sqrt$upper,
    fwkc_cal_linear_lower = cal_linear$lower,
    fwkc_cal_linear_upper = cal_linear$upper,
    fwkc_cal_adaptive_lower = cal_adaptive$lower,
    fwkc_cal_adaptive_upper = cal_adaptive$upper,
    fwkc_legacy_linear_contract_lower = cal_legacy$lower,
    fwkc_legacy_linear_contract_upper = cal_legacy$upper,
    calibration_lambda_se_sqrt = cal_sqrt$lambda_se,
    calibration_lambda_variance_linear = cal_sqrt$lambda_variance,
    calibration_lambda_adaptive_se = cal_adaptive$lambda_adaptive_se,
    calibration_n_eff = cal_adaptive$n_eff,
    calibration_identity_diff = calibration_identity_diff,

    # Components
    bayes_mean = if (core_ok) core0$bayes_mean else NA_real_,
    bayes_se = if (core_ok) core0$bayes_se else NA_real_,
    bayes_rhat = if (core_ok) core0$bayes_rhat else NA_real_,
    bayes_convergence_ok = if (core_ok) core0$bayes_convergence_ok else NA,
    ml_mean = if (core_ok) core0$ml_mean else NA_real_,
    ml_se = if (core_ok) core0$ml_se else NA_real_,
    rf_oob_mse = if (core_ok) core0$rf_oob_mse else NA_real_,
    w_bayes = if (core_ok) core0$w_bayes else NA_real_,
    w_ml = if (core_ok) core0$w_ml else NA_real_,
    pseudo_se_empirical = if (core_ok) core0$pseudo_se_empirical else NA_real_,
    pseudo_se_bootstrap = if (core_ok) core0$pseudo_se_bootstrap else NA_real_,
    pseudo_mean_cov = if (core_ok) core0$pseudo_mean_cov else NA_real_,
    component_boot_cov = component_cov,
    component_boot_cor = component_cor,

    # Descriptive comparators / operating characteristics
    PABAK = direct0$PABAK,
    Gwet_AC1 = direct0$Gwet_AC1,
    prevalence = direct0$prevalence,
    Se = direct0$Se,
    Sp = direct0$Sp,

    # Paired comparison
    paired_tests = pt,

    # Preserved baseline diagnostics, explicitly non-primary
    legacy_estimated_n = legacy_n,
    legacy_predictive_power = legacy_power,
    internal_reference_inclusion = internal_inclusion,

    # Runtime / validity
    core_seconds = core_seconds,
    direct_boot_seconds = direct_boot_seconds,
    fwkc_boot_seconds = fwkc_boot_seconds,
    valid_direct_boot = valid_direct,
    valid_boot = valid_fwkc,
    direct_failed = !direct_success,
    fwkc_failed = !fwkc_success,
    failed = !direct_success && !fwkc_success,
    B_boot_requested = cfg$B_boot,
    boot = if (keep_boot) boot_df else NULL
  )
}
###############################################################################
# 19) REAL-DATA ANALYSIS
###############################################################################

run_real_data <- function(df, cfg) {
  cfg_real <- cfg
  cfg_real$B_boot <- cfg$B_boot_real
  cfg_real$n_mc <- cfg$n_mc
  cfg_real$checkpoint_every_boot <- as.integer(cfg$checkpoint_every_boot_real)
  c_grid <- seq(0.1, 0.9, by = 0.1)

  rows <- vector("list", length(c_grid))
  tests <- vector("list", length(c_grid))
  real_stage_start <- proc.time()[["elapsed"]]

  save_parent_progress <- function() {
    done_rows <- rows[!vapply(rows, is.null, logical(1))]
    if (length(done_rows)>0) {
      save_csv_safe(rbind_fill(done_rows),
                    file.path(cfg$output_dir,"real_data","real_data_results_IN_PROGRESS.csv"))
    }
    done_tests <- tests[!vapply(tests,is.null,logical(1))]
    if (length(done_tests)>0) {
      save_csv_safe(rbind_fill(done_tests),
                    file.path(cfg$output_dir,"real_data","paired_tests_IN_PROGRESS.csv"))
    }
    invisible(NULL)
  }

  if (parallel_stage_enabled(cfg_real,"real_data",length(c_grid))) {
    cl <- get_parallel_cluster(cfg_real)
    workers <- resolve_parallel_workers(cfg_real,length(c_grid))
    cfg_worker <- cfg_real
    cfg_worker$parallel_enabled <- FALSE
    cfg_worker$parallel_workers <- 1L
    cfg_worker$write_progress_status <- FALSE
    cfg_worker$verbose <- isTRUE(cfg$parallel_worker_progress)
    cfg_worker$console_progress <- isTRUE(cfg$parallel_worker_progress)

    manual_batch <- suppressWarnings(as.integer(cfg$parallel_scenario_batch_size))
    if (is.finite(manual_batch) && manual_batch>0L) {
      batch_size <- manual_batch
    } else {
      batch_size <- workers * max(1L,as.integer(cfg$parallel_batch_multiplier))
    }
    batch_size <- max(1L,min(batch_size,length(c_grid)))
    batches <- split(seq_along(c_grid),ceiling(seq_along(c_grid)/batch_size))

    log_message("PARALLEL STAGE START | Real-data c-grid | workers=",workers,
                " | tasks=",length(c_grid)," | B=",cfg_real$B_boot,
                " | mode=",cfg$parallel_mode," | statistical settings unchanged")

    for (b in seq_along(batches)) {
      ids <- batches[[b]]
      log_message("PARALLEL BATCH START | Real-data c-grid | batch=",b,"/",length(batches),
                  " | c=",paste(c_grid[ids],collapse=","))
      raw <- parallel::parLapplyLB(
        cl, ids, fwkc_real_data_task,
        c_grid=c_grid, df=df, cfg_real=cfg_worker,
        show_boot_progress=isTRUE(cfg$parallel_worker_progress)
      )
      errors <- character(0)
      for (item in raw) {
        if (isTRUE(item$ok)) {
          rows[[item$index]] <- item$row
          tests[[item$index]] <- item$test
          log_message("REAL-DATA c COMPLETE | c=",c_grid[item$index]," | worker_pid=",item$pid)
        } else {
          errors <- c(errors,paste0("c-index ",item$index,": ",item$error))
        }
      }
      save_parent_progress()
      completed <- sum(!vapply(rows,is.null,logical(1)))
      progress_message(
        stage="Real-data c-grid [parallel]",current=completed,total=length(c_grid),cfg=cfg,
        stage_start=real_stage_start,
        detail=paste0("batch=",b,"/",length(batches),"; workers=",workers,
                      "; completed=",completed,"/",length(c_grid))
      )
      log_message("PARALLEL BATCH END | Real-data c-grid | batch=",b,"/",length(batches))
      if (length(errors)>0) {
        stop("Parallel real-data stage failed after saving completed outputs:\n- ",
             paste(errors,collapse="\n- "))
      }
    }
    log_message("PARALLEL STAGE COMPLETE | Real-data c-grid | workers=",workers,
                " | elapsed=",format_duration(proc.time()[["elapsed"]]-real_stage_start))
  } else {
    for (i in seq_along(c_grid)) {
      cc <- c_grid[i]
      log_message("Real-data analysis START | c=",cc," | ",i,"/",length(c_grid))
      item <- fwkc_real_data_task(i,c_grid,df,cfg_real,show_boot_progress=TRUE)
      if (!isTRUE(item$ok)) stop(item$error)
      rows[[i]] <- item$row
      tests[[i]] <- item$test
      save_parent_progress()
      progress_message(stage="Real-data c-grid",current=i,total=length(c_grid),cfg=cfg,
                       stage_start=real_stage_start,detail=paste0("c=",cc,"; sequential"))
    }
  }

  real_results <- rbind_fill(rows)
  paired_results <- rbind_fill(tests[!vapply(tests,is.null,logical(1))])
  save_csv_safe(real_results,file.path(cfg$output_dir,"real_data","real_data_results.csv"))
  if (!is.null(paired_results) && nrow(paired_results)>0) {
    save_csv_safe(paired_results,file.path(cfg$output_dir,"real_data","paired_tests.csv"))
  }
  safe_write_lines(
    paste0("completed: ",format(Sys.time(),"%Y-%m-%d %H:%M:%S")),
    file.path(cfg$output_dir,"real_data","REAL_DATA_COMPLETED.flag")
  )
  list(results=real_results,paired_tests=paired_results)
}

###############################################################################
# 20) DGM CATALOG + GENERATORS
###############################################################################

make_dgm_catalog <- function(df) {
  anchor <- compute_metrics(df$D, df$T, c_val = 0.5)

  data.frame(
    dgm_id = c(
      "anchor_iid",
      "moderate_iid",
      "high_iid",
      "sensitivity_dominant_iid",
      "specificity_dominant_iid",
      "moderate_clustered",
      "moderate_mixture"
    ),
    dgm_type = c(
      "iid",
      "iid",
      "iid",
      "iid",
      "iid",
      "clustered_beta",
      "mixture_tradeoff"
    ),
    Se_true = c(
      anchor$Se,
      0.75,
      0.90,
      0.95,
      0.70,
      0.75,
      0.75
    ),
    Sp_true = c(
      anchor$Sp,
      0.75,
      0.90,
      0.70,
      0.95,
      0.75,
      0.75
    ),
    source = c(
      "real-data anchor",
      "prespecified non-circular",
      "prespecified non-circular",
      "prespecified non-circular",
      "prespecified non-circular",
      "prespecified heterogeneous DGM",
      "prespecified mixture-heterogeneity DGM"
    ),
    stringsAsFactors = FALSE
  )
}

simulate_dgm_iid <- function(n, p_true, Se_true, Sp_true) {
  D <- rbinom(n, 1, p_true)
  T <- integer(n)

  id1 <- D == 1
  id0 <- D == 0

  if (sum(id1) > 0) {
    T[id1] <- rbinom(
      sum(id1),
      1,
      Se_true
    )
  }

  if (sum(id0) > 0) {
    T[id0] <- rbinom(
      sum(id0),
      1,
      1 - Sp_true
    )
  }

  data.frame(
    D = factor(D, levels = c(0, 1)),
    T = factor(T, levels = c(0, 1))
  )
}

simulate_dgm_clustered_beta <- function(
  n,
  p_true,
  Se_true,
  Sp_true,
  cluster_size,
  phi
) {
  n_clusters <- ceiling(n / cluster_size)
  cluster <- rep(seq_len(n_clusters), each = cluster_size)
  cluster <- cluster[seq_len(n)]

  # Cluster-specific operating characteristics.
  # Marginal means remain Se_true and Sp_true.
  se_cluster <- rbeta(
    n_clusters,
    Se_true * phi,
    (1 - Se_true) * phi
  )

  sp_cluster <- rbeta(
    n_clusters,
    Sp_true * phi,
    (1 - Sp_true) * phi
  )

  D <- rbinom(n, 1, p_true)
  T <- integer(n)

  for (j in seq_len(n_clusters)) {
    id <- which(cluster == j)

    id1 <- id[D[id] == 1]
    id0 <- id[D[id] == 0]

    if (length(id1) > 0) {
      T[id1] <- rbinom(
        length(id1),
        1,
        se_cluster[j]
      )
    }

    if (length(id0) > 0) {
      T[id0] <- rbinom(
        length(id0),
        1,
        1 - sp_cluster[j]
      )
    }
  }

  data.frame(
    D = factor(D, levels = c(0, 1)),
    T = factor(T, levels = c(0, 1))
  )
}

simulate_dgm_mixture_tradeoff <- function(
  n,
  p_true,
  Se_true,
  Sp_true,
  delta
) {
  delta_max <- min(
    Se_true - 0.02,
    0.98 - Se_true,
    Sp_true - 0.02,
    0.98 - Sp_true
  )
  delta_use <- max(0, min(delta, delta_max))

  G <- rbinom(n, 1, 0.5)
  se_i <- Se_true + ifelse(G == 1, delta_use, -delta_use)
  sp_i <- Sp_true + ifelse(G == 1, -delta_use, delta_use)

  D <- rbinom(n, 1, p_true)
  prob_T1 <- ifelse(D == 1, se_i, 1 - sp_i)
  T <- rbinom(n, 1, clamp(prob_T1, 1e-8, 1 - 1e-8))

  data.frame(
    D = factor(D, levels = c(0, 1)),
    T = factor(T, levels = c(0, 1))
  )
}

simulate_dgm <- function(
  n,
  p_true,
  Se_true,
  Sp_true,
  dgm_type,
  cfg
) {
  if (dgm_type == "iid") {
    return(
      simulate_dgm_iid(
        n = n,
        p_true = p_true,
        Se_true = Se_true,
        Sp_true = Sp_true
      )
    )
  }

  if (dgm_type == "clustered_beta") {
    return(
      simulate_dgm_clustered_beta(
        n = n,
        p_true = p_true,
        Se_true = Se_true,
        Sp_true = Sp_true,
        cluster_size = cfg$cluster_size,
        phi = cfg$cluster_phi
      )
    )
  }

  if (dgm_type == "mixture_tradeoff") {
    return(
      simulate_dgm_mixture_tradeoff(
        n = n,
        p_true = p_true,
        Se_true = Se_true,
        Sp_true = Sp_true,
        delta = cfg$mixture_delta
      )
    )
  }

  stop("Unknown dgm_type: ", dgm_type)
}

###############################################################################
# 21) SCENARIO GRID
###############################################################################

make_main_scenarios <- function(catalog, cfg) {
  cat_main <- catalog[
    catalog$dgm_id %in% cfg$main_dgm_ids,
    ,
    drop = FALSE
  ]

  if (cfg$scenario_design == "full_factorial") {
    base_grid <- expand.grid(
      n = cfg$n_values,
      p_true = cfg$p_values,
      c_val = cfg$c_values,
      stringsAsFactors = FALSE
    )

  } else if (cfg$scenario_design == "reviewer_balanced") {
    # Five deliberately chosen prevalence/weight combinations:
    # - both weighting directions under low prevalence,
    # - a balanced central condition,
    # - both weighting directions under high prevalence.
    stress_pairs <- data.frame(
      p_true = c(0.1, 0.1, 0.5, 0.9, 0.9),
      c_val = c(0.2, 0.8, 0.5, 0.2, 0.8)
    )

    base_grid <- merge(
      data.frame(n = cfg$n_values),
      stress_pairs,
      by = NULL
    )

  } else {
    stop(
      "Unknown scenario_design: ",
      cfg$scenario_design
    )
  }

  out <- rbind_fill(
    lapply(
      seq_len(nrow(cat_main)),
      function(i) {
        z <- base_grid
        z$dgm_id <- cat_main$dgm_id[i]
        z$dgm_type <- cat_main$dgm_type[i]
        z$Se_true <- cat_main$Se_true[i]
        z$Sp_true <- cat_main$Sp_true[i]
        z$dgm_source <- cat_main$source[i]
        z
      }
    )
  )

  out$scenario_id <- seq_len(nrow(out))

  if (!is.na(cfg$scenario_cap)) {
    # A balanced cap across DGM families, not simply the first rows.
    keep <- integer(0)

    split_ids <- split(
      seq_len(nrow(out)),
      out$dgm_id
    )

    target_each <- max(
      1,
      floor(cfg$scenario_cap /
        length(split_ids))
    )

    for (ids in split_ids) {
      keep <- c(
        keep,
        head(ids, target_each)
      )
    }

    if (length(keep) < cfg$scenario_cap) {
      remaining <- setdiff(
        seq_len(nrow(out)),
        keep
      )

      keep <- c(
        keep,
        head(
          remaining,
          cfg$scenario_cap -
            length(keep)
        )
      )
    }

    out <- out[
      sort(
        unique(
          head(keep, cfg$scenario_cap)
        )
      ),
      ,
      drop = FALSE
    ]
  }

  rownames(out) <- NULL
  out
}

###############################################################################
# 22) VALID FREQUENTIST COVERAGE ACROSS INDEPENDENT DGM DATASETS
###############################################################################

scenario_checkpoint_path <- function(cfg, scenario_id) {
  file.path(
    cfg$output_dir,
    "checkpoints",
    sprintf("scenario_%04d.rds", scenario_id)
  )
}

scenario_detail_path <- function(cfg, scenario_id) {
  file.path(
    cfg$output_dir,
    "simulation",
    "detail",
    sprintf("scenario_%04d.csv", scenario_id)
  )
}


scenario_rep_journal_dir <- function(cfg, scenario_id) {
  file.path(
    cfg$output_dir,
    "checkpoints",
    sprintf("scenario_%04d_reps", scenario_id)
  )
}

scenario_rep_journal_path <- function(cfg, scenario_id, dgm_rep) {
  file.path(
    scenario_rep_journal_dir(cfg, scenario_id),
    sprintf("rep_%05d.rds", as.integer(dgm_rep))
  )
}

save_dgm_rep_journal <- function(cfg, scenario_id, dgm_rep, signature, row) {
  if (!isTRUE(cfg$dgm_rep_journal_enabled)) return(invisible(NULL))
  path <- scenario_rep_journal_path(cfg, scenario_id, dgm_rep)
  safe_save_rds(
    list(
      signature = signature,
      dgm_rep = as.integer(dgm_rep),
      row = row,
      saved_at = Sys.time()
    ),
    path
  )
  invisible(path)
}

recover_dgm_rep_journal <- function(cfg, scenario_id, signature, rows) {
  if (!isTRUE(cfg$dgm_rep_journal_enabled)) {
    return(list(rows = rows, recovered = 0L))
  }
  d <- scenario_rep_journal_dir(cfg, scenario_id)
  if (!dir.exists(d)) return(list(rows = rows, recovered = 0L))

  files <- list.files(d, pattern = "^rep_[0-9]+\\.rds$", full.names = TRUE)
  if (length(files) == 0L) return(list(rows = rows, recovered = 0L))

  recovered <- 0L
  for (f in sort(files)) {
    obj <- tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(obj) || is.null(obj$signature) || !identical(obj$signature, signature) ||
        is.null(obj$dgm_rep) || is.null(obj$row)) next
    m <- suppressWarnings(as.integer(obj$dgm_rep))
    if (!is.finite(m) || m < 1L || m > length(rows)) next
    rows[[m]] <- obj$row
    recovered <- recovered + 1L
  }
  list(rows = rows, recovered = recovered)
}

cleanup_dgm_rep_journal <- function(cfg, scenario_id) {
  if (!isTRUE(cfg$cleanup_dgm_rep_journal_on_complete)) return(invisible(FALSE))
  d <- scenario_rep_journal_dir(cfg, scenario_id)
  if (dir.exists(d)) {
    try(unlink(d, recursive = TRUE, force = TRUE), silent = TRUE)
  }
  invisible(TRUE)
}

checkpoint_signature <- function(cfg, s) {
  seed_id <- if ("seed_scenario_id" %in% names(s)) s$seed_scenario_id[1] else s$scenario_id[1]
  fields <- c(
    script_version = as.character(cfg$script_version),
    run_mode = as.character(cfg$run_mode),
    seed = as.character(cfg$seed),
    scenario_id = as.character(s$scenario_id[1]),
    seed_scenario_id = as.character(seed_id),
    dgm_id = as.character(s$dgm_id[1]),
    dgm_type = as.character(s$dgm_type[1]),
    n = as.character(s$n[1]),
    p_true = format(as.numeric(s$p_true[1]), digits = 17),
    c_val = format(as.numeric(s$c_val[1]), digits = 17),
    Se_true = format(as.numeric(s$Se_true[1]), digits = 17),
    Sp_true = format(as.numeric(s$Sp_true[1]), digits = 17),
    M_dgm = as.character(cfg$M_dgm),
    B_boot = as.character(cfg$B_boot),
    n_mc = as.character(cfg$n_mc),
    n_raters = as.character(cfg$n_raters),
    prior_type = as.character(cfg$prior_type),
    uncertainty_generator = as.character(cfg$uncertainty_generator),
    pseudo_mode = as.character(cfg$pseudo_mode),
    bayes_engine = as.character(cfg$bayes_engine),
    gibbs_sampler = as.character(cfg$gibbs_sampler),
    bayes_chains = as.character(cfg$bayes_chains),
    bayes_iter = as.character(cfg$bayes_iter),
    bayes_burn = as.character(cfg$bayes_burn),
    transformation = as.character(cfg$transformation),
    ml_layer = as.character(cfg$ml_layer),
    rf_ntree = as.character(cfg$rf_ntree),
    fusion_rule = as.character(cfg$fusion_rule),
    primary_ci = as.character(cfg$primary_ci),
    calibration_mode = as.character(cfg$calibration_mode),
    n0 = as.character(cfg$n0),
    cluster_size = as.character(cfg$cluster_size),
    cluster_phi = as.character(cfg$cluster_phi),
    mixture_delta = as.character(cfg$mixture_delta)
  )
  paste(names(fields), fields, sep = "=", collapse = "|")
}

evaluate_scenario <- function(s, cfg) {
  cp_path <- scenario_checkpoint_path(cfg, s$scenario_id)
  detail_path <- scenario_detail_path(cfg, s$scenario_id)

  rows <- vector("list", cfg$M_dgm)
  start_m <- 1L
  dgm_stage_start <- proc.time()[["elapsed"]]
  sig <- checkpoint_signature(cfg, s)
  seed_id <- if ("seed_scenario_id" %in% names(s)) s$seed_scenario_id[1] else s$scenario_id[1]

  if (isTRUE(cfg$resume) && file.exists(cp_path)) {
    cp <- tryCatch(
      readRDS(cp_path),
      error = function(e) NULL
    )

    if (!is.null(cp) && !is.null(cp$rows) &&
        !is.null(cp$signature) && identical(cp$signature, sig)) {
      completed <- length(cp$rows)

      if (completed > 0) {
        for (j in seq_len(completed)) {
          rows[[j]] <- cp$rows[[j]]
        }

        start_m <- completed + 1L

        if (!is.null(cp$rng_state)) {
          assign(
            ".Random.seed",
            cp$rng_state,
            envir = .GlobalEnv
          )
        }

        log_message(
          "Resuming scenario ",
          s$scenario_id,
          " from DGM replicate ",
          start_m
        )
      }
    }
  }

  # Recover any per-replicate atomic journal entries newer than the most recent
  # batched full checkpoint.  This keeps crash loss near zero without rewriting
  # the entire growing scenario object after every DGM replicate.
  jr <- recover_dgm_rep_journal(cfg, s$scenario_id, sig, rows)
  rows <- jr$rows
  completed_flags <- !vapply(rows, is.null, logical(1))
  contiguous_completed <- 0L
  if (length(completed_flags) > 0L) {
    first_missing <- which(!completed_flags)[1]
    contiguous_completed <- if (is.na(first_missing)) length(rows) else first_missing - 1L
  }
  if (contiguous_completed + 1L > start_m) {
    start_m <- contiguous_completed + 1L
    log_message(
      "Recovered scenario ", s$scenario_id,
      " through DGM replicate ", contiguous_completed,
      " from atomic replicate journal; next=", start_m
    )
  }

  k_true <- true_kappa_from_params(
    p = s$p_true,
    Se = s$Se_true,
    Sp = s$Sp_true,
    c_val = s$c_val
  )

  if (start_m <= cfg$M_dgm) {
    for (m in start_m:cfg$M_dgm) {
      # Independent, reproducible DGM seed separated from the inner
      # bootstrap/Monte-Carlo seed.
      set_fwkc_seed(
        cfg$seed +
          10000000 +
          100000 * seed_id +
          m
      )

      dat <- simulate_dgm(
        n = s$n,
        p_true = s$p_true,
        Se_true = s$Se_true,
        Sp_true = s$Sp_true,
        dgm_type = s$dgm_type,
        cfg = cfg
      )

      cfg_fit <- cfg
      if (!isTRUE(cfg$checkpoint_inner_bootstrap_simulation)) {
        cfg_fit$checkpoint_every_boot <- 0L
      }
      fit <- pipeline_bootstrap_inference(
        df_in = dat,
        c_val = s$c_val,
        cfg = cfg_fit,
        keep_boot = FALSE,
        seed_offset =
          100000 +
          1000 * seed_id +
          m
      )

      if (is.null(fit) || isTRUE(fit$failed)) {
        rows[[m]] <- data.frame(
          dgm_rep = m,
          scenario_id = s$scenario_id,
          dgm_id = s$dgm_id,
          dgm_type = s$dgm_type,
          n = s$n,
          p_true = s$p_true,
          c_val = s$c_val,
          Se_true = s$Se_true,
          Sp_true = s$Sp_true,
          Kappa_true = k_true,
          fit_failed = TRUE,
          stringsAsFactors = FALSE
        )

      } else {
        cover_direct_boot <- coverage_indicator(
          k_true,
          fit$direct_boot_lower,
          fit$direct_boot_upper
        )

        cover_direct_delta <- coverage_indicator(
          k_true,
          fit$direct_delta_lower,
          fit$direct_delta_upper
        )

        cover_direct_logit <- coverage_indicator(
          k_true,
          fit$direct_logit_lower,
          fit$direct_logit_upper
        )

        cover_direct_basic <- coverage_indicator(
          k_true,
          fit$direct_basic_lower,
          fit$direct_basic_upper
        )

        cover_direct_bca <- coverage_indicator(
          k_true,
          fit$direct_bca_lower,
          fit$direct_bca_upper
        )

        cover_fwkc <- coverage_indicator(
          k_true,
          fit$fwkc_lower,
          fit$fwkc_upper
        )

        cover_fwkc_cal <- coverage_indicator(
          k_true,
          fit$fwkc_cal_lower,
          fit$fwkc_cal_upper
        )
        cover_fwkc_cal_sqrt <- coverage_indicator(
          k_true, fit$fwkc_cal_sqrt_lower, fit$fwkc_cal_sqrt_upper
        )
        cover_fwkc_cal_linear <- coverage_indicator(
          k_true, fit$fwkc_cal_linear_lower, fit$fwkc_cal_linear_upper
        )
        cover_fwkc_cal_adaptive <- coverage_indicator(
          k_true, fit$fwkc_cal_adaptive_lower, fit$fwkc_cal_adaptive_upper
        )
        cover_fwkc_legacy <- coverage_indicator(
          k_true, fit$fwkc_legacy_linear_contract_lower,
          fit$fwkc_legacy_linear_contract_upper
        )

        rows[[m]] <- data.frame(
          dgm_rep = m,
          scenario_id = s$scenario_id,
          dgm_id = s$dgm_id,
          dgm_type = s$dgm_type,
          n = s$n,
          p_true = s$p_true,
          c_val = s$c_val,
          Se_true = s$Se_true,
          Sp_true = s$Sp_true,
          Kappa_true = k_true,

          Kappa_Direct = fit$direct_point,
          Direct_Boot_Lower = fit$direct_boot_lower,
          Direct_Boot_Upper = fit$direct_boot_upper,
          Direct_Delta_Lower = fit$direct_delta_lower,
          Direct_Delta_Upper = fit$direct_delta_upper,
          Direct_Logit_Lower = fit$direct_logit_lower,
          Direct_Logit_Upper = fit$direct_logit_upper,
          Direct_Basic_Lower = fit$direct_basic_lower,
          Direct_Basic_Upper = fit$direct_basic_upper,
          Direct_BCa_Lower = fit$direct_bca_lower,
          Direct_BCa_Upper = fit$direct_bca_upper,

          Kappa_Bayes = fit$bayes_mean,
          Bayes_Rhat = fit$bayes_rhat,
          Bayes_Convergence_OK = fit$bayes_convergence_ok,
          Kappa_ML = fit$ml_mean,
          Kappa_FWKC = fit$fwkc_point,
          FWKC_Boot_SE = fit$fwkc_boot_se,
          FWKC_Lower = fit$fwkc_lower,
          FWKC_Upper = fit$fwkc_upper,
          FWKC_Cal_Lower = fit$fwkc_cal_lower,
          FWKC_Cal_Upper = fit$fwkc_cal_upper,
          FWKC_Cal_Sqrt_Lower = fit$fwkc_cal_sqrt_lower,
          FWKC_Cal_Sqrt_Upper = fit$fwkc_cal_sqrt_upper,
          FWKC_Cal_LinearSE_Lower = fit$fwkc_cal_linear_lower,
          FWKC_Cal_LinearSE_Upper = fit$fwkc_cal_linear_upper,
          FWKC_Cal_Adaptive_Lower = fit$fwkc_cal_adaptive_lower,
          FWKC_Cal_Adaptive_Upper = fit$fwkc_cal_adaptive_upper,
          FWKC_Legacy_LinearContract_Lower = fit$fwkc_legacy_linear_contract_lower,
          FWKC_Legacy_LinearContract_Upper = fit$fwkc_legacy_linear_contract_upper,
          Lambda_SE_Sqrt = fit$calibration_lambda_se_sqrt,
          Lambda_Variance_Linear = fit$calibration_lambda_variance_linear,
          Calibration_Identity_Diff = fit$calibration_identity_diff,

          Cover_Direct_Boot = cover_direct_boot,
          Cover_Direct_Delta = cover_direct_delta,
          Cover_Direct_Logit = cover_direct_logit,
          Cover_Direct_Basic = cover_direct_basic,
          Cover_Direct_BCa = cover_direct_bca,
          Cover_FWKC = cover_fwkc,
          Cover_FWKC_Cal = cover_fwkc_cal,
          Cover_FWKC_Cal_Sqrt = cover_fwkc_cal_sqrt,
          Cover_FWKC_Cal_LinearSE = cover_fwkc_cal_linear,
          Cover_FWKC_Cal_Adaptive = cover_fwkc_cal_adaptive,
          Cover_FWKC_Legacy_LinearContract = cover_fwkc_legacy,

          Empirical_Prevalence = fit$prevalence,
          PABAK = fit$PABAK,
          Gwet_AC1 = fit$Gwet_AC1,

          Pseudo_SE_Empirical = fit$pseudo_se_empirical,
          Pseudo_SE_Bootstrap = fit$pseudo_se_bootstrap,
          Component_Boot_Cov = fit$component_boot_cov,
          Component_Boot_Cor = fit$component_boot_cor,

          Direct_Boot_Seconds = fit$direct_boot_seconds,
          FWKC_Core_Seconds = fit$core_seconds,
          FWKC_Boot_Seconds = fit$fwkc_boot_seconds,

          Valid_Direct_Boot = fit$valid_direct_boot,
          Valid_FWKC_Boot = fit$valid_boot,
          Valid_Boot = fit$valid_boot,
          Direct_failed = fit$direct_failed,
          FWKC_failed = fit$fwkc_failed,
          InternalRef_Inclusion = fit$internal_reference_inclusion,
          Legacy_Estimated_n = fit$legacy_estimated_n,
          Legacy_Estimated_n_vs_Truth =
            if (isTRUE(cfg$retain_legacy_diagnostics)) {
              legacy_analytic_n_estimate(
                fit$fwkc_boot_se,
                abs(fit$fwkc_point - k_true),
                alpha = cfg$alpha,
                power = cfg$target_power
              )
            } else NA_real_,
          Legacy_Predictive_Power = fit$legacy_predictive_power,
          fit_failed = fit$failed,
          stringsAsFactors = FALSE
        )
      }

      # Replicate-level durability with O(1) write size.
      save_dgm_rep_journal(
        cfg = cfg,
        scenario_id = s$scenario_id,
        dgm_rep = m,
        signature = sig,
        row = rows[[m]]
      )

      if (
        m %% cfg$checkpoint_every_dgm == 0 ||
          m == cfg$M_dgm
      ) {
        completed_rows <- rows[seq_len(m)]

        safe_save_rds(
          list(
            signature = sig,
            rows = completed_rows,
            rng_state = .Random.seed,
            scenario = s,
            config = cfg
          ),
          cp_path
        )

        detail_now <- rbind_fill(completed_rows)
        save_csv_safe(detail_now, detail_path)
      }

      if (
        isTRUE(cfg$verbose) &&
          should_report_progress(m, cfg$M_dgm, cfg)
      ) {
        progress_message(
          stage = paste0("Scenario ", s$scenario_id, " DGM"),
          current = m,
          total = cfg$M_dgm,
          cfg = cfg,
          stage_start = dgm_stage_start,
          detail = paste0(
            s$dgm_id,
            "; n=", s$n,
            "; p=", s$p_true,
            "; c=", s$c_val
          )
        )
      }
    }
  }

  detail <- rbind_fill(rows)

  save_csv_safe(detail, detail_path)
  cleanup_dgm_rep_journal(cfg, s$scenario_id)

  summarize_scenario(detail, s, cfg)
}

safe_mean <- function(x) {
  if (all(!is.finite(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

safe_max <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  max(x)
}

safe_rmse <- function(est, truth) {
  ok <- is.finite(est)
  if (!any(ok)) return(NA_real_)
  sqrt(mean((est[ok] - truth)^2))
}

safe_mae <- function(est, truth) {
  ok <- is.finite(est)
  if (!any(ok)) return(NA_real_)
  mean(abs(est[ok] - truth))
}

coverage_mcse <- function(cov_rate, n_valid) {
  if (!is.finite(cov_rate) || n_valid <= 0) return(NA_real_)
  sqrt(cov_rate * (1 - cov_rate) / n_valid)
}


wilson_interval_binary <- function(x, conf_level = 0.95) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n <= 0) return(c(lower = NA_real_, upper = NA_real_))

  s <- sum(x == 1)
  phat <- s / n
  z <- qnorm(1 - (1 - conf_level) / 2)
  den <- 1 + z^2 / n
  center <- (phat + z^2 / (2 * n)) / den
  half <- z * sqrt(
    phat * (1 - phat) / n + z^2 / (4 * n^2)
  ) / den

  c(
    lower = max(0, center - half),
    upper = min(1, center + half)
  )
}

coverage_with_failures_as_noncoverage <- function(x, total_expected) {
  if (!is.finite(total_expected) || total_expected <= 0) return(NA_real_)
  sum(x == 1, na.rm = TRUE) / total_expected
}

coverage_indicator <- function(
  truth,
  lower,
  upper
) {
  if (
    !is.finite(truth) ||
      !is.finite(lower) ||
      !is.finite(upper)
  ) {
    return(NA_integer_)
  }

  as.integer(
    truth >= lower &&
      truth <= upper
  )
}

scenario_summary_failure_row <- function(s, cfg) {
  # Full stable schema: downstream anchor/QC/sensitivity summaries must never
  # lose columns merely because every replicate in a scenario failed.
  cols <- c(
    "scenario_id","dgm_id","dgm_type","dgm_source","n","p_true","c_val","Se_true","Sp_true","Kappa_true",
    "M_requested","M_valid","M_valid_direct","M_valid_fwkc","M_valid_paired",
    "direct_dataset_success_rate","fwkc_dataset_success_rate","paired_dataset_success_rate","direct_failure_rate","fwkc_failure_rate",
    "coverage_direct_boot","coverage_direct_boot_ci_lower","coverage_direct_boot_ci_upper","coverage_direct_boot_failure_as_noncoverage","mcse_direct_boot",
    "coverage_direct_delta","mcse_direct_delta","coverage_direct_logit","mcse_direct_logit","coverage_direct_basic","mcse_direct_basic",
    "coverage_direct_bca","mcse_direct_bca","coverage_fwkc","coverage_fwkc_ci_lower","coverage_fwkc_ci_upper",
    "coverage_fwkc_failure_as_noncoverage","mcse_fwkc","coverage_fwkc_calibrated","coverage_fwkc_calibrated_ci_lower",
    "coverage_fwkc_calibrated_ci_upper","coverage_fwkc_calibrated_failure_as_noncoverage","mcse_fwkc_calibrated",
    "coverage_fwkc_cal_sqrt","coverage_fwkc_cal_linear_se","coverage_fwkc_cal_adaptive_sqrt","coverage_fwkc_legacy_linear_contract",
    "mean_FWKC_minus_Direct","p_t_paired_independent_DGM","p_wilcox_paired_independent_DGM","p_sign_paired_independent_DGM",
    "p_wilcox_absolute_error_independent_DGM","bias_direct","bias_bayes","bias_ml","bias_fwkc","rmse_direct","rmse_bayes","rmse_ml","rmse_fwkc",
    "mae_direct","mae_bayes","mae_ml","mae_fwkc","mean_width_direct_boot","mean_width_direct_delta","mean_width_direct_logit",
    "mean_width_direct_basic","mean_width_direct_bca","mean_width_fwkc","mean_width_fwkc_calibrated","mean_width_fwkc_cal_sqrt",
    "mean_width_fwkc_cal_linear_se","mean_width_fwkc_cal_adaptive_sqrt","mean_width_fwkc_legacy_linear_contract",
    "mean_lambda_se_sqrt","mean_lambda_variance_linear","max_calibration_identity_diff","mean_PABAK","mean_Gwet_AC1",
    "mean_pseudo_se_empirical","mean_pseudo_se_bootstrap","mean_component_boot_cov","mean_component_boot_cor","mean_bayes_rhat",
    "bayes_convergence_rate","mean_legacy_estimated_n","mean_legacy_estimated_n_vs_truth","mean_legacy_predictive_power",
    "mean_seconds_direct_boot","mean_seconds_fwkc_core","mean_seconds_fwkc_boot","prior_type","uncertainty_generator","pseudo_mode",
    "bayes_engine","transformation","ml_layer","fusion_rule","primary_ci","calibration_mode","n0","B_boot_inner","M_dgm_target"
  )
  z <- as.list(setNames(rep(NA, length(cols)), cols))
  z$scenario_id <- s$scenario_id
  z$dgm_id <- as.character(s$dgm_id)
  z$dgm_type <- as.character(s$dgm_type)
  z$dgm_source <- if ("dgm_source" %in% names(s)) as.character(s$dgm_source) else NA_character_
  z$n <- as.numeric(s$n); z$p_true <- as.numeric(s$p_true); z$c_val <- as.numeric(s$c_val)
  z$Se_true <- as.numeric(s$Se_true); z$Sp_true <- as.numeric(s$Sp_true)
  z$Kappa_true <- true_kappa_from_params(s$p_true,s$Se_true,s$Sp_true,s$c_val)
  z$M_requested <- as.integer(cfg$M_dgm)
  z$M_valid <- 0L; z$M_valid_direct <- 0L; z$M_valid_fwkc <- 0L; z$M_valid_paired <- 0L
  z$direct_dataset_success_rate <- 0; z$fwkc_dataset_success_rate <- 0; z$paired_dataset_success_rate <- 0
  z$direct_failure_rate <- 1; z$fwkc_failure_rate <- 1
  z$coverage_direct_boot_failure_as_noncoverage <- 0
  z$coverage_fwkc_failure_as_noncoverage <- 0
  z$prior_type <- cfg$prior_type; z$uncertainty_generator <- cfg$uncertainty_generator
  z$pseudo_mode <- cfg$pseudo_mode; z$bayes_engine <- cfg$bayes_engine
  z$transformation <- cfg$transformation; z$ml_layer <- cfg$ml_layer; z$fusion_rule <- cfg$fusion_rule
  z$primary_ci <- cfg$primary_ci; z$calibration_mode <- cfg$calibration_mode
  z$n0 <- cfg$n0; z$B_boot_inner <- cfg$B_boot; z$M_dgm_target <- cfg$M_dgm
  as.data.frame(z, stringsAsFactors=FALSE, check.names=FALSE)
}

summarize_scenario <- function(detail, s, cfg) {
  valid <- detail
  M_total <- cfg$M_dgm

  # Use a vectorized failure filter.
  if ("fit_failed" %in% names(detail)) {
    valid <- detail[
      is.na(detail$fit_failed) | detail$fit_failed == FALSE,
      ,
      drop = FALSE
    ]
  }

  if (nrow(valid) == 0) {
    return(scenario_summary_failure_row(s, cfg))
  }

  truth <- valid$Kappa_true[1]
  Mv <- nrow(valid)
  M_direct <- sum(is.finite(valid$Kappa_Direct))
  M_fwkc <- sum(is.finite(valid$Kappa_FWKC))
  M_paired <- sum(is.finite(valid$Kappa_Direct) & is.finite(valid$Kappa_FWKC))

  cov_db <- safe_mean(valid$Cover_Direct_Boot)
  cov_dd <- safe_mean(valid$Cover_Direct_Delta)
  cov_dl <- safe_mean(valid$Cover_Direct_Logit)
  cov_dbasic <- safe_mean(valid$Cover_Direct_Basic)
  cov_dbca <- safe_mean(valid$Cover_Direct_BCa)
  cov_f <- safe_mean(valid$Cover_FWKC)
  cov_fc <- safe_mean(valid$Cover_FWKC_Cal)
  cov_fc_sqrt <- safe_mean(valid$Cover_FWKC_Cal_Sqrt)
  cov_fc_linear <- safe_mean(valid$Cover_FWKC_Cal_LinearSE)
  cov_fc_adaptive <- safe_mean(valid$Cover_FWKC_Cal_Adaptive)
  cov_fc_legacy <- safe_mean(valid$Cover_FWKC_Legacy_LinearContract)

  ci_cov_db <- wilson_interval_binary(valid$Cover_Direct_Boot, cfg$coverage_conf_level)
  ci_cov_f <- wilson_interval_binary(valid$Cover_FWKC, cfg$coverage_conf_level)
  ci_cov_fc <- wilson_interval_binary(valid$Cover_FWKC_Cal, cfg$coverage_conf_level)

  diff_est <- valid$Kappa_FWKC - valid$Kappa_Direct
  ok_diff <- is.finite(diff_est)

  t_p_independent <- NA_real_
  wilcox_p_independent <- NA_real_
  sign_p_independent <- NA_real_

  if (sum(ok_diff) >= 5) {
    tt <- tryCatch(
      t.test(diff_est[ok_diff], mu = 0),
      error = function(e) NULL
    )

    wt <- tryCatch(
      wilcox.test(
        diff_est[ok_diff],
        mu = 0,
        exact = FALSE
      ),
      error = function(e) NULL
    )

    n_pos <- sum(diff_est[ok_diff] > 0)
    n_nonzero <- sum(diff_est[ok_diff] != 0)

    st <- if (n_nonzero > 0) {
      tryCatch(
        binom.test(
          n_pos,
          n_nonzero,
          p = 0.5
        ),
        error = function(e) NULL
      )
    } else {
      NULL
    }

    if (!is.null(tt)) t_p_independent <- tt$p.value
    if (!is.null(wt)) wilcox_p_independent <- wt$p.value
    if (!is.null(st)) sign_p_independent <- st$p.value
  }

  abs_err_direct <- abs(valid$Kappa_Direct - truth)
  abs_err_fwkc <- abs(valid$Kappa_FWKC - truth)

  err_test <- if (sum(ok_diff) >= 5) {
    tryCatch(
      wilcox.test(
        abs_err_fwkc[ok_diff],
        abs_err_direct[ok_diff],
        paired = TRUE,
        exact = FALSE
      ),
      error = function(e) NULL
    )
  } else {
    NULL
  }

  data.frame(
    scenario_id = s$scenario_id,
    dgm_id = s$dgm_id,
    dgm_type = s$dgm_type,
    dgm_source = s$dgm_source,
    n = s$n,
    p_true = s$p_true,
    c_val = s$c_val,
    Se_true = s$Se_true,
    Sp_true = s$Sp_true,
    Kappa_true = truth,
    M_requested = cfg$M_dgm,
    M_valid = Mv,
    M_valid_direct = M_direct,
    M_valid_fwkc = M_fwkc,
    M_valid_paired = M_paired,
    direct_dataset_success_rate = M_direct / cfg$M_dgm,
    fwkc_dataset_success_rate = M_fwkc / cfg$M_dgm,
    paired_dataset_success_rate = M_paired / cfg$M_dgm,
    direct_failure_rate = 1 - M_direct / cfg$M_dgm,
    fwkc_failure_rate = 1 - M_fwkc / cfg$M_dgm,

    coverage_direct_boot = cov_db,
    coverage_direct_boot_ci_lower = unname(ci_cov_db["lower"]),
    coverage_direct_boot_ci_upper = unname(ci_cov_db["upper"]),
    coverage_direct_boot_failure_as_noncoverage =
      coverage_with_failures_as_noncoverage(valid$Cover_Direct_Boot, M_total),
    mcse_direct_boot = coverage_mcse(
      cov_db,
      sum(is.finite(valid$Cover_Direct_Boot))
    ),

    coverage_direct_delta = cov_dd,
    mcse_direct_delta = coverage_mcse(
      cov_dd,
      sum(is.finite(valid$Cover_Direct_Delta))
    ),

    coverage_direct_logit = cov_dl,
    mcse_direct_logit = coverage_mcse(
      cov_dl,
      sum(is.finite(valid$Cover_Direct_Logit))
    ),

    coverage_direct_basic = cov_dbasic,
    mcse_direct_basic = coverage_mcse(
      cov_dbasic,
      sum(is.finite(valid$Cover_Direct_Basic))
    ),

    coverage_direct_bca = cov_dbca,
    mcse_direct_bca = coverage_mcse(
      cov_dbca,
      sum(is.finite(valid$Cover_Direct_BCa))
    ),

    coverage_fwkc = cov_f,
    coverage_fwkc_ci_lower = unname(ci_cov_f["lower"]),
    coverage_fwkc_ci_upper = unname(ci_cov_f["upper"]),
    coverage_fwkc_failure_as_noncoverage =
      coverage_with_failures_as_noncoverage(valid$Cover_FWKC, M_total),
    mcse_fwkc = coverage_mcse(
      cov_f,
      sum(is.finite(valid$Cover_FWKC))
    ),

    coverage_fwkc_calibrated = cov_fc,
    coverage_fwkc_calibrated_ci_lower = unname(ci_cov_fc["lower"]),
    coverage_fwkc_calibrated_ci_upper = unname(ci_cov_fc["upper"]),
    coverage_fwkc_calibrated_failure_as_noncoverage =
      coverage_with_failures_as_noncoverage(valid$Cover_FWKC_Cal, M_total),
    mcse_fwkc_calibrated = coverage_mcse(
      cov_fc,
      sum(is.finite(valid$Cover_FWKC_Cal))
    ),
    coverage_fwkc_cal_sqrt = cov_fc_sqrt,
    coverage_fwkc_cal_linear_se = cov_fc_linear,
    coverage_fwkc_cal_adaptive_sqrt = cov_fc_adaptive,
    coverage_fwkc_legacy_linear_contract = cov_fc_legacy,

    mean_FWKC_minus_Direct =
      safe_mean(diff_est),
    p_t_paired_independent_DGM =
      t_p_independent,
    p_wilcox_paired_independent_DGM =
      wilcox_p_independent,
    p_sign_paired_independent_DGM =
      sign_p_independent,
    p_wilcox_absolute_error_independent_DGM =
      if (is.null(err_test)) NA_real_ else err_test$p.value,

    bias_direct =
      safe_mean(valid$Kappa_Direct - truth),
    bias_bayes =
      safe_mean(valid$Kappa_Bayes - truth),
    bias_ml =
      safe_mean(valid$Kappa_ML - truth),
    bias_fwkc =
      safe_mean(valid$Kappa_FWKC - truth),

    rmse_direct =
      safe_rmse(valid$Kappa_Direct, truth),
    rmse_bayes =
      safe_rmse(valid$Kappa_Bayes, truth),
    rmse_ml =
      safe_rmse(valid$Kappa_ML, truth),
    rmse_fwkc =
      safe_rmse(valid$Kappa_FWKC, truth),

    mae_direct =
      safe_mae(valid$Kappa_Direct, truth),
    mae_bayes =
      safe_mae(valid$Kappa_Bayes, truth),
    mae_ml =
      safe_mae(valid$Kappa_ML, truth),
    mae_fwkc =
      safe_mae(valid$Kappa_FWKC, truth),

    mean_width_direct_boot =
      safe_mean(
        valid$Direct_Boot_Upper -
          valid$Direct_Boot_Lower
      ),

    mean_width_direct_delta =
      safe_mean(
        valid$Direct_Delta_Upper -
          valid$Direct_Delta_Lower
      ),

    mean_width_direct_logit =
      safe_mean(
        valid$Direct_Logit_Upper -
          valid$Direct_Logit_Lower
      ),

    mean_width_direct_basic =
      safe_mean(
        valid$Direct_Basic_Upper -
          valid$Direct_Basic_Lower
      ),

    mean_width_direct_bca =
      safe_mean(
        valid$Direct_BCa_Upper -
          valid$Direct_BCa_Lower
      ),

    mean_width_fwkc =
      safe_mean(
        valid$FWKC_Upper -
          valid$FWKC_Lower
      ),

    mean_width_fwkc_calibrated =
      safe_mean(
        valid$FWKC_Cal_Upper -
          valid$FWKC_Cal_Lower
      ),
    mean_width_fwkc_cal_sqrt =
      safe_mean(valid$FWKC_Cal_Sqrt_Upper - valid$FWKC_Cal_Sqrt_Lower),
    mean_width_fwkc_cal_linear_se =
      safe_mean(valid$FWKC_Cal_LinearSE_Upper - valid$FWKC_Cal_LinearSE_Lower),
    mean_width_fwkc_cal_adaptive_sqrt =
      safe_mean(valid$FWKC_Cal_Adaptive_Upper - valid$FWKC_Cal_Adaptive_Lower),
    mean_width_fwkc_legacy_linear_contract =
      safe_mean(valid$FWKC_Legacy_LinearContract_Upper -
        valid$FWKC_Legacy_LinearContract_Lower),
    mean_lambda_se_sqrt = safe_mean(valid$Lambda_SE_Sqrt),
    mean_lambda_variance_linear = safe_mean(valid$Lambda_Variance_Linear),
    max_calibration_identity_diff = safe_max(valid$Calibration_Identity_Diff),

    mean_PABAK =
      safe_mean(valid$PABAK),

    mean_Gwet_AC1 =
      safe_mean(valid$Gwet_AC1),

    mean_pseudo_se_empirical =
      safe_mean(valid$Pseudo_SE_Empirical),

    mean_pseudo_se_bootstrap =
      safe_mean(valid$Pseudo_SE_Bootstrap),

    mean_component_boot_cov =
      safe_mean(valid$Component_Boot_Cov),

    mean_component_boot_cor =
      safe_mean(valid$Component_Boot_Cor),

    mean_bayes_rhat =
      if ("Bayes_Rhat" %in% names(valid)) safe_mean(valid$Bayes_Rhat) else NA_real_,
    bayes_convergence_rate =
      if ("Bayes_Convergence_OK" %in% names(valid)) {
        safe_mean(as.numeric(valid$Bayes_Convergence_OK))
      } else NA_real_,

    mean_legacy_estimated_n =
      if ("Legacy_Estimated_n" %in% names(valid)) safe_mean(valid$Legacy_Estimated_n) else NA_real_,
    mean_legacy_estimated_n_vs_truth =
      if ("Legacy_Estimated_n_vs_Truth" %in% names(valid)) safe_mean(valid$Legacy_Estimated_n_vs_Truth) else NA_real_,
    mean_legacy_predictive_power =
      if ("Legacy_Predictive_Power" %in% names(valid)) safe_mean(valid$Legacy_Predictive_Power) else NA_real_,

    mean_seconds_direct_boot =
      safe_mean(valid$Direct_Boot_Seconds),

    mean_seconds_fwkc_core =
      safe_mean(valid$FWKC_Core_Seconds),

    mean_seconds_fwkc_boot =
      safe_mean(valid$FWKC_Boot_Seconds),

    prior_type = cfg$prior_type,
    uncertainty_generator = cfg$uncertainty_generator,
    pseudo_mode = cfg$pseudo_mode,
    bayes_engine = cfg$bayes_engine,
    transformation = cfg$transformation,
    ml_layer = cfg$ml_layer,
    fusion_rule = cfg$fusion_rule,
    primary_ci = cfg$primary_ci,
    calibration_mode = cfg$calibration_mode,
    n0 = cfg$n0,
    B_boot_inner = cfg$B_boot,
    M_dgm_target = cfg$M_dgm,
    stringsAsFactors = FALSE
  )
}

run_main_simulation <- function(df, cfg) {
  cfg_main <- cfg
  cfg_main$B_boot <- cfg$B_boot_dgm
  cfg_main$n_mc <- cfg$n_mc
  catalog <- make_dgm_catalog(df)
  scenarios <- make_main_scenarios(catalog, cfg)

  save_csv_safe(
    catalog,
    file.path(cfg$output_dir, "simulation", "dgm_catalog.csv")
  )
  save_csv_safe(
    scenarios,
    file.path(cfg$output_dir, "simulation", "scenario_grid.csv")
  )

  log_message(
    "Independent-DGM scenarios to run: ", nrow(scenarios),
    " | parallel_workers=", resolve_parallel_workers(cfg, nrow(scenarios))
  )

  summaries <- run_scenario_grid_compute(
    scenarios = scenarios,
    cfg = cfg_main,
    stage_key = "main_simulation",
    stage_label = "Main simulation scenarios",
    checkpoint_csv = file.path(
      cfg$output_dir, "simulation", "main_results_checkpoint.csv"
    )
  )

  ans <- rbind_fill(summaries)
  save_csv_safe(
    ans,
    file.path(cfg$output_dir, "simulation", "main_results_final.csv")
  )
  ans
}

###############################################################################
# 22B) MANUSCRIPT REAL-DATA ANCHOR-SAMPLE-SIZE BRIDGE
###############################################################################

# This tier is intentionally separate from the primary n=30,50,100 continuity
# grid. It answers a distinct question: at the ACTUAL real-data sample size,
# does the inferential framework behave as expected in an information-rich
# setting and does finite-sample calibration become inactive? The default uses
# the observed prevalence plus the full c-grid. A full p-by-c grid is available
# as an optional sensitivity design, but is not required for the primary claim.
make_anchor_bridge_scenarios <- function(df, catalog, cfg) {
  if (!isTRUE(cfg$run_anchor_sample_size_bridge)) return(data.frame())

  anchor <- catalog[catalog$dgm_id == "anchor_iid", , drop = FALSE]
  if (nrow(anchor) != 1L) {
    stop("Anchor bridge requires exactly one anchor_iid row in the DGM catalog.")
  }

  m_obs <- compute_metrics(df$D, df$T, 0.5)
  anchor_n <- nrow(df)
  p_obs <- m_obs$prevalence

  if (!is.finite(anchor_n) || anchor_n < 2 || !is.finite(p_obs) || p_obs <= 0 || p_obs >= 1) {
    stop("Observed real-data sample size/prevalence is invalid for the anchor bridge.")
  }

  if (identical(cfg$anchor_bridge_design, "observed_prevalence_c_grid")) {
    base_grid <- data.frame(
      n = rep(anchor_n, length(cfg$anchor_bridge_c_values)),
      p_true = rep(p_obs, length(cfg$anchor_bridge_c_values)),
      c_val = cfg$anchor_bridge_c_values,
      stringsAsFactors = FALSE
    )
  } else if (identical(cfg$anchor_bridge_design, "full_pc_grid")) {
    base_grid <- expand.grid(
      n = anchor_n,
      p_true = cfg$anchor_bridge_p_values,
      c_val = cfg$anchor_bridge_c_values,
      stringsAsFactors = FALSE
    )
  } else {
    stop("Unknown anchor_bridge_design: ", cfg$anchor_bridge_design)
  }

  base_grid$dgm_id <- anchor$dgm_id[1]
  base_grid$dgm_type <- anchor$dgm_type[1]
  base_grid$Se_true <- anchor$Se_true[1]
  base_grid$Sp_true <- anchor$Sp_true[1]
  base_grid$dgm_source <- "real_data_anchor_sample_size_bridge"
  base_grid$scenario_id <- 900000L + seq_len(nrow(base_grid))
  base_grid$seed_scenario_id <- 900000L + seq_len(nrow(base_grid))
  base_grid$observed_anchor_n <- anchor_n
  base_grid$observed_anchor_prevalence <- p_obs
  base_grid$manuscript_anchor_n <- cfg$manuscript_anchor_n
  rownames(base_grid) <- NULL
  base_grid
}

safe_df_numeric <- function(df, name) {
  if (is.null(df) || !is.data.frame(df) || !(name %in% names(df))) return(numeric(0))
  x <- suppressWarnings(as.numeric(df[[name]]))
  x[is.finite(x)]
}

safe_df_mean <- function(df, name) {
  x <- safe_df_numeric(df, name)
  if (length(x) == 0L) NA_real_ else mean(x)
}

safe_df_min <- function(df, name) {
  x <- safe_df_numeric(df, name)
  if (length(x) == 0L) NA_real_ else min(x)
}

safe_df_max <- function(df, name) {
  x <- safe_df_numeric(df, name)
  if (length(x) == 0L) NA_real_ else max(x)
}

run_anchor_sample_size_bridge <- function(df, cfg) {
  if (!isTRUE(cfg$run_anchor_sample_size_bridge)) return(NULL)

  catalog <- make_dgm_catalog(df)
  scenarios <- make_anchor_bridge_scenarios(df, catalog, cfg)
  if (is.null(scenarios) || nrow(scenarios) == 0) return(NULL)

  cfg2 <- cfg
  cfg2$M_dgm <- cfg$anchor_bridge_M
  cfg2$B_boot <- cfg$anchor_bridge_B
  cfg2$n_mc <- cfg$anchor_bridge_n_mc
  cfg2$output_dir <- file.path(cfg$output_dir, "anchor_bridge")

  for (subdir in c("simulation", "simulation/detail", "checkpoints", "parallel")) {
    dir.create(file.path(cfg2$output_dir, subdir), recursive = TRUE, showWarnings = FALSE)
  }

  save_csv_safe(
    scenarios,
    file.path(cfg2$output_dir, "simulation", "anchor_sample_size_scenario_grid.csv")
  )

  anchor_meta <- data.frame(
    observed_input_n = nrow(df),
    manuscript_anchor_n = cfg$manuscript_anchor_n,
    observed_prevalence = unique(scenarios$observed_anchor_prevalence)[1],
    Se_anchor = unique(scenarios$Se_true)[1],
    Sp_anchor = unique(scenarios$Sp_true)[1],
    n0 = cfg$n0,
    expected_fixed_lambda_var = min(1, nrow(df) / cfg$n0),
    expected_fixed_lambda_se = sqrt(min(1, nrow(df) / cfg$n0)),
    design = cfg$anchor_bridge_design,
    M_dgm = cfg2$M_dgm,
    B_boot = cfg2$B_boot,
    n_mc = cfg2$n_mc,
    stringsAsFactors = FALSE
  )
  save_csv_safe(
    anchor_meta,
    file.path(cfg2$output_dir, "anchor_sample_size_metadata.csv")
  )

  log_message(
    "Anchor-sample-size bridge: n=", nrow(df),
    " | observed prevalence=", sprintf("%.4f", anchor_meta$observed_prevalence),
    " | scenarios=", nrow(scenarios),
    " | M=", cfg2$M_dgm,
    " | B=", cfg2$B_boot,
    " | n_mc=", cfg2$n_mc,
    " | parallel_workers=", resolve_parallel_workers(cfg2, nrow(scenarios))
  )

  summaries <- run_scenario_grid_compute(
    scenarios = scenarios,
    cfg = cfg2,
    stage_key = "anchor_bridge",
    stage_label = "Anchor sample-size bridge",
    checkpoint_csv = file.path(
      cfg2$output_dir, "simulation", "anchor_sample_size_results_checkpoint.csv"
    )
  )

  ans <- rbind_fill(summaries)
  save_csv_safe(
    ans,
    file.path(cfg2$output_dir, "simulation", "anchor_sample_size_results_final.csv")
  )

  required_anchor_cols <- c(
    "coverage_direct_boot","coverage_fwkc","rmse_direct","rmse_fwkc",
    "mean_lambda_se_sqrt","mean_lambda_variance_linear","M_valid",
    "direct_dataset_success_rate","fwkc_dataset_success_rate"
  )
  schema_complete <- all(required_anchor_cols %in% names(ans))
  bridge_summary <- data.frame(
    observed_input_n = nrow(df),
    manuscript_anchor_n = cfg$manuscript_anchor_n,
    scenarios_completed = nrow(ans),
    scenarios_with_valid_inference = if ("M_valid" %in% names(ans)) sum(ans$M_valid > 0, na.rm=TRUE) else 0L,
    mean_direct_coverage = safe_df_mean(ans,"coverage_direct_boot"),
    mean_fwkc_coverage = safe_df_mean(ans,"coverage_fwkc"),
    mean_direct_rmse = safe_df_mean(ans,"rmse_direct"),
    mean_fwkc_rmse = safe_df_mean(ans,"rmse_fwkc"),
    mean_lambda_se_sqrt = safe_df_mean(ans,"mean_lambda_se_sqrt"),
    mean_lambda_variance_linear = safe_df_mean(ans,"mean_lambda_variance_linear"),
    minimum_direct_success_rate = safe_df_min(ans,"direct_dataset_success_rate"),
    minimum_fwkc_success_rate = safe_df_min(ans,"fwkc_dataset_success_rate"),
    summary_schema_complete = schema_complete,
    stringsAsFactors = FALSE
  )
  log_message(
    "ANCHOR QUALITY CHECK | scenarios=",nrow(ans),
    " | valid_scenarios=",bridge_summary$scenarios_with_valid_inference,
    " | min_direct_success=",signif(bridge_summary$minimum_direct_success_rate,4),
    " | min_fwkc_success=",signif(bridge_summary$minimum_fwkc_success_rate,4),
    " | lambda_SE_mean=",signif(bridge_summary$mean_lambda_se_sqrt,6),
    " | schema_complete=",schema_complete
  )
  save_csv_safe(
    bridge_summary,
    file.path(cfg2$output_dir, "anchor_sample_size_bridge_summary.csv")
  )

  list(results = ans, scenarios = scenarios, metadata = anchor_meta, summary = bridge_summary)
}

# Produce a manuscript-facing bridge table containing the three original
# finite-sample sizes plus the real-data anchor size. The scopes are labelled
# explicitly because n=30/50/100 average over the full 9x9 p-by-c grid whereas
# the default n=620 bridge averages over all c at the observed real prevalence.
make_sample_size_bridge_comparison <- function(main_results, anchor_bridge, cfg) {
  if (is.null(main_results) || nrow(main_results) == 0 ||
      is.null(anchor_bridge) || !is.list(anchor_bridge) ||
      is.null(anchor_bridge$results) || nrow(anchor_bridge$results) == 0) {
    return(NULL)
  }

  safe_mean2 <- function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0) NA_real_ else mean(x)
  }

  primary_rows <- lapply(sort(unique(main_results$n)), function(nn) {
    z <- main_results[main_results$n == nn, , drop = FALSE]
    data.frame(
      n = nn,
      scope = "primary_full_9x9_p_by_c_grid",
      scenarios = nrow(z),
      mean_coverage_direct = safe_mean2(z$coverage_direct_boot),
      mean_coverage_fwkc = safe_mean2(z$coverage_fwkc),
      mean_rmse_direct = safe_mean2(z$rmse_direct),
      mean_rmse_fwkc = safe_mean2(z$rmse_fwkc),
      mean_lambda_se_sqrt = safe_mean2(z$mean_lambda_se_sqrt),
      mean_lambda_variance_linear = safe_mean2(z$mean_lambda_variance_linear),
      stringsAsFactors = FALSE
    )
  })

  abr <- anchor_bridge$results
  anchor_row <- data.frame(
    n = unique(abr$n)[1],
    scope = paste0("anchor_observed_prevalence_c_grid_p=", sprintf("%.4f", unique(abr$p_true)[1])),
    scenarios = nrow(abr),
    mean_coverage_direct = safe_mean2(abr$coverage_direct_boot),
    mean_coverage_fwkc = safe_mean2(abr$coverage_fwkc),
    mean_rmse_direct = safe_mean2(abr$rmse_direct),
    mean_rmse_fwkc = safe_mean2(abr$rmse_fwkc),
    mean_lambda_se_sqrt = safe_mean2(abr$mean_lambda_se_sqrt),
    mean_lambda_variance_linear = safe_mean2(abr$mean_lambda_variance_linear),
    stringsAsFactors = FALSE
  )

  out <- rbind_fill(c(primary_rows, list(anchor_row)))
  out <- out[order(out$n), , drop = FALSE]

  save_csv_safe(
    out,
    file.path(cfg$output_dir, "anchor_bridge", "sample_size_bridge_comparison.csv")
  )

  if (isTRUE(cfg$make_plots) && nrow(out) >= 2) {
    tryCatch({
      png(file.path(cfg$output_dir, "plots", "sample_size_bridge_coverage.png"),
          width=1200, height=800, res=120)
      matplot(
        out$n,
        cbind(out$mean_coverage_direct, out$mean_coverage_fwkc),
        type="b", pch=c(1,19), lty=1,
        xlab="Sample size (n)", ylab="Mean empirical coverage", ylim=c(0,1)
      )
      abline(h=cfg$nominal_coverage,lty=2)
      legend("bottomright",
             legend=c("Direct bootstrap","FWKC primary","Nominal"),
             pch=c(1,19,NA), lty=c(1,1,2), bty="n")
      dev.off()
    }, error=function(e) {
      try(dev.off(), silent=TRUE)
      log_message("Sample-size bridge coverage plot failed: ", conditionMessage(e))
    })

    tryCatch({
      png(file.path(cfg$output_dir, "plots", "sample_size_bridge_calibration.png"),
          width=1200, height=800, res=120)
      plot(
        out$n, out$mean_lambda_se_sqrt,
        type="b", pch=19,
        xlab="Sample size (n)", ylab="Mean fixed calibration factor (SE scale)",
        ylim=c(0,1.05)
      )
      abline(h=1,lty=2)
      dev.off()
    }, error=function(e) {
      try(dev.off(), silent=TRUE)
      log_message("Sample-size bridge calibration plot failed: ", conditionMessage(e))
    })
  }

  out
}

###############################################################################
# 23) SENSITIVITY ANALYSES
###############################################################################

make_robustness_scenarios <- function(catalog, cfg) {
  cat_r <- catalog[
    catalog$dgm_id %in% cfg$robustness_dgm_ids,
    ,
    drop = FALSE
  ]

  if (nrow(cat_r) == 0) {
    return(data.frame())
  }

  # Prespecified 3x3 stress grid spanning both extremes and the center of the
  # p and c axes. The exhaustive 9x9 grid remains reserved for the primary
  # continuity analysis so the revision preserves the submitted 81-scenario
  # structure at each sample size without multiplying computational cost by all
  # DGM families.
  base_grid <- expand.grid(
    n = cfg$n_values,
    p_true = cfg$robustness_p_values,
    c_val = cfg$robustness_c_values,
    stringsAsFactors = FALSE
  )

  out <- rbind_fill(
    lapply(
      seq_len(nrow(cat_r)),
      function(i) {
        z <- base_grid
        z$dgm_id <- cat_r$dgm_id[i]
        z$dgm_type <- cat_r$dgm_type[i]
        z$Se_true <- cat_r$Se_true[i]
        z$Sp_true <- cat_r$Sp_true[i]
        z$dgm_source <- cat_r$source[i]
        z
      }
    )
  )

  out$scenario_id <- seq_len(nrow(out))
  out$seed_scenario_id <- 10000L + out$scenario_id

  if (!is.na(cfg$robustness_scenario_cap)) {
    keep <- integer(0)
    split_ids <- split(seq_len(nrow(out)), out$dgm_id)
    target_each <- max(
      1,
      floor(cfg$robustness_scenario_cap / length(split_ids))
    )

    for (ids in split_ids) {
      keep <- c(keep, head(ids, target_each))
    }

    if (length(keep) < cfg$robustness_scenario_cap) {
      remaining <- setdiff(seq_len(nrow(out)), keep)
      keep <- c(
        keep,
        head(
          remaining,
          cfg$robustness_scenario_cap - length(keep)
        )
      )
    }

    out <- out[
      sort(unique(head(keep, cfg$robustness_scenario_cap))),
      ,
      drop = FALSE
    ]
  }

  rownames(out) <- NULL
  out
}

run_dgm_robustness <- function(df, cfg) {
  if (!isTRUE(cfg$run_dgm_robustness)) return(NULL)

  catalog <- make_dgm_catalog(df)
  scenarios <- make_robustness_scenarios(catalog, cfg)
  if (is.null(scenarios) || nrow(scenarios) == 0) return(NULL)

  cfg2 <- cfg
  cfg2$M_dgm <- cfg$robustness_M
  cfg2$B_boot <- cfg$robustness_B
  cfg2$n_mc <- cfg$n_mc
  cfg2$output_dir <- file.path(cfg$output_dir, "robustness")

  for (subdir in c("simulation", "simulation/detail", "checkpoints", "parallel")) {
    dir.create(file.path(cfg2$output_dir, subdir), recursive = TRUE, showWarnings = FALSE)
  }

  save_csv_safe(
    catalog[catalog$dgm_id %in% cfg$robustness_dgm_ids, , drop = FALSE],
    file.path(cfg2$output_dir, "simulation", "dgm_catalog_robustness.csv")
  )
  save_csv_safe(
    scenarios,
    file.path(cfg2$output_dir, "simulation", "scenario_grid_robustness.csv")
  )

  log_message(
    "Non-circular DGM robustness scenarios to run: ", nrow(scenarios),
    " | M=", cfg2$M_dgm,
    " | B=", cfg2$B_boot,
    " | parallel_workers=", resolve_parallel_workers(cfg2, nrow(scenarios))
  )

  summaries <- run_scenario_grid_compute(
    scenarios = scenarios,
    cfg = cfg2,
    stage_key = "robustness",
    stage_label = "DGM robustness scenarios",
    checkpoint_csv = file.path(
      cfg2$output_dir, "simulation", "robustness_results_checkpoint.csv"
    )
  )

  ans <- rbind_fill(summaries)
  save_csv_safe(
    ans,
    file.path(cfg2$output_dir, "simulation", "robustness_results_final.csv")
  )
  ans
}

make_sensitivity_scenarios <- function(catalog, cfg) {
  # Six prespecified stress scenarios span low/high prevalence, both weighting
  # directions, asymmetric operating characteristics, and structural DGM
  # misspecification. These are NOT selected after examining the results.
  desired <- data.frame(
    dgm_id = c(
      "moderate_iid",
      "moderate_iid",
      "sensitivity_dominant_iid",
      "specificity_dominant_iid",
      "moderate_clustered",
      "moderate_mixture"
    ),
    n = c(30, 50, 30, 30, 50, 50),
    p_true = c(0.1, 0.9, 0.1, 0.9, 0.1, 0.9),
    c_val = c(0.9, 0.1, 0.9, 0.1, 0.9, 0.1),
    stringsAsFactors = FALSE
  )

  desired <- desired[
    seq_len(min(nrow(desired), cfg$sensitivity_scenarios)),
    ,
    drop = FALSE
  ]

  idx <- match(desired$dgm_id, catalog$dgm_id)

  desired$dgm_type <- catalog$dgm_type[idx]
  desired$Se_true <- catalog$Se_true[idx]
  desired$Sp_true <- catalog$Sp_true[idx]
  desired$dgm_source <- "sensitivity"
  desired$scenario_id <- 800000 + seq_len(nrow(desired))

  desired
}

run_variant_scenarios <- function(
  scenarios,
  cfg,
  label,
  variant_index
) {
  cfg2 <- cfg
  cfg2$M_dgm <- cfg$sensitivity_M
  if (!startsWith(label, "bboot_")) cfg2$B_boot <- cfg$sensitivity_B
  if (!startsWith(label, "nmc_")) cfg2$n_mc <- cfg$sensitivity_n_mc
  # Variant-specific checkpoint trees allow safe resume without changing seeds/statistics.
  cfg2$resume <- cfg$resume
  cfg2$verbose <- isTRUE(cfg$parallel_worker_progress)

  safe_label <- gsub("[^A-Za-z0-9_-]", "_", label)
  cfg2$output_dir <- file.path(
    cfg$output_dir,
    "sensitivity",
    sprintf("%02d_%s", variant_index, safe_label)
  )
  for (subdir in c("simulation", "simulation/detail", "checkpoints", "parallel")) {
    dir.create(file.path(cfg2$output_dir, subdir), recursive = TRUE, showWarnings = FALSE)
  }

  scenarios2 <- scenarios
  scenarios2$seed_scenario_id <- scenarios2$scenario_id
  scenarios2$scenario_id <- variant_index * 100 + seq_len(nrow(scenarios2))

  out <- run_scenario_grid_compute(
    scenarios = scenarios2,
    cfg = cfg2,
    stage_key = "sensitivity",
    stage_label = paste0("Sensitivity ", label),
    checkpoint_csv = file.path(cfg2$output_dir, "variant_summary_IN_PROGRESS.csv"),
    extra_columns = list(sensitivity_label = label)
  )

  ans_variant <- rbind_fill(out)
  save_csv_safe(
    ans_variant,
    file.path(cfg2$output_dir, "variant_summary_FINAL.csv")
  )
  ans_variant
}

run_sensitivity <- function(df, cfg) {
  if (!isTRUE(cfg$run_sensitivity)) {
    return(NULL)
  }

  catalog <- make_dgm_catalog(df)
  scenarios <- make_sensitivity_scenarios(catalog, cfg)

  all_results <- list()
  variant_index <- 0L
  sensitivity_total_variants <- 34L
  sensitivity_start <- proc.time()[["elapsed"]]

  add_variant <- function(label, cfg_variant) {
    variant_index <<- variant_index + 1L

    log_message(
      "Sensitivity variant START | ", variant_index, "/",
      sensitivity_total_variants, " | ", label
    )

    all_results[[label]] <<- run_variant_scenarios(
      scenarios = scenarios,
      cfg = cfg_variant,
      label = label,
      variant_index = variant_index
    )

    combined_now <- rbind_fill(all_results)
    save_csv_safe(
      combined_now,
      file.path(
        cfg$output_dir,
        "sensitivity",
        "sensitivity_results_IN_PROGRESS.csv"
      )
    )

    progress_message(
      stage = "Sensitivity variants",
      current = variant_index,
      total = sensitivity_total_variants,
      cfg = cfg,
      stage_start = sensitivity_start,
      detail = paste0("completed=", label)
    )
  }

  # Proper Beta-prior sensitivity.
  for (v in c("jeffreys", "uniform", "legacy")) {
    c2 <- cfg
    c2$uncertainty_generator <- "beta_posterior"
    c2$prior_type <- v

    add_variant(
      paste0("prior_", v),
      c2
    )
  }

  # Nonparametric/empirical uncertainty-generator sensitivity.
  for (v in c(
    "beta_posterior",
    "empirical_bootstrap"
  )) {
    c2 <- cfg
    c2$uncertainty_generator <- v

    add_variant(
      paste0("uncertainty_", v),
      c2
    )
  }

  # Posterior-predictive vs parameter-only propagation.
  for (v in c(
    "posterior_predictive",
    "parameter_only"
  )) {
    c2 <- cfg
    c2$pseudo_mode <- v

    add_variant(
      paste0("pseudo_", v),
      c2
    )
  }

  # Fusion-rule sensitivity.
  for (v in c(
    "inverse_variance",
    "inverse_se",
    "equal",
    "bayes_only",
    "ml_only"
  )) {
    c2 <- cfg
    c2$fusion_rule <- v

    add_variant(
      paste0("fusion_", v),
      c2
    )
  }

  # Transformation sensitivity.
  for (v in c(
    "logit11",
    "fisher_z",
    "identity"
  )) {
    c2 <- cfg
    c2$transformation <- v

    add_variant(
      paste0("transform_", v),
      c2
    )
  }

  # ML comparator.
  for (v in c("rf", "spline", "none")) {
    c2 <- cfg
    c2$ml_layer <- v

    if (v == "none") {
      c2$fusion_rule <- "bayes_only"
    }

    add_variant(
      paste0("ml_", v),
      c2
    )
  }

  # Calibration-rule sensitivity for the SECONDARY interval.
  for (v in c(
    "none",
    "sqrt_se_expand",
    "linear_se_expand",
    "adaptive_sqrt_expand",
    "legacy_linear_contract"
  )) {
    c2 <- cfg
    c2$calibration_mode <- v

    add_variant(
      paste0("calibration_", v),
      c2
    )
  }

  # n0 sensitivity for the SECONDARY calibrated interval only.
  for (v in c(75, 100, 125, 150)) {
    c2 <- cfg
    c2$n0 <- v

    add_variant(
      paste0("n0_", v),
      c2
    )
  }

  # Monte Carlo integration sensitivity: checks that the final result is not
  # materially driven by an arbitrary n_mc choice.
  for (v in c(250, 500, 1000, 2000)) {
    c2 <- cfg
    c2$n_mc <- v

    add_variant(
      paste0("nmc_", v),
      c2
    )
  }

  # Bootstrap-replicate sensitivity for the full-pipeline interval.
  for (v in c(100, 200, 500)) {
    c2 <- cfg
    c2$B_boot <- v

    add_variant(
      paste0("bboot_", v),
      c2
    )
  }

  ans <- rbind_fill(all_results)

  save_csv_safe(
    ans,
    file.path(
      cfg$output_dir,
      "sensitivity",
      "sensitivity_results.csv"
    )
  )

  ans
}

###############################################################################
# 24) EXTERNAL REAL-DATA GENERALIZABILITY ANALYSIS
###############################################################################

expand_counts_to_data <- function(TP, FN, FP, TN) {
  D <- c(rep(1, TP), rep(1, FN), rep(0, FP), rep(0, TN))
  T <- c(rep(1, TP), rep(0, FN), rep(1, FP), rep(0, TN))
  data.frame(D=factor(D,levels=c(0,1)),T=factor(T,levels=c(0,1)))
}

validate_external_counts_table <- function(ext) {
  required <- c("dataset","TP","FN","FP","TN")
  if (!all(required %in% names(ext))) {
    stop("External data must contain columns: ",paste(required,collapse=", "))
  }
  for (nm in c("TP","FN","FP","TN")) ext[[nm]] <- suppressWarnings(as.numeric(ext[[nm]]))
  bad <- !is.finite(ext$TP) | !is.finite(ext$FN) | !is.finite(ext$FP) | !is.finite(ext$TN) |
    ext$TP<0 | ext$FN<0 | ext$FP<0 | ext$TN<0 |
    abs(ext$TP-round(ext$TP))>1e-8 | abs(ext$FN-round(ext$FN))>1e-8 |
    abs(ext$FP-round(ext$FP))>1e-8 | abs(ext$TN-round(ext$TN))>1e-8
  if (any(bad)) stop("External counts must be finite non-negative integers; bad row(s): ",paste(which(bad),collapse=","))
  ext$TP <- as.integer(round(ext$TP)); ext$FN <- as.integer(round(ext$FN))
  ext$FP <- as.integer(round(ext$FP)); ext$TN <- as.integer(round(ext$TN))
  npos <- ext$TP+ext$FN; nneg <- ext$FP+ext$TN
  if (any(npos<=0 | nneg<=0)) stop("Each external table must contain both reference-positive and reference-negative subjects.")
  if (any(!nzchar(trimws(as.character(ext$dataset))))) stop("Every external row needs a non-empty dataset identifier.")
  ext$dataset <- make.unique(as.character(ext$dataset))
  ext
}

auditc_external_counts_builtin <- function() {
  # Verified 14-study AuditC diagnostic-accuracy tables.
  # Source: Kriston et al. (2008), Ann Intern Med 149:879-888;
  # also distributed as mada::AuditC.
  # Column order follows the FWKC convention TP, FN, FP, TN.
  ext <- data.frame(
    dataset = sprintf("AuditC_study_%02d", 1:14),
    TP = c(47L,126L,19L,36L,130L,84L,68L,752L,59L,142L,137L,57L,34L,152L),
    FN = c(9L,51L,10L,3L,19L,2L,0L,0L,5L,50L,24L,3L,1L,51L),
    FP = c(101L,272L,12L,78L,211L,68L,112L,3226L,55L,571L,107L,103L,21L,88L),
    TN = c(738L,1543L,192L,276L,959L,89L,423L,2977L,136L,2788L,358L,437L,56L,264L),
    stringsAsFactors = FALSE
  )
  # Fixed integrity checks make accidental editing visible immediately.
  expected_sums <- c(TP=1843L,FN=228L,FP=5025L,TN=11236L)
  actual_sums <- c(TP=sum(ext$TP),FN=sum(ext$FN),FP=sum(ext$FP),TN=sum(ext$TN))
  if (nrow(ext) != 14L || !identical(as.integer(actual_sums), as.integer(expected_sums))) {
    stop(
      "Built-in AuditC integrity check failed. Expected 14 rows and totals TP/FN/FP/TN=",
      paste(expected_sums,collapse="/"), "; got rows=",nrow(ext)," totals=",
      paste(actual_sums,collapse="/")
    )
  }
  ext$external_source <- "Verified built-in copy of mada::AuditC"
  ext$external_source_type <- "published_cross_domain_real_diagnostic_accuracy"
  ext$external_reference <- paste(
    "Kriston L, Hoelzel L, Weiser A, Berner M, Haerter M (2008).",
    "Meta-analysis: Are 3 Questions Enough to Detect Unhealthy Alcohol Use?",
    "Ann Intern Med 149:879-888."
  )
  ext$external_data_note <- "14 independent published primary-study 2x2 diagnostic tables; cross-domain generalizability only."
  ext
}

load_external_validation_source <- function(cfg) {
  mode <- as.character(cfg$external_validation_source)[1]
  if (is.na(mode) || !nzchar(mode) || identical(mode,"none")) return(NULL)

  if (identical(mode,"AuditC_builtin_verified")) {
    ext <- auditc_external_counts_builtin()
  } else if (identical(mode,"mada_AuditC")) {
    if (!requireNamespace("mada", quietly=TRUE)) {
      stop(
        "External validation source mada_AuditC was requested, but package 'mada' is not installed. ",
        "Install it BEFORE the long run with install.packages('mada')."
      )
    }
    e <- new.env(parent=emptyenv())
    utils::data(list=cfg$external_mada_dataset, package="mada", envir=e)
    if (!exists(cfg$external_mada_dataset,envir=e,inherits=FALSE)) {
      stop("Could not load mada::",cfg$external_mada_dataset," for external validation.")
    }
    ext <- as.data.frame(get(cfg$external_mada_dataset,envir=e,inherits=FALSE))
    ext$dataset <- paste0("AuditC_study_",sprintf("%02d",seq_len(nrow(ext))))
    ext$external_source <- "mada::AuditC"
    ext$external_source_type <- "published_cross_domain_real_diagnostic_accuracy"
    ext$external_reference <- paste(
      "Kriston L, Hoelzel L, Weiser A, Berner M, Haerter M (2008).",
      "Meta-analysis: Are 3 Questions Enough to Detect Unhealthy Alcohol Use?",
      "Ann Intern Med 149:879-888."
    )
    ext$mada_version <- as.character(utils::packageVersion("mada"))
  } else if (identical(mode,"file")) {
    f <- cfg$external_counts_file
    if (is.na(f) || !nzchar(f) || !file.exists(f)) stop("external_validation_source='file' requires an existing external_counts_file.")
    extn <- tolower(tools::file_ext(f))
    ext <- if (extn %in% c("xlsx","xls")) as.data.frame(readxl::read_excel(f)) else utils::read.csv(f,stringsAsFactors=FALSE)
    if (!("external_source" %in% names(ext))) ext$external_source <- normalizePath(f,winslash="/",mustWork=FALSE)
    if (!("external_source_type" %in% names(ext))) ext$external_source_type <- "user_supplied_independent_real_data"
    if (!("external_reference" %in% names(ext))) ext$external_reference <- NA_character_
  } else {
    stop("Unknown external_validation_source: ",mode)
  }

  ext <- validate_external_counts_table(ext)
  mx <- suppressWarnings(as.integer(cfg$external_max_studies))
  if (is.finite(mx) && mx>0L && nrow(ext)>mx) ext <- ext[seq_len(mx),,drop=FALSE]
  rownames(ext) <- NULL
  ext
}

external_validation_preflight <- function(cfg) {
  ext <- load_external_validation_source(cfg)
  required <- identical(cfg$run_mode,"publication") && isTRUE(cfg$external_validation_required_in_publication)
  if (is.null(ext)) {
    if (required) stop("Publication external-validation gate is enabled but no external source is configured.")
    log_message("External real-data validation not configured; limitation will be recorded.")
    return(NULL)
  }
  log_message(
    "EXTERNAL VALIDATION PREFLIGHT PASSED | source=",cfg$external_validation_source,
    " | studies=",nrow(ext)," | c_values=",paste(cfg$external_c_values,collapse=","),
    " | interpretation=cross-domain real-data generalizability (not coronary clinical transport validation)"
  )
  ext
}

primary_counts_from_df <- function(df) {
  D <- as.integer(as.character(df$D)); T <- as.integer(as.character(df$T))
  c(TP=sum(D==1&T==1),FN=sum(D==1&T==0),FP=sum(D==0&T==1),TN=sum(D==0&T==0))
}

fwkc_external_task <- function(task, cfg_ext) {
  tryCatch({
    rr <- task$row
    cc <- task$c_val
    dat <- expand_counts_to_data(rr$TP,rr$FN,rr$FP,rr$TN)
    fit <- pipeline_bootstrap_inference(
      dat,cc,cfg_ext,keep_boot=FALSE,
      seed_offset=970000 + 1000L*as.integer(task$study_index) + as.integer(round(100*cc)),
      show_boot_progress=isTRUE(cfg_ext$parallel_worker_progress),
      progress_label=paste0("External bootstrap ",rr$dataset," c=",cc)
    )
    failed <- is.null(fit) || isTRUE(fit$failed)
    row <- data.frame(
      dataset=as.character(rr$dataset), study_index=as.integer(task$study_index), c_val=cc,
      n=nrow(dat), TP=rr$TP,FN=rr$FN,FP=rr$FP,TN=rr$TN,
      external_source=if("external_source"%in%names(rr)) as.character(rr$external_source) else NA_character_,
      external_source_type=if("external_source_type"%in%names(rr)) as.character(rr$external_source_type) else NA_character_,
      external_reference=if("external_reference"%in%names(rr)) as.character(rr$external_reference) else NA_character_,
      fit_success=!failed,
      Kappa_Direct=if(failed) NA_real_ else fit$direct_point,
      Direct_Lower=if(failed) NA_real_ else fit$direct_boot_lower,
      Direct_Upper=if(failed) NA_real_ else fit$direct_boot_upper,
      Kappa_FWKC=if(failed) NA_real_ else fit$fwkc_point,
      FWKC_Lower=if(failed) NA_real_ else fit$fwkc_lower,
      FWKC_Upper=if(failed) NA_real_ else fit$fwkc_upper,
      FWKC_Calibrated_Lower=if(failed) NA_real_ else fit$fwkc_cal_lower,
      FWKC_Calibrated_Upper=if(failed) NA_real_ else fit$fwkc_cal_upper,
      PABAK=if(failed) NA_real_ else fit$PABAK,
      Gwet_AC1=if(failed) NA_real_ else fit$Gwet_AC1,
      stringsAsFactors=FALSE
    )
    list(ok=TRUE,row=row,error=NA_character_,pid=Sys.getpid())
  }, error=function(e) list(ok=FALSE,row=NULL,error=conditionMessage(e),pid=Sys.getpid()))
}

run_external_validation <- function(df, cfg) {
  ext <- load_external_validation_source(cfg)
  if (is.null(ext)) {
    log_message("External validation skipped: no independent external source configured.")
    return(NULL)
  }
  dir.create(file.path(cfg$output_dir,"external"),recursive=TRUE,showWarnings=FALSE)
  save_csv_safe(ext,file.path(cfg$output_dir,"external","external_validation_input.csv"))

  pc <- primary_counts_from_df(df)
  ext$matches_primary_counts <- apply(ext[,c("TP","FN","FP","TN"),drop=FALSE],1,function(z) all(as.numeric(z)==as.numeric(pc)))
  if (any(ext$matches_primary_counts)) {
    stop("External validation rejected: at least one external table exactly matches the primary manuscript 2x2 counts.")
  }
  provenance <- data.frame(
    source_mode=cfg$external_validation_source,
    source_label=unique(ext$external_source)[1],
    source_type=unique(ext$external_source_type)[1],
    reference=unique(ext$external_reference)[1],
    studies=nrow(ext),
    interpretation="External cross-domain real-data generalizability; not same-disease coronary clinical transport validation",
    stringsAsFactors=FALSE
  )
  save_csv_safe(provenance,file.path(cfg$output_dir,"external","external_validation_provenance.csv"))

  tasks <- list(); q <- 0L
  for (i in seq_len(nrow(ext))) for (cc in cfg$external_c_values) {
    q <- q+1L; tasks[[q]] <- list(row=ext[i,,drop=FALSE],study_index=i,c_val=cc)
  }
  cfg_ext <- cfg
  cfg_ext$B_boot <- cfg$B_boot_external
  cfg_ext$n_mc <- cfg$n_mc
  cfg_ext$checkpoint_every_boot <- as.integer(cfg$checkpoint_every_boot_external)
  cfg_ext$output_dir <- file.path(cfg$output_dir,"external")
  dir.create(file.path(cfg_ext$output_dir,"checkpoints"),recursive=TRUE,showWarnings=FALSE)

  results <- vector("list",length(tasks)); t0 <- proc.time()[["elapsed"]]
  if (parallel_stage_enabled(cfg_ext,"external",length(tasks))) {
    cl <- get_parallel_cluster(cfg_ext); workers <- resolve_parallel_workers(cfg_ext,length(tasks))
    cfg_worker <- cfg_ext; cfg_worker$parallel_enabled <- FALSE; cfg_worker$parallel_workers <- 1L
    batch_size <- max(1L,min(length(tasks),workers*max(1L,as.integer(cfg$parallel_batch_multiplier))))
    batches <- split(seq_along(tasks),ceiling(seq_along(tasks)/batch_size))
    log_message("PARALLEL STAGE START | External real-data validation | workers=",workers," | tasks=",length(tasks)," | B=",cfg_ext$B_boot)
    for (b in seq_along(batches)) {
      ids <- batches[[b]]
      raw <- parallel::parLapplyLB(cl,tasks[ids],fwkc_external_task,cfg_ext=cfg_worker)
      errors <- character(0)
      for (j in seq_along(ids)) {
        item <- raw[[j]]; ii <- ids[j]
        if (isTRUE(item$ok)) results[[ii]] <- item$row else errors <- c(errors,paste0("task ",ii,": ",item$error))
      }
      done <- results[!vapply(results,is.null,logical(1))]
      if(length(done)>0) save_csv_safe(rbind_fill(done),file.path(cfg$output_dir,"external","external_validation_IN_PROGRESS.csv"))
      progress_message("External validation [parallel]",length(done),length(tasks),cfg,t0,paste0("batch=",b,"/",length(batches)))
      if(length(errors)>0) stop("External validation parallel errors after saving completed tasks:\n- ",paste(errors,collapse="\n- "))
    }
  } else {
    for(i in seq_along(tasks)) {
      item <- fwkc_external_task(tasks[[i]],cfg_ext)
      if(!isTRUE(item$ok)) stop(item$error)
      results[[i]] <- item$row
      save_csv_safe(rbind_fill(results[seq_len(i)]),file.path(cfg$output_dir,"external","external_validation_IN_PROGRESS.csv"))
      progress_message("External validation",i,length(tasks),cfg,t0,paste0("dataset=",item$row$dataset,"; c=",item$row$c_val))
    }
  }
  ans <- rbind_fill(results)
  save_csv_safe(ans,file.path(cfg$output_dir,"external","external_validation.csv"))

  summ <- aggregate(fit_success ~ dataset,data=ans,FUN=function(z) mean(as.numeric(z),na.rm=TRUE))
  names(summ)[2] <- "fit_success_rate"
  summ$n <- vapply(summ$dataset,function(d) unique(ans$n[ans$dataset==d])[1],numeric(1))
  save_csv_safe(summ,file.path(cfg$output_dir,"external","external_validation_summary.csv"))
  log_message("EXTERNAL VALIDATION COMPLETE | studies=",length(unique(ans$dataset))," | tasks=",nrow(ans)," | min_fit_success=",signif(min(summ$fit_success_rate),4))
  ans
}

###############################################################################
# 25) PRACTICAL BENEFIT + SCALABILITY
###############################################################################

make_practical_benefit_table <- function(main_results, cfg) {
  if (is.null(main_results) || nrow(main_results) == 0) {
    return(NULL)
  }

  x <- main_results

  x$coverage_gain_fwkc_vs_direct <-
    x$coverage_fwkc -
    x$coverage_direct_boot

  x$rmse_gain_direct_minus_fwkc <-
    x$rmse_direct -
    x$rmse_fwkc

  x$extra_seconds_fwkc_vs_direct <-
    x$mean_seconds_fwkc_boot -
    x$mean_seconds_direct_boot

  x$stress_regime <- ifelse(
    x$n <= 50 &
      (x$p_true <= 0.1 | x$p_true >= 0.9),
    "small_imbalanced",
    ifelse(
      x$n <= 50,
      "small_other",
      ifelse(
        x$p_true <= 0.1 | x$p_true >= 0.9,
        "imbalanced_other",
        "other"
      )
    )
  )

  out <- aggregate(
    cbind(
      coverage_gain_fwkc_vs_direct,
      rmse_gain_direct_minus_fwkc,
      extra_seconds_fwkc_vs_direct
    ) ~ n + stress_regime,
    data = x,
    FUN = function(z) mean(z, na.rm = TRUE)
  )

  save_csv_safe(
    out,
    file.path(
      cfg$output_dir,
      "simulation",
      "practical_benefit_summary.csv"
    )
  )

  out
}

run_scalability_benchmark <- function(cfg) {
  if (!isTRUE(cfg$run_scalability)) {
    return(NULL)
  }

  rows <- list()
  k <- 0L
  scalability_total <- length(cfg$scalability_n) * cfg$scalability_reps
  scalability_start <- proc.time()[["elapsed"]]

  for (nn in cfg$scalability_n) {
    for (r in seq_len(cfg$scalability_reps)) {
      set_fwkc_seed(
        cfg$seed +
          70000000 +
          nn * 10 +
          r
      )

      dat <- simulate_dgm_iid(
        n = nn,
        p_true = 0.5,
        Se_true = 0.80,
        Sp_true = 0.85
      )

      # Benchmark configuration uses the SAME estimator architecture but a
      # deliberately small bootstrap count so runtime scaling can be measured
      # without turning the benchmark itself into the main simulation.
      cb <- cfg
      cb$B_boot <- cfg$scalability_B
      cb$n_mc <- cfg$scalability_n_mc
      cb$compute_bca <- FALSE
      cb$run_sensitivity <- FALSE
      cb$run_scalability <- FALSE

      t_direct <- proc.time()[["elapsed"]]
      direct_delta_ci(
        dat,
        c_val = 0.5,
        alpha = cfg$alpha
      )
      direct_delta_seconds <- elapsed_seconds(t_direct)

      t_core <- proc.time()[["elapsed"]]
      core <- fwkc_core(
        dat,
        c_val = 0.5,
        cfg = cb
      )
      fwkc_core_seconds <- elapsed_seconds(t_core)

      t_pipe <- proc.time()[["elapsed"]]
      full <- pipeline_bootstrap_inference(
        df_in = dat,
        c_val = 0.5,
        cfg = cb,
        keep_boot = FALSE,
        seed_offset =
          80000000 +
          nn * 10 +
          r
      )
      full_seconds <- elapsed_seconds(t_pipe)

      k <- k + 1L
      rows[[k]] <- data.frame(
        n = nn,
        replicate = r,
        bayes_engine = cb$bayes_engine,
        B_boot = cb$B_boot,
        n_mc = cb$n_mc,
        direct_delta_seconds =
          direct_delta_seconds,
        fwkc_core_seconds =
          fwkc_core_seconds,
        fwkc_full_bootstrap_seconds =
          full_seconds,
        fwkc_point =
          if (is.null(core)) NA_real_ else core$point,
        pipeline_failed =
          is.null(full) || isTRUE(full$failed),
        stringsAsFactors = FALSE
      )

      save_csv_safe(
        rbind_fill(rows),
        file.path(
          cfg$output_dir,
          "simulation",
          "scalability_detail_IN_PROGRESS.csv"
        )
      )

      progress_message(
        stage = "Scalability benchmark",
        current = k,
        total = scalability_total,
        cfg = cfg,
        stage_start = scalability_start,
        detail = paste0("n=", nn, "; replicate=", r)
      )
    }
  }

  detail <- rbind_fill(rows)

  summary <- aggregate(
    cbind(
      direct_delta_seconds,
      fwkc_core_seconds,
      fwkc_full_bootstrap_seconds
    ) ~ n,
    data = detail,
    FUN = function(z) mean(z, na.rm = TRUE)
  )

  save_csv_safe(
    detail,
    file.path(
      cfg$output_dir,
      "simulation",
      "scalability_detail.csv"
    )
  )

  save_csv_safe(
    summary,
    file.path(
      cfg$output_dir,
      "simulation",
      "scalability_summary.csv"
    )
  )

  list(
    detail = detail,
    summary = summary
  )
}

###############################################################################
# 25B) PUBLICATION QUALITY-CONTROL SUMMARY
###############################################################################

build_publication_qc <- function(
  real_data, main_results, anchor_bridge, anchor_sample_check, robustness,
  sensitivity, external, bayes_crosscheck, cfg
) {
  checks <- list(); k <- 0L
  finite_min <- function(x) { x<-x[is.finite(x)]; if(length(x)==0) NA_real_ else min(x) }
  finite_max <- function(x) { x<-x[is.finite(x)]; if(length(x)==0) NA_real_ else max(x) }
  add_check <- function(name,value,threshold,pass,note) {
    k <<- k+1L
    checks[[k]] <<- data.frame(
      check=name, value=as.character(value), threshold=as.character(threshold),
      pass=isTRUE(pass), note=note, stringsAsFactors=FALSE
    )
  }

  add_check("run_mode_publication",cfg$run_mode,"publication",
            identical(cfg$run_mode,"publication"),"Only publication preset is manuscript-final.")

  pplan <- parallel_runtime_plan(cfg)
  add_check(
    "parallel_execution_is_computational_only",
    paste0("mode=",pplan$mode[1],"; workers=",pplan$effective_scenario_workers[1]),
    "statistical_design_changed=FALSE",
    identical(pplan$statistical_design_changed[1], FALSE),
    "PSOCK scheduling may change wall-clock time but must not change estimand, M, B, n_mc, chains, DGMs or QC."
  )

  bx_ok <- !is.null(bayes_crosscheck) && nrow(bayes_crosscheck)>0 &&
    "pass" %in% names(bayes_crosscheck) && all(bayes_crosscheck$pass)
  add_check("bayes_engine_crosscheck",if(bx_ok)"all_pass" else "not_all_pass","all pass",
            bx_ok,"Blocked Gibbs and independent same-prior collapsed CmdStan HMC must agree within the prespecified tolerance, R-hat gate and required sampler diagnostics.")

  bx_div <- if(!is.null(bayes_crosscheck) && nrow(bayes_crosscheck)>0 &&
               "brms_divergent_transitions" %in% names(bayes_crosscheck)) {
    sum(bayes_crosscheck$brms_divergent_transitions, na.rm=TRUE)
  } else NA_real_
  add_check(
    "bayes_crosscheck_divergent_transitions",
    bx_div,
    if(isTRUE(cfg$bayes_crosscheck_require_zero_divergences)) "0" else "diagnostic_only",
    if(isTRUE(cfg$bayes_crosscheck_require_zero_divergences)) is.finite(bx_div) && bx_div==0 else TRUE,
    "Divergent transitions are recorded explicitly; publication requires zero when configured."
  )

  bx_td <- if(!is.null(bayes_crosscheck) && nrow(bayes_crosscheck)>0 &&
              "brms_max_treedepth_hits" %in% names(bayes_crosscheck)) {
    sum(bayes_crosscheck$brms_max_treedepth_hits, na.rm=TRUE)
  } else NA_real_
  add_check(
    "bayes_crosscheck_max_treedepth_hits",
    bx_td,
    if(isTRUE(cfg$bayes_crosscheck_require_zero_max_treedepth)) "0" else "diagnostic_only",
    if(isTRUE(cfg$bayes_crosscheck_require_zero_max_treedepth)) is.finite(bx_td) && bx_td==0 else TRUE,
    "Max-treedepth hits are reported separately; they are a hard gate only when configured."
  )

  rr <- if(!is.null(real_data)) real_data$results else NULL
  if(!is.null(rr) && nrow(rr)>0) {
    direct_ok <- if("Direct_failed" %in% names(rr)) mean(!rr$Direct_failed,na.rm=TRUE) else NA_real_
    fwkc_ok <- if("FWKC_failed" %in% names(rr)) mean(!rr$FWKC_failed,na.rm=TRUE) else NA_real_
    add_check("real_data_direct_success",direct_ok,"1.00",is.finite(direct_ok)&&direct_ok==1,
              "Every reported c should have valid Direct inference.")
    add_check("real_data_fwkc_success",fwkc_ok,"1.00",is.finite(fwkc_ok)&&fwkc_ok==1,
              "Every reported c should have valid FWKC inference or be explicitly flagged.")
  }

  full_grid_ok <- !is.null(main_results) && nrow(main_results)>=243 &&
    setequal(round(unique(main_results$p_true),10),round(seq(0.1,0.9,0.1),10)) &&
    setequal(round(unique(main_results$c_val),10),round(seq(0.1,0.9,0.1),10)) &&
    setequal(unique(main_results$n),c(30,50,100))
  add_check("complete_primary_grid",if(full_grid_ok)"243_or_more" else "incomplete","243 anchor scenarios",
            full_grid_ok,"Primary continuity analysis must preserve the submitted full p-by-c grid.")

  anchor_n_ok <- !is.null(anchor_sample_check) && nrow(anchor_sample_check)>0 &&
    isTRUE(anchor_sample_check$exact_match[1])
  add_check(
    "manuscript_anchor_sample_size",
    if(is.null(anchor_sample_check)) "not_checked" else anchor_sample_check$observed_input_n[1],
    cfg$manuscript_anchor_n,
    anchor_n_ok,
    "Publication input must match the n=620 total reported in manuscript Table 2."
  )

  abr <- if(!is.null(anchor_bridge) && is.list(anchor_bridge)) anchor_bridge$results else NULL
  bridge_ok <- !is.null(abr) && nrow(abr)>0 &&
    all(abr$n == cfg$manuscript_anchor_n) &&
    setequal(round(unique(abr$c_val),10), round(cfg$anchor_bridge_c_values,10))
  add_check(
    "anchor_sample_size_bridge_present",
    if(bridge_ok) paste0("n=",cfg$manuscript_anchor_n,"; scenarios=",nrow(abr)) else "missing_or_incomplete",
    "observed n + full c-grid",
    bridge_ok,
    "A separate large-sample bridge links the simulation framework directly to the real-data anchor."
  )

  anchor_lambda_se <- safe_df_numeric(abr,"mean_lambda_se_sqrt")
  anchor_lambda_var <- safe_df_numeric(abr,"mean_lambda_variance_linear")
  bridge_lambda_ok <- bridge_ok &&
    length(anchor_lambda_se)==nrow(abr) && length(anchor_lambda_var)==nrow(abr) &&
    all(abs(anchor_lambda_se-1)<=1e-12) && all(abs(anchor_lambda_var-1)<=1e-12)
  lambda_value <- if (length(anchor_lambda_se)>0L) {
    paste(signif(range(anchor_lambda_se),6),collapse=" to ")
  } else "NO_FINITE_VALUES"
  add_check(
    "anchor_size_fixed_calibration_inactive",lambda_value,
    "finite lambda_SE=lambda_VAR=1 for every anchor scenario",
    bridge_lambda_ok,
    "At n=620>n0, calibration must be finite and exactly inactive; empty/all-NA values are an explicit FAIL."
  )
  anchor_min_direct <- safe_df_min(abr,"direct_dataset_success_rate")
  anchor_min_fwkc <- safe_df_min(abr,"fwkc_dataset_success_rate")
  add_check(
    "anchor_inference_success",
    paste0("direct=",signif(anchor_min_direct,4),"; fwkc=",signif(anchor_min_fwkc,4)),
    ">=0.90 for both",
    is.finite(anchor_min_direct)&&anchor_min_direct>=0.90&&is.finite(anchor_min_fwkc)&&anchor_min_fwkc>=0.90,
    "Anchor bridge must contain genuinely valid inferential results, not merely completed scenario objects."
  )

  if(!is.null(main_results) && nrow(main_results)>0) {
    min_direct <- finite_min(main_results$direct_dataset_success_rate)
    min_fwkc <- finite_min(main_results$fwkc_dataset_success_rate)
    max_mcse_f <- finite_max(main_results$mcse_fwkc)
    max_mcse_d <- finite_max(main_results$mcse_direct_boot)
    min_conv <- finite_min(main_results$bayes_convergence_rate)
    add_check("minimum_direct_dataset_success_rate",min_direct,">=0.90",
              is.finite(min_direct)&&min_direct>=0.90,"Direct benchmark validity must not be hidden.")
    add_check("minimum_fwkc_dataset_success_rate",min_fwkc,">=0.90",
              is.finite(min_fwkc)&&min_fwkc>=0.90,"Low FWKC success is a reportable limitation.")
    add_check("maximum_fwkc_coverage_mcse",max_mcse_f,"<=0.02",
              is.finite(max_mcse_f)&&max_mcse_f<=0.02,"Coverage requires adequate Monte Carlo precision.")
    add_check("maximum_direct_coverage_mcse",max_mcse_d,"<=0.02",
              is.finite(max_mcse_d)&&max_mcse_d<=0.02,"Coverage requires adequate Monte Carlo precision.")
    add_check("minimum_bayesian_convergence_rate",min_conv,">=0.95",
              is.finite(min_conv)&&min_conv>=0.95,"Nonconverged Bayesian fits count as FWKC failures.")
  }

  robust_types <- if(!is.null(robustness)&&nrow(robustness)>0) unique(robustness$dgm_type) else character(0)
  add_check("noncircular_robustness_present",length(robust_types),">=3 structural DGM types",
            length(robust_types)>=3,"Reviewer-facing robustness must extend beyond the real-data anchor.")

  required_sensitivity_labels <- c(
    "prior_jeffreys","prior_uniform","prior_legacy","uncertainty_beta_posterior","uncertainty_empirical_bootstrap",
    "pseudo_posterior_predictive","pseudo_parameter_only","fusion_inverse_variance","fusion_inverse_se","fusion_equal",
    "fusion_bayes_only","fusion_ml_only","transform_logit11","transform_fisher_z","transform_identity",
    "ml_rf","ml_spline","ml_none","calibration_none","calibration_sqrt_se_expand","calibration_linear_se_expand",
    "calibration_adaptive_sqrt_expand","calibration_legacy_linear_contract","n0_75","n0_100","n0_125","n0_150",
    "nmc_250","nmc_500","nmc_1000","nmc_2000","bboot_100","bboot_200","bboot_500"
  )
  sens_labels <- if(!is.null(sensitivity)&&"sensitivity_label"%in%names(sensitivity)) unique(as.character(sensitivity$sensitivity_label)) else character(0)
  sens_labels_ok <- all(required_sensitivity_labels %in% sens_labels)
  sens_mvalid <- safe_df_numeric(sensitivity,"M_valid")
  sens_direct <- safe_df_numeric(sensitivity,"direct_dataset_success_rate")
  sens_fwkc <- safe_df_numeric(sensitivity,"fwkc_dataset_success_rate")
  sens_valid_ok <- !is.null(sensitivity)&&nrow(sensitivity)>0 &&
    length(sens_mvalid)==nrow(sensitivity) && all(sens_mvalid>0) &&
    length(sens_direct)==nrow(sensitivity) && length(sens_fwkc)==nrow(sensitivity) &&
    all(sens_direct>=0.90) && all(sens_fwkc>=0.90)
  add_check("sensitivity_variant_set_complete",paste0(length(sens_labels)," labels"),paste0(length(required_sensitivity_labels)," required"),
            sens_labels_ok,"All prespecified sensitivity families must be represented; row count alone is insufficient.")
  add_check("sensitivity_valid_inference",
            if(length(sens_mvalid)) paste0("min_M_valid=",min(sens_mvalid),"; min_direct=",signif(min(sens_direct),4),"; min_fwkc=",signif(min(sens_fwkc),4)) else "NO_VALID_ROWS",
            "all rows M_valid>0 and success rates>=0.90",
            sens_valid_ok,
            "All-NA/zero-valid sensitivity rows are now an explicit FAIL rather than a false PASS.")

  # V25 formal n_mc=500 stability gate. The primary integration count is accepted
  # only if prespecified stress scenarios remain stable relative to n_mc=1000 and
  # n_mc=2000. Pilot/test outputs are never used here; this uses the current run.
  nmc_stability_ok <- FALSE
  nmc_stability_value <- "UNAVAILABLE"
  if (!is.null(sensitivity) && nrow(sensitivity) > 0 &&
      all(c("sensitivity_label","n","p_true","c_val","bias_fwkc","rmse_fwkc","mae_fwkc",
            "fwkc_dataset_success_rate") %in% names(sensitivity))) {
    base500 <- sensitivity[sensitivity$sensitivity_label=="nmc_500",,drop=FALSE]
    cmp_list <- list()
    kk <- 0L
    for (lab in c("nmc_1000","nmc_2000")) {
      ref <- sensitivity[sensitivity$sensitivity_label==lab,,drop=FALSE]
      if (nrow(base500)>0 && nrow(ref)>0) {
        for (ii in seq_len(nrow(base500))) {
          hit <- which(ref$n==base500$n[ii] &
                       abs(ref$p_true-base500$p_true[ii])<=1e-12 &
                       abs(ref$c_val-base500$c_val[ii])<=1e-12)
          if (length(hit)>0L) {
            jj <- hit[1L]
            kk <- kk + 1L
            cmp_list[[kk]] <- data.frame(
              reference=lab,
              n=base500$n[ii], p_true=base500$p_true[ii], c_val=base500$c_val[ii],
              abs_bias_diff=abs(base500$bias_fwkc[ii]-ref$bias_fwkc[jj]),
              abs_rmse_diff=abs(base500$rmse_fwkc[ii]-ref$rmse_fwkc[jj]),
              abs_mae_diff=abs(base500$mae_fwkc[ii]-ref$mae_fwkc[jj]),
              min_success=min(base500$fwkc_dataset_success_rate[ii],ref$fwkc_dataset_success_rate[jj],na.rm=TRUE),
              stringsAsFactors=FALSE
            )
          }
        }
      }
    }
    cmp <- if(length(cmp_list)) rbind_fill(cmp_list) else NULL
    if (!is.null(cmp) && nrow(cmp)>0) {
      max_bias_diff <- finite_max(cmp$abs_bias_diff)
      max_rmse_diff <- finite_max(cmp$abs_rmse_diff)
      max_mae_diff <- finite_max(cmp$abs_mae_diff)
      min_nmc_success <- finite_min(cmp$min_success)
      nmc_stability_ok <- is.finite(max_bias_diff) &&
        max_bias_diff <= cfg$nmc_stability_max_abs_bias_diff &&
        is.finite(max_rmse_diff) &&
        max_rmse_diff <= cfg$nmc_stability_max_abs_rmse_diff &&
        is.finite(max_mae_diff) &&
        max_mae_diff <= cfg$nmc_stability_max_abs_mae_diff &&
        is.finite(min_nmc_success) && min_nmc_success >= 0.90
      nmc_stability_value <- paste0(
        "max|bias|diff=",signif(max_bias_diff,4),
        "; max|RMSE|diff=",signif(max_rmse_diff,4),
        "; max|MAE|diff=",signif(max_mae_diff,4),
        "; min_success=",signif(min_nmc_success,4)
      )
      save_csv_safe(cmp,file.path(cfg$output_dir,"sensitivity","nmc_500_stability_audit.csv"))
    }
  }
  add_check(
    "nmc_500_primary_stability",
    nmc_stability_value,
    paste0("|bias diff|<=",cfg$nmc_stability_max_abs_bias_diff,
           "; |RMSE diff|<=",cfg$nmc_stability_max_abs_rmse_diff,
           "; |MAE diff|<=",cfg$nmc_stability_max_abs_mae_diff,
           "; success>=0.90 vs n_mc=1000/2000"),
    if (identical(cfg$publication_profile,"strict_max_precision")) TRUE else nmc_stability_ok,
    "The balanced-official profile may use n_mc=500 only when the prespecified higher-n_mc sensitivity audit confirms practical numerical stability."
  )

  ext_ok <- !is.null(external) && nrow(external)>0 && all(c("dataset","c_val","fit_success") %in% names(external))
  ext_datasets <- if(ext_ok) unique(external$dataset) else character(0)
  ext_c_ok <- ext_ok && all(vapply(ext_datasets,function(d) {
    setequal(round(unique(external$c_val[external$dataset==d]),10),round(cfg$external_c_values,10))
  },logical(1)))
  ext_success <- if(ext_ok) aggregate(fit_success~dataset,data=external,FUN=function(z) mean(as.numeric(z),na.rm=TRUE)) else NULL
  ext_min_success <- if(!is.null(ext_success)&&nrow(ext_success)>0) min(ext_success$fit_success,na.rm=TRUE) else NA_real_
  ext_pass <- ext_ok && ext_c_ok && is.finite(ext_min_success) && ext_min_success>=cfg$external_min_dataset_success_rate
  ext_source_configured <- !is.null(cfg$external_validation_source) &&
    !is.na(cfg$external_validation_source) &&
    nzchar(as.character(cfg$external_validation_source)) &&
    !identical(as.character(cfg$external_validation_source),"none")
  ext_gate_pass <- if (ext_source_configured) {
    ext_pass
  } else {
    !(identical(cfg$run_mode,"publication") && isTRUE(cfg$external_validation_required_in_publication))
  }
  add_check(
    "external_real_data_generalizability",
    if(ext_ok) paste0("datasets=",length(ext_datasets),"; min_success=",signif(ext_min_success,4)) else "NOT_RUN",
    paste0("configured external source must complete full c-grid with success>=",cfg$external_min_dataset_success_rate),
    ext_gate_pass,
    "A configured external source must pass even in test mode. Cross-domain published external real diagnostic data strengthen generalizability but do not constitute same-disease coronary clinical transport validation."
  )

  secondary_precision_ok <-
    cfg$anchor_bridge_M >= cfg$publication_min_anchor_M &&
    cfg$anchor_bridge_B >= cfg$publication_min_anchor_B &&
    cfg$anchor_bridge_n_mc >= cfg$publication_min_anchor_n_mc &&
    cfg$robustness_M >= cfg$publication_min_robustness_M &&
    cfg$robustness_B >= cfg$publication_min_robustness_B &&
    cfg$sensitivity_M >= cfg$publication_min_sensitivity_M &&
    cfg$sensitivity_B >= cfg$publication_min_sensitivity_B &&
    cfg$sensitivity_n_mc >= cfg$publication_min_sensitivity_n_mc &&
    cfg$sensitivity_scenarios >= cfg$publication_min_sensitivity_scenarios &&
    cfg$B_boot_external >= cfg$publication_min_B_external &&
    cfg$covariance_boot_B >= cfg$publication_min_covariance_B
  add_check(
    "secondary_publication_precision",
    paste0("anchor=",cfg$anchor_bridge_M,"/",cfg$anchor_bridge_B,"/",cfg$anchor_bridge_n_mc,
           "; robust=",cfg$robustness_M,"/",cfg$robustness_B,
           "; sensitivity=",cfg$sensitivity_M,"/",cfg$sensitivity_B,"/",cfg$sensitivity_n_mc,
           "/scen=",cfg$sensitivity_scenarios,
           "; externalB=",cfg$B_boot_external,"; covB=",cfg$covariance_boot_B),
    "meets profile-specific secondary minima",
    secondary_precision_ok,
    "Secondary/robustness layers may be lighter than the primary coverage experiment, but their manuscript-grade minima are still enforced."
  )

  precision_ok <- cfg$M_dgm>=cfg$publication_min_M &&
    cfg$B_boot_dgm>=cfg$publication_min_B_dgm &&
    cfg$B_boot_real>=cfg$publication_min_B_real &&
    cfg$n_mc>=cfg$publication_min_n_mc
  add_check("publication_precision_preset",
            paste(cfg$M_dgm,cfg$B_boot_dgm,cfg$B_boot_real,cfg$n_mc,sep="/"),
            paste(cfg$publication_min_M,cfg$publication_min_B_dgm,cfg$publication_min_B_real,cfg$publication_min_n_mc,sep="/"),
            precision_ok,"Do not label reduced development runs as manuscript-final.")

  out <- rbind_fill(checks)
  save_csv_safe(out,file.path(cfg$output_dir,"PUBLICATION_QC.csv"))
  out
}

write_reportability_gate <- function(publication_qc, cfg) {
  ready_path <- file.path(cfg$output_dir, "PUBLICATION_READY.flag")
  fail_path <- file.path(cfg$output_dir, "RESULTS_NOT_REPORTABLE.flag")

  if (file.exists(ready_path)) file.remove(ready_path)
  if (file.exists(fail_path)) file.remove(fail_path)

  pass_all <- !is.null(publication_qc) &&
    nrow(publication_qc) > 0 &&
    all(publication_qc$pass %in% TRUE)

  stamp <- paste0(
    "run_mode=", cfg$run_mode, "\n",
    "timestamp=", format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"), "\n",
    "all_publication_qc_pass=", pass_all, "\n"
  )

  if (pass_all) {
    safe_write_lines(
      c(
        "FWKC RESULTS PASSED ALL PROGRAMMATIC PUBLICATION QC GATES.",
        "This flag does not replace scientific/editorial peer review.",
        stamp
      ),
      ready_path
    )
  } else {
    failed_names <- if (!is.null(publication_qc) && nrow(publication_qc) > 0) {
      publication_qc$check[!publication_qc$pass]
    } else {
      "publication_qc_unavailable"
    }
    safe_write_lines(
      c(
        "FWKC RESULTS ARE NOT REPORTABLE UNDER THE PRESPECIFIED QC GATES.",
        paste0("Failed checks: ", paste(failed_names, collapse = ", ")),
        stamp
      ),
      fail_path
    )
  }

  pass_all
}

###############################################################################
# 26) PLOTS
###############################################################################

make_reviewer_readiness_report <- function(
  cfg, validation, main_results, anchor_bridge, anchor_sample_check,
  sample_size_bridge_comparison = NULL,
  robustness, sensitivity, external, bayes_crosscheck, publication_qc
) {
  matrix_ok <- !is.null(validation$max_algebraic_difference) &&
    is.finite(validation$max_algebraic_difference) &&
    validation$max_algebraic_difference <= 1e-10

  full_grid_ok <- !is.null(main_results) && nrow(main_results) >= 243 &&
    setequal(round(unique(main_results$p_true),10), round(seq(0.1,0.9,0.1),10)) &&
    setequal(round(unique(main_results$c_val),10), round(seq(0.1,0.9,0.1),10)) &&
    setequal(unique(main_results$n), c(30,50,100))

  dgm_types <- unique(c(
    if (!is.null(main_results) && nrow(main_results)>0) main_results$dgm_type else character(0),
    if (!is.null(robustness) && nrow(robustness)>0) robustness$dgm_type else character(0)
  ))
  multiple_dgm_ok <- length(dgm_types) >= 3
  sensitivity_ok <- !is.null(sensitivity) && nrow(sensitivity) > 0 &&
    "M_valid" %in% names(sensitivity) && all(is.finite(sensitivity$M_valid)) && all(sensitivity$M_valid > 0)
  shrinkage_ok <- !is.null(validation$shrinkage_check) &&
    max(validation$shrinkage_check$identity_difference, na.rm=TRUE) <= 1e-12
  crosscheck_ok <- !is.null(bayes_crosscheck) && nrow(bayes_crosscheck)>0 &&
    "pass" %in% names(bayes_crosscheck) && all(bayes_crosscheck$pass)
  qc_ok <- !is.null(publication_qc) && nrow(publication_qc)>0 &&
    all(publication_qc$pass %in% TRUE)
  external_ok <- !is.null(external) && nrow(external)>0 &&
    all(c("dataset","c_val","fit_success") %in% names(external)) &&
    all(is.finite(as.numeric(external$fit_success))) &&
    all(as.logical(external$fit_success))
  anchor_n_ok <- !is.null(anchor_sample_check) && nrow(anchor_sample_check)>0 &&
    isTRUE(anchor_sample_check$exact_match[1])
  anchor_bridge_results <- if(!is.null(anchor_bridge) && is.list(anchor_bridge)) anchor_bridge$results else NULL
  anchor_bridge_ok <- !is.null(anchor_bridge_results) && nrow(anchor_bridge_results)>0 &&
    all(anchor_bridge_results$n == cfg$manuscript_anchor_n)

  out <- data.frame(
    reviewer_issue = c(
      "Weighted-kappa estimand / weight-matrix derivation",
      "Proper Beta-binomial Se/Sp uncertainty",
      "Independent-DGM frequentist coverage",
      "Complete submitted 9x9 p-by-c grid at n=30,50,100",
      "Direct benchmark independent of FWKC success",
      "Bias/RMSE/MAE by estimator stage",
      "Multiple non-circular / heterogeneous DGMs",
      "Fusion/transformation/ML/prior/calibration sensitivity",
      "Pseudo-rater covariance + full-pipeline bootstrap",
      "Finite-sample calibration scale identity",
      "Bayesian convergence and same-prior Gibbs-vs-collapsed-CmdStan-HMC cross-check",
      "Computational cost/scalability/practical benefit",
      "Prevalence-robust descriptive comparators",
      "Preliminary multiclass/ordinal extension path",
      "Manuscript Table 2 real-data sample size and anchor-size bridge",
      "External independent real-data validation",
      "Programmatic publication QC gate"
    ),
    code_status = c(
      if (matrix_ok) "PASS" else "CHECK",
      if (cfg$prior_type=="jeffreys" && cfg$uncertainty_generator=="beta_posterior") "PASS" else "CHECK",
      if (!is.null(main_results) && nrow(main_results)>0) "PASS" else "CHECK",
      if (full_grid_ok) "PASS" else "CHECK_OR_TRUNCATED_RUN",
      if (all(c("direct_dataset_success_rate","fwkc_dataset_success_rate") %in% names(main_results))) "PASS" else "CHECK",
      if (all(c("bias_direct","bias_bayes","bias_ml","bias_fwkc","rmse_direct","rmse_fwkc") %in% names(main_results))) "PASS" else "CHECK",
      if (multiple_dgm_ok) "PASS" else "CHECK",
      if (sensitivity_ok) "PASS" else "SKIPPED",
      if (all(c("mean_component_boot_cov","mean_pseudo_se_bootstrap") %in% names(main_results))) "PASS" else "CHECK",
      if (shrinkage_ok) "PASS" else "CHECK",
      if (crosscheck_ok) "PASS" else "CHECK",
      if (isTRUE(cfg$run_scalability)) "PASS" else "CHECK",
      if (all(c("mean_PABAK","mean_Gwet_AC1") %in% names(main_results))) "PASS" else "CHECK",
      "PASS_PRELIMINARY_ONLY",
      if (anchor_n_ok && anchor_bridge_ok) "PASS" else "CHECK",
      if (external_ok) "PASS_CROSS_DOMAIN_EXTERNAL_REAL_DATA" else "NOT_SUPPLIED",
      if (qc_ok) "PASS" else "CHECK"
    ),
    manuscript_action = c(
      "Show W=[[1,c],[1-c,1]] and algebraic equivalence in Methods/response.",
      "Describe Jeffreys posterior as primary and empirical/bootstrap/Uniform/legacy as sensitivity/audit.",
      "Report coverage only across independent DGM datasets.",
      "Retain the full continuity grid and label the separate robustness tier explicitly.",
      "Report method-specific valid counts/success rates; do not condition Direct on FWKC success.",
      "Report bias and RMSE with coverage; avoid variance-only superiority claims.",
      "Describe iid, clustered-beta and mixture-tradeoff misspecification mechanisms.",
      "Use sensitivity outputs to answer robustness concerns without post-hoc selection.",
      "Explain covariance diagnostics and full-pipeline bootstrap propagation.",
      "Distinguish lambda_var from lambda_se; calibration remains secondary.",
      "Report R-hat criteria, same-prior Gibbs-vs-collapsed-CmdStan-HMC agreement, divergent transitions, ESS and max-treedepth diagnostics from the predeclared engine cross-check.",
      "Report runtime/scalability and practical-benefit trade-offs.",
      "Treat PABAK/Gwet AC1 as descriptive comparators, not estimators of kappa(c).",
      "State that multiclass/ordinal code is a preliminary estimand extension, not a validated FWKC architecture.",
      "Report Table 2 n=620 explicitly and describe the separate observed-prevalence anchor-size bridge; retain n=30,50,100 as the primary finite-sample grid.",
      "Report the published AuditC analysis as cross-domain external real-data generalizability; do not describe it as coronary-disease clinical transport validation.",
      "Use PUBLICATION_READY.flag only after all prespecified QC checks pass."
    ),
    stringsAsFactors = FALSE
  )
  save_csv_safe(out, file.path(cfg$output_dir,"validation","reviewer_readiness.csv"))
  out
}

make_plots <- function(main_results, cfg) {
  if (
    !isTRUE(cfg$make_plots) ||
      is.null(main_results) ||
      nrow(main_results) == 0
  ) {
    return(invisible(NULL))
  }

  nvals <- sort(unique(main_results$n))

  aggregate_by_n <- function(varname) {
    vapply(
      nvals,
      function(nn) {
        x <- main_results[
          main_results$n == nn,
          varname
        ]
        if (all(!is.finite(x))) NA_real_ else mean(x, na.rm = TRUE)
      },
      numeric(1)
    )
  }

  # Coverage
  png(
    file.path(cfg$output_dir, "plots", "coverage_by_n.png"),
    width = 1200,
    height = 800,
    res = 120
  )

  matplot(
    nvals,
    cbind(
      aggregate_by_n("coverage_direct_boot"),
      aggregate_by_n("coverage_direct_delta"),
      aggregate_by_n("coverage_direct_logit"),
      aggregate_by_n("coverage_direct_bca"),
      aggregate_by_n("coverage_fwkc"),
      aggregate_by_n("coverage_fwkc_calibrated")
    ),
    type = "b",
    pch = c(1, 2, 3, 4, 19, 17),
    lty = 1,
    xlab = "Sample size (n)",
    ylab = "Mean coverage",
    ylim = c(0, 1)
  )

  abline(h = 0.95, lty = 2)

  legend(
    "bottomright",
    legend = c(
      "Direct percentile bootstrap",
      "Direct delta/Wald",
      "Direct transformed-logit",
      "Direct BCa",
      "FWKC primary",
      "FWKC calibrated sensitivity",
      "Nominal 0.95"
    ),
    pch = c(1, 2, 3, 4, 19, 17, NA),
    lty = c(1, 1, 1, 1, 1, 1, 2),
    bty = "n"
  )

  dev.off()

  # RMSE
  png(
    file.path(cfg$output_dir, "plots", "rmse_by_n.png"),
    width = 1200,
    height = 800,
    res = 120
  )

  matplot(
    nvals,
    cbind(
      aggregate_by_n("rmse_direct"),
      aggregate_by_n("rmse_fwkc")
    ),
    type = "b",
    pch = c(1, 19),
    lty = 1,
    xlab = "Sample size (n)",
    ylab = "Mean RMSE"
  )

  legend(
    "topright",
    legend = c("Direct", "FWKC"),
    pch = c(1, 19),
    lty = 1,
    bty = "n"
  )

  dev.off()

  # CI width
  png(
    file.path(cfg$output_dir, "plots", "ci_width_by_n.png"),
    width = 1200,
    height = 800,
    res = 120
  )

  matplot(
    nvals,
    cbind(
      aggregate_by_n("mean_width_direct_boot"),
      aggregate_by_n("mean_width_fwkc"),
      aggregate_by_n("mean_width_fwkc_calibrated")
    ),
    type = "b",
    pch = c(1, 19, 17),
    lty = 1,
    xlab = "Sample size (n)",
    ylab = "Mean confidence-interval width"
  )

  legend(
    "topright",
    legend = c(
      "Direct bootstrap",
      "FWKC primary",
      "FWKC calibrated"
    ),
    pch = c(1, 19, 17),
    lty = 1,
    bty = "n"
  )

  dev.off()

  # Runtime
  png(
    file.path(cfg$output_dir, "plots", "runtime_by_n.png"),
    width = 1200,
    height = 800,
    res = 120
  )

  matplot(
    nvals,
    cbind(
      aggregate_by_n("mean_seconds_direct_boot"),
      aggregate_by_n("mean_seconds_fwkc_boot")
    ),
    type = "b",
    pch = c(1, 19),
    lty = 1,
    xlab = "Sample size (n)",
    ylab = "Mean seconds per independent DGM dataset"
  )

  legend(
    "topleft",
    legend = c("Direct bootstrap", "FWKC bootstrap"),
    pch = c(1, 19),
    lty = 1,
    bty = "n"
  )

  dev.off()


  # Coverage-gain heatmaps by sample size, averaged over DGM families.
  for (nn in sort(unique(main_results$n))) {
    sub <- main_results[
      main_results$n == nn,
      ,
      drop = FALSE
    ]

    ps <- sort(unique(sub$p_true))
    cs <- sort(unique(sub$c_val))

    Z <- matrix(
      NA_real_,
      nrow = length(ps),
      ncol = length(cs)
    )

    for (ip in seq_along(ps)) {
      for (ic in seq_along(cs)) {
        z <- sub[
          sub$p_true == ps[ip] &
            sub$c_val == cs[ic],
          ,
          drop = FALSE
        ]

        if (nrow(z) > 0) {
          Z[ip, ic] <- mean(
            z$coverage_fwkc -
              z$coverage_direct_boot,
            na.rm = TRUE
          )
        }
      }
    }

    finite_z <- Z[is.finite(Z)]

    # Verification mode intentionally runs only a few scenarios. It may
    # therefore contain only one prevalence or one c value for a given n.
    # image()/contour() require a genuine 2-D grid, so skip the heatmap when
    # the grid is not large enough instead of terminating the full analysis.
    if (
      length(ps) < 2 ||
        length(cs) < 2 ||
        length(finite_z) < 4
    ) {
      log_message(
        "Coverage-gain heatmap skipped for n=",
        nn,
        ": insufficient p-by-c grid in run_mode=",
        cfg$run_mode,
        "."
      )
      next
    }

    png(
      file.path(
        cfg$output_dir,
        "plots",
        sprintf(
          "coverage_gain_heatmap_n%s.png",
          nn
        )
      ),
      width = 1100,
      height = 800,
      res = 120
    )

    image(
      x = cs,
      y = ps,
      z = t(Z),
      xlab = "Weight index c",
      ylab = "Disease prevalence",
      main = paste(
        "FWKC - direct bootstrap coverage gain, n =",
        nn
      )
    )

    # A constant surface has no contour levels. image() is still valid, but
    # contour() should be omitted in that case.
    if (
      length(finite_z) >= 4 &&
        diff(range(finite_z)) > 0
    ) {
      contour(
        x = cs,
        y = ps,
        z = t(Z),
        add = TRUE,
        drawlabels = TRUE
      )
    }

    dev.off()
  }

  invisible(NULL)
}

###############################################################################
# 26B) BAYES-ENGINE CROSS-CHECK + REPRODUCIBILITY METADATA
###############################################################################


###############################################################################
# 26B-V25) SAME-PRIOR COLLAPSED CMDSTAN HMC CROSS-CHECK + LIVE DIAGNOSTICS
###############################################################################

.fwkc_collapsed_hmc_state <- new.env(parent = emptyenv())

collapsed_cmdstan_model_code <- function() {
  c(
    "data {",
    "  int<lower=2> M;",
    "  int<lower=2> R;",
    "  matrix[M,R] Z;",
    "  real mu0;",
    "  real<lower=0> mu_sd;",
    "  real<lower=0> a_sigma;",
    "  real<lower=0> b_sigma;",
    "  real<lower=0> a_tau;",
    "  real<lower=0> b_tau;",
    "}",
    "parameters {",
    "  real mu;",
    "  vector[R] u_raw;",
    "  real<lower=0> sigma2;",
    "  real<lower=0> tau_r2;",
    "  real<lower=0> tau_mc2;",
    "}",
    "transformed parameters {",
    "  vector[R] u = sqrt(tau_r2) * u_raw;",
    "}",
    "model {",
    "  mu ~ normal(mu0, mu_sd);",
    "  u_raw ~ std_normal();",
    "  sigma2 ~ inv_gamma(a_sigma, b_sigma);",
    "  tau_r2 ~ inv_gamma(a_tau, b_tau);",
    "  tau_mc2 ~ inv_gamma(a_tau, b_tau);",
    "  {",
    "    real s2 = sigma2;",
    "    real tm = tau_mc2;",
    "    real denom = s2 + R * tm;",
    "    real logdet = (R - 1) * log(s2) + log(denom);",
    "    real corrcoef = tm / (s2 * denom);",
    "    for (m in 1:M) {",
    "      vector[R] rr = to_vector(Z[m]) - rep_vector(mu, R) - u;",
    "      real sr = sum(rr);",
    "      target += -0.5 * (logdet + dot_self(rr) / s2 - corrcoef * square(sr));",
    "    }",
    "  }",
    "}"
  )
}

collapsed_cmdstan_stan_file <- function(cfg) {
  d <- file.path(cfg$output_dir, "validation", "cmdstan_same_prior_crosscheck")
  ensure_parent_writable(file.path(d, ".write_probe"))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(d, "fwkc_crossed_gaussian_collapsed_same_prior.stan")
  code <- collapsed_cmdstan_model_code()
  rewrite <- TRUE
  if (file.exists(path)) {
    old <- tryCatch(readLines(path, warn = FALSE), error = function(e) character(0))
    rewrite <- !identical(old, code)
  }
  if (rewrite) safe_write_lines(code, path)
  path
}

get_collapsed_cmdstan_model <- function(cfg) {
  if (exists("model", envir = .fwkc_collapsed_hmc_state, inherits = FALSE)) {
    return(get("model", envir = .fwkc_collapsed_hmc_state, inherits = FALSE))
  }
  if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    stop("V25 same-prior HMC cross-check requires cmdstanr.")
  }
  stan_file <- collapsed_cmdstan_stan_file(cfg)
  write_live_step(
    "Bayes cross-check: compile/load collapsed CmdStan model",
    paste0("stan_file=", stan_file), cfg, force = TRUE
  )
  t0 <- proc.time()[["elapsed"]]
  model <- cmdstanr::cmdstan_model(
    stan_file,
    compile = TRUE,
    force_recompile = FALSE
  )
  assign("model", model, envir = .fwkc_collapsed_hmc_state)
  log_message(
    "CMDSTAN MODEL READY | elapsed=", format_duration(proc.time()[["elapsed"]] - t0),
    " | executable=", tryCatch(model$exe_file(), error = function(e) "unknown")
  )
  model
}

extract_cmdstan_direct_diagnostics <- function(fit, max_treedepth) {
  out <- list(
    divergent_transitions = NA_integer_,
    total_postwarmup_transitions = NA_integer_,
    divergence_rate = NA_real_,
    max_treedepth_hits = NA_integer_,
    min_ebfmi = NA_real_,
    diagnostics_source = "cmdstanr_direct"
  )
  sd <- tryCatch(
    fit$sampler_diagnostics(format = "draws_array"),
    error = function(e) NULL
  )
  if (is.null(sd) || length(dim(sd)) < 3L) {
    out$diagnostics_source <- "cmdstanr_direct_unavailable"
    return(out)
  }
  vars <- dimnames(sd)[[3]]
  if (is.null(vars)) return(out)
  if ("divergent__" %in% vars) {
    dv <- sd[, , "divergent__", drop = TRUE]
    out$divergent_transitions <- as.integer(sum(dv > 0, na.rm = TRUE))
    out$total_postwarmup_transitions <- as.integer(sum(is.finite(dv)))
    if (out$total_postwarmup_transitions > 0L) {
      out$divergence_rate <- out$divergent_transitions / out$total_postwarmup_transitions
    }
  }
  if ("treedepth__" %in% vars) {
    td <- sd[, , "treedepth__", drop = TRUE]
    out$max_treedepth_hits <- as.integer(sum(td >= as.integer(max_treedepth), na.rm = TRUE))
  }
  if ("energy__" %in% vars) {
    en <- sd[, , "energy__", drop = FALSE]
    vals <- numeric(0)
    for (ch in seq_len(dim(en)[2])) {
      e <- as.numeric(en[, ch, 1])
      e <- e[is.finite(e)]
      if (length(e) >= 3L && is.finite(var(e)) && var(e) > 0) {
        vals <- c(vals, mean(diff(e)^2) / var(e))
      }
    }
    if (length(vals) > 0L) out$min_ebfmi <- min(vals, na.rm = TRUE)
  }
  out
}

collapsed_cmdstan_hmc_pool <- function(raters, cfg, c_val, attempt_id = 1L) {
  K <- do.call(cbind, lapply(raters, function(rr) rr$Kappa))
  K <- K[complete.cases(K), , drop = FALSE]
  if (nrow(K) < 5L || ncol(K) < 2L) {
    return(list(
      mean = NA_real_, se = NA_real_, draws = numeric(0),
      engine = "cmdstan_collapsed_same_prior_insufficient_data",
      rhat_mu = NA_real_, convergence_ok = FALSE,
      divergent_transitions = NA_integer_, total_postwarmup_transitions = NA_integer_,
      divergence_rate = NA_real_, max_treedepth_hits = NA_integer_,
      min_ebfmi = NA_real_, ess_bulk_intercept = NA_real_, ess_tail_intercept = NA_real_,
      diagnostics_source = "none"
    ))
  }
  Z <- apply(K, 2, transform_kappa, method = cfg$transformation)
  Z <- as.matrix(Z)
  M <- nrow(Z); R <- ncol(Z)
  if (!all(is.finite(Z))) stop("Non-finite transformed pseudo-rater values reached V25 HMC cross-check.")

  model <- get_collapsed_cmdstan_model(cfg)
  cores <- resolve_brms_crosscheck_cores(cfg)
  iter_sampling <- as.integer(cfg$bayes_crosscheck_brms_iter - cfg$bayes_crosscheck_brms_warmup)
  if (iter_sampling < 20L) stop("Cross-check post-warmup sampling iterations are too small.")
  seed_hmc <- normalize_fwkc_seed(cfg$seed + 1090000 + round(1000 * c_val) + attempt_id)
  outdir <- file.path(cfg$output_dir, "validation", "cmdstan_same_prior_crosscheck", "csv")
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  base <- paste0("c", gsub("\\.", "p", format(c_val, trim=TRUE, scientific=FALSE)), "_a", attempt_id)

  write_live_step(
    paste0("Bayes cross-check c=", c_val, ": CmdStan HMC sampling"),
    paste0(
      "attempt=", attempt_id,
      "; M=", M, "; R=", R,
      "; chains=", cfg$bayes_chains,
      "; parallel_chains=", cores,
      "; warmup=", cfg$bayes_crosscheck_brms_warmup,
      "; sampling=", iter_sampling,
      "; adapt_delta=", cfg$brms_adapt_delta,
      "; max_treedepth=", cfg$brms_max_treedepth,
      "; refresh=", cfg$bayes_crosscheck_cmdstan_refresh
    ), cfg, force = TRUE
  )
  log_message(
    "CMDSTAN LIVE PROGRESS FOLLOWS | c=", c_val,
    " | attempt=", attempt_id,
    " | refresh every ", cfg$bayes_crosscheck_cmdstan_refresh,
    " iterations per chain. If console appears quiet, inspect CURRENT_STEP.txt."
  )

  t0 <- proc.time()[["elapsed"]]
  fit <- model$sample(
    data = list(
      M = M, R = R, Z = Z,
      mu0 = cfg$bayes_mu0, mu_sd = cfg$bayes_mu_sd,
      a_sigma = cfg$bayes_a_sigma, b_sigma = cfg$bayes_b_sigma,
      a_tau = cfg$bayes_a_tau, b_tau = cfg$bayes_b_tau
    ),
    seed = seed_hmc,
    chains = as.integer(cfg$bayes_chains),
    parallel_chains = as.integer(cores),
    iter_warmup = as.integer(cfg$bayes_crosscheck_brms_warmup),
    iter_sampling = iter_sampling,
    adapt_delta = as.numeric(cfg$brms_adapt_delta),
    max_treedepth = as.integer(cfg$brms_max_treedepth),
    refresh = as.integer(cfg$bayes_crosscheck_cmdstan_refresh),
    output_dir = outdir,
    output_basename = base
  )
  elapsed <- proc.time()[["elapsed"]] - t0
  write_live_step(
    paste0("Bayes cross-check c=", c_val, ": HMC sampling finished"),
    paste0("attempt=", attempt_id, "; elapsed=", format_duration(elapsed)), cfg, force = TRUE
  )

  da <- tryCatch(fit$draws(variables = "mu", format = "draws_array"), error = function(e) NULL)
  if (is.null(da)) stop("CmdStan HMC completed but mu draws could not be extracted.")
  mu_mat <- da[, , "mu", drop = TRUE]
  if (is.null(dim(mu_mat))) mu_mat <- matrix(mu_mat, ncol = cfg$bayes_chains)
  pdiag <- posterior_chain_diagnostics(mu_mat)
  rhat_mu <- safe_scalar_numeric(pdiag$rhat)
  ess_bulk <- safe_scalar_numeric(pdiag$ess_bulk)
  ess_tail <- safe_scalar_numeric(pdiag$ess_tail)
  mu_draws <- as.numeric(mu_mat)
  k_draws <- inverse_transform_kappa(mu_draws, cfg$transformation)
  sdiag <- extract_cmdstan_direct_diagnostics(fit, cfg$brms_max_treedepth)

  log_message(
    "CMDSTAN HMC SUMMARY | c=", c_val,
    " | attempt=", attempt_id,
    " | elapsed=", format_duration(elapsed),
    " | Rhat=", signif(rhat_mu, 6),
    " | ESS_bulk/tail=", signif(ess_bulk, 6), "/", signif(ess_tail, 6),
    " | divergences=", sdiag$divergent_transitions, "/", sdiag$total_postwarmup_transitions,
    " | maxTD_hits=", sdiag$max_treedepth_hits,
    " | min_EBFMI=", signif(sdiag$min_ebfmi, 6)
  )

  list(
    mean = mean(k_draws, na.rm = TRUE),
    se = sd(k_draws, na.rm = TRUE),
    draws = k_draws,
    engine = "cmdstan_collapsed_same_prior",
    rhat_mu = rhat_mu,
    convergence_ok = is.finite(rhat_mu) && rhat_mu <= cfg$bayes_rhat_max,
    divergent_transitions = sdiag$divergent_transitions,
    total_postwarmup_transitions = sdiag$total_postwarmup_transitions,
    divergence_rate = sdiag$divergence_rate,
    max_treedepth_hits = sdiag$max_treedepth_hits,
    min_ebfmi = sdiag$min_ebfmi,
    ess_bulk_intercept = ess_bulk,
    ess_tail_intercept = ess_tail,
    diagnostics_source = sdiag$diagnostics_source,
    elapsed_seconds = elapsed,
    cmdstan_csv_files = if (isTRUE(cfg$bayes_crosscheck_keep_cmdstan_csv)) {
      tryCatch(paste(fit$output_files(), collapse = ";"), error = function(e) NA_character_)
    } else NA_character_
  )
}

run_bayes_engine_crosscheck <- function(df, cfg) {
  if (!isTRUE(cfg$run_bayes_engine_crosscheck)) return(NULL)

  needed <- c("cmdstanr", "posterior")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  cmdstan_ver <- if (length(missing) == 0L) {
    tryCatch(cmdstanr::cmdstan_version(error_on_NA = FALSE), error = function(e) NA_character_)
  } else NA_character_
  cmdstan_ok <- length(cmdstan_ver) > 0L && all(!is.na(cmdstan_ver))

  validation_dir <- file.path(cfg$output_dir, "validation")
  ensure_parent_writable(file.path(validation_dir, ".write_probe"))
  progress_path <- file.path(validation_dir, "bayes_engine_crosscheck_progress.csv")
  final_path <- file.path(validation_dir, "bayes_engine_crosscheck.csv")
  attempts_path <- file.path(validation_dir, "bayes_engine_crosscheck_attempts.csv")
  fail_flag <- file.path(validation_dir, "BAYES_CROSSCHECK_FAILED.flag")
  pass_flag <- file.path(validation_dir, "BAYES_CROSSCHECK_PASSED.flag")
  if (file.exists(pass_flag)) try(file.remove(pass_flag), silent = TRUE)

  if (length(missing) > 0L || !cmdstan_ok) {
    msg <- paste0(
      "V25 same-prior CmdStan HMC cross-check unavailable. Missing/configuration issue: ",
      paste(missing, collapse = ", "),
      if (length(missing) == 0L) " CmdStan not configured." else ""
    )
    safe_write_lines(msg, fail_flag)
    if (identical(cfg$run_mode, "publication") && isTRUE(cfg$require_bayes_crosscheck_in_publication)) stop(msg)
    return(data.frame(c_val=NA_real_, status="unavailable", failure_reason=msg, stringsAsFactors=FALSE))
  }

  if (!identical(cfg$bayes_crosscheck_engine, "cmdstan_collapsed_same_prior")) {
    stop("Publication v25 requires bayes_crosscheck_engine='cmdstan_collapsed_same_prior'.")
  }

  crosscheck_signature <- paste(
    "v25_cmdstan_collapsed_same_prior_adaptive",
    cfg$transformation, cfg$n_mc, cfg$n_raters,
    cfg$bayes_chains, cfg$bayes_iter, cfg$bayes_burn, cfg$bayes_thin,
    cfg$bayes_crosscheck_brms_iter, cfg$bayes_crosscheck_brms_warmup,
    cfg$bayes_crosscheck_brms_adapt_delta, cfg$bayes_crosscheck_brms_max_treedepth,
    cfg$bayes_crosscheck_rescue_iter, cfg$bayes_crosscheck_rescue_warmup,
    cfg$bayes_crosscheck_rescue_adapt_delta, cfg$bayes_crosscheck_rescue_max_treedepth,
    cfg$bayes_crosscheck_min_ess_bulk, cfg$bayes_crosscheck_min_ess_tail,
    cfg$bayes_mu0, cfg$bayes_mu_sd, cfg$bayes_a_sigma, cfg$bayes_b_sigma,
    cfg$bayes_a_tau, cfg$bayes_b_tau,
    sep = "|"
  )

  previous <- NULL
  if (isTRUE(cfg$resume) && isTRUE(cfg$bayes_crosscheck_resume_passed) && file.exists(progress_path)) {
    previous <- tryCatch(read.csv(progress_path, stringsAsFactors = FALSE), error = function(e) NULL)
  }
  rows <- vector("list", length(cfg$bayes_crosscheck_c))
  all_attempt_rows <- list()

  save_progress <- function() {
    done <- rows[!vapply(rows, is.null, logical(1))]
    if (length(done) > 0L) {
      out_now <- rbind_fill(done)
      save_csv_safe(out_now, progress_path)
      save_csv_safe(out_now, final_path)
      safe_save_rds(out_now, file.path(validation_dir, "bayes_engine_crosscheck.rds"))
    }
    if (length(all_attempt_rows) > 0L) {
      att <- rbind_fill(all_attempt_rows)
      save_csv_safe(att, attempts_path)
      safe_save_rds(att, file.path(validation_dir, "bayes_engine_crosscheck_attempts.rds"))
    }
    invisible(NULL)
  }

  format_num <- function(x, digits=6) {
    x <- safe_scalar_numeric(x)
    if (!is.finite(x)) return("NA")
    format(signif(x, digits), trim=TRUE, scientific=FALSE)
  }

  evaluate_attempt <- function(cc, attempt_id, cfg_h, gfit, gen) {
    t0 <- proc.time()[["elapsed"]]
    hfit <- collapsed_cmdstan_hmc_pool(gen$raters, cfg_h, cc, attempt_id)
    diff <- abs(gfit$mean - hfit$mean)
    combined_se <- sqrt(gfit$se^2 + hfit$se^2)
    tol <- max(cfg$bayes_crosscheck_abs_floor,
               cfg$bayes_crosscheck_se_multiplier * combined_se, na.rm=TRUE)
    mean_ok <- is.finite(diff) && is.finite(tol) && diff <= tol
    gibbs_ok <- isTRUE(gfit$convergence_ok)
    hmc_rhat_ok <- isTRUE(hfit$convergence_ok)
    div_n <- hfit$divergent_transitions
    n_trans <- hfit$total_postwarmup_transitions
    div_ok <- if (isTRUE(cfg$bayes_crosscheck_require_zero_divergences)) {
      is.finite(div_n) && div_n == 0L
    } else TRUE
    td_n <- hfit$max_treedepth_hits
    td_ok <- if (isTRUE(cfg$bayes_crosscheck_require_zero_max_treedepth)) {
      is.finite(td_n) && td_n == 0L
    } else TRUE
    ess_bulk_ok <- is.finite(hfit$ess_bulk_intercept) &&
      hfit$ess_bulk_intercept >= cfg$bayes_crosscheck_min_ess_bulk
    ess_tail_ok <- is.finite(hfit$ess_tail_intercept) &&
      hfit$ess_tail_intercept >= cfg$bayes_crosscheck_min_ess_tail
    ess_ok <- ess_bulk_ok && ess_tail_ok

    reasons <- character(0)
    if (!mean_ok) reasons <- c(reasons, paste0("mean_difference_exceeds_tolerance(diff=",format_num(diff),",tol=",format_num(tol),")"))
    if (!gibbs_ok) reasons <- c(reasons, paste0("gibbs_rhat_failed(Rhat=",format_num(gfit$rhat_mu),",limit=",cfg$bayes_rhat_max,")"))
    if (!hmc_rhat_ok) reasons <- c(reasons, paste0("hmc_rhat_failed(Rhat=",format_num(hfit$rhat_mu),",limit=",cfg$bayes_rhat_max,")"))
    if (!div_ok) reasons <- c(reasons, paste0("hmc_divergences=",ifelse(is.finite(div_n),div_n,"NA"),ifelse(is.finite(n_trans),paste0("/",n_trans),"")))
    if (!td_ok) reasons <- c(reasons, paste0("hmc_max_treedepth_hits=",ifelse(is.finite(td_n),td_n,"NA")))
    if (!ess_bulk_ok) reasons <- c(reasons, paste0("hmc_ESS_bulk=",format_num(hfit$ess_bulk_intercept),"<",cfg$bayes_crosscheck_min_ess_bulk))
    if (!ess_tail_ok) reasons <- c(reasons, paste0("hmc_ESS_tail=",format_num(hfit$ess_tail_intercept),"<",cfg$bayes_crosscheck_min_ess_tail))
    if (length(reasons) == 0L) reasons <- "none"
    pass <- mean_ok && gibbs_ok && hmc_rhat_ok && div_ok && td_ok && ess_ok

    data.frame(
      c_val = cc,
      crosscheck_engine = "cmdstan_collapsed_same_prior",
      crosscheck_signature = crosscheck_signature,
      attempt = attempt_id,
      adapt_delta = cfg_h$brms_adapt_delta,
      max_treedepth = cfg_h$brms_max_treedepth,
      gibbs_mean = gfit$mean,
      hmc_mean = hfit$mean,
      brms_mean = hfit$mean, # backward-compatible alias for existing workbook/QC code
      gibbs_se = gfit$se,
      hmc_se = hfit$se,
      brms_se = hfit$se,
      gibbs_rhat = gfit$rhat_mu,
      hmc_rhat = hfit$rhat_mu,
      brms_rhat = hfit$rhat_mu,
      abs_difference = diff,
      tolerance = tol,
      mean_agreement_ok = mean_ok,
      gibbs_rhat_ok = gibbs_ok,
      hmc_rhat_ok = hmc_rhat_ok,
      brms_rhat_ok = hmc_rhat_ok,
      hmc_divergent_transitions = div_n,
      brms_divergent_transitions = div_n,
      hmc_total_postwarmup_transitions = n_trans,
      brms_total_postwarmup_transitions = n_trans,
      hmc_divergence_rate = hfit$divergence_rate,
      brms_divergence_rate = hfit$divergence_rate,
      hmc_max_treedepth_hits = td_n,
      brms_max_treedepth_hits = td_n,
      hmc_min_ebfmi = hfit$min_ebfmi,
      brms_min_ebfmi = hfit$min_ebfmi,
      hmc_ess_bulk_mu = hfit$ess_bulk_intercept,
      brms_ess_bulk_intercept = hfit$ess_bulk_intercept,
      hmc_ess_tail_mu = hfit$ess_tail_intercept,
      brms_ess_tail_intercept = hfit$ess_tail_intercept,
      diagnostics_source = hfit$diagnostics_source,
      divergence_ok = div_ok,
      treedepth_ok = td_ok,
      ess_bulk_ok = ess_bulk_ok,
      ess_tail_ok = ess_tail_ok,
      ess_ok = ess_ok,
      pass = pass,
      status = if (pass) "pass" else "review_required",
      failure_reason = paste(reasons, collapse = "; "),
      elapsed_seconds = proc.time()[["elapsed"]] - t0,
      stringsAsFactors = FALSE
    )
  }

  log_message(
    "Bayes cross-check V25 PLAN | engine=collapsed same-prior CmdStan HMC",
    " | c_values=", paste(cfg$bayes_crosscheck_c, collapse=","),
    " | Gibbs chains=", cfg$bayes_chains,
    " | HMC chains=", cfg$bayes_chains,
    " | iter/warmup=", cfg$bayes_crosscheck_brms_iter, "/", cfg$bayes_crosscheck_brms_warmup,
    " | HMC refresh=", cfg$bayes_crosscheck_cmdstan_refresh,
    " | first_control=", cfg$bayes_crosscheck_brms_adapt_delta, "/", cfg$bayes_crosscheck_brms_max_treedepth,
    " | rescue=", cfg$bayes_crosscheck_rescue_adapt_delta, "/", cfg$bayes_crosscheck_rescue_max_treedepth,
    " | SAME model + SAME priors as Gibbs; MC-index effect exactly marginalized"
  )

  for (i in seq_along(cfg$bayes_crosscheck_c)) {
    cc <- cfg$bayes_crosscheck_c[i]
    c_t0 <- proc.time()[["elapsed"]]

    if (!is.null(previous) && nrow(previous) > 0L && all(c("c_val","pass","crosscheck_signature") %in% names(previous))) {
      hit <- which(abs(as.numeric(previous$c_val) - cc) <= 1e-12 &
                   previous$pass %in% TRUE & previous$crosscheck_signature == crosscheck_signature)
      if (length(hit) > 0L) {
        rows[[i]] <- previous[hit[1], , drop=FALSE]
        log_message("Bayes cross-check RESUME | c=",cc," | signature-matched prior PASS reused")
        save_progress(); next
      }
    }

    write_live_step(paste0("Bayes cross-check c=",cc," step 1/6"), "generate matched pseudo-raters", cfg, TRUE)
    set_fwkc_seed(cfg$seed + 880000 + i)
    cfg_gen <- cfg; cfg_gen$bayes_engine <- "gibbs"
    gen_t0 <- proc.time()[["elapsed"]]
    gen <- generate_pseudo_raters(df, cc, cfg_gen)
    log_message("CROSSCHECK STEP DONE | c=",cc," | pseudo-raters | elapsed=",format_duration(proc.time()[["elapsed"]]-gen_t0))
    if (is.null(gen)) {
      rows[[i]] <- data.frame(c_val=cc,crosscheck_engine="cmdstan_collapsed_same_prior",
                              crosscheck_signature=crosscheck_signature,attempt=0L,pass=FALSE,
                              status="pseudo_rater_generation_failed",failure_reason="pseudo_rater_generation_failed",
                              stringsAsFactors=FALSE)
      save_progress()
      stop("Publication Bayes cross-check failed at pseudo-rater generation for c=",cc,". Result saved.")
    }

    write_live_step(paste0("Bayes cross-check c=",cc," step 2/6"), "blocked/collapsed Gibbs fit", cfg, TRUE)
    gt0 <- proc.time()[["elapsed"]]
    cfg_g <- cfg; cfg_g$bayes_engine <- "gibbs"
    gfit <- bayesian_pool(gen$raters, cfg_g)
    log_message(
      "GIBBS CROSSCHECK SUMMARY | c=",cc,
      " | mean=",format_num(gfit$mean)," | se=",format_num(gfit$se),
      " | Rhat=",format_num(gfit$rhat_mu),
      " | ESS_bulk/tail=",format_num(gfit$ess_bulk_mu),"/",format_num(gfit$ess_tail_mu),
      " | elapsed=",format_duration(proc.time()[["elapsed"]]-gt0)
    )
    if (!isTRUE(gfit$convergence_ok)) {
      rows[[i]] <- data.frame(
        c_val=cc,crosscheck_engine="cmdstan_collapsed_same_prior",crosscheck_signature=crosscheck_signature,
        attempt=0L,gibbs_mean=gfit$mean,gibbs_se=gfit$se,gibbs_rhat=gfit$rhat_mu,
        pass=FALSE,status="review_required",
        failure_reason=paste0("gibbs_rhat_failed(Rhat=",format_num(gfit$rhat_mu),",limit=",cfg$bayes_rhat_max,")"),
        stringsAsFactors=FALSE
      )
      save_progress()
      stop("Publication cross-check stopped before HMC because Gibbs did not converge at c=",cc,". Result saved.")
    }

    write_live_step(paste0("Bayes cross-check c=",cc," step 3/6"), "compile/reuse low-dimensional same-prior CmdStan model", cfg, TRUE)
    get_collapsed_cmdstan_model(cfg)

    cfg_h1 <- cfg
    cfg_h1$brms_adapt_delta <- cfg$bayes_crosscheck_brms_adapt_delta
    cfg_h1$brms_max_treedepth <- cfg$bayes_crosscheck_brms_max_treedepth
    write_live_step(paste0("Bayes cross-check c=",cc," step 4/6"), "HMC attempt 1; live chain progress should print below", cfg, TRUE)
    a1 <- evaluate_attempt(cc, 1L, cfg_h1, gfit, gen)
    all_attempt_rows[[length(all_attempt_rows)+1L]] <- a1
    save_progress()

    sampler_problem <- !isTRUE(a1$hmc_rhat_ok[1]) || !isTRUE(a1$divergence_ok[1]) ||
      !isTRUE(a1$treedepth_ok[1]) || !isTRUE(a1$ess_ok[1])
    final_attempt <- a1; rescue_used <- FALSE
    if (!isTRUE(a1$pass[1]) && isTRUE(cfg$bayes_crosscheck_auto_rescue) && sampler_problem) {
      cfg_h2 <- cfg_h1
      cfg_h2$brms_adapt_delta <- cfg$bayes_crosscheck_rescue_adapt_delta
      cfg_h2$brms_max_treedepth <- cfg$bayes_crosscheck_rescue_max_treedepth
      cfg_h2$bayes_crosscheck_brms_iter <- cfg$bayes_crosscheck_rescue_iter
      cfg_h2$bayes_crosscheck_brms_warmup <- cfg$bayes_crosscheck_rescue_warmup
      write_live_step(paste0("Bayes cross-check c=",cc," step 5/6"),
                      paste0("HMC rescue; reason=",a1$failure_reason[1],
                             "; iter/warmup=",cfg_h2$bayes_crosscheck_brms_iter,"/",cfg_h2$bayes_crosscheck_brms_warmup,
                             "; compiled model reused"), cfg, TRUE)
      a2 <- evaluate_attempt(cc, 2L, cfg_h2, gfit, gen)
      all_attempt_rows[[length(all_attempt_rows)+1L]] <- a2
      final_attempt <- a2; rescue_used <- TRUE
      save_progress()
    } else {
      write_live_step(paste0("Bayes cross-check c=",cc," step 5/6"), "rescue not required", cfg, TRUE)
    }

    rows[[i]] <- final_attempt
    rows[[i]]$rescue_used <- rescue_used
    rows[[i]]$elapsed_seconds_total_c <- proc.time()[["elapsed"]] - c_t0
    save_progress()
    rr <- rows[[i]]
    write_live_step(paste0("Bayes cross-check c=",cc," step 6/6"), "save diagnostics and apply publication gate", cfg, TRUE)
    log_message(
      "Bayes cross-check END | c=",cc,
      " | engine=",rr$crosscheck_engine[1],
      " | status=",rr$status[1],
      " | Gibbs/HMC=",format_num(rr$gibbs_mean),"/",format_num(rr$hmc_mean),
      " | diff/tol=",format_num(rr$abs_difference),"/",format_num(rr$tolerance),
      " | Rhat(G/H)=",format_num(rr$gibbs_rhat),"/",format_num(rr$hmc_rhat),
      " | divergences=",ifelse(is.finite(rr$hmc_divergent_transitions),rr$hmc_divergent_transitions,"NA"),
      "/",ifelse(is.finite(rr$hmc_total_postwarmup_transitions),rr$hmc_total_postwarmup_transitions,"NA"),
      " | maxTD_hits=",ifelse(is.finite(rr$hmc_max_treedepth_hits),rr$hmc_max_treedepth_hits,"NA"),
      " | ESS_bulk/tail=",format_num(rr$hmc_ess_bulk_mu),"/",format_num(rr$hmc_ess_tail_mu),
      " | reason=",rr$failure_reason[1],
      " | elapsed_total_c=",format_duration(rr$elapsed_seconds_total_c[1])
    )

    if (identical(cfg$run_mode,"publication") && isTRUE(cfg$require_bayes_crosscheck_in_publication) &&
        !isTRUE(rr$pass[1]) && isTRUE(cfg$bayes_crosscheck_fail_fast)) {
      safe_write_lines(c(
        "FWKC v25 publication same-prior HMC cross-check FAILED FAST.",
        paste0("c=",cc), paste0("reason=",rr$failure_reason[1]),
        paste0("Gibbs_Rhat=",rr$gibbs_rhat[1]), paste0("HMC_Rhat=",rr$hmc_rhat[1]),
        paste0("HMC_divergences=",rr$hmc_divergent_transitions[1]),
        paste0("attempts_file=",attempts_path)
      ), fail_flag)
      stop("Publication same-prior CmdStan HMC cross-check failed at c=",cc,
           " AFTER saving diagnostics. Reason: ",rr$failure_reason[1],".")
    }
  }

  out <- rbind_fill(rows)
  save_csv_safe(out, final_path)
  safe_save_rds(out, file.path(validation_dir,"bayes_engine_crosscheck.rds"))
  if (identical(cfg$run_mode,"publication") && isTRUE(cfg$require_bayes_crosscheck_in_publication) &&
      (nrow(out) != length(cfg$bayes_crosscheck_c) || any(!out$pass))) {
    safe_write_lines(c("FWKC v25 publication cross-check FAILED.","Inspect bayes_engine_crosscheck.csv."),fail_flag)
    stop("Publication gate failed: Gibbs-vs-collapsed-CmdStan-HMC cross-check did not pass.")
  }
  if (file.exists(fail_flag)) try(file.remove(fail_flag), silent=TRUE)
  safe_write_lines(c(
    "FWKC v25 same-prior collapsed CmdStan HMC cross-check PASSED all prespecified c values.",
    paste0("c_values=",paste(cfg$bayes_crosscheck_c,collapse=",")),
    paste0("completed=",format(Sys.time(),"%Y-%m-%d %H:%M:%S %z"))
  ), pass_flag)
  write_live_step("Bayes cross-check complete", "all prespecified c values passed", cfg, TRUE)
  out
}

write_reproducibility_metadata <- function(cfg) {
  if (!isTRUE(cfg$write_session_info)) return(invisible(NULL))

  safe_write_lines(
    capture.output(sessionInfo()),
    file.path(cfg$output_dir, "SESSION_INFO.txt")
  )
  safe_save_rds(cfg, file.path(cfg$output_dir, "CONFIG_USED.rds"))
  safe_write_lines(
    capture.output(str(cfg)),
    file.path(cfg$output_dir, "CONFIG_USED.txt")
  )

  pkg_names <- c(
    "R", "readxl", "openxlsx", "randomForest",
    "brms", "cmdstanr", "posterior"
  )
  pkg_versions <- vapply(
    pkg_names,
    function(pkg) {
      if (pkg == "R") return(as.character(getRversion()))
      if (!requireNamespace(pkg, quietly = TRUE)) return(NA_character_)
      as.character(packageVersion(pkg))
    },
    character(1)
  )
  save_csv_safe(
    data.frame(package = pkg_names, version = pkg_versions),
    file.path(cfg$output_dir, "PACKAGE_VERSIONS.csv")
  )
  invisible(NULL)
}

###############################################################################
# 27) REVIEWER-COMPLIANCE MANIFEST + TECHNICAL PIPELINE DIAGRAM
###############################################################################

build_reviewer_manifest <- function(cfg) {
  data.frame(
    reviewer = c(
      rep("Reviewer 1", 15),
      rep("Reviewer 2", 7)
    ),
    item = c(
      as.character(1:15),
      c("Estimand", "Beta posterior", "Coverage", "RF specification",
        "Broader benchmarks", "Presentation", "External validation")
    ),
    requirement = c(
      "Quantify implicit bias across the multi-layer pipeline",
      "Justify Beta uncertainty and assess nonparametric robustness",
      "Assess sensitivity to fusion weighting",
      "Specify RF features/overfitting control and compare simpler model",
      "Assess transformation sensitivity",
      "Assess pseudo-rater covariance reliability and bootstrap covariance",
      "Assess n0/shrinkage sensitivity and adaptive alternative",
      "Reduce dependence on one real-data anchor",
      "Use multiple DGMs / misspecification stress tests",
      "Use nonparametric paired inference when bootstrap is non-normal",
      "Support coverage rationale with transparent inferential diagnostics",
      "Quantify computational cost and scalability",
      "Show when added complexity gives practical benefit",
      "Provide preliminary ordinal/multiclass extension path",
      "Compare descriptive prevalence-robust agreement measures",
      "Derive kappa(c) from a standard weighted-agreement matrix",
      "Replace Beta(n*Se,n*(1-Se)) as the primary uncertainty model",
      "Evaluate coverage across independent DGM datasets",
      "Fully specify RF and prevent target leakage/overfitting",
      "Benchmark genuine competing inference methods and bias/RMSE/MAE",
      "Use a genuine technical diagram and reproducible visual outputs",
      "Allow additional real datasets without fabricating validation data"
    ),
    implementation = c(
      "Stage-specific bias/RMSE/MAE for Direct, Bayesian, ML, and fused FWKC",
      "Jeffreys/Uniform Beta-binomial posterior plus empirical-bootstrap sensitivity",
      "Baseline inverse-SE preserved as primary; inverse-variance/equal/Bayes-only/ML-only sensitivity",
      "Se, Sp and other-rater kappas; target excluded; OOB RF; cross-fitted spline",
      "logit11, Fisher-z and identity sensitivity",
      "empirical covariance diagnostic + Monte-Carlo bootstrap of covariance + full pipeline bootstrap",
      "n0 grid plus adaptive minimum-stratum calibration; calibration remains secondary",
      "Full anchor continuity grid plus separate prespecified non-circular robustness tier and optional external counts",
      "iid accuracy regimes plus clustered-beta and mixture-tradeoff heterogeneous DGMs",
      "paired bootstrap CI/centered-bootstrap p-value; Wilcoxon only across independent DGM datasets",
      "primary full-pipeline bootstrap, competing CIs, independent-method validity, MCSE for empirical coverage",
      "separate direct/FWKC timings and scalability benchmark",
      "coverage/RMSE gain versus extra runtime in stress regimes",
      "generic weighted-kappa estimand function for K-category tables, explicitly preliminary",
      "PABAK and Gwet AC1 reported descriptively, not as the same estimand",
      "W=[[1,c],[1-c,1]] weighted agreement plus loss/count/population equivalence tests",
      "Primary Se/Sp posterior uses TP,FN,TN,FP with Jeffreys prior; legacy retained only for sensitivity",
      "M_dgm independent datasets per scenario; bootstrap replicates never counted as coverage replicates",
      "symmetric leave-one-rater-out RF with OOB predictions and sparse-sample safeguards",
      "Wald/delta, transformed delta, percentile, basic and BCa competitors; Direct remains estimable when FWKC fails; stage error metrics",
      "base-R pipeline diagram plus coverage/RMSE/runtime plots",
      "external_counts_file accepts dataset,TP,FN,FP,TN when genuine external data are supplied"
    ),
    primary_output = c(
      "simulation/main_results_final.csv",
      "sensitivity/sensitivity_results.csv",
      "sensitivity/sensitivity_results.csv",
      "simulation/main_results_final.csv; sensitivity/sensitivity_results.csv",
      "sensitivity/sensitivity_results.csv",
      "simulation/detail/*.csv; real_data/real_data_results.csv",
      "sensitivity/sensitivity_results.csv",
      "simulation/scenario_grid.csv; robustness/simulation/scenario_grid_robustness.csv",
      "simulation/dgm_catalog.csv; simulation/main_results_final.csv",
      "real_data/paired_tests.csv; simulation/main_results_final.csv",
      "simulation/main_results_final.csv",
      "simulation/scalability_summary.csv",
      "simulation/practical_benefit_summary.csv",
      "function weighted_kappa_multiclass()",
      "real_data/real_data_results.csv; simulation/detail/*.csv",
      "validation/estimand_equivalence.csv; validation/weight_matrix_derivation.csv",
      "sensitivity/sensitivity_results.csv",
      "simulation/detail/*.csv; simulation/main_results_final.csv",
      "RF_OOB_MSE and ML sensitivity outputs",
      "simulation/main_results_final.csv",
      "plots/FWKC_pipeline_technical.png and statistical plots",
      "external/external_validation.csv when external_counts_file is supplied"
    ),
    stringsAsFactors = FALSE
  )
}

write_reviewer_manifest <- function(cfg) {
  manifest <- build_reviewer_manifest(cfg)

  if (isTRUE(cfg$write_reviewer_manifest)) {
    save_csv_safe(
      manifest,
      file.path(cfg$output_dir, "reviewer_to_code_manifest.csv")
    )
  }

  manifest
}

make_pipeline_technical_diagram <- function(cfg) {
  if (!isTRUE(cfg$make_pipeline_diagram)) return(invisible(FALSE))

  path <- file.path(
    cfg$output_dir,
    "plots",
    "FWKC_pipeline_technical.png"
  )

  png(path, width = 1800, height = 1100, res = 150)
  on.exit(dev.off(), add = TRUE)

  par(mar = c(1, 1, 3, 1))
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1))
  title("FWKC reviewer-ready inferential pipeline")

  box_node <- function(x, y, label, w = 0.20, h = 0.10) {
    rect(x - w/2, y - h/2, x + w/2, y + h/2, border = "black", col = "white")
    text(x, y, label, cex = 0.78)
  }

  arrow_node <- function(x0, y0, x1, y1) {
    arrows(x0, y0, x1, y1, length = 0.08)
  }

  box_node(0.12, 0.78, "Observed binary data\n2x2 table")
  box_node(0.38, 0.78, "Target estimand\nloss / weight-matrix / closed form")
  box_node(0.66, 0.82, "Proper Se/Sp uncertainty\nBeta-binomial or empirical")
  box_node(0.66, 0.62, "Matched pseudo-raters\nposterior predictive")
  box_node(0.38, 0.54, "Bayesian pooling\ncrossed rater + MC effects")
  box_node(0.66, 0.42, "ML aggregation\nOOB RF / cross-fitted spline")
  box_node(0.38, 0.30, "Fusion\nwith sensitivity rules")
  box_node(0.12, 0.30, "Full-pipeline bootstrap\nprimary FWKC uncertainty")
  box_node(0.12, 0.10, "Independent DGM repeats\ncoverage + bias/RMSE/MAE")
  box_node(0.66, 0.14, "Competing methods +\nsensitivity + scalability")

  arrow_node(0.22, 0.78, 0.28, 0.78)
  arrow_node(0.48, 0.78, 0.56, 0.81)
  arrow_node(0.66, 0.77, 0.66, 0.68)
  arrow_node(0.58, 0.61, 0.46, 0.56)
  arrow_node(0.66, 0.57, 0.66, 0.48)
  arrow_node(0.58, 0.42, 0.46, 0.33)
  arrow_node(0.28, 0.30, 0.22, 0.30)
  arrow_node(0.12, 0.25, 0.12, 0.16)
  arrow_node(0.22, 0.10, 0.56, 0.14)

  text(
    0.5, 0.02,
    "Coverage is computed across independent DGM datasets; bootstrap replicates quantify within-dataset uncertainty.",
    cex = 0.72
  )

  invisible(TRUE)
}

###############################################################################
# 28) SUMMARY WORKBOOK
###############################################################################

config_to_df <- function(cfg) {
  vals <- vapply(
    cfg,
    function(x) {
      if (is.data.frame(x)) {
        return("<data.frame>")
      }
      if (length(x) == 0L) return("")
      paste(x, collapse = ",")
    },
    character(1)
  )

  data.frame(
    key = names(vals),
    value = unname(vals),
    stringsAsFactors = FALSE
  )
}

write_summary_workbook <- function(
  validation, real_data, main_results, anchor_bridge, anchor_sample_check,
  sample_size_bridge_comparison, robustness, sensitivity, external, benefit,
  scalability, reviewer_manifest,
  readiness, bayes_crosscheck, publication_qc, cfg
) {
  wb <- createWorkbook()
  addWorksheet(wb,"Config"); writeData(wb,"Config",config_to_df(cfg))
  addWorksheet(wb,"Parallel_Plan"); writeData(wb,"Parallel_Plan",parallel_runtime_plan(cfg))
  addWorksheet(wb,"Estimand_Check"); writeData(wb,"Estimand_Check",validation$equivalence)
  addWorksheet(wb,"Reference_Discrepancy"); writeData(wb,"Reference_Discrepancy",validation$reference_discrepancy)
  addWorksheet(wb,"Weight_Matrix"); writeData(wb,"Weight_Matrix",validation$weight_matrix)
  if(!is.null(validation$shrinkage_check)) {
    addWorksheet(wb,"Shrinkage_Check"); writeData(wb,"Shrinkage_Check",validation$shrinkage_check)
  }
  addWorksheet(wb,"Reviewer_Map"); writeData(wb,"Reviewer_Map",reviewer_manifest)
  if(!is.null(readiness)) { addWorksheet(wb,"Reviewer_Readiness"); writeData(wb,"Reviewer_Readiness",readiness) }
  if(!is.null(bayes_crosscheck)) { addWorksheet(wb,"Bayes_Crosscheck"); writeData(wb,"Bayes_Crosscheck",bayes_crosscheck) }
  if(!is.null(publication_qc)) { addWorksheet(wb,"Publication_QC"); writeData(wb,"Publication_QC",publication_qc) }
  addWorksheet(wb,"Real_Data"); writeData(wb,"Real_Data",real_data$results)
  if(!is.null(real_data$paired_tests)&&nrow(real_data$paired_tests)>0) {
    addWorksheet(wb,"Paired_Tests"); writeData(wb,"Paired_Tests",real_data$paired_tests)
  }
  if(!is.null(main_results)) { addWorksheet(wb,"Main_Simulation"); writeData(wb,"Main_Simulation",main_results) }
  if(!is.null(anchor_sample_check)) {
    addWorksheet(wb,"Anchor_N_Check"); writeData(wb,"Anchor_N_Check",anchor_sample_check)
  }
  if(!is.null(anchor_bridge) && is.list(anchor_bridge)) {
    if(!is.null(anchor_bridge$results)) { addWorksheet(wb,"Anchor_N_Bridge"); writeData(wb,"Anchor_N_Bridge",anchor_bridge$results) }
    if(!is.null(anchor_bridge$summary)) { addWorksheet(wb,"Anchor_N_Summary"); writeData(wb,"Anchor_N_Summary",anchor_bridge$summary) }
  }
  if(!is.null(sample_size_bridge_comparison)) {
    addWorksheet(wb,"N_30_50_100_620"); writeData(wb,"N_30_50_100_620",sample_size_bridge_comparison)
  }
  if(!is.null(robustness)) { addWorksheet(wb,"DGM_Robustness"); writeData(wb,"DGM_Robustness",robustness) }
  if(!is.null(sensitivity)) { addWorksheet(wb,"Sensitivity"); writeData(wb,"Sensitivity",sensitivity) }
  if(!is.null(external)) { addWorksheet(wb,"External"); writeData(wb,"External",external) }
  if(!is.null(benefit)) { addWorksheet(wb,"Practical_Benefit"); writeData(wb,"Practical_Benefit",benefit) }
  if(!is.null(scalability)&&!is.null(scalability$summary)) {
    addWorksheet(wb,"Scalability"); writeData(wb,"Scalability",scalability$summary)
    addWorksheet(wb,"Scalability_Detail"); writeData(wb,"Scalability_Detail",scalability$detail)
  }
  saveWorkbook(wb,file.path(cfg$output_dir,"FWKC_FINAL_V25_HYBRID_SUMMARY.xlsx"),overwrite=TRUE)
}

###############################################################################
# 29) MAIN
###############################################################################

main <- function(cfg) {
  log_message("FWKC final v25 collapsed-CmdStan verbose-progress persistent-PSOCK analysis started | run_mode=", cfg$run_mode)
  log_message("RESULTS DIRECTORY: ", cfg$output_dir)
  log_message("EMERGENCY I/O DIRECTORY: ", io_emergency_root())
  on.exit(stop_parallel_cluster(), add = TRUE)

  # V25 preflight: prove that all critical result directories can be created,
  # written, committed and read BEFORE starting the expensive Bayesian cross-check.
  run_io_self_test(cfg)
  safe_write_lines(
    c(
      paste0("results_directory: ", cfg$output_dir),
      paste0("emergency_io_directory: ", io_emergency_root()),
      paste0("started: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
      paste0("run_mode: ", cfg$run_mode),
      paste0("script_version: ", cfg$script_version),
      paste0("parallel_mode: ", cfg$parallel_mode),
      paste0("parallel_effective_workers: ", resolve_parallel_workers(cfg)),
      paste0("brms_local_cores: ", resolve_brms_cores(cfg)),
      paste0("brms_crosscheck_cores: ", resolve_brms_crosscheck_cores(cfg)),
      paste0("manuscript_anchor_n_expected: ", cfg$manuscript_anchor_n),
      paste0("anchor_bridge_design: ", cfg$anchor_bridge_design)
    ),
    file.path(cfg$output_dir,"RESULTS_LOCATION.txt")
  )

  main_stage_start <- proc.time()[["elapsed"]]
  total_main_stages <- 13L; main_stage <- 0L
  mark_main_stage <- function(label) {
    main_stage <<- main_stage + 1L
    progress_message("Overall pipeline",main_stage,total_main_stages,cfg,
                     stage_start=main_stage_start,detail=label)
  }

  write_reproducibility_metadata(cfg)
  parallel_plan <- write_parallel_plan(cfg)
  log_message(
    "PARALLEL RUNTIME PLAN | enabled=",parallel_plan$parallel_enabled[1],
    " | mode=",parallel_plan$mode[1],
    " | scenario_workers=",parallel_plan$effective_scenario_workers[1],
    " | brms_local_cores=",parallel_plan$brms_local_cores[1],
    " | brms_crosscheck_cores=",parallel_plan$brms_crosscheck_cores[1],
    " | M/B_DGM/B_REAL/n_mc=",parallel_plan$statistical_M_dgm[1],"/",
    parallel_plan$statistical_B_boot_dgm[1],"/",parallel_plan$statistical_B_boot_real[1],"/",
    parallel_plan$statistical_n_mc[1]," (UNCHANGED)"
  )
  mark_main_stage("reproducibility metadata + parallel runtime plan")
  reviewer_manifest <- write_reviewer_manifest(cfg)
  diagram_status <- tryCatch(make_pipeline_technical_diagram(cfg),error=function(e){
    log_message("Pipeline diagram failed: ",conditionMessage(e)); FALSE
  })
  mark_main_stage("reviewer manifest + technical diagram")

  df <- load_input_data(cfg)
  external_preflight <- external_validation_preflight(cfg)
  anchor_sample_check <- validate_manuscript_anchor_sample(df,cfg)
  mark_main_stage("input data + provenance + manuscript anchor n check")

  validation <- validate_estimand(cfg)
  posterior_diag_selftest <- run_posterior_diagnostics_selftest(cfg)
  bayes_crosscheck <- run_bayes_engine_crosscheck(df,cfg)
  safe_save_rds(list(validation=validation,posterior_diag_selftest=posterior_diag_selftest,
                     bayes_crosscheck=bayes_crosscheck),
                file.path(cfg$output_dir,"stage_archive","stage_01_validation_bayes.rds"))
  mark_main_stage("estimand validation + Bayes cross-check")

  preflight_flag <- file.path(cfg$output_dir, "PREFLIGHT_PASSED_RERUN_TO_CONTINUE.flag")
  if (identical(cfg$run_mode, "publication") &&
      isTRUE(cfg$publication_preflight_auto) &&
      !file.exists(preflight_flag)) {
    safe_write_lines(
      c(
        "FWKC V25 PUBLICATION PREFLIGHT PASSED.",
        "I/O self-test, posterior diagnostics API self-test, input provenance, estimand checks and all prespecified Bayes cross-check c values passed.",
        "No main simulation was started in this first preflight run.",
        "RERUN THE IDENTICAL SCRIPT: signature-matched Bayes PASS rows will be reused and the full publication analysis will continue.",
        paste0("timestamp=", format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"))
      ),
      preflight_flag
    )
    log_message(
      "PUBLICATION PREFLIGHT PASSED | no long simulation started | ",
      "rerun the identical V25 script to continue; passed Bayes checks will be reused safely."
    )
    return(invisible(list(
      preflight_only = TRUE,
      validation = validation,
      bayes_crosscheck = bayes_crosscheck,
      output_dir = cfg$output_dir,
      emergency_io_directory = io_emergency_root()
    )))
  }

  real_data <- run_real_data(df,cfg)
  safe_save_rds(real_data,file.path(cfg$output_dir,"stage_archive","stage_02_real_data.rds"))
  mark_main_stage("real-data analysis")

  main_results <- run_main_simulation(df,cfg)
  safe_save_rds(main_results,file.path(cfg$output_dir,"stage_archive","stage_03_primary_simulation.rds"))
  mark_main_stage("full-grid independent-DGM primary simulation")

  anchor_bridge <- run_anchor_sample_size_bridge(df,cfg)
  sample_size_bridge_comparison <- make_sample_size_bridge_comparison(main_results,anchor_bridge,cfg)
  safe_save_rds(
    list(anchor_bridge=anchor_bridge,comparison=sample_size_bridge_comparison),
    file.path(cfg$output_dir,"stage_archive","stage_03b_anchor_sample_size_bridge.rds")
  )
  mark_main_stage("real-data anchor sample-size bridge + n=30/50/100/620 comparison")

  robustness <- run_dgm_robustness(df,cfg)
  safe_save_rds(robustness,file.path(cfg$output_dir,"stage_archive","stage_04_dgm_robustness.rds"))
  mark_main_stage("non-circular DGM robustness")

  sensitivity <- run_sensitivity(df,cfg)
  safe_save_rds(sensitivity,file.path(cfg$output_dir,"stage_archive","stage_05_sensitivity.rds"))
  mark_main_stage("sensitivity analyses")

  external <- run_external_validation(df,cfg)
  safe_save_rds(external,file.path(cfg$output_dir,"stage_archive","stage_06_external.rds"))
  mark_main_stage("external validation / limitation recorded")

  benefit <- make_practical_benefit_table(main_results,cfg)
  scalability <- run_scalability_benchmark(cfg)
  mark_main_stage("practical benefit + scalability")

  publication_qc <- build_publication_qc(
    real_data=real_data, main_results=main_results,
    anchor_bridge=anchor_bridge, anchor_sample_check=anchor_sample_check,
    robustness=robustness, sensitivity=sensitivity, external=external,
    bayes_crosscheck=bayes_crosscheck, cfg=cfg
  )
  readiness <- make_reviewer_readiness_report(
    cfg=cfg, validation=validation, main_results=main_results,
    anchor_bridge=anchor_bridge, anchor_sample_check=anchor_sample_check,
    sample_size_bridge_comparison=sample_size_bridge_comparison,
    robustness=robustness, sensitivity=sensitivity, external=external,
    bayes_crosscheck=bayes_crosscheck, publication_qc=publication_qc
  )
  mark_main_stage("publication QC + reviewer readiness")

  plot_status <- tryCatch({ make_plots(main_results,cfg); TRUE },error=function(e){
    log_message("Plot generation did not complete: ",conditionMessage(e),
                ". Core statistical outputs remain saved."); FALSE
  })

  write_summary_workbook(
    validation=validation, real_data=real_data, main_results=main_results,
    anchor_bridge=anchor_bridge, anchor_sample_check=anchor_sample_check,
    sample_size_bridge_comparison=sample_size_bridge_comparison,
    robustness=robustness, sensitivity=sensitivity, external=external,
    benefit=benefit, scalability=scalability, reviewer_manifest=reviewer_manifest,
    readiness=readiness, bayes_crosscheck=bayes_crosscheck,
    publication_qc=publication_qc, cfg=cfg
  )
  reportable <- write_reportability_gate(publication_qc,cfg)
  mark_main_stage("plots + workbook + reportability gate")

  all_results <- list(
    config=cfg, validation=validation, real_data=real_data,
    anchor_sample_check=anchor_sample_check,
    main_results=main_results, anchor_bridge=anchor_bridge,
    sample_size_bridge_comparison=sample_size_bridge_comparison,
    robustness=robustness, sensitivity=sensitivity,
    external=external, benefit=benefit, scalability=scalability,
    reviewer_manifest=reviewer_manifest, readiness=readiness,
    bayes_crosscheck=bayes_crosscheck, publication_qc=publication_qc,
    parallel_plan=parallel_plan,
    reportable=reportable, diagram_status=diagram_status, plot_status=plot_status
  )
  safe_save_rds(all_results,file.path(cfg$output_dir,"FWKC_FINAL_V25_HYBRID_ALL_RESULTS.rds"))
  mark_main_stage("final RDS archive saved")

  if(identical(cfg$run_mode,"publication") && !isTRUE(reportable)) {
    stop("Publication QC gate failed after all available outputs were saved. ",
         "Inspect PUBLICATION_QC.csv and RESULTS_NOT_REPORTABLE.flag before manuscript use.")
  }

  log_message("FWKC final v25 collapsed-CmdStan verbose-progress persistent-PSOCK analysis completed.")
  invisible(all_results)
}

###############################################################################
# 30) EXECUTION
###############################################################################

if (isTRUE(config$run_analysis)) {
  message(
    "FWKC v25 collapsed-CmdStan verbose-progress persistent-PSOCK is starting in run_mode = ",
    config$run_mode,
    ". In verification mode this is a functional test, not manuscript-grade inference.\n",
    "Parallel mode: ", config$parallel_mode,
    " | scenario workers planned: ", resolve_parallel_workers(config),
    " | regular local cores: ", resolve_brms_cores(config),
    " | HMC cross-check cores: ", resolve_brms_crosscheck_cores(config), "\n",
    "Results will be stored in: ", config$output_dir
  )

  save_abort_state <- function(kind, msg) {
    state <- list(
      kind = kind,
      message = msg,
      time = Sys.time(),
      config = config,
      random_seed = if (exists(".Random.seed", envir = .GlobalEnv)) .Random.seed else NULL
    )
    lines <- c(
      paste0("kind: ", kind),
      paste0("time: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
      paste0("message: ", msg),
      "Completed inner-bootstrap / DGM checkpoints remain available in the checkpoints directory."
    )

    ok <- tryCatch({
      ensure_parent_writable(file.path(config$output_dir, "RUN_ABORTED_STATE.rds"))
      safe_save_rds(state, file.path(config$output_dir, "RUN_ABORTED_STATE.rds"))
      safe_write_lines(lines, file.path(config$output_dir, "RUN_ABORTED.txt"))
      TRUE
    }, error = function(ioe) {
      warning(
        "Could not persist abort state in the configured output directory: ",
        conditionMessage(ioe), call. = FALSE
      )
      FALSE
    })

    if (!isTRUE(ok)) {
      # Emergency copy only; this must NEVER replace/mask the original error.
      fallback_dir <- file.path(tempdir(), "FWKC_v25_emergency")
      try(dir.create(fallback_dir, recursive = TRUE, showWarnings = FALSE), silent = TRUE)
      try(saveRDS(state, file.path(fallback_dir, "RUN_ABORTED_STATE.rds")), silent = TRUE)
      try(writeLines(lines, file.path(fallback_dir, "RUN_ABORTED.txt")), silent = TRUE)
      message("Emergency abort-state fallback attempted at: ", fallback_dir)
    }
    invisible(ok)
  }

  results <- tryCatch(
    main(config),
    interrupt = function(e) {
      save_abort_state("interrupt", conditionMessage(e))
      stop(e)
    },
    error = function(e) {
      save_abort_state("error", conditionMessage(e))
      stop(e)
    }
  )
} else {
  message(
    "Script loaded without execution because config$run_analysis = FALSE."
  )
}
###############################################################################
# END
###############################################################################
