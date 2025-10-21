context("Integration tests of est_target_param_components")

test_that("recovers true cohort mean outcomes on staircase data", {
  # Problem size and staircase layout
  Tval <- 10L
  r <- 2L
  window <- 3L

  # Build staircase cohorts and deterministic context
  outcomes <- make_outcomes(Tval)
  cohort_indices <- make_staircase_observed_indices(Tval, window)
  C <- length(cohort_indices)
  units_per_cohort <- 5L
  units_by_cohort <- make_units_by_cohort(n_cohorts = C, units_per_cohort = units_per_cohort, prefix = "u")

  # One shared ctx used for both panel generation and truth construction
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, include_outcome_fes = TRUE)

  # Deterministic, noise-free panel from the ctx; then shuffle rows
  panel <- build_panel_from_indices_factor(
    outcomes = outcomes,
    cohort_indices = cohort_indices,
    units_by_cohort = units_by_cohort,
    include_covariates = FALSE,
    include_auxiliary = FALSE,
    r = r,
    rotate = FALSE,
    ctx = ctx
  )
  set.seed(42)
  panel <- panel[sample(nrow(panel))]

  # Panel container
  panel_obj <- UnbalancedPanel$new(
    panel_df = panel,
    unit_id_col = "unit_id",
    outcome_id_col = "outcome_id",
    outcome_value_col = "y",
    model_rank = r
  )

  # Estimator spec (PC, no outcome FEs)
  est_specs <- list(
    pc = list(
      factor_model_estimator = "principal_components",
      include_outcome_fes = TRUE,
      r = r
    )
  )

  # Run end-to-end estimator (both via imputation and direct composition)
  res_imp <- est_target_param_components(
    panel_obj,
    est_specs = est_specs,
    num_threads = 1L,
    est_outcome_means_via_imputation = TRUE
  )
  # res_dir <- est_target_param_components(
  #   panel_obj,
  #   est_specs = est_specs,
  #   num_threads = 1L,
  #   est_outcome_means_via_imputation = FALSE
  # )
  M_hat_imp <- res_imp$outcome_means$pc$mean_outcomes()  # C x T
  # M_hat_dir <- res_dir$outcome_means$pc$mean_outcomes()  # C x T

  # Align cohort order using exact index match from original to panel order
  ooi_panel <- panel_obj$get_observed_outcome_indices()
  panel_idx_for_orig <- match_cohorts_panel_order(cohort_indices, ooi_panel)

  # Ground truth per cohort using the same generator as the panel (average per-unit outcomes over full T)
  true_M <- matrix(NA_real_, nrow = C, ncol = length(outcomes))
  for (c in seq_along(cohort_indices)) {
    cp <- panel_idx_for_orig[c]
    unit_ids <- units_by_cohort[[c]]
    Y_full <- expected_Y_for_units_ctx(
      ctx, c,
      unit_ids = unit_ids,
      T_idx = seq_len(length(outcomes))
    ) # N x T
    true_M[cp, ] <- colMeans(Y_full)
  }

  expect_equal(M_hat_imp, true_M, tolerance = comp_rel_tol(1e-8, M_hat_imp, true_M), scale = 1)
  # expect_equal(M_hat_dir, true_M, tolerance = comp_rel_tol(1e-8, M_hat_dir, true_M), scale = 1)
  # expect_equal(M_hat_imp, M_hat_dir, tolerance = comp_rel_tol(1e-8, M_hat_imp, M_hat_dir), scale = 1)

  # Also test the constituent steps yield the same result
  est1 <- est_cohort_specific_params(panel_obj, est_specs = est_specs, num_threads = 1L)
  
  # Take the single spec from the list and aggregate
  fmes_by_cohort <- est1$cohort_specific_factor_ests[[1]]
  # Equal weights by size are default when not specified; construct equal weights explicitly
  w_equal <- CohortWeightEstimates$new(make_cohort_weight_estimates_cpp(rep(1, C)))
  fme_agg <- aggregate_factor_model_params(fmes_by_cohort, panel_obj$get_observed_outcome_indices(), w_equal)
  imp_comps <- comp_imputation_components(panel_obj, fme_agg, est1$cohort_outcome_means, num_threads = 1L)
  ome <- estimate_outcome_means_across_cohorts(
    imp_comps,
    panel_obj$get_observed_outcome_indices(),
    est1$cohort_outcome_means
  )

  # Projection equivalence of aggregated factors to true global factors
  P_true <- projection_matrix_r(ctx$true_factors)
  P_hat  <- projection_matrix_r(imp_comps$G())
  expect_equal(P_hat, P_true, tolerance = comp_rel_tol(5e-7, P_hat, P_true))

  # Check cohort-specific factor estimates and observed outcome means against truth
  for (c in seq_along(cohort_indices)) {
    cp <- panel_idx_for_orig[c]
    observed_idxs <- cohort_indices[[c]]
    unit_ids <- units_by_cohort[[c]]

    G_hat_c <- matrix(0.0, nrow = length(outcomes), ncol = r)
    G_hat_c[observed_idxs, ] <- fmes_by_cohort[[cp]]$G()
    P_hat_c <- projection_matrix_r(G_hat_c)
    G_true_c <- matrix(0.0, nrow = length(outcomes), ncol = r)
    G_true_c[observed_idxs, ] <- ctx$true_factors[observed_idxs, ]
    P_true_c <- projection_matrix_r(G_true_c)
    expect_equal(P_hat_c, P_true_c, tolerance = comp_rel_tol(1e-7, P_hat_c, P_true_c))

    Y_c <- expected_Y_for_units_ctx(ctx, c, unit_ids = unit_ids, T_idx = observed_idxs) # N x T_c
    m_true_c <- colMeans(Y_c)
    m_hat_c <- est1$cohort_outcome_means[[cp]]$observed_outcome_means()
    expect_equal(m_hat_c, as.numeric(m_true_c), tolerance = comp_rel_tol(1e-8, m_hat_c, m_true_c), scale = 1)
  }

  expect_equal(ome$mean_outcomes(), true_M, tolerance = comp_rel_tol(1e-8, ome$mean_outcomes(), true_M), scale = 1)

  # One-shot and three-step pipelines must match exactly
  expect_equal(M_hat_imp, ome$mean_outcomes(), tolerance = comp_rel_tol(1e-8, M_hat_imp, ome$mean_outcomes()), scale = 1)
  # expect_equal(M_hat_dir, ome$mean_outcomes(), tolerance = comp_rel_tol(1e-8, M_hat_dir, ome$mean_outcomes()), scale = 1)
})

test_that("pipeline runs reasonably fast on a larger panel (optional perf check)", {
  skip_on_cran()
  if (!isTRUE(getOption("apm_run_perf_tests", FALSE))) skip("Set options(apm_run_perf_tests = TRUE) to enable perf checks.")

  n_threads <- get_cpp_default_concurrency()
  print(sprintf("Using up to %d threads", n_threads))

  Tval <- 30L
  r <- 2L
  window <- 3L

  outcomes <- make_outcomes(Tval)
  cohort_indices <- make_staircase_observed_indices(Tval, window)
  C <- length(cohort_indices)
  units_per_cohort <- 1000L
  units_by_cohort <- make_units_by_cohort(n_cohorts = C, units_per_cohort = units_per_cohort)

  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, include_outcome_fes = TRUE)
  panel <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, ctx = ctx)
  panel <- panel[sample(nrow(panel))]

  print(sprintf("Panel size: %d", nrow(panel)))

  set_apm_threads(1L)
  panel_construction_time <- system.time(panel_obj <- UnbalancedPanel$new(panel, "unit_id", "outcome_id", "y", model_rank = r))["elapsed"]
  # TODO: figure out data.table concurrency; right now multithreaded is slower than single-threaded
  # set_apm_threads(n_threads)
  # panel_construction_time_multi <- system.time(panel_obj_multi <- UnbalancedPanel$new(panel, "unit_id", "outcome_id", "y", model_rank = r))["elapsed"]

  print(sprintf("Single-threaded panel construction time: %f", panel_construction_time))
  # print(sprintf("Multi-threaded panel construction time: %f", panel_construction_time_multi))

  # Soft perf sanity: multi-thread not egregiously slower than single-thread
  # expect_lt(panel_construction_time_multi, panel_construction_time * 2.0 + 1.0)

  est_specs <- list(pc = list(factor_model_estimator = "principal_components", include_outcome_fes = TRUE, r = r))

  t1 <- system.time(res1_imp <- est_target_param_components(panel_obj, est_specs = est_specs, num_threads = 1L, est_outcome_means_via_imputation = TRUE))["elapsed"]
  t2 <- system.time(res2_imp <- est_target_param_components(panel_obj, est_specs = est_specs, num_threads = n_threads, est_outcome_means_via_imputation = TRUE))["elapsed"]
  # Also compute direct composition (no imputation) variants
  # res1_dir <- est_target_param_components(panel_obj, est_specs = est_specs, num_threads = 1L, est_outcome_means_via_imputation = FALSE)
  # res2_dir <- est_target_param_components(panel_obj, est_specs = est_specs, num_threads = n_threads, est_outcome_means_via_imputation = FALSE)

  print(sprintf("Elapsed time (1 thread): %f", t1))
  print(sprintf("Elapsed time (%d threads): %f", n_threads, t2))

  M1_imp <- res1_imp$outcome_means$pc$mean_outcomes()
  M2_imp <- res2_imp$outcome_means$pc$mean_outcomes()
  # M1_dir <- res1_dir$outcome_means$pc$mean_outcomes()
  # M2_dir <- res2_dir$outcome_means$pc$mean_outcomes()
  expect_equal(M1_imp, M2_imp, tolerance = comp_rel_tol(1e-6, M1_imp, M2_imp))
  # expect_equal(M1_dir, M2_dir, tolerance = comp_rel_tol(1e-6, M1_dir, M2_dir))
  # expect_equal(M1_imp, M1_dir, tolerance = comp_rel_tol(1e-6, M1_imp, M1_dir))
  # expect_equal(M2_imp, M2_dir, tolerance = comp_rel_tol(1e-6, M2_imp, M2_dir))

  # Align cohort order using exact index match from original to panel order
  ooi_panel <- panel_obj$get_observed_outcome_indices()
  panel_idx_for_orig <- match_cohorts_panel_order(cohort_indices, ooi_panel)
  expect_true(!any(is.na(panel_idx_for_orig)))

  # Correctness: both single- and multi-threaded results match truth computed via the generator
  true_M <- matrix(NA_real_, nrow = C, ncol = length(outcomes))
  for (c in seq_along(cohort_indices)) {
    cp <- panel_idx_for_orig[c]
    unit_ids <- units_by_cohort[[c]]
    Y_full <- expected_Y_for_units_ctx(
      ctx, c,
      unit_ids = unit_ids,
      T_idx = seq_len(length(outcomes))
    ) # N x T
    true_M[cp, ] <- colMeans(Y_full)
  }

  expect_equal(M1_imp, true_M, tolerance = comp_rel_tol(1e-4, M1_imp, true_M), scale = 1)
  expect_equal(M2_imp, true_M, tolerance = comp_rel_tol(1e-4, M2_imp, true_M), scale = 1)
  # expect_equal(M1_dir, true_M, tolerance = comp_rel_tol(1e-4, M1_dir, true_M), scale = 1)
  # expect_equal(M2_dir, true_M, tolerance = comp_rel_tol(1e-4, M2_dir, true_M), scale = 1)

  # Soft perf sanity: multi-thread not egregiously slower than single-thread
  expect_lt(as.numeric(t2), as.numeric(t1) * 2.0 + 1.0)
})