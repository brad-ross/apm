context("Integration tests of est_target_param_components")

test_that("recovers true cohort mean outcomes on staircase data", {
  # Problem size and staircase layout
  Tval <- 12L
  r <- 2L
  window <- 3L

  # Build staircase cohorts and deterministic context
  outcomes <- make_outcomes(Tval)
  cohort_indices <- make_staircase_observed_indices(Tval, window)
  C <- length(cohort_indices)
  units_per_cohort <- 5L
  units_by_cohort <- make_units_by_cohort(n_cohorts = C, units_per_cohort = units_per_cohort, prefix = "u")

  # One shared ctx used for both panel generation and truth construction
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE)

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
      include_outcome_fes = FALSE,
      r = r
    )
  )

  # Run end-to-end estimator
  res <- est_target_param_components(panel_obj, est_specs = est_specs, num_threads = 1L)
  M_hat <- res$pc$mean_outcomes()  # C x T

  # Align cohort order using exact index match from original to panel order
  ooi_panel <- panel_obj$get_observed_outcome_indices()
  panel_idx_for_orig <- match_cohorts_panel_order(cohort_indices, ooi_panel)

  # Ground truth per cohort using the same ctx (global basis)
  true_M <- matrix(NA_real_, nrow = C, ncol = length(outcomes))
  for (c in seq_along(cohort_indices)) {
    cp <- panel_idx_for_orig[c]
    unit_ids <- units_by_cohort[[c]]
    L_mat <- do.call(rbind, lapply(unit_ids, function(u) unit_loading_from_all_units(u, ctx$all_units, ctx$r, c, C)))
    # Mean loadings are defined in the original global basis; no cohort rotation is applied
    lbar <- colMeans(L_mat)
    true_M[cp, ] <- as.numeric(ctx$true_factors %*% lbar)
  }

  expect_equal(M_hat, true_M, tolerance = comp_rel_tol(1e-8, M_hat, true_M), scale = 1)

  # Also test the constituent steps yield the same result
  est1 <- est_cohort_specific_params(panel_obj, est_specs = est_specs, num_threads = 1L)
  
  # Take the single spec from the list and aggregate
  fmes_by_cohort <- est1$cohort_specific_factor_ests[[1]]
  # Equal weights by size are default when not specified; construct equal weights explicitly
  w_equal <- CohortWeightEstimates$new(make_cohort_weight_estimates_cpp(rep(1, C)))
  fme_agg <- aggregate_factor_model_params(fmes_by_cohort, panel_obj$get_observed_outcome_indices(), w_equal)
  ome <- estimate_outcome_means_across_cohorts(
    fme_agg,
    panel_obj$get_observed_outcome_indices(),
    est1$cohort_outcome_means
  )

  # Projection equivalence of aggregated factors to true global factors
  P_true <- projection_matrix_r(ctx$true_factors)
  P_hat  <- projection_matrix_r(fme_agg$G())
  expect_equal(P_hat, P_true, tolerance = comp_rel_tol(1e-8, P_hat, P_true))

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
  expect_equal(M_hat, ome$mean_outcomes(), tolerance = comp_rel_tol(1e-8, M_hat, ome$mean_outcomes()), scale = 1)
})

test_that("pipeline runs reasonably fast on a larger panel (optional perf check)", {
  skip_on_cran()
  if (!isTRUE(getOption("apm_run_perf_tests", FALSE))) skip("Set options(apm_run_perf_tests = TRUE) to enable perf checks.")

  Tval <- 30L
  r <- 2L
  window <- 3L

  outcomes <- make_outcomes(Tval)
  cohort_indices <- make_staircase_observed_indices(Tval, window)
  C <- length(cohort_indices)
  units_per_cohort <- 1000L
  units_by_cohort <- make_units_by_cohort(n_cohorts = C, units_per_cohort = units_per_cohort)

  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE)
  panel <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, ctx = ctx)
  panel <- panel[sample(nrow(panel))]

  panel_construction_time <- system.time(panel_obj <- UnbalancedPanel$new(panel, "unit_id", "outcome_id", "y", model_rank = r))["elapsed"]
  set_apm_threads(get_cpp_default_concurrency())
  panel_construction_time_multi <- system.time(panel_obj_multi <- UnbalancedPanel$new(panel, "unit_id", "outcome_id", "y", model_rank = r))["elapsed"]

  print(sprintf("Single-threaded panel construction time: %f", panel_construction_time))
  print(sprintf("Multi-threaded panel construction time: %f", panel_construction_time_multi))

  # Soft perf sanity: multi-thread not egregiously slower than single-thread
  expect_lt(panel_construction_time_multi, panel_construction_time * 2.0 + 1.0)

  est_specs <- list(pc = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = r))

  t1 <- system.time(res1 <- est_target_param_components(panel_obj, est_specs = est_specs, num_threads = 1L))["elapsed"]
  t2 <- system.time(res2 <- est_target_param_components(panel_obj, est_specs = est_specs))["elapsed"]

  print(sprintf("Elapsed time (1 thread): %f", t1))
  print(sprintf("Elapsed time (%d threads): %f", get_cpp_default_concurrency(), t2))

  M1 <- res1$pc$mean_outcomes()
  M2 <- res2$pc$mean_outcomes()
  expect_equal(M1, M2, tolerance = comp_rel_tol(1e-10, M1, M2))

  # Align cohort order using exact index match from original to panel order
  ooi_panel <- panel_obj$get_observed_outcome_indices()
  panel_idx_for_orig <- match_cohorts_panel_order(cohort_indices, ooi_panel)
  expect_true(!any(is.na(panel_idx_for_orig)))

  # Correctness: both single- and multi-threaded results match truth
  true_M <- matrix(NA_real_, nrow = C, ncol = length(outcomes))
  for (c in seq_along(cohort_indices)) {
    cp <- panel_idx_for_orig[c]
    unit_ids <- units_by_cohort[[c]]
    L_mat <- do.call(rbind, lapply(unit_ids, function(u) unit_loading_from_all_units(u, ctx$all_units, ctx$r)))
    lbar <- colMeans(L_mat)
    true_M[cp, ] <- as.numeric(ctx$true_factors %*% lbar)
  }

  expect_equal(M1, true_M, tolerance = comp_rel_tol(1e-8, M1, true_M), scale = 1)
  expect_equal(M2, true_M, tolerance = comp_rel_tol(1e-8, M2, true_M), scale = 1)

  # Soft perf sanity: multi-thread not egregiously slower than single-thread
  expect_lt(as.numeric(t2), as.numeric(t1) * 2.0 + 1.0)
})