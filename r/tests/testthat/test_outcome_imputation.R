testthat::test_that("comp_imputation_components recovers g0 and L without covariates", {
  # Build synthetic panel similar to C++ tests
  T <- 7L; r <- 2L; T_c <- 3L; units_per <- 5L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = FALSE, include_outcome_fes = TRUE)
  G_true <- ctx$true_factors
  # Embed g0 in panel generating process via ctx
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = FALSE, include_auxiliary = FALSE,
                                              r = r, rotate = TRUE, ctx = ctx)

  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r)

  # Build FactorModelEstimates via helper (pass ctx$g0)
  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0))

  # Run comp
  N <- nrow(panel$get_unit_cohorts())
  wb <- get_weighted_bootstrap_draws(N, 1L, type = "multinomial", seed = 1L)
  out <- comp_imputation_components(panel, fme, weighted_bootstrap = wb)

  # Check span(G) unchanged (projection matrices equal)
  P_out <- projection_matrix_r(out$G())
  P_true <- projection_matrix_r(G_true)
  testthat::expect_equal(P_out, P_true, tolerance = 1e-9)

  # g0 should be equal (already orthogonalized against span(G))
  testthat::expect_true(out$has_g0())
  testthat::expect_equal(as.numeric(out$g0()), as.numeric(ctx$g0), tolerance = 1e-6)

  # L present and finite with expected dims: N x r
  L <- out$L()
  unit_cohorts <- panel$get_unit_cohorts()
  N <- nrow(unit_cohorts)
  testthat::expect_equal(dim(L), c(N, r))
  testthat::expect_true(all(is.finite(L)))
})

testthat::test_that("comp_imputation_components recovers alpha with covariates (no g0)", {
  T <- 7L; r <- 2L; T_c <- 3L; units_per <- 5L; q <- 2L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, include_outcome_fes = FALSE)
  a_true <- c(0.5, 1.0)
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = TRUE, include_auxiliary = FALSE,
                                              r = r, rotate = FALSE, ctx = ctx, a = a_true)

  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r, covar_cols = c("cov1", "cov2"))

  G_true <- ctx$true_factors
  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, a = a_true))

  N <- nrow(panel$get_unit_cohorts())
  wb <- get_weighted_bootstrap_draws(N, 1L, type = "multinomial", seed = 1L)
  out <- comp_imputation_components(panel, fme, weighted_bootstrap = wb)

  testthat::expect_true(out$has_a())
  testthat::expect_equal(as.numeric(out$a()), as.numeric(a_true), tolerance = 1e-4)
  P_out <- projection_matrix_r(out$G())
  P_true <- projection_matrix_r(G_true)
  testthat::expect_equal(P_out, P_true, tolerance = 1e-9)
})

testthat::test_that("comp_imputation_components recovers alpha with covariates and FE", {
  T <- 7L; r <- 2L; T_c <- 3L; units_per <- 5L; q <- 2L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, include_outcome_fes = TRUE)
  G_true <- ctx$true_factors
  a_true <- c(0.5, 1.0)

  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = TRUE, include_auxiliary = FALSE,
                                              r = r, rotate = FALSE, ctx = ctx, a = a_true)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r, covar_cols = c("cov1", "cov2"))

  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0, a = rep(0, q)))
  out <- comp_imputation_components(panel, fme)

  # Span(G) unchanged
  P_out <- projection_matrix_r(out$G())
  P_true <- projection_matrix_r(G_true)
  testthat::expect_equal(P_out, P_true, tolerance = 1e-9)

  # Alpha recovered
  testthat::expect_true(out$has_a())
  testthat::expect_equal(as.numeric(out$a()), as.numeric(a_true), tolerance = 1e-4)

  # g0 equals its orthogonal projection against span(G)
  g0_exp <- as.numeric(ctx$g0 - G_true %*% solve(crossprod(G_true), crossprod(G_true, ctx$g0)))
  testthat::expect_true(out$has_g0())
  testthat::expect_equal(as.numeric(out$g0()), g0_exp, tolerance = 1e-6)

  # L present with expected dims N x r
  L <- out$L()
  N <- nrow(panel$get_unit_cohorts())
  testthat::expect_equal(dim(L), c(N, r))
  testthat::expect_true(all(is.finite(L)))
})

testthat::test_that("comp_imputation_components with cohort sufficient stats (covariates and FE)", {
  T <- 8L; r <- 2L; T_c <- 3L; units_per <- 5L; q <- 2L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, include_outcome_fes = TRUE)
  G_true <- ctx$true_factors
  a_true <- c(0.5, 1.0)

  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = TRUE, include_auxiliary = FALSE,
                                              r = r, rotate = TRUE, ctx = ctx, a = a_true)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r, covar_cols = c("cov1", "cov2"))

  a0 <- rep(0, q)
  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0, a = a0))

  specs <- list(pca_fe = list(factor_model_estimator = "principal_components", include_outcome_fes = TRUE, r = r))
  cse <- est_cohort_specific_params(panel, specs)
  suff_stats <- cse$cohort_outcome_means

  out <- comp_imputation_components(panel, fme, cohort_outcome_mean_suff_stat_ests = suff_stats)

  # Span(G) unchanged
  P_out <- projection_matrix_r(out$G())
  P_true <- projection_matrix_r(G_true)
  testthat::expect_equal(P_out, P_true, tolerance = 1e-9)

  # Alpha and g0 recovered (g0 up to orthogonal projection against span(G))
  testthat::expect_true(out$has_a())
  testthat::expect_equal(as.numeric(out$a()), as.numeric(a_true), tolerance = 1e-8)
  g0_exp <- as.numeric(ctx$g0 - G_true %*% solve(crossprod(G_true), crossprod(G_true, ctx$g0)))
  testthat::expect_true(out$has_g0())
  testthat::expect_equal(as.numeric(out$g0()), g0_exp, tolerance = 1e-8)

  # L dims C x r; validate equals cohort mean unit loadings in panel order
  L <- out$L()
  Cval <- length(cohort_indices)
  testthat::expect_equal(dim(L), c(Cval, r))
  # Map original cohort indices to panel cohort ids
  ooi_panel <- panel$get_observed_outcome_indices()
  panel_idx_for_orig <- match_cohorts_panel_order(cohort_indices, ooi_panel)
  testthat::expect_true(!any(is.na(panel_idx_for_orig)))
  for (c in seq_len(Cval)) {
    cp <- panel_idx_for_orig[c]
    unit_ids <- units_by_cohort[[c]]
    l_mat <- do.call(rbind, lapply(unit_ids, function(u) unit_loading_from_all_units(u, ctx$all_units, ctx$r, c, length(units_by_cohort))))
    mean_l <- colMeans(l_mat)
    testthat::expect_equal(as.numeric(L[cp, ]), as.numeric(mean_l), tolerance = 1e-5)
  }
})

testthat::test_that("comp_imputation_components handles all-ones factors with FE and no covariates", {
  T <- 6L; r <- 1L; T_c <- 3L; units_per <- 4L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = FALSE, include_outcome_fes = TRUE)
  G_true <- matrix(1, nrow = T, ncol = r)
  # build panel using ctx (which embeds g0)
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = FALSE, include_auxiliary = FALSE,
                                              r = r, rotate = FALSE, ctx = ctx)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r)

  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0))
  out <- comp_imputation_components(panel, fme)

  # G is ones, no a, g0 present
  testthat::expect_equal(out$G(), G_true)
  testthat::expect_false(out$has_a())
  testthat::expect_true(out$has_g0())
})

testthat::test_that("comp_imputation_components handles by-spec map dispatch", {
  T <- 6L; r <- 2L; T_c <- 3L; units_per <- 4L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = FALSE, include_outcome_fes = TRUE)
  G_true <- ctx$true_factors
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = FALSE, include_auxiliary = FALSE,
                                              r = r, rotate = TRUE, ctx = ctx)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r)

  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0))

  N <- nrow(panel$get_unit_cohorts())
  wb <- get_weighted_bootstrap_draws(N, 1L, type = "multinomial", seed = 1L)
  res <- comp_imputation_components(panel, list(spec1 = fme, spec2 = fme), weighted_bootstrap = wb)
  testthat::expect_true(all(sort(names(res)) == c("spec1", "spec2")))
  for (k in names(res)) {
    outk <- res[[k]]
    testthat::expect_true(outk$has_g0())
    testthat::expect_equal(as.numeric(outk$g0()), as.numeric(ctx$g0), tolerance = 1e-6)
    P_out <- projection_matrix_r(outk$G())
    P_true <- projection_matrix_r(G_true)
    testthat::expect_equal(P_out, P_true, tolerance = 1e-9)
  }
})

testthat::test_that("comp_imputation_components LSMR recovers alpha with covariates and FE", {
  T <- 7L; r <- 2L; T_c <- 3L; units_per <- 5L; q <- 2L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, include_outcome_fes = TRUE)
  G_true <- ctx$true_factors
  a_true <- c(0.5, 1.0)

  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = TRUE, include_auxiliary = FALSE,
                                              r = r, rotate = FALSE, ctx = ctx, a = a_true)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r, covar_cols = c("cov1", "cov2"))

  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0, a = rep(0, q)))
  out <- comp_imputation_components(panel, fme, imputation_options = list(solver = "lsmr"))

  # Span(G) unchanged
  P_out <- projection_matrix_r(out$G())
  P_true <- projection_matrix_r(G_true)
  testthat::expect_equal(P_out, P_true, tolerance = 1e-9)

  # Alpha recovered
  testthat::expect_true(out$has_a())
  testthat::expect_equal(as.numeric(out$a()), as.numeric(a_true), tolerance = 1e-4)

  # g0 equals its orthogonal projection against span(G)
  g0_exp <- as.numeric(ctx$g0 - G_true %*% solve(crossprod(G_true), crossprod(G_true, ctx$g0)))
  testthat::expect_true(out$has_g0())
  testthat::expect_equal(as.numeric(out$g0()), g0_exp, tolerance = 1e-6)

  # L present with expected dims N x r
  L <- out$L()
  N <- nrow(panel$get_unit_cohorts())
  testthat::expect_equal(dim(L), c(N, r))
  testthat::expect_true(all(is.finite(L)))
})

testthat::test_that("comp_imputation_components LSMR recovers g0 and L without covariates (FE present)", {
  T <- 7L; r <- 2L; T_c <- 3L; units_per <- 5L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, include_outcome_fes = TRUE)
  G_true <- ctx$true_factors

  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = FALSE, include_auxiliary = FALSE,
                                              r = r, rotate = TRUE, ctx = ctx)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r)

  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0))
  out <- comp_imputation_components(panel, fme, imputation_options = list(solver = "lsmr"))

  # Span(G) unchanged; no alpha; g0 projection recovered
  P_out <- projection_matrix_r(out$G())
  P_true <- projection_matrix_r(G_true)
  testthat::expect_equal(P_out, P_true, tolerance = 1e-9)
  testthat::expect_false(out$has_a())
  g0_exp <- as.numeric(ctx$g0 - G_true %*% solve(crossprod(G_true), crossprod(G_true, ctx$g0)))
  testthat::expect_true(out$has_g0())
  testthat::expect_equal(as.numeric(out$g0()), g0_exp, tolerance = 1e-6)

  # L present with expected dims N x r
  L <- out$L()
  N <- nrow(panel$get_unit_cohorts())
  testthat::expect_equal(dim(L), c(N, r))
  testthat::expect_true(all(is.finite(L)))
})

testthat::test_that("comp_imputation_components LSMR handles all-ones G with FE and covariates", {
  T <- 6L; r <- 1L; T_c <- 3L; units_per <- 4L; q <- 2L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = FALSE, include_outcome_fes = TRUE)
  G_true <- matrix(1, nrow = T, ncol = r)
  a_true <- c(0.5, 1.0)

  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = TRUE, include_auxiliary = FALSE,
                                              r = r, rotate = FALSE, ctx = ctx, a = a_true)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r, covar_cols = c("cov1", "cov2"))

  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0, a = rep(0, q)))
  out <- comp_imputation_components(panel, fme, imputation_options = list(solver = "lsmr"))

  testthat::expect_equal(out$G(), G_true)
  testthat::expect_true(out$has_g0())
  testthat::expect_true(out$has_a())

  # Validate L entries via mean residual per unit over observed outcomes
  L <- out$L()  # N x 1
  unit_cohorts <- panel$get_unit_cohorts()
  units <- unit_cohorts[["unit_id"]]
  a_hat <- as.numeric(out$a())
  g0_hat <- as.numeric(out$g0())
  for (i in seq_len(nrow(unit_cohorts))) {
    u <- units[i]
    rows <- panel_dt[panel_dt$unit_id == u & is.finite(panel_dt$y), ]
    if (nrow(rows) == 0L) {
      next
    }
    t_idx <- match(rows$outcome_id, outcomes)
    resid <- rows$y - g0_hat[t_idx] - (a_hat[1] * rows$cov1 + a_hat[2] * rows$cov2)
    mean_resid <- mean(resid)
    testthat::expect_equal(as.numeric(L[i, 1]), as.numeric(mean_resid), tolerance = 1e-6)
  }
})

testthat::test_that("comp_imputation_components LSMR handles all-ones G with FE only (no covariates)", {
  T <- 6L; r <- 1L; T_c <- 3L; units_per <- 4L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = FALSE, include_outcome_fes = TRUE)
  G_true <- matrix(1, nrow = T, ncol = r)

  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = FALSE, include_auxiliary = FALSE,
                                              r = r, rotate = FALSE, ctx = ctx)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r)

  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0))
  out <- comp_imputation_components(panel, fme, imputation_options = list(solver = "lsmr"))

  testthat::expect_equal(out$G(), G_true)
  testthat::expect_false(out$has_a())
  testthat::expect_true(out$has_g0())

  # Validate L entries via mean residual per unit over observed outcomes
  L <- out$L()  # N x 1
  unit_cohorts <- panel$get_unit_cohorts()
  units <- unit_cohorts[["unit_id"]]
  g0_hat <- as.numeric(out$g0())
  for (i in seq_len(nrow(unit_cohorts))) {
    u <- units[i]
    rows <- panel_dt[panel_dt$unit_id == u & is.finite(panel_dt$y), ]
    if (nrow(rows) == 0L) {
      next
    }
    t_idx <- match(rows$outcome_id, outcomes)
    resid <- rows$y - g0_hat[t_idx]
    mean_resid <- mean(resid)
    testthat::expect_equal(as.numeric(L[i, 1]), as.numeric(mean_resid), tolerance = 1e-6)
  }
})

testthat::test_that("comp_imputation_components LSMR with cohort sufficient stats (covariates and FE)", {
  T <- 8L; r <- 2L; T_c <- 3L; units_per <- 5L; q <- 2L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE, include_outcome_fes = TRUE)
  G_true <- ctx$true_factors
  a_true <- c(0.5, 1.0)

  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = TRUE, include_auxiliary = FALSE,
                                              r = r, rotate = TRUE, ctx = ctx, a = a_true)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r, covar_cols = c("cov1", "cov2"))

  a0 <- rep(0, q)
  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = ctx$g0, a = a0))

  specs <- list(pca_fe = list(factor_model_estimator = "principal_components", include_outcome_fes = TRUE, r = r))
  cse <- est_cohort_specific_params(panel, specs)
  suff_stats <- cse$cohort_outcome_means

  out <- comp_imputation_components(panel, fme, cohort_outcome_mean_suff_stat_ests = suff_stats, imputation_options = list(solver = "lsmr"))

  # Span(G) unchanged
  P_out <- projection_matrix_r(out$G())
  P_true <- projection_matrix_r(G_true)
  testthat::expect_equal(P_out, P_true, tolerance = 1e-9)

  # Alpha and g0 recovered (g0 up to orthogonal projection against span(G))
  testthat::expect_true(out$has_a())
  testthat::expect_equal(as.numeric(out$a()), as.numeric(a_true), tolerance = 1e-8)
  g0_exp <- as.numeric(ctx$g0 - G_true %*% solve(crossprod(G_true), crossprod(G_true, ctx$g0)))
  testthat::expect_true(out$has_g0())
  testthat::expect_equal(as.numeric(out$g0()), g0_exp, tolerance = 1e-8)

  # L dims C x r; validate equals cohort mean unit loadings in panel order (LSMR)
  L <- out$L()
  Cval <- length(cohort_indices)
  testthat::expect_equal(dim(L), c(Cval, r))
  ooi_panel <- panel$get_observed_outcome_indices()
  panel_idx_for_orig <- match_cohorts_panel_order(cohort_indices, ooi_panel)
  testthat::expect_true(!any(is.na(panel_idx_for_orig)))
  for (c in seq_len(Cval)) {
    cp <- panel_idx_for_orig[c]
    unit_ids <- units_by_cohort[[c]]
    l_mat <- do.call(rbind, lapply(unit_ids, function(u) unit_loading_from_all_units(u, ctx$all_units, ctx$r, c, length(units_by_cohort))))
    mean_l <- colMeans(l_mat)
    testthat::expect_equal(as.numeric(L[cp, ]), as.numeric(mean_l), tolerance = 1e-5)
  }
})