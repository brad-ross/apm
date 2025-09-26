testthat::test_that("comp_imputation_components recovers g0 and L without covariates", {
  # Build synthetic panel similar to C++ tests
  T <- 7L; r <- 2L; T_c <- 3L; units_per <- 5L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE)
  # Generate g0 per C++ tests logic: arbitrary then orthogonalize w.r.t. span(G)
  set.seed(1)
  g0_raw <- as.numeric(seq_len(T)) / (T + 1)
  G_true <- ctx$true_factors
  g0_proj <- as.numeric(g0_raw - G_true %*% solve(crossprod(G_true), crossprod(G_true, g0_raw)))

  # Embed g0 in panel generating process
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = FALSE, include_auxiliary = FALSE,
                                              r = r, rotate = TRUE, ctx = ctx, g0 = g0_proj)

  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r)

  # Build FactorModelEstimates via helper (pass orthogonalized g0)
  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = g0_proj))

  # Run comp
  out <- comp_imputation_components(panel, fme)

  # Check span(G) unchanged (projection matrices equal)
  P_out <- projection_matrix_r(out$G())
  P_true <- projection_matrix_r(G_true)
  testthat::expect_equal(P_out, P_true, tolerance = 1e-9)

  # g0 should be equal (already orthogonalized against span(G))
  testthat::expect_true(out$has_g0())
  testthat::expect_equal(as.numeric(out$g0()), as.numeric(g0_proj), tolerance = 1e-6)

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
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE)
  a_true <- c(0.5, 1.0)
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = TRUE, include_auxiliary = FALSE,
                                              r = r, rotate = TRUE, ctx = ctx, a = a_true)

  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r, covar_cols = c("cov1", "cov2"))

  G_true <- ctx$true_factors
  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, a = a_true))

  out <- comp_imputation_components(panel, fme)

  testthat::expect_true(out$has_a())
  testthat::expect_equal(as.numeric(out$a()), as.numeric(a_true), tolerance = 1e-4)
  P_out <- projection_matrix_r(out$G())
  P_true <- projection_matrix_r(G_true)
  testthat::expect_equal(P_out, P_true, tolerance = 1e-9)
})

testthat::test_that("comp_imputation_components handles by-spec map dispatch", {
  T <- 6L; r <- 2L; T_c <- 3L; units_per <- 4L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), units_per)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = TRUE)
  g0_raw <- seq_len(T) / (T + 1)
  G_true <- ctx$true_factors
  g0_proj <- as.numeric(g0_raw - G_true %*% solve(crossprod(G_true), crossprod(G_true, g0_raw)))
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                              include_covariates = FALSE, include_auxiliary = FALSE,
                                              r = r, rotate = TRUE, ctx = ctx, g0 = g0_proj)
  panel <- UnbalancedPanel$new(panel_dt, unit_id_col = "unit_id", outcome_id_col = "outcome_id", outcome_value_col = "y", model_rank = r)

  fme <- FactorModelEstimates$new(make_factor_model_estimates_cpp(G_true, g0 = g0_proj))

  res <- comp_imputation_components(panel, list(spec1 = fme, spec2 = fme))
  testthat::expect_true(all(sort(names(res)) == c("spec1", "spec2")))
  for (k in names(res)) {
    outk <- res[[k]]
    testthat::expect_true(outk$has_g0())
    testthat::expect_equal(as.numeric(outk$g0()), as.numeric(g0_proj), tolerance = 1e-6)
    P_out <- projection_matrix_r(outk$G())
    P_true <- projection_matrix_r(G_true)
    testthat::expect_equal(P_out, P_true, tolerance = 1e-9)
  }
})


