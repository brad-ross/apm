context("Testing cohort-specific parameter estimation helpers")

library(data.table)

test_that("validate_est_specs_ enforces required fields and values", {
    # valid
    expect_silent(apm:::validate_est_specs_(list(list(
        name = "principal_components",
        include_outcome_fes = FALSE,
        r = 2
    ))))

    # missing field
    expect_error(apm:::validate_est_specs_(list(list(
        name = "principal_components",
        r = 2
    ))), regexp = "missing fields")

    # unsupported name
    expect_error(apm:::validate_est_specs_(list(list(
        name = "other",
        include_outcome_fes = TRUE,
        r = 2
    ))), regexp = "Only 'principal_components'")

    # include_outcome_fes must be logical(1)
    expect_error(apm:::validate_est_specs_(list(list(
        name = "principal_components",
        include_outcome_fes = c(TRUE, FALSE),
        r = 2
    ))))

    # r must be positive integer-like
    expect_error(apm:::validate_est_specs_(list(list(
        name = "principal_components",
        include_outcome_fes = FALSE,
        r = 0
    ))))
})

test_that("new_estimator_from_spec_ instantiates correct estimator", {
    spec1 <- list(name = "principal_components", include_outcome_fes = FALSE, r = 1)
    e1 <- apm:::new_estimator_from_spec_(spec1, T_c = 3, q = 0, bootstrap = NULL)
    expect_true(inherits(e1, "PCEstimator"))
    expect_equal(e1$r(), 1L)
    expect_equal(e1$T_c(), 3L)
    expect_equal(e1$q(), 0L)

    spec2 <- list(name = "principal_components", include_outcome_fes = TRUE, r = 2)
    e2 <- apm:::new_estimator_from_spec_(spec2, T_c = 4, q = 2, bootstrap = NULL)
    expect_true(inherits(e2, "PCEstimatorWithFEs"))
    expect_equal(e2$r(), 2L)
    expect_equal(e2$T_c(), 4L)
    expect_equal(e2$q(), 2L)
})

test_that("build_Y_X_for_group_ reshapes Y and X correctly (with covariates)", {
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3)
    units_by_cohort <- make_units_by_cohort(3, 2)

    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = TRUE, r = 2L)

    obj <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        covar_cols = c("cov1", "cov2")
    )

    pp <- obj$get_processed_panel()
    T_idx <- obj$get_observed_outcome_indices()[[1]]
    sd <- pp[cohort_id == 1L]

    yx <- apm:::build_Y_X_for_group_(sd, T_idx, obj$get_covar_cols())

    expect_true(is.matrix(yx$Y))
    expect_equal(dim(yx$Y), c(2L, length(T_idx)))
    expect_true(!any(is.na(yx$Y)))

    # Contents: construct expected Y from factor model used in helper
    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    expected_Y <- expected_Y_for_units_ctx(ctx, cohort_id = 1L, unit_ids = units_by_cohort[[1]], T_idx = T_idx)
    expect_equal(max(abs(yx$Y - expected_Y)), 0, tolerance = 1e-12)

    expect_true(is.array(yx$X))
    expect_equal(dim(yx$X), c(2L, length(outcomes), 2L))

    # cov1 is unit index within global unit ordering (u1..u6) -> cohort 1 has 1,2
    expected_cov1 <- matrix(c(rep(1L, length(outcomes)), rep(2L, length(outcomes))), nrow = 2, byrow = TRUE)
    expect_equal(yx$X[, , 1], expected_cov1)
    # cov2 is cohort id constant (=1) across units/outcomes
    expect_equal(yx$X[, , 2], matrix(1L, nrow = 2, ncol = length(outcomes)))
})

test_that("build_Y_X_for_group_ handles q=0 (no covariates)", {
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3)
    units_by_cohort <- make_units_by_cohort(3, 2)

    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, r = 2L)

    obj <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2
    )

    pp <- obj$get_processed_panel()
    T_idx <- obj$get_observed_outcome_indices()[[1]]
    sd <- pp[cohort_id == 1L]

    yx <- apm:::build_Y_X_for_group_(sd, T_idx, obj$get_covar_cols())
    expect_true(is.matrix(yx$Y))
    expect_equal(dim(yx$Y), c(2L, length(T_idx)))
    expect_null(yx$X)

    # Contents: expected Y as in the covariate case
    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    expected_Y <- expected_Y_for_units_ctx(ctx, cohort_id = 1L, unit_ids = units_by_cohort[[1]], T_idx = T_idx)
    expect_equal(max(abs(yx$Y - expected_Y)), 0, tolerance = 1e-12)
})

test_that("est_cohort_specific_params integrates estimators per cohort", {
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3)
    units_by_cohort <- make_units_by_cohort(3, 2)

    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = TRUE, r = 2L)

    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        covar_cols = c("cov1", "cov2")
    )

    est_specs <- list(
        pca = list(name = "principal_components", include_outcome_fes = FALSE, r = 2L),
        pca_fe = list(name = "principal_components", include_outcome_fes = TRUE, r = 2L)
    )

    res <- est_cohort_specific_params(panel, est_specs)

    # outer structure keys
    expect_setequal(names(res), c("cohort_specific_factor_ests", "cohort_outcome_means"))

    # factor ests keyed by spec name, then cohort ids
    f <- res$cohort_specific_factor_ests
    expect_equal(sort(names(f)), sort(c("pca", "pca_fe")))
    # lists are indexed by numeric cohort_id
    expect_equal(length(f[["pca"]]), length(cohort_indices))
    expect_equal(length(f[["pca_fe"]]), length(cohort_indices))

    # check one cohort's factor model outputs
    out_no_fe <- f[["pca"]][[1]]
    out_fe <- f[["pca_fe"]][[1]]

    expect_true(inherits(out_no_fe, "FactorModelEstimates"))
    expect_true(inherits(out_fe, "FactorModelEstimates"))

    # Dimensions and attributes
    expect_equal(nrow(out_no_fe$G()), length(cohort_indices[[1]]))
    expect_equal(ncol(out_no_fe$G()), 2L)
    expect_false(out_no_fe$has_g0())

    expect_equal(nrow(out_fe$G()), length(cohort_indices[[1]]))
    expect_equal(ncol(out_fe$G()), 2L)
    expect_true(out_fe$has_g0())

    # Content: estimated cohort factor spans should match true rotated spans
    T_idx <- cohort_indices[[1]]
    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    G1_true <- ctx$cohort_G_list[[1]]

    proj_est <- projection_matrix_r(out_no_fe$G())
    proj_true <- projection_matrix_r(G1_true)
    expect_equal(proj_est, proj_true, tolerance = 1e-9)

    # outcome mean sufficient statistics per cohort
    oms <- res$cohort_outcome_means
    expect_equal(length(oms), length(cohort_indices))
    expect_true(inherits(oms[[1]], "OutcomeMeanSuffStatEstimates"))
    # observed means length should equal number of observed outcomes for cohort 1
    expect_equal(length(oms[[1]]$observed_outcome_means()), length(cohort_indices[[1]]))

    # observed means should equal true outcome means across units for cohort 1
    expected_Y_mat <- expected_Y_for_units_ctx(ctx, cohort_id = 1L, unit_ids = units_by_cohort[[1]], T_idx = T_idx)
    expected_means <- colMeans(expected_Y_mat)
    expect_equal(oms[[1]]$observed_outcome_means(), as.numeric(expected_means), tolerance = 1e-12)
})