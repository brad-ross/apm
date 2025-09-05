context("Testing cohort-specific parameter estimation")

library(data.table)

# test_that("est_cohort_specific_params validates spec fields and estimator name", {
#     outcomes <- make_outcomes(3)
#     cohort_indices <- make_staircase_observed_indices(3, 2)
#     units_by_cohort <- make_units_by_cohort(2, 2)

#     panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, r = 1L)
#     panel <- UnbalancedPanel$new(
#         panel_df = panel_dt,
#         unit_id_col = "unit_id",
#         outcome_id_col = "outcome_id",
#         outcome_value_col = "y",
#         model_rank = 1,
#         min_cohort_size = 1
#     )

#     # missing field r
#     est_specs_missing <- list(list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE))
#     expect_error(est_cohort_specific_params(panel, est_specs_missing), regexp = "missing fields")

#     # unsupported estimator name should error from C++ core
#     est_specs_bad <- list(list(factor_model_estimator = "other", include_outcome_fes = FALSE, r = 1L))
#     expect_error(est_cohort_specific_params(panel, est_specs_bad))
# })

test_that("wrapper returns FactorModelEstimates with expected dimensions", {
    outcomes <- make_outcomes(4)
    cohort_indices <- make_staircase_observed_indices(4, 2)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), 2)

    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, r = 2L)
    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 1
    )

    est_specs <- list(spec = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L))
    res <- est_cohort_specific_params(panel, est_specs)
    out <- res$cohort_specific_factor_ests[[1]][[1]]
    expect_true(inherits(out, "FactorModelEstimates"))
    expect_equal(ncol(out$G()), 2L)
    expect_equal(nrow(out$G()), length(cohort_indices[[1]]))
})

test_that("covariate means are computed with expected dimensions and values", {
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

    est_specs <- list(pc = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L))
    res <- est_cohort_specific_params(panel, est_specs)
    oms <- res$cohort_outcome_means[[1]]
    CM <- oms$covar_means()
    expect_true(is.matrix(CM))
    expect_equal(dim(CM), c(length(cohort_indices[[1]]), 2L))
    # cov1 across cohort 1 has two units with values 1 and 2 at all outcomes -> mean 1.5
    expect_true(all(CM[, 1] == 1.5))
    # cov2 is cohort id constant (=1)
    expect_true(all(CM[, 2] == 1))
})

test_that("q=0 flows without covariate means", {
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3)
    units_by_cohort <- make_units_by_cohort(3, 2)

    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, r = 2L)

    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2
    )

    est_specs <- list(pc = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L))
    res <- est_cohort_specific_params(panel, est_specs)
    oms <- res$cohort_outcome_means[[1]]
    expect_null(oms$covar_means())
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
        pca = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L),
        pca_fe = list(factor_model_estimator = "principal_components", include_outcome_fes = TRUE, r = 2L)
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

test_that("bootstrap flows through and produces replicates", {
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3)
    units_by_cohort <- make_units_by_cohort(3, 3)

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

    # N equals number of unique units in processed panel
    N <- nrow(unique(panel$get_processed_panel()[, .(unit_idx)]))
    B <- 3L
    wb <- get_weighted_bootstrap_draws(N = N, B = B, type = "bayesian", seed = 123L)

    est_specs <- list(
        pca = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L)
    )

    res <- est_cohort_specific_params(panel, est_specs, bootstrap = wb)

    # Factor model bootstrap replicates present per cohort
    f <- res$cohort_specific_factor_ests[["pca"]]
    expect_equal(length(f), length(cohort_indices))
    expect_true(all(vapply(f, function(e) e$num_bootstraps() == B, logical(1))))

    # Outcome sufficient stats also have bootstrap replicates
    oms <- res$cohort_outcome_means
    expect_equal(length(oms), length(cohort_indices))
    expect_true(all(vapply(oms, function(e) e$num_bootstraps() == B, logical(1))))
})