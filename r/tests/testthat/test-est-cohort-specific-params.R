context("Testing cohort-specific parameter estimation")

library(data.table)

# test_that("est_cohort_specific_params validates spec fields and estimator name", {
#     T <- 3L
#     T_c <- 2L
#     outcomes <- make_outcomes(T)
#     cohort_indices <- make_staircase_observed_indices(T, T_c)
#     units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

#     panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, r = 1L)
#     panel <- UnbalancedPanel$new(
#         panel_df = panel_dt,
#         unit_id_col = "unit_id",
#         outcome_id_col = "outcome_id",
#         outcome_value_col = "y",
#         model_rank = 1,
#         min_cohort_size = 1,
#         sort_cohorts_lexicographically = TRUE
#     )

#     # missing field r
#     est_specs_missing <- list(list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE))
#     expect_error(est_cohort_specific_params(panel, est_specs_missing), regexp = "missing fields")

#     # unsupported estimator name should error from C++ core
#     est_specs_bad <- list(list(factor_model_estimator = "other", include_outcome_fes = FALSE, r = 1L))
#     expect_error(est_cohort_specific_params(panel, est_specs_bad))
# })

test_that("wrapper returns FactorModelEstimates with expected dimensions", {
    T <- 4L
    T_c <- 2L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, ctx = ctx)
    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 1,
        sort_cohorts_lexicographically = TRUE
    )

    est_specs <- list(spec = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L))
    res <- est_cohort_specific_params(panel, est_specs)
    out <- res$cohort_specific_factor_ests[[1]][[1]]
    expect_true(inherits(out, "FactorModelEstimates"))
    expect_equal(ncol(out$G()), 2L)
    expect_equal(nrow(out$G()), length(cohort_indices[[1]]))
})

test_that("cohort weights default to equal (1/C) without bootstrap", {
    T <- 4L
    T_c <- 3L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, ctx = ctx)
    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 1,
        sort_cohorts_lexicographically = TRUE
    )

    est_specs <- list(spec = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L))
    res <- est_cohort_specific_params(panel, est_specs)

    expect_true("cohort_weights" %in% names(res))
    w <- res$cohort_weights[["spec"]]
    expect_true(inherits(w, "CohortWeightEstimates"))
    expect_false(w$has_bootstrap())
    C <- length(cohort_indices)
    expect_equal(as.numeric(w$cohort_weights()), rep(1 / C, C))
})

test_that("covariate means are computed with expected dimensions and values", {
    T <- 5L
    T_c <- 3L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = TRUE, ctx = ctx)

    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        covar_cols = c("cov1", "cov2"),
        sort_cohorts_lexicographically = TRUE
    )

    est_specs <- list(pc = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L))
    res <- est_cohort_specific_params(panel, est_specs)
    oms <- res$cohort_outcome_means[[1]]
    CM <- oms$covar_means()
    expect_true(is.matrix(CM))
    expect_equal(dim(CM), c(length(outcomes), 2L))
    # cov1 across cohort 1 has two units with values 1 and 2 at all outcomes -> mean 1.5
    expect_true(all(CM[, 1] == 2))
    # cov2 is cohort id constant (=1)
    expect_true(all(CM[, 2] == 1))
})

test_that("q=0 flows without covariate means", {
    T <- 5L
    T_c <- 3L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, ctx = ctx)

    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        sort_cohorts_lexicographically = TRUE
    )

    est_specs <- list(pc = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L))
    res <- est_cohort_specific_params(panel, est_specs)
    oms <- res$cohort_outcome_means[[1]]
    expect_null(oms$covar_means())
})

test_that("est_cohort_specific_params integrates estimators per cohort", {
    T <- 12L
    T_c <- 3L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, ctx = ctx)

    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        sort_cohorts_lexicographically = TRUE
    )
    

    est_specs <- list(
        pca = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L),
        pca_fe = list(factor_model_estimator = "principal_components", include_outcome_fes = TRUE, r = 2L)
    )

    res <- est_cohort_specific_params(panel, est_specs)

    # outer structure keys: auxiliary means are optional
    expect_true(all(c("cohort_specific_factor_ests", "cohort_outcome_means", "cohort_weights") %in% names(res)))
    expect_false("cohort_auxiliary_means" %in% names(res))

    # factor ests keyed by spec name, then cohort ids
    f <- res$cohort_specific_factor_ests
    expect_equal(sort(names(f)), sort(c("pca", "pca_fe")))
    # lists are indexed by numeric cohort_id
    expect_equal(length(f[["pca"]]), length(cohort_indices))
    expect_equal(length(f[["pca_fe"]]), length(cohort_indices))

    # outcome mean sufficient statistics per cohort
    oms <- res$cohort_outcome_means
    expect_equal(length(oms), length(cohort_indices))

    # Build index-based mapping from original cohorts to panel cohorts by exact match of observed outcome indices
    ooi_panel <- panel$get_observed_outcome_indices()
    panel_idx_for_orig <- match_cohorts_panel_order(cohort_indices, ooi_panel)
    expect_true(!any(is.na(panel_idx_for_orig)))

    for (c in seq_along(cohort_indices)) {
        cp <- panel_idx_for_orig[c]
        T_idx <- cohort_indices[[c]]

        # check one cohort's factor model outputs
        out_no_fe <- f[["pca"]][[cp]]
        out_fe <- f[["pca_fe"]][[cp]]

        expect_true(inherits(out_no_fe, "FactorModelEstimates"))
        expect_true(inherits(out_fe, "FactorModelEstimates"))

        # Dimensions and attributes
        expect_equal(nrow(out_no_fe$G()), length(T_idx))
        expect_equal(ncol(out_no_fe$G()), 2L)
        expect_false(out_no_fe$has_g0())

        expect_equal(nrow(out_fe$G()), length(T_idx))
        expect_equal(ncol(out_fe$G()), 2L)
        expect_true(out_fe$has_g0())

        # Content: estimated cohort factor spans should match true rotated spans
        G1_true <- ctx$true_factors[T_idx, , drop = FALSE]

        proj_est <- projection_matrix_r(out_no_fe$G())
        proj_true <- projection_matrix_r(G1_true)
        expect_equal(proj_est, proj_true, tolerance = 1e-9)

        expect_true(inherits(oms[[cp]], "OutcomeMeanSuffStatEstimates"))
        # observed means length should equal number of observed outcomes for cohort c
        expect_equal(length(oms[[cp]]$observed_outcome_means()), length(T_idx))

        # observed means should equal true outcome means across units for cohort c
        expected_Y_mat <- expected_Y_for_units_ctx(ctx, cohort_id = c, unit_ids = units_by_cohort[[c]], T_idx = T_idx)
        expected_means <- colMeans(expected_Y_mat)
        expect_equal(oms[[cp]]$observed_outcome_means(), as.numeric(expected_means), tolerance = 1e-12)
    }
})

test_that("bootstrap flows through and produces replicates", {
    T <- 5L
    T_c <- 3L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = TRUE, ctx = ctx)

    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        covar_cols = c("cov1", "cov2"),
        sort_cohorts_lexicographically = TRUE
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

test_that("cohort weights with bootstrap: equal vs by_size behave as expected", {
    T <- 5L
    T_c <- 3L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, ctx = ctx)

    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        sort_cohorts_lexicographically = TRUE
    )

    N <- nrow(unique(panel$get_processed_panel()[, .(unit_idx)]))
    B <- 5L
    wb <- get_weighted_bootstrap_draws(N = N, B = B, type = "bayesian", seed = 321L)

    est_specs <- list(
        pca_equal = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L, cohort_weighting = "equal"),
        pca_by    = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L, cohort_weighting = "by_size")
    )

    res <- est_cohort_specific_params(panel, est_specs, bootstrap = wb)

    # Equal weights: point and each bootstrap draw are 1/C
    C <- length(cohort_indices)
    eq <- rep(1 / C, C)
    w_eq <- res$cohort_weights[["pca_equal"]]
    expect_true(inherits(w_eq, "CohortWeightEstimates"))
    expect_true(w_eq$has_bootstrap())
    expect_equal(as.numeric(w_eq$cohort_weights()), eq)
    for (b in seq_len(B)) {
        expect_equal(as.numeric(w_eq$bootstrap_cohort_weights(b)), eq)
    }

    # By-size weights: each bootstrap draw should sum to 1; point equals rowMeans of bootstraps
    w_by <- res$cohort_weights[["pca_by"]]
    expect_true(inherits(w_by, "CohortWeightEstimates"))
    expect_true(w_by$has_bootstrap())
    boots <- sapply(seq_len(B), function(b) as.numeric(w_by$bootstrap_cohort_weights(b)))
    expect_equal(dim(boots), c(C, B))
    # columns sums are 1
    expect_true(all(abs(colSums(boots) - 1) < 1e-10))
    # point equals average across draws
    expect_equal(as.numeric(w_by$cohort_weights()), rowMeans(boots), tolerance = 1e-10)
    # values in [0,1]
    expect_true(all(boots >= 0 & boots <= 1))
})


test_that("masking drops outcome and returns correct masked mean", {
    T <- 5L
    T_c <- 3L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort,
                                                include_covariates = FALSE, ctx = ctx)
    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 1,
        sort_cohorts_lexicographically = TRUE
    )

    est_specs <- list(pc = list(factor_model_estimator = "principal_components",
                                include_outcome_fes = FALSE, r = 2L))

    # Mask outcome 5 for cohort 3 (1-based)
    res <- est_cohort_specific_params(panel, est_specs, cohort_outcomes_to_mask = list("3" = 5L))

    # Masked observed outcome indices present and cohort 3 drops 5 -> remains {3,4}
    expect_true("masked_observed_outcome_indices" %in% names(res))
    moi <- res$masked_observed_outcome_indices
    expect_equal(length(moi), length(cohort_indices))
    expect_equal(as.integer(moi[[3]]), c(3L, 4L))

    # Factor dims: cohort 3 loses one observed row; cohorts 1 and 2 unchanged
    f_pc <- res$cohort_specific_factor_ests[["pc"]]
    expect_equal(nrow(f_pc[[1]]$G()), length(cohort_indices[[1]]))
    expect_equal(nrow(f_pc[[2]]$G()), length(cohort_indices[[2]]))
    expect_equal(nrow(f_pc[[3]]$G()), length(cohort_indices[[3]]) - 1L)

    # Masked means: for cohort 3, outcome 5 mean equals empirical mean from processed panel
    expect_true("masked_cohort_outcome_means" %in% names(res))
    mm <- res$masked_cohort_outcome_means[["3"]]
    expect_true(is.numeric(mm))
    expect_equal(length(mm), 1L)

    pp <- panel$get_processed_panel()
    sub <- pp[cohort_id == 3 & outcome_idx == 5]
    expect_true(nrow(sub) > 0)
    exp_mean <- mean(sub$y)
    expect_equal(as.numeric(mm[1]), exp_mean, tolerance = 1e-12)
})


test_that("auxiliary means: dimensions and values without covariates", {
    T <- 5L
    T_c <- 3L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, include_auxiliary = TRUE, ctx = ctx)

    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        auxiliary_cols = c("aux1", "aux2"),
        sort_cohorts_lexicographically = TRUE
    )

    est_specs <- list(pc = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 2L))
    res <- est_cohort_specific_params(panel, est_specs)

    aux <- res$cohort_auxiliary_means
    expect_equal(length(aux), length(cohort_indices))

    pp <- panel$get_processed_panel()
    uc <- panel$get_unit_cohorts()
    for (c in seq_along(cohort_indices)) {
        # Expected pop share: share of units in cohort c
        units_c <- nrow(uc[cohort_id == c])
        share_exp <- if (nrow(uc) > 0) units_c / nrow(uc) else 0

        est <- aux[[c]]
        expect_true(inherits(est, "CohortAuxiliaryDataMeanEstimates"))
        expect_false(est$has_bootstrap())
        expect_equal(est$cohort_pop_share(), share_exp, tolerance = 1e-12)

        # Determine Tc from observed_outcome_indices fixture
        M <- est$auxiliary_means()
        expect_equal(dim(M), c(T, 2L))

        # Compare by actual outcome index present in cohort c
        for (k in seq_len(length(cohort_indices[[c]]))) {
            t <- cohort_indices[[c]][k]
            sub <- pp[cohort_id == c & outcome_idx == t]
            expect_true(nrow(sub) > 0)
            expect_equal(M[t, 1], mean(sub$aux1), tolerance = 1e-12)
            expect_equal(M[t, 2], mean(sub$aux2), tolerance = 1e-12)
        }
    }
})

test_that("auxiliary means: bootstrap replicates present and shares valid", {
    T <- 5L
    T_c <- 3L
    outcomes <- make_outcomes(T)
    cohort_indices <- make_staircase_observed_indices(T, T_c)
    units_by_cohort <- make_units_by_cohort(length(cohort_indices), T_c)

    ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 1L, rotate = TRUE)
    panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, include_auxiliary = TRUE, ctx = ctx)
    # Check unobserved outcomes present with NA y before overwriting aux columns
    unobs <- setdiff(outcomes, outcomes[cohort_indices[[1]]])
    if (length(unobs) > 0) {
        u1 <- units_by_cohort[[1]][1]
        sub_unobs <- panel_dt[unit_id == u1 & outcome_id %in% unobs]
        expect_equal(nrow(sub_unobs), length(unobs))
        expect_true(all(is.na(sub_unobs$y)))
        expect_true(all(c("aux1", "aux2") %in% names(sub_unobs)))
    }
    panel_dt[, aux1 := 1.0]

    panel <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 1,
        min_cohort_size = 1,
        auxiliary_cols = c("aux1"),
        sort_cohorts_lexicographically = TRUE
    )

    N <- nrow(unique(panel$get_processed_panel()[, .(unit_idx)]))
    B <- 2L
    wb <- get_weighted_bootstrap_draws(N = N, B = B, type = "bayesian", seed = 123L)

    est_specs <- list(pc = list(factor_model_estimator = "principal_components", include_outcome_fes = FALSE, r = 1L))
    res <- est_cohort_specific_params(panel, est_specs, bootstrap = wb)

    aux <- res$cohort_auxiliary_means
    expect_equal(length(aux), length(cohort_indices))
    # Build unit->cohort map and expected shares per draw from bootstrap weights
    pp <- unique(panel$get_processed_panel()[, .(unit_idx, cohort_id)])
    W <- wb$weights()
    exp_draw <- function(b) {
        sapply(seq_along(cohort_indices), function(c) {
            u_in_c <- pp[cohort_id == c, unit_idx]
            sum(W[u_in_c, b])
        })
    }
    exp1 <- exp_draw(1L)
    exp2 <- exp_draw(2L)

    for (c in seq_along(cohort_indices)) {
        est <- aux[[c]]
        expect_true(est$has_bootstrap())
        expect_equal(est$num_bootstraps(), 2L)
        expect_true(est$cohort_pop_share(b = 1L) >= 0 && est$cohort_pop_share(b = 1L) <= 1)
        expect_true(est$cohort_pop_share(b = 2L) >= 0 && est$cohort_pop_share(b = 2L) <= 1)
        expect_equal(est$cohort_pop_share(b = 1L), exp1[c], tolerance = 1e-12)
        expect_equal(est$cohort_pop_share(b = 2L), exp2[c], tolerance = 1e-12)
    }
})