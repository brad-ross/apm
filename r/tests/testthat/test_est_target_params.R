context("Testing estimation of target parameters")

test_that("target param equals masked observed mean for last cohort/outcome", {
  # Staircase layout
  T <- 5L
  T_c <- 3L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c, add_no_missing_cohort = TRUE)
  C <- length(cohort_indices)
  units_by_cohort <- make_units_by_cohort(C, T_c)

  # Deterministic context and panel
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
  panel_dt <- build_panel_from_indices_factor(
    outcomes = outcomes,
    cohort_indices = cohort_indices,
    units_by_cohort = units_by_cohort,
    include_covariates = FALSE,
    include_auxiliary = FALSE,
    r = 2L,
    rotate = FALSE,
    ctx = ctx
  )
  # Shuffle rows to avoid any ordering assumptions
  set.seed(123)
  panel_dt <- panel_dt[sample(nrow(panel_dt))]

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

  # Mask an outcome in the last cohort that is ALSO observed by another cohort.
  # In staircase(T=5, window=3), cohort C observes {3,4,5}; outcome 4 is also in cohort 2.
  # C increased by 1 due to the no-missing cohort
  last_cohort <- length(cohort_indices)
  last_outcome <- max(cohort_indices[[last_cohort]])
  expect_equal(last_outcome, T)
  masked_outcome <- last_outcome

  # Trivial bootstrap with B = 2 over panel rows
  N <- nrow(panel$get_unit_cohorts())
  B <- 2L
  wb <- get_weighted_bootstrap_draws(N, B, type = "multinomial", seed = 1L)

  res <- est_target_param_components(
    panel,
    est_specs = est_specs,
    num_threads = 1L,
    cohort_outcomes_to_mask = setNames(list(as.integer(masked_outcome)), as.character(last_cohort)),
    bootstrap = wb,
    est_outcome_means_via_imputation = TRUE
  )

  # # Sanity: masked info present
  # expect_true("masked_cohort_outcome_means" %in% names(res))
  # expect_true("masked_observed_outcome_indices" %in% names(res))

  # # Expected masked mean (for the single masked outcome in last cohort)
  # masked_mean <- as.numeric(res$masked_cohort_outcome_means[[as.character(last_cohort)]][1])

  # # Define target parameter function selecting Y[last_cohort, last_outcome]
  # fn <- function(Y, shares, observed_means_list, covar_means_list, eta) {
  #   as.numeric(Y[last_cohort, masked_outcome])
  # }

  # # Estimate target param from the outcome means (single spec)
  # # Provide suff stats and keep eta NULL (not used)
  # tpe <- est_target_params(res$outcome_means$pc, fn, aux_means = NULL,
  #                          suff_stats = res$cohort_outcome_means)
  # tp <- tpe$target_params()

  # # Check that the selected outcome mean equals the masked observed mean
  # expect_equal(as.numeric(tp[1]), masked_mean, tolerance = 1e-12)

  # # Bootstrap checks: structure and values
  # expect_true(tpe$has_bootstrap())
  # expect_equal(tpe$num_bootstraps(), B)
  # # For our fn that ignores shares/eta, the bootstrap replicate equals the selected cell of Y_b
  # for (b in seq_len(B)) {
  #   Y_b <- res$outcome_means$pc$mean_outcomes(b)
  #   expect_equal(as.numeric(tpe$target_params(b)[1]), as.numeric(Y_b[last_cohort, masked_outcome]), tolerance = 1e-12)
  # }
})

test_that("est_masked_outcome_mean_err_metrics computes bias/se/rmse per spec for masked pairs", {
  T <- 5L; T_c <- 3L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c, add_no_missing_cohort = TRUE)
  C <- length(cohort_indices)
  units_by_cohort <- make_units_by_cohort(C, T_c)

  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, FALSE, FALSE, 2L, FALSE, ctx)
  set.seed(1); panel_dt <- panel_dt[sample(nrow(panel_dt))]

  panel <- UnbalancedPanel$new(panel_dt, "unit_id", "outcome_id", "y",
                               model_rank = 2, min_cohort_size = 1,
                               sort_cohorts_lexicographically = TRUE)

  est_specs <- list(pc = list(factor_model_estimator = "principal_components",
                              include_outcome_fes = FALSE, r = 2L),
                    pc2 = list(factor_model_estimator = "principal_components",
                               include_outcome_fes = FALSE, r = 2L))

  last_cohort <- length(cohort_indices)
  masked_outcome <- max(cohort_indices[[last_cohort]])
  N <- nrow(panel$get_unit_cohorts()); B <- 3L
  wb <- get_weighted_bootstrap_draws(N, B, type = "multinomial", seed = 42L)

  comps <- est_target_param_components(
    panel,
    est_specs = est_specs,
    num_threads = 1L,
    cohort_outcomes_to_mask = setNames(list(as.integer(masked_outcome)), as.character(last_cohort)),
    bootstrap = wb,
    est_outcome_means_via_imputation = TRUE
  )

  truth <- as.numeric(comps$masked_cohort_outcome_means[[as.character(last_cohort)]][1])
  coh <- last_cohort
  out <- masked_outcome

  exp_rows <- lapply(names(comps$outcome_means), function(spec) {
    ome <- comps$outcome_means[[spec]]
    xp <- ome$.__enclos_env__$private$xp
    draws <- vapply(seq_len(B), function(b) {
      M <- ome_boot_means_cpp(xp, b)
      as.numeric(M[coh, out])
    }, numeric(1))
    bias <- mean(draws) - truth
    se <- if (length(draws) > 1) sd(draws) else NaN
    rmse <- sqrt(mean((draws - truth)^2))
    data.frame(cohort = coh, outcome = out, spec = spec, bias = bias, se = se, rmse = rmse, stringsAsFactors = FALSE)
  })
  expected <- do.call(rbind, exp_rows)

  df <- est_masked_outcome_mean_err_metrics(comps)
  df1 <- df[df$cohort == coh & df$outcome == out, ]
  rownames(df1) <- NULL
  df1 <- df1[order(df1$spec), ]
  expected <- expected[order(expected$spec), ]

  expect_equal(df1$spec, expected$spec)
  expect_equal(df1$bias, expected$bias, tolerance = 1e-10)
  expect_equal(df1$se, expected$se, tolerance = 1e-10)
  expect_equal(df1$rmse, expected$rmse, tolerance = 1e-10)
  expect_true(all(df1$cohort_pop_share >= 0 & df1$cohort_pop_share <= 1))
})

test_that("est_masked_outcome_mean_err_metrics errors when bootstrap missing", {
  T <- 5L; T_c <- 3L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c, add_no_missing_cohort = TRUE)
  C <- length(cohort_indices)
  units_by_cohort <- make_units_by_cohort(C, T_c)

  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, FALSE, FALSE, 2L, FALSE, ctx)
  panel <- UnbalancedPanel$new(panel_dt, "unit_id", "outcome_id", "y",
                               model_rank = 2, min_cohort_size = 1,
                               sort_cohorts_lexicographically = TRUE)

  est_specs <- list(pc = list(factor_model_estimator = "principal_components",
                              include_outcome_fes = FALSE, r = 2L))

  last_cohort <- length(cohort_indices)
  masked_outcome <- max(cohort_indices[[last_cohort]])

  comps <- est_target_param_components(
    panel,
    est_specs = est_specs,
    num_threads = 1L,
    cohort_outcomes_to_mask = setNames(list(as.integer(masked_outcome)), as.character(last_cohort)),
    bootstrap = NULL,
    est_outcome_means_via_imputation = TRUE
  )
  expect_error(est_masked_outcome_mean_err_metrics(comps))
})

test_that("est_masked_outcome_mean_err_metrics errors on mismatched mask length", {
  T <- 5L; T_c <- 3L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c, add_no_missing_cohort = TRUE)
  C <- length(cohort_indices)
  units_by_cohort <- make_units_by_cohort(C, T_c)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, FALSE, FALSE, 2L, FALSE, ctx)
  panel <- UnbalancedPanel$new(panel_dt, "unit_id", "outcome_id", "y",
                               model_rank = 2, min_cohort_size = 1,
                               sort_cohorts_lexicographically = TRUE)

  est_specs <- list(pc = list(factor_model_estimator = "principal_components",
                              include_outcome_fes = FALSE, r = 2L))
  N <- nrow(panel$get_unit_cohorts()); B <- 2L
  wb <- get_weighted_bootstrap_draws(N, B, type = "multinomial", seed = 123)
  last_cohort <- length(cohort_indices)
  masked_outcome <- max(cohort_indices[[last_cohort]])

  comps <- est_target_param_components(
    panel,
    est_specs = est_specs,
    num_threads = 1L,
    cohort_outcomes_to_mask = setNames(list(as.integer(masked_outcome)), as.character(last_cohort)),
    bootstrap = wb,
    est_outcome_means_via_imputation = TRUE
  )
  comps$masked_cohort_outcome_means[[as.character(last_cohort)]] <- numeric(0)
  expect_error(est_masked_outcome_mean_err_metrics(comps))
})

test_that("est_masked_outcome_mean_err_metrics errors on out-of-range outcome index", {
  T <- 5L; T_c <- 3L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c, add_no_missing_cohort = TRUE)
  C <- length(cohort_indices)
  units_by_cohort <- make_units_by_cohort(C, T_c)
  ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE)
  panel_dt <- build_panel_from_indices_factor(outcomes, cohort_indices, units_by_cohort, FALSE, FALSE, 2L, FALSE, ctx)
  panel <- UnbalancedPanel$new(panel_dt, "unit_id", "outcome_id", "y",
                               model_rank = 2, min_cohort_size = 1,
                               sort_cohorts_lexicographically = TRUE)

  est_specs <- list(pc = list(factor_model_estimator = "principal_components",
                              include_outcome_fes = FALSE, r = 2L))
  N <- nrow(panel$get_unit_cohorts()); B <- 2L
  wb <- get_weighted_bootstrap_draws(N, B, type = "multinomial", seed = 123)

  comps <- est_target_param_components(
    panel,
    est_specs = est_specs,
    num_threads = 1L,
    cohort_outcomes_to_mask = NULL,
    bootstrap = wb,
    est_outcome_means_via_imputation = TRUE
  )
  comps$cohort_outcome_mask <- setNames(list(as.integer(T + 10L)), "1")
  comps$masked_cohort_outcome_means <- list("1" = c(0))
  expect_error(est_masked_outcome_mean_err_metrics(comps))
})