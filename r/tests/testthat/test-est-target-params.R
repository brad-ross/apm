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
  N <- nrow(panel_dt)
  B <- 2L
  wb <- get_weighted_bootstrap_draws(N, B, type = "multinomial", seed = 1L)

  res <- est_target_param_components(
    panel,
    est_specs = est_specs,
    num_threads = 1L,
    cohort_outcomes_to_mask = setNames(list(as.integer(masked_outcome)), as.character(last_cohort)),
    bootstrap = wb
  )

  # Sanity: masked info present
  expect_true("masked_cohort_outcome_means" %in% names(res))
  expect_true("masked_observed_outcome_indices" %in% names(res))

  # Expected masked mean (for the single masked outcome in last cohort)
  masked_mean <- as.numeric(res$masked_cohort_outcome_means[[as.character(last_cohort)]][1])

  # Define target parameter function selecting Y[last_cohort, last_outcome]
  fn <- function(Y, shares, observed_means_list, covar_means_list, eta) {
    as.numeric(Y[last_cohort, masked_outcome])
  }

  # Estimate target param from the outcome means (single spec)
  # Provide suff stats and keep eta NULL (not used)
  tpe <- est_target_params(res$outcome_means$pc, fn, aux_means = NULL,
                           suff_stats = res$cohort_outcome_means)
  tp <- tpe$target_params()

  # Check that the selected outcome mean equals the masked observed mean
  expect_equal(as.numeric(tp[1]), masked_mean, tolerance = 1e-12)

  # Bootstrap checks: structure and values
  expect_true(tpe$has_bootstrap())
  expect_equal(tpe$num_bootstraps(), B)
  # For our fn that ignores shares/eta, the bootstrap replicate equals the selected cell of Y_b
  for (b in seq_len(B)) {
    Y_b <- res$outcome_means$pc$mean_outcomes(b)
    expect_equal(as.numeric(tpe$target_params(b)[1]), as.numeric(Y_b[last_cohort, masked_outcome]), tolerance = 1e-12)
  }
})