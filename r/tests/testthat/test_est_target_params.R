context("Testing estimation of target parameters")

setup_target_fixture <- function(num_specs = 1L) {
  T <- 5L
  T_c <- 3L
  outcomes <- make_outcomes(T)
  cohort_indices <- make_staircase_observed_indices(T, T_c, add_no_missing_cohort = TRUE)
  C <- length(cohort_indices)
  units_by_cohort <- make_units_by_cohort(C, T_c)

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

  base_spec <- list(factor_model_estimator = "principal_components",
                    include_outcome_fes = FALSE,
                    r = 2L)
  spec_names <- c("pc", if (num_specs > 1L) paste0("pc", seq_len(num_specs - 1L) + 1L))
  est_specs <- setNames(vector("list", length(spec_names)), spec_names)
  for (nm in spec_names) est_specs[[nm]] <- base_spec

  comps <- est_target_param_components(
    panel,
    est_specs = est_specs,
    num_threads = 1L,
    cohort_outcomes_to_mask = setNames(list(as.integer(max(cohort_indices[[length(cohort_indices)]]))),
                                       as.character(length(cohort_indices))),
    bootstrap = get_weighted_bootstrap_draws(nrow(panel$get_unit_cohorts()), 2L, type = "multinomial", seed = 1L),
    est_outcome_means_via_imputation = TRUE
  )
  list(panel = panel, comps = comps)
}

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

test_that("get_target_param_diff_ests returns zero vector when inputs match", {
  fixture <- setup_target_fixture(num_specs = 1L)
  comps <- fixture$comps
  fn <- function(Y, shares, observed_means_list, covar_means_list, eta_list) {
    colMeans(Y)
  }
  tpe <- est_target_params(
    outcome_means = comps$outcome_means[[1]],
    fn = fn,
    aux_means = comps$cohort_auxiliary_means,
    suff_stats = comps$cohort_outcome_mean_ests
  )
  diff <- get_target_param_diff_ests(tpe, tpe)
  expect_true(inherits(diff, "TargetParameterEstimates"))
  expect_equal(as.numeric(diff$target_params()), rep(0, tpe$p()), tolerance = 1e-12)
  if (diff$has_bootstrap()) {
    M <- diff$boots_matrix()
    expect_true(all(abs(M) < 1e-12))
  }
})

test_that("combine_target_param_ests concatenates two estimates", {
  fixture <- setup_target_fixture(num_specs = 1L)
  comps <- fixture$comps
  fn <- function(Y, shares, observed_means_list, covar_means_list, eta_list) {
    colMeans(Y)
  }
  tpe <- est_target_params(
    outcome_means = comps$outcome_means[[1]],
    fn = fn,
    aux_means = comps$cohort_auxiliary_means,
    suff_stats = comps$cohort_outcome_mean_ests
  )
  combined <- combine_target_param_ests(tpe, tpe)
  expect_true(inherits(combined, "TargetParameterEstimates"))
  p <- tpe$p()
  expect_equal(combined$p(), 2L * p)
  tp_comb <- combined$target_params()
  expect_equal(as.numeric(tp_comb[seq_len(p)]), as.numeric(tpe$target_params()), tolerance = 1e-10)
  expect_equal(as.numeric(tp_comb[(p + 1):(2 * p)]), as.numeric(tpe$target_params()), tolerance = 1e-10)
  boots <- combined$boots_matrix()
  expect_equal(nrow(boots), 2L * p)
  expect_equal(ncol(boots), tpe$num_bootstraps())
  expect_equal(boots[seq_len(p), , drop = FALSE], tpe$boots_matrix(), tolerance = 1e-10)
  expect_equal(boots[(p + 1):(2 * p), , drop = FALSE], tpe$boots_matrix(), tolerance = 1e-10)
})

test_that("combine_target_param_ests combines a list and validates inputs", {
  fixture <- setup_target_fixture(num_specs = 1L)
  comps <- fixture$comps
  fn <- function(Y, shares, observed_means_list, covar_means_list, eta_list) colMeans(Y)
  tpe <- est_target_params(
    outcome_means = comps$outcome_means[[1]],
    fn = fn,
    aux_means = comps$cohort_auxiliary_means,
    suff_stats = comps$cohort_outcome_mean_ests
  )
  combined <- combine_target_param_ests(list(tpe, tpe, tpe))
  expect_true(inherits(combined, "TargetParameterEstimates"))
  expect_equal(combined$p(), 3L * tpe$p())
  boots <- combined$boots_matrix()
  expect_equal(nrow(boots), 3L * tpe$p())
  expect_equal(ncol(boots), tpe$num_bootstraps())

  expect_error(combine_target_param_ests(list(tpe), tpe))
  expect_error(combine_target_param_ests(list(tpe, "oops")))
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

test_that("est_target_params multi-spec reuses shared cohort inputs", {
  fixture <- setup_target_fixture(num_specs = 2L)
  comps <- fixture$comps
  fn <- function(Y, shares, observed_means_list, covar_means_list, eta_list) {
    colMeans(Y)
  }
  res <- est_target_params(
    outcome_means = comps$outcome_means,
    fn = fn,
    aux_means = comps$cohort_auxiliary_means,
    suff_stats = comps$cohort_outcome_mean_ests
  )
  expect_identical(sort(names(res)), sort(names(comps$outcome_means)))
  expect_true(all(vapply(res, function(e) inherits(e, "TargetParameterEstimates"), logical(1))))
})

test_that("subset method extracts vector of indices", {
  fixture <- setup_target_fixture(num_specs = 1L)
  comps <- fixture$comps
  fn <- function(Y, shares, observed_means_list, covar_means_list, eta_list) colMeans(Y)
  tpe <- est_target_params(comps$outcome_means[[1]], fn,
                           aux_means = comps$cohort_auxiliary_means,
                           suff_stats = comps$cohort_outcome_mean_ests)

  idx <- c(1L, 3L)
  sub <- tpe$subset(idx)

  expect_true(inherits(sub, "TargetParameterEstimates"))
  expect_equal(sub$p(), length(idx))
  expect_equal(as.numeric(sub$target_params()), as.numeric(tpe$target_params()[idx]), tolerance = 1e-12)
  if (tpe$has_bootstrap()) {
    expect_equal(sub$num_bootstraps(), tpe$num_bootstraps())
    expect_equal(sub$boots_matrix(), tpe$boots_matrix()[idx, , drop = FALSE], tolerance = 1e-12)
  }
})

test_that("subset method handles single index and validates inputs", {
  fixture <- setup_target_fixture(num_specs = 1L)
  comps <- fixture$comps
  fn <- function(Y, shares, observed_means_list, covar_means_list, eta_list) colMeans(Y)
  tpe <- est_target_params(comps$outcome_means[[1]], fn,
                           aux_means = comps$cohort_auxiliary_means,
                           suff_stats = comps$cohort_outcome_mean_ests)

  idx <- 2L
  sub <- tpe$subset(idx)

  expect_true(inherits(sub, "TargetParameterEstimates"))
  expect_equal(sub$p(), 1L)
  expect_equal(as.numeric(sub$target_params()), as.numeric(tpe$target_params()[idx]), tolerance = 1e-12)
  if (tpe$has_bootstrap()) {
    expect_equal(sub$num_bootstraps(), tpe$num_bootstraps())
    expect_equal(sub$boots_matrix(), matrix(tpe$boots_matrix()[idx, , drop = FALSE], nrow = 1),
                 tolerance = 1e-12)
  }

  expect_error(tpe$subset(c(0L, 1L)))
  expect_error(tpe$subset(c(1L, tpe$p() + 1L)))
})