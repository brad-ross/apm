context("FGW bipartite match attribution")

setup_match_attr_fixture <- function() {
  T <- 3L
  cohort_indices <- list(
    as.integer(c(1L, 2L)),
    as.integer(c(2L, 3L))
  )
  outcomes <- make_outcomes(T)
  units_by_cohort <- make_units_by_cohort(length(cohort_indices), 2L)
  panel_dt <- build_panel_from_indices(
    outcomes = outcomes,
    cohort_indices = cohort_indices,
    units_by_cohort = units_by_cohort,
    include_covariates = FALSE,
    include_auxiliary = FALSE
  )
  panel <- UnbalancedPanel$new(
    panel_df = panel_dt,
    unit_id_col = "unit_id",
    outcome_id_col = "outcome_id",
    outcome_value_col = "y",
    model_rank = 1,
    min_cohort_size = 1,
    sort_cohorts_lexicographically = TRUE
  )
  est_specs <- list(pc = list(
    factor_model_estimator = "principal_components",
    include_outcome_fes = FALSE,
    r = 1L
  ))
  comps <- est_target_param_components(
    panel = panel,
    est_specs = est_specs,
    num_threads = 1L,
    est_outcome_means_via_imputation = FALSE
  )
  list(panel = panel, comps = comps)
}

test_that("FGW wrapper returns expected ratio", {
  fixture <- setup_match_attr_fixture()
  panel <- fixture$panel
  comps <- fixture$comps
  obs_idx <- panel$get_observed_outcome_indices()

  tpe <- est_fgw_bipartite_match_outcome_diff_params(
    outcome_means = comps$outcome_means$pc,
    outcome_idx_1 = 1L,
    outcome_idx_2 = 3L,
    observed_outcome_indices = obs_idx,
    suff_stats = comps$cohort_outcome_mean_ests
  )
  params <- tpe$target_params()
  expect_length(params, 2L)
  expect_equal(sum(params), 1.0, tolerance = 1e-8)

  Y <- comps$outcome_means$pc$mean_outcomes()
  shares <- vapply(comps$cohort_outcome_mean_ests, function(s) s$cohort_pop_share(), numeric(1))

  obs_avg <- function(idx) {
    mask <- vapply(obs_idx, function(v) idx %in% v, logical(1))
    if (!any(mask)) return(0)
    weights <- shares[mask]
    values <- as.numeric(Y[mask, idx, drop = FALSE])
    sum(weights * values) / sum(weights)
  }
  pop_avg <- function(idx) {
    values <- as.numeric(Y[, idx, drop = FALSE])
    sum(shares * values) / sum(shares)
  }

  expected_ratio <- (pop_avg(1L) - pop_avg(3L)) / (obs_avg(1L) - obs_avg(3L))
  expect_equal(params[1], expected_ratio, tolerance = 1e-8)
  expect_equal(params[2], 1 - expected_ratio, tolerance = 1e-8)
})

test_that("FGW wrapper errors on invalid outcome index", {
  fixture <- setup_match_attr_fixture()
  panel <- fixture$panel
  comps <- fixture$comps
  obs_idx <- panel$get_observed_outcome_indices()

  expect_error(
    est_fgw_bipartite_match_outcome_diff_params(
      outcome_means = comps$outcome_means$pc,
      outcome_idx_1 = 10L,
      outcome_idx_2 = 3L,
      observed_outcome_indices = obs_idx,
      suff_stats = comps$cohort_outcome_mean_ests
    ),
    "not observed",
    ignore.case = TRUE
  )
})

