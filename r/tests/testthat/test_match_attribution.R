context("FGW bipartite match attribution")

setup_match_attr_fixture <- function(num_specs = 1L) {
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
  base_spec <- list(
    factor_model_estimator = "principal_components",
    include_outcome_fes = FALSE,
    r = 1L
  )
  spec_names <- c("pc", if (num_specs > 1L) paste0("pc", seq_len(num_specs - 1L) + 1L))
  est_specs <- setNames(vector("list", length(spec_names)), spec_names)
  for (nm in spec_names) est_specs[[nm]] <- base_spec
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
  obs_means <- lapply(comps$cohort_outcome_mean_ests, function(s) s$observed_outcome_means())

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

  observed_value <- function(c_idx, outcome_idx) {
    pos <- match(outcome_idx, obs_idx[[c_idx]])
    obs_means[[c_idx]][pos]
  }

  obs_avg <- function(idx) {
    mask <- vapply(obs_idx, function(v) idx %in% v, logical(1))
    if (!any(mask)) return(0)
    numer <- 0
    denom <- 0
    for (c in which(mask)) {
      numer <- numer + shares[c] * observed_value(c, idx)
      denom <- denom + shares[c]
    }
    numer / denom
  }
  pop_avg <- function(idx) {
    values <- as.numeric(Y[, idx, drop = FALSE])
    sum(shares * values) / sum(shares)
  }

  expected_ratio <- (pop_avg(1L) - pop_avg(3L)) / (obs_avg(1L) - obs_avg(3L))
  expect_equal(params[1], expected_ratio, tolerance = 1e-8)
  expect_equal(params[2], 1 - expected_ratio, tolerance = 1e-8)
})

test_that("FGW wrapper handles grouped outcomes with weights", {
  fixture <- setup_match_attr_fixture()
  panel <- fixture$panel
  comps <- fixture$comps
  obs_idx <- panel$get_observed_outcome_indices()
  obs_means <- lapply(comps$cohort_outcome_mean_ests, function(s) s$observed_outcome_means())

  outcome_weights <- c(1, 2, 3)
  grp1 <- c(1L, 2L)
  grp2 <- c(2L, 3L)

  tpe <- est_fgw_bipartite_match_outcome_diff_params(
    outcome_means = comps$outcome_means$pc,
    observed_outcome_indices = obs_idx,
    suff_stats = comps$cohort_outcome_mean_ests,
    outcome_indices_1 = grp1,
    outcome_indices_2 = grp2,
    outcome_weights = outcome_weights
  )

  params <- tpe$target_params()
  expect_length(params, 2L)
  expect_equal(sum(params), 1.0, tolerance = 1e-8)

  cohort_weights <- vapply(comps$cohort_outcome_mean_ests, function(s) s$cohort_pop_share(), numeric(1))
  mean_outcomes <- comps$outcome_means$pc$mean_outcomes()

  observed_group_avg <- function(group_idxs) {
    total_weight <- 0
    total_value <- 0
    for (c in seq_along(obs_idx)) {
      cohort_subset <- intersect(group_idxs, obs_idx[[c]])
      if (!length(cohort_subset)) next
      local_positions <- match(cohort_subset, obs_idx[[c]])
      local_means <- obs_means[[c]][local_positions]
      local_weights <- outcome_weights[cohort_subset]
      row_weight_sum <- sum(local_weights)
      row_weighted_total <- sum(local_weights * local_means)
      total_weight <- total_weight + cohort_weights[c] * row_weight_sum
      total_value <- total_value + cohort_weights[c] * row_weighted_total
    }
    total_value / total_weight
  }

  obs_group1 <- observed_group_avg(grp1)
  obs_group2 <- observed_group_avg(grp2)

  pop_group1 <- (cohort_weights[1] * (1 * mean_outcomes[1, 1] + 2 * mean_outcomes[1, 2]) +
                   cohort_weights[2] * (1 * mean_outcomes[2, 1] + 2 * mean_outcomes[2, 2])) /
    ((cohort_weights[1] + cohort_weights[2]) * 3)
  pop_group2 <- (cohort_weights[1] * (2 * mean_outcomes[1, 2] + 3 * mean_outcomes[1, 3]) +
                   cohort_weights[2] * (2 * mean_outcomes[2, 2] + 3 * mean_outcomes[2, 3])) /
    ((cohort_weights[1] + cohort_weights[2]) * 5)

  expected_ratio <- (pop_group1 - pop_group2) / (obs_group1 - obs_group2)

  expect_equal(params[1], expected_ratio, tolerance = 1e-8)
  expect_equal(params[2], 1 - expected_ratio, tolerance = 1e-8)
})

test_that("FGW wrapper can disable observed outcome means usage", {
  fixture <- setup_match_attr_fixture()
  panel <- fixture$panel
  comps <- fixture$comps
  obs_idx <- panel$get_observed_outcome_indices()

  tpe <- est_fgw_bipartite_match_outcome_diff_params(
    outcome_means = comps$outcome_means$pc,
    outcome_idx_1 = 1L,
    outcome_idx_2 = 3L,
    observed_outcome_indices = obs_idx,
    suff_stats = comps$cohort_outcome_mean_ests,
    use_observed_outcome_means = FALSE
  )
  params <- tpe$target_params()
  expect_length(params, 2L)
  expect_equal(sum(params), 1.0, tolerance = 1e-8)

  Y <- comps$outcome_means$pc$mean_outcomes()
  shares <- vapply(comps$cohort_outcome_mean_ests, function(s) s$cohort_pop_share(), numeric(1))

  obs_avg_raw <- function(idx) {
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

  expected_ratio <- (pop_avg(1L) - pop_avg(3L)) / (obs_avg_raw(1L) - obs_avg_raw(3L))
  expect_equal(params[1], expected_ratio, tolerance = 1e-8)
  expect_equal(params[2], 1 - expected_ratio, tolerance = 1e-8)
})

# test_that("FGW wrapper errors on invalid outcome index", {
#   fixture <- setup_match_attr_fixture()
#   panel <- fixture$panel
#   comps <- fixture$comps
#   obs_idx <- panel$get_observed_outcome_indices()

#   expect_error(
#     est_fgw_bipartite_match_outcome_diff_params(
#       outcome_means = comps$outcome_means$pc,
#       outcome_idx_1 = 10L,
#       outcome_idx_2 = 3L,
#       observed_outcome_indices = obs_idx,
#       suff_stats = comps$cohort_outcome_mean_ests
#     ),
#     "not observed",
#     ignore.case = TRUE
#   )
# })

test_that("FGW multi-spec wrapper reuses shared cohort inputs", {
  fixture <- setup_match_attr_fixture(num_specs = 2L)
  panel <- fixture$panel
  comps <- fixture$comps
  obs_idx <- panel$get_observed_outcome_indices()

  res <- est_fgw_bipartite_match_outcome_diff_params(
    outcome_means = comps$outcome_means,
    outcome_idx_1 = 1L,
    outcome_idx_2 = 2L,
    observed_outcome_indices = obs_idx,
    suff_stats = comps$cohort_outcome_mean_ests
  )
  expect_identical(sort(names(res)), sort(names(comps$outcome_means)))
  expect_true(all(vapply(res, function(e) inherits(e, "TargetParameterEstimates"), logical(1))))
})