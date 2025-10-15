# Shared helpers are defined in helper-test-utils.R

context("Testing apm core functionality")

test_that("APM alignment works for a staircase pattern", {
  true_factors <- make_true_factors_5x2()
  observed_outcome_indices <- make_staircase_observed_indices(5, 3)
  
  run_alignment_test_r(true_factors, observed_outcome_indices)
})

test_that("APM alignment with weights works for a staircase pattern", {
  true_factors <- make_true_factors_5x2()
  observed_outcome_indices <- make_staircase_observed_indices(5, 3)
  
  cohort_weights <- c(1.0, 2.0, 1.0)
  
  run_alignment_test_r(
    true_factors, 
    observed_outcome_indices, 
    cohort_weights = cohort_weights
  )
})

test_that("APM alignment works for a non-contiguous pattern", {
  true_factors <- matrix(c(
    0.1, 0.6,
    0.2, 0.7,
    0.3, 0.8,
    0.4, 0.9,
    0.5, 1.0
  ), nrow = 5, ncol = 2, byrow = TRUE)
  
  observed_outcome_indices <- list(
    c(1, 2, 3),
    c(2, 3, 4),
    c(1, 3, 5) # Non-contiguous outcomes
  )
  
  run_alignment_test_r(true_factors, observed_outcome_indices)
}) 

# setup_estimation_test_data_r is provided by helper-test-utils.R

test_that("EstimateMeans_FactorsCovariatesFixedEffects", {
  data <- setup_estimation_test_data_r()
  
  true_m <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  true_m[1,] <- (data$G %*% data$l_c[[1]] + data$X_c_vec[[1]] %*% data$a + data$g_0)
  true_m[2,] <- (data$G %*% data$l_c[[2]] + data$X_c_vec[[2]] %*% data$a + data$g_0)
  
  m_c_vec <- list(
    true_m[1, data$observed_outcome_indices[[1]]],
    true_m[2, data$observed_outcome_indices[[2]]]
  )
  
  estimated_m <- impute_outcomes_across_cohorts_from_obs_outcomes(
    data$G, data$observed_outcome_indices, m_c_vec, 
    g_0 = data$g_0, a = data$a, X_c_vec = data$X_c_vec)
  
  expect_equal(estimated_m, true_m, tolerance = 1e-9)
})

test_that("EstimateMeans_FactorsAndFixedEffects", {
  data <- setup_estimation_test_data_r()
  
  true_m <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  true_m[1,] <- (data$G %*% data$l_c[[1]] + data$g_0)
  true_m[2,] <- (data$G %*% data$l_c[[2]] + data$g_0)
  
  m_c_vec <- list(
    true_m[1, data$observed_outcome_indices[[1]]],
    true_m[2, data$observed_outcome_indices[[2]]]
  )
  
  estimated_m <- impute_outcomes_across_cohorts_from_obs_outcomes(
    data$G, data$observed_outcome_indices, m_c_vec, g_0 = data$g_0)
  
  expect_equal(estimated_m, true_m, tolerance = 1e-9)
})

test_that("EstimateMeans_FactorsAndCovariates", {
  data <- setup_estimation_test_data_r()
  
  true_m <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  true_m[1,] <- (data$G %*% data$l_c[[1]] + data$X_c_vec[[1]] %*% data$a)
  true_m[2,] <- (data$G %*% data$l_c[[2]] + data$X_c_vec[[2]] %*% data$a)
  
  m_c_vec <- list(
    true_m[1, data$observed_outcome_indices[[1]]],
    true_m[2, data$observed_outcome_indices[[2]]]
  )
  
  estimated_m <- impute_outcomes_across_cohorts_from_obs_outcomes(
    data$G, data$observed_outcome_indices, m_c_vec, 
    a = data$a, X_c_vec = data$X_c_vec)
  
  expect_equal(estimated_m, true_m, tolerance = 1e-9)
})

test_that("EstimateMeans_FactorsOnly", {
  data <- setup_estimation_test_data_r()
  
  true_m <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  true_m[1,] <- (data$G %*% data$l_c[[1]])
  true_m[2,] <- (data$G %*% data$l_c[[2]])
  
  m_c_vec <- list(
    true_m[1, data$observed_outcome_indices[[1]]],
    true_m[2, data$observed_outcome_indices[[2]]]
  )
  
  estimated_m <- impute_outcomes_across_cohorts_from_obs_outcomes(
    data$G, data$observed_outcome_indices, m_c_vec)
  
  expect_equal(estimated_m, true_m, tolerance = 1e-9)
}) 

test_that("ImputeOutcomesAcrossCohorts_FactorsOnly", {
  data <- setup_estimation_test_data_r()

  L <- matrix(0, nrow = data$Cval, ncol = ncol(data$G))
  for (c in seq_len(data$Cval)) L[c, ] <- data$l_c[[c]]

  expected <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  for (c in seq_len(data$Cval)) expected[c, ] <- (data$G %*% data$l_c[[c]])

  got <- impute_outcomes_across_cohorts(data$G, L)
  expect_equal(got, expected, tolerance = 1e-12)
})

test_that("ImputeOutcomesAcrossCohorts_FactorsAndFixedEffects", {
  data <- setup_estimation_test_data_r()

  L <- matrix(0, nrow = data$Cval, ncol = ncol(data$G))
  for (c in seq_len(data$Cval)) L[c, ] <- data$l_c[[c]]

  expected <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  for (c in seq_len(data$Cval)) expected[c, ] <- (data$G %*% data$l_c[[c]] + data$g_0)

  got <- impute_outcomes_across_cohorts(data$G, L, g_0 = data$g_0)
  expect_equal(got, expected, tolerance = 1e-12)
})

test_that("ImputeOutcomesAcrossCohorts_FactorsAndCovariates", {
  data <- setup_estimation_test_data_r()

  L <- matrix(0, nrow = data$Cval, ncol = ncol(data$G))
  for (c in seq_len(data$Cval)) L[c, ] <- data$l_c[[c]]

  expected <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  for (c in seq_len(data$Cval)) expected[c, ] <- (data$G %*% data$l_c[[c]] + data$X_c_vec[[c]] %*% data$a)

  got <- impute_outcomes_across_cohorts(data$G, L, a = data$a, X_c = data$X_c_vec)
  expect_equal(got, expected, tolerance = 1e-12)
})

test_that("ImputeOutcomesAcrossCohorts_AllComponents", {
  data <- setup_estimation_test_data_r()

  L <- matrix(0, nrow = data$Cval, ncol = ncol(data$G))
  for (c in seq_len(data$Cval)) L[c, ] <- data$l_c[[c]]

  expected <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  for (c in seq_len(data$Cval)) expected[c, ] <- (data$G %*% data$l_c[[c]] + data$g_0 + data$X_c_vec[[c]] %*% data$a)

  got <- impute_outcomes_across_cohorts(data$G, L, g_0 = data$g_0, a = data$a, X_c = data$X_c_vec)
  expect_equal(got, expected, tolerance = 1e-12)
})

#===============================================================================
# O3 Algorithm Tests
#===============================================================================

# canonicalize_o3_output is provided by helper-test-utils.R

test_that("o3_algorithm works for staircase pattern", {
  observed_outcome_indices <- list(
    c(1, 2, 3),
    c(2, 3, 4),
    c(3, 4, 5)
  )
  r <- 2
  
  result <- o3_algorithm(observed_outcome_indices, r)
  
  # The initial state is returned, then all cohorts merge into one super-cohort
  # in a single iteration.
  expected <- list(
    list(c(1), c(2), c(3)),
    list(c(1, 2, 3))
  )
  
  expect_equal(canonicalize_o3_output(result), canonicalize_o3_output(expected))
  expect_true(aligned_factors_identified(observed_outcome_indices, r))
})

test_that("o3_algorithm works for non-contiguous pattern", {
  observed_outcome_indices <- list(
    c(1, 2, 3), # Cohort 1
    c(2, 3, 4), # Cohort 2
    c(1, 4, 5)  # Cohort 3
  )
  r <- 2
  
  result <- o3_algorithm(observed_outcome_indices, r)
  
  # Expected merge history (initial state is returned)
  # 1. Cohorts 1 and 2 merge -> {{1,2}, {3}}
  # 2. Cohort 3 merges with {1,2} -> {{1,2,3}}
  expected <- list(
    list(c(1), c(2), c(3)),
    list(c(1, 2), c(3)),
    list(c(1, 2, 3))
  )
  
  expect_equal(canonicalize_o3_output(result), canonicalize_o3_output(expected))
  expect_true(aligned_factors_identified(observed_outcome_indices, r))
})

test_that("aligned_factors_identified returns FALSE for unconnected cohorts", {
  observed_outcome_indices <- list(
    c(1, 2), # Cohort 1
    c(2, 3, 4), # Cohort 2
    c(3, 4, 5)     # This cohort has no overlap with the others
  )
  r <- 2
  
  result <- o3_algorithm(observed_outcome_indices, r)
  
  # Expected: initial state, then cohorts 1 and 2 merge, but 3 remains separate.
  expected <- list(
    list(c(1), c(2), c(3)),
    list(c(1), c(2, 3))
  )
  
  expect_equal(canonicalize_o3_output(result), canonicalize_o3_output(expected))
  expect_false(aligned_factors_identified(observed_outcome_indices, r))
})

test_that("aligned_factors_identified returns FALSE when no merges occur", {
  observed_outcome_indices <- list(
    c(1, 2),
    c(2, 3),
    c(3, 4, 5)
  )
  r <- 2
  
  result <- o3_algorithm(observed_outcome_indices, r)
  
  # No merges occur; result should contain only the initial state.
  expected <- list(
    list(c(1), c(2), c(3))
  )
  expect_equal(canonicalize_o3_output(result), canonicalize_o3_output(expected))
  expect_false(aligned_factors_identified(observed_outcome_indices, r))
}) 