# Helper function to compute a projection matrix in R
# Uses QR decomposition for numerical stability, mirroring the C++ implementation.
projection_matrix_r <- function(X) {
  if (ncol(X) == 0) {
    return(matrix(0, nrow = nrow(X), ncol = nrow(X)))
  }
  svd_X <- svd(X)
  # Only keep singular vectors corresponding to nonzero singular values (numerical tolerance)
  tol <- max(dim(X)) * max(svd_X$d) * .Machine$double.eps
  rank <- sum(svd_X$d > tol)
  if (rank == 0) {
    return(matrix(0, nrow = nrow(X), ncol = nrow(X)))
  }
  U <- svd_X$u[, 1:rank, drop = FALSE]
  U %*% t(U)
}

generate_symmetric_rotation_matrices <- function(C, r) {
  lapply(1:C, function(c) {
    R <- matrix(0, nrow = r, ncol = r)
    for (i in 1:r) {
      for (j in 1:r) {
        R[i, j] <- 0.1 * c * i + 0.1 * j
      }
    }
    R <- (R + t(R)) / 2  # Make symmetric
    diag(R) <- diag(R) + r
    R
  })
}

# Helper function to run the core alignment test logic
run_alignment_test_r <- function(true_factors, 
                                 observed_outcome_indices,
                                 cohort_weights = NULL) {
  # Generate deterministic, full-rank rotation matrices
  C <- length(observed_outcome_indices)
  r <- ncol(true_factors)
  rotation_matrices <- generate_symmetric_rotation_matrices(C, r)
  
  # Create rotated cohort-specific factor matrices
  cohort_factor_matrices <- lapply(1:C, function(c) {
    true_factors[observed_outcome_indices[[c]], ] %*% rotation_matrices[[c]]
  })
  
  # Run the alignment function
  aligned_factors <- align_factors_using_apm(
    cohort_factor_matrices,
    observed_outcome_indices,
    cohort_weights = cohort_weights
  )
  
  # Check the result by comparing projection matrices
  proj_aligned <- projection_matrix_r(aligned_factors)
  proj_true <- projection_matrix_r(true_factors)
  
  expect_equal(proj_aligned, proj_true, tolerance = 1e-9)
}

context("Testing apm core functionality")

test_that("APM alignment works for a staircase pattern", {
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
    c(3, 4, 5)
  )
  
  run_alignment_test_r(true_factors, observed_outcome_indices)
})

test_that("APM alignment with weights works for a staircase pattern", {
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
    c(3, 4, 5)
  )
  
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

# Helper function to set up data for estimation tests
setup_estimation_test_data_r <- function() {
  list(
    Tval = 4, r = 2, Cval = 2, q = 2,
    G = matrix(c(1, 1, 2, 4, 3, 9, 4, 16), nrow = 4, ncol = 2, byrow = TRUE),
    a = c(0.5, 1.0),
    g_0 = seq(0.1, 0.4, by = 0.1),
    observed_outcome_indices = list(c(1, 2, 3), c(2, 3, 4)),
    l_c = list(c(1.0, 2.0), c(3.0, 4.0)),
    X_c_vec = list(
      matrix(c(0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.4, 0.8), nrow = 4, ncol = 2, byrow = TRUE),
      matrix(c(1.1, 1.5, 1.2, 1.6, 1.3, 1.7, 1.4, 1.8), nrow = 4, ncol = 2, byrow = TRUE)
    )
  )
}

test_that("EstimateMeans_FactorsCovariatesFixedEffects", {
  data <- setup_estimation_test_data_r()
  
  true_m <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  true_m[1,] <- (data$G %*% data$l_c[[1]] + data$X_c_vec[[1]] %*% data$a + data$g_0)
  true_m[2,] <- (data$G %*% data$l_c[[2]] + data$X_c_vec[[2]] %*% data$a + data$g_0)
  
  m_c_vec <- list(
    true_m[1, data$observed_outcome_indices[[1]]],
    true_m[2, data$observed_outcome_indices[[2]]]
  )
  
  estimated_m <- estimate_outcome_means_across_cohorts(
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
  
  estimated_m <- estimate_outcome_means_across_cohorts(
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
  
  estimated_m <- estimate_outcome_means_across_cohorts(
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
  
  estimated_m <- estimate_outcome_means_across_cohorts(
    data$G, data$observed_outcome_indices, m_c_vec)
  
  expect_equal(estimated_m, true_m, tolerance = 1e-9)
}) 

#===============================================================================
# O3 Algorithm Tests
#===============================================================================

# Helper function to canonicalize the nested list output from o3_algorithm
# for stable comparison. It sorts the inner vectors and then the outer list.
canonicalize_o3_output <- function(output) {
  lapply(output, function(iteration) {
    # Sort the numbers within each super-cohort vector
    sorted_scs <- lapply(iteration, sort)
    # Sort the list of super-cohorts to have a canonical order
    sorted_scs[order(sapply(sorted_scs, function(x) paste(x, collapse = "_")))]
  })
}

test_that("o3_algorithm works for staircase pattern", {
  observed_outcome_indices <- list(
    c(1, 2, 3),
    c(2, 3, 4),
    c(3, 4, 5)
  )
  r <- 2
  
  result <- o3_algorithm(observed_outcome_indices, r)
  
  # The initial state is not returned. All cohorts merge into one super-cohort
  # in a single iteration.
  expected <- list(
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
  
  # Expected merge history (initial state is not returned)
  # 1. Cohorts 1 and 2 merge -> {{1,2}, {3}}
  # 2. Cohort 3 merges with {1,2} -> {{1,2,3}}
  expected <- list(
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
  
  # Expected: cohorts 1 and 2 merge, but 3 remains separate.
  expected <- list(
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
  
  # No iterations should occur, so the result should be an empty list.
  expect_equal(result, list())
  expect_false(aligned_factors_identified(observed_outcome_indices, r))
}) 