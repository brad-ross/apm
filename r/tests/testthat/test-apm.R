library(testthat)

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
run_alignment_test_r <- function(true_factors, observed_outcome_indices) {
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
    observed_outcome_indices
  )
  
  # Check the result by comparing projection matrices
  proj_aligned <- projection_matrix_r(aligned_factors)
  proj_true <- projection_matrix_r(true_factors)
  
  expect_equal(proj_aligned, proj_true, tolerance = 1e-9)
}


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