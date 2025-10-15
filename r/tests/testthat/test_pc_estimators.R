proj <- function(X) {
  X %*% solve(crossprod(X)) %*% t(X)
}

expect_same_subspace <- function(G1, G2, tol = 1e-9) {
  P1 <- proj(G1)
  P2 <- proj(G2)
  testthat::expect_true(max(abs(P1 - P2)) < tol, info = paste0("P1=\n", capture.output(print(P1)), "\nP2=\n", capture.output(print(P2))))
}

context("Testing principal component factor model estimators")

#==============================================================================
# PCEstimator tests (no fixed effects)
#==============================================================================

test_that("PCEstimator: SingleBatch_NoBootstrap_PrincipalDirection", {
  T_c <- 2L; r <- 1L
  est <- PCEstimator$new(r, T_c)

  Y <- rbind(
    c(1, 0),
    c(1, 0),
    c(0, 2),
    c(0, 2)
  )
  unit_idxs <- 1:4

  est$add_data(unit_idxs, Y)
  out <- est$estimate()

  expected_G <- matrix(c(0, 1), ncol = 1)
  expect_same_subspace(out$G(), expected_G)
  expect_false(out$has_bootstrap())
})

test_that("PCEstimator: SplitBatch_Invariance", {
  T_c <- 2L; r <- 1L
  est <- PCEstimator$new(r, T_c)

  Y1 <- rbind(c(1, 0), c(1, 0))
  idx1 <- c(1L, 2L)

  Y2 <- rbind(c(0, 2), c(0, 2))
  idx2 <- c(3L, 4L)

  est$add_data(idx1, Y1)
  est$add_data(idx2, Y2)

  out <- est$estimate()
  expected_G <- matrix(c(0, 1), ncol = 1)
  expect_same_subspace(out$G(), expected_G)
})

test_that("PCEstimator: AddDatum_Equivalence", {
  T_c <- 2L; r <- 1L
  est <- PCEstimator$new(r, T_c)

  Y <- rbind(
    c(1, 0),
    c(1, 0),
    c(0, 2),
    c(0, 2)
  )

  for (i in 1:4) {
    est$add_datum(i, Y[i, ])
  }

  out <- est$estimate()
  expected_G <- matrix(c(0, 1), ncol = 1)
  expect_same_subspace(out$G(), expected_G)
})

test_that("PCEstimator: Bootstrap_ReproducibleWithSeed_AndSizes", {
  T_c <- 2L; r <- 1L; B <- 2L

  Y <- rbind(
    c(1, 0),
    c(1, 0),
    c(0, 2),
    c(0, 2)
  )
  unit_idxs <- 1:4

  wb1 <- WeightedBootstrap$new(N = nrow(Y), B = B, type = "multinomial", seed = 123)
  wb2 <- WeightedBootstrap$new(N = nrow(Y), B = B, type = "multinomial", seed = 123)

  est1 <- PCEstimator$new(r, T_c, bootstrap = wb1)
  est1$add_data(unit_idxs, Y)
  out1 <- est1$estimate()

  est2 <- PCEstimator$new(r, T_c, bootstrap = wb2)
  est2$add_data(unit_idxs, Y)
  out2 <- est2$estimate()

  expect_true(out1$has_bootstrap())
  expect_equal(out1$num_bootstraps(), B)

  # Point estimate subspace agrees across runs
  expect_same_subspace(out1$G(), out2$G())

  # Replicate agreement across runs with same seed
  for (b in 1:B) {
    expect_equal(out1$G(b), out2$G(b))
  }
})

test_that("PCEstimator: QHandling_DimensionsEnforced_CovariatesIgnored", {
  T_c <- 2L; r <- 1L; q <- 1L

  Y <- rbind(
    c(1, 0),
    c(1, 0),
    c(0, 2),
    c(0, 2)
  )
  unit_idxs <- 1:4

  X1 <- array(1, dim = c(nrow(Y), T_c, q))
  X2 <- array(0, dim = c(nrow(Y), T_c, q))

  est1 <- PCEstimator$new(r, T_c, q = q)
  est1$add_data(unit_idxs, Y, X1)
  out1 <- est1$estimate()

  est2 <- PCEstimator$new(r, T_c, q = q)
  est2$add_data(unit_idxs, Y, X2)
  out2 <- est2$estimate()

  expect_same_subspace(out1$G(), out2$G())

#   # Invalid shapes should throw
#   est_bad_q0 <- PCEstimator$new(r, T_c) # q==0
#   X_bad_q0 <- array(1, dim = c(nrow(Y), T_c, 1))
#   expect_error(est_bad_q0$add_data(unit_idxs, Y, X_bad_q0))

#   est_bad_dims <- PCEstimator$new(r, T_c, q = q)
#   X_wrong_slices <- array(1, dim = c(nrow(Y), T_c, 2))
#   expect_error(est_bad_dims$add_data(unit_idxs, Y, X_wrong_slices))
#   X_wrong_rows <- array(1, dim = c(nrow(Y) + 1L, T_c, q))
#   expect_error(est_bad_dims$add_data(unit_idxs, Y, X_wrong_rows))
})

test_that("PCEstimator: RankZero_CornerCase", {
  T_c <- 2L; r <- 0L
  est <- PCEstimator$new(r, T_c)

  Y <- rbind(c(1, 2), c(3, 4))
  unit_idxs <- c(1L, 2L)

  est$add_data(unit_idxs, Y)
  out <- est$estimate()

  G <- out$G()
  expect_equal(nrow(G), T_c)
  expect_equal(ncol(G), 0L)
  expect_false(out$has_g0())
  expect_false(out$has_bootstrap())
})

# test_that("PCEstimator: InputValidationAndConstructorChecks", {
#   expect_error(PCEstimator$new(3L, 2L))

#   T_c <- 2L; r <- 1L
#   est <- PCEstimator$new(r, T_c)

#   unit_idxs <- c(1L, 2L)
#   Y_wrong_cols <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, byrow = TRUE) # 2 x 3
#   expect_error(est$add_data(unit_idxs, Y_wrong_cols))

#   Y <- matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)
#   X_nonempty <- array(1, dim = c(2, T_c, 1))
#   expect_error(est$add_data(unit_idxs, Y, X_nonempty)) # q==0 but X non-empty
# })

# #==============================================================================
# # PCEstimatorWithFEs tests (subtract means)
# #==============================================================================

test_that("PCEstimatorWithFEs: SingleBatch_MeansAndCovariance_NoBootstrap", {
  T_c <- 2L; r <- 1L
  est <- PCEstimatorWithFEs$new(r, T_c)

  Y <- rbind(
    c(1, 2),
    c(1, 0),
    c(0, 2),
    c(0, 0)
  )
  unit_idxs <- 1:4

  est$add_data(unit_idxs, Y)
  out <- est$estimate()

  expect_true(out$has_g0())
  expected_mu <- c(0.5, 1.0)
  expect_equal(out$g0()[, 1], expected_mu, tolerance = 1e-12)

  expected_G <- matrix(c(0, 1), ncol = 1)
  expect_same_subspace(out$G(), expected_G)
})

test_that("PCEstimatorWithFEs: SplitBatch_Invariance", {
  T_c <- 2L; r <- 1L
  est <- PCEstimatorWithFEs$new(r, T_c)

  Y1 <- rbind(c(1, 2), c(1, 0))
  idx1 <- c(1L, 2L)

  Y2 <- rbind(c(0, 2), c(0, 0))
  idx2 <- c(3L, 4L)

  est$add_data(idx1, Y1)
  est$add_data(idx2, Y2)

  out <- est$estimate()

  expect_true(out$has_g0())
  expected_mu <- c(0.5, 1.0)
  expect_equal(out$g0()[, 1], expected_mu, tolerance = 1e-12)

  expected_G <- matrix(c(0, 1), ncol = 1)
  expect_same_subspace(out$G(), expected_G)
})

test_that("PCEstimatorWithFEs: AddDatum_Equivalence", {
  T_c <- 2L; r <- 1L
  est <- PCEstimatorWithFEs$new(r, T_c)

  Y <- rbind(
    c(1, 2),
    c(1, 0),
    c(0, 2),
    c(0, 0)
  )

  for (i in 1:4) {
    est$add_datum(i, Y[i, ])
  }

  out <- est$estimate()
  expected_mu <- c(0.5, 1.0)
  expect_true(out$has_g0())
  expect_equal(out$g0()[, 1], expected_mu, tolerance = 1e-12)

  expected_G <- matrix(c(0, 1), ncol = 1)
  expect_same_subspace(out$G(), expected_G)
})

test_that("PCEstimatorWithFEs: Bootstrap_ReproducibleWithSeed_AndSizes", {
  T_c <- 2L; r <- 1L; B <- 2L

  Y <- rbind(
    c(1, 2),
    c(1, 0),
    c(0, 2),
    c(0, 0)
  )
  unit_idxs <- 1:4

  wb1 <- WeightedBootstrap$new(N = nrow(Y), B = B, type = "multinomial", seed = 777)
  wb2 <- WeightedBootstrap$new(N = nrow(Y), B = B, type = "multinomial", seed = 777)

  est1 <- PCEstimatorWithFEs$new(r, T_c, bootstrap = wb1)
  est1$add_data(unit_idxs, Y)
  out1 <- est1$estimate()

  est2 <- PCEstimatorWithFEs$new(r, T_c, bootstrap = wb2)
  est2$add_data(unit_idxs, Y)
  out2 <- est2$estimate()

  expect_true(out1$has_bootstrap())
  expect_equal(out1$num_bootstraps(), B)
  expect_true(out1$has_g0())

  # Point estimates identical under same seed
  expect_equal(out1$g0(), out2$g0(), tolerance = 1e-12)
  expect_same_subspace(out1$G(), out2$G())

  # Replicates identical under same seed
  for (b in 1:B) {
    expect_equal(out1$g0(b), out2$g0(b), tolerance = 1e-12)
    expect_equal(out1$G(b), out2$G(b))
  }
})

test_that("PCEstimatorWithFEs: QHandling_DimensionsEnforced_CovariatesIgnored", {
  T_c <- 2L; r <- 1L; q <- 1L

  Y <- rbind(
    c(1, 2),
    c(1, 0),
    c(0, 2),
    c(0, 0)
  )
  unit_idxs <- 1:4

  X1 <- array(1, dim = c(nrow(Y), T_c, q))
  X2 <- array(0, dim = c(nrow(Y), T_c, q))

  est1 <- PCEstimatorWithFEs$new(r, T_c, q = q)
  est1$add_data(unit_idxs, Y, X1)
  out1 <- est1$estimate()

  est2 <- PCEstimatorWithFEs$new(r, T_c, q = q)
  est2$add_data(unit_idxs, Y, X2)
  out2 <- est2$estimate()

  expect_true(out1$has_g0())
  expect_true(out2$has_g0())
  expect_equal(out1$g0(), out2$g0(), tolerance = 1e-12)
  expect_same_subspace(out1$G(), out2$G())

#   # Invalid shapes should throw
#   est_bad_q0 <- PCEstimatorWithFEs$new(r, T_c) # q==0
#   X_bad_q0 <- array(1, dim = c(nrow(Y), T_c, 1))
#   expect_error(est_bad_q0$add_data(unit_idxs, Y, X_bad_q0))

#   est_bad_dims <- PCEstimatorWithFEs$new(r, T_c, q = q)
#   X_wrong_slices <- array(1, dim = c(nrow(Y), T_c, 2))
#   expect_error(est_bad_dims$add_data(unit_idxs, Y, X_wrong_slices))
#   X_wrong_rows <- array(1, dim = c(nrow(Y) + 1L, T_c, q))
#   expect_error(est_bad_dims$add_data(unit_idxs, Y, X_wrong_rows))
})

test_that("PCEstimatorWithFEs: RankZero_CornerCase", {
  T_c <- 2L; r <- 0L
  est <- PCEstimatorWithFEs$new(r, T_c)

  Y <- rbind(c(1, 2), c(3, 4))
  unit_idxs <- c(1L, 2L)

  est$add_data(unit_idxs, Y)
  out <- est$estimate()

  G <- out$G()
  expect_equal(nrow(G), T_c)
  expect_equal(ncol(G), 0L)
  expect_true(out$has_g0())
  expected_mu <- c(2.0, 3.0)
  expect_equal(out$g0()[, 1], expected_mu, tolerance = 1e-12)
  expect_false(out$has_bootstrap())
})