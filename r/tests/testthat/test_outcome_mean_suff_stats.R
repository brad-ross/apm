context("Testing streaming outcome mean sufficient statistics estimator")

#==============================================================================
# Outcome means without covariates
#==============================================================================

test_that("OutcomeMeans: SingleBatch_NoBootstrap", {
  T_c <- 4L
  Y <- rbind(
    c(11, 21, 31, 41),
    c(12, 22, 32, 42),
    c(13, 23, 33, 43),
    c(14, 24, 34, 44),
    c(15, 25, 35, 45),
    c(16, 26, 36, 46)
  )
  unit_idxs <- 1:nrow(Y)

  est <- OutcomeMeanSuffStatEstimator$new(T_c)
  est$add_data(unit_idxs, Y)
  out <- est$estimate(total_units = nrow(Y))

  expect_false(out$has_bootstrap())
  expect_equal(out$num_bootstraps(), 0L)
  expect_equal(out$T_c(), T_c)
  expect_equal(out$T(), 0L)
  expect_equal(out$q(), 0L)

  expected_means <- colMeans(Y)
  expect_equal(out$observed_outcome_means(), expected_means, tolerance = 1e-12)
  expect_null(out$covar_means())
  # Population share equals N / total_units = 1
  expect_equal(out$cohort_pop_share(), 1, tolerance = 1e-12)
})

test_that("OutcomeMeans: SplitBatch_Invariance", {
  T_c <- 2L
  Y <- rbind(
    c(1, 0),
    c(1, 0),
    c(0, 2),
    c(0, 2)
  )

  est1 <- OutcomeMeanSuffStatEstimator$new(T_c)
  est1$add_data(1:2, Y[1:2, , drop = FALSE])
  est1$add_data(3:4, Y[3:4, , drop = FALSE])
  out1 <- est1$estimate(total_units = nrow(Y))

  est2 <- OutcomeMeanSuffStatEstimator$new(T_c)
  est2$add_data(1:4, Y)
  out2 <- est2$estimate(total_units = nrow(Y))

  expect_equal(out1$observed_outcome_means(), out2$observed_outcome_means(), tolerance = 1e-12)
  # Shares equal and equal to 1
  expect_equal(out1$cohort_pop_share(), 1, tolerance = 1e-12)
  expect_equal(out2$cohort_pop_share(), 1, tolerance = 1e-12)
})

test_that("OutcomeMeans: AddDatum_Equivalence", {
  T_c <- 2L
  Y <- rbind(
    c(1, 0),
    c(1, 0),
    c(0, 2),
    c(0, 2)
  )

  est1 <- OutcomeMeanSuffStatEstimator$new(T_c)
  est1$add_data(1:4, Y)
  out1 <- est1$estimate(total_units = nrow(Y))

  est2 <- OutcomeMeanSuffStatEstimator$new(T_c)
  for (i in 1:nrow(Y)) est2$add_datum(i, Y[i, ])
  out2 <- est2$estimate(total_units = nrow(Y))

  expect_equal(out1$observed_outcome_means(), out2$observed_outcome_means(), tolerance = 1e-12)
  expect_equal(out1$cohort_pop_share(), 1, tolerance = 1e-12)
  expect_equal(out2$cohort_pop_share(), 1, tolerance = 1e-12)
})

#==============================================================================
# Bootstrap reproducibility and sizes
#==============================================================================

test_that("OutcomeMeans: Bootstrap_ReproducibleWithSeed_AndSizes", {
  T_c <- 3L; B <- 5L
  Y <- rbind(
    c(11, 21, 31),
    c(12, 22, 32),
    c(13, 23, 33),
    c(14, 24, 34),
    c(15, 25, 35),
    c(16, 26, 36)
  )
  unit_idxs <- 1:nrow(Y)

  wb1 <- get_weighted_bootstrap_draws(nrow(Y), B, type = "multinomial", seed = 999)
  wb2 <- get_weighted_bootstrap_draws(nrow(Y), B, type = "multinomial", seed = 999)

  est1 <- OutcomeMeanSuffStatEstimator$new(T_c, bootstrap = wb1)
  est1$add_data(unit_idxs, Y)
  out1 <- est1$estimate(total_units = nrow(Y))

  est2 <- OutcomeMeanSuffStatEstimator$new(T_c, bootstrap = wb2)
  est2$add_data(unit_idxs, Y)
  out2 <- est2$estimate(total_units = nrow(Y))

  expect_true(out1$has_bootstrap())
  expect_equal(out1$num_bootstraps(), B)

  # Point estimates identical across runs
  expect_equal(out1$observed_outcome_means(), out2$observed_outcome_means(), tolerance = 1e-12)

  # Replicate agreement across runs with same seed
  for (b in 1:B) {
    expect_equal(out1$observed_outcome_means(b), out2$observed_outcome_means(b), tolerance = 1e-12)
    # Bootstrap shares should be 1 when all units are included
    expect_equal(out1$cohort_pop_share(b), 1, tolerance = 1e-12)
    expect_equal(out2$cohort_pop_share(b), 1, tolerance = 1e-12)
  }
})

#==============================================================================
# Covariates handling
#==============================================================================

test_that("OutcomeMeans: Covariates_Means_NoBootstrap", {
  N <- 5L; T_c <- 3L; q <- 2L
  Y <- matrix(rep(seq_len(T_c), each = N), nrow = N, ncol = T_c)
  X <- array(0, dim = c(N, T_c, q))
  for (k in 1:q) for (j in 1:T_c) X[, j, k] <- 100 * k + j + seq_len(N)

  est <- OutcomeMeanSuffStatEstimator$new(T_c, T = T_c, q = q)
  est$add_data(1:N, Y, X)
  out <- est$estimate(total_units = N)

  expect_equal(out$T(), T_c)
  expect_equal(out$q(), q)
  expect_equal(out$cohort_pop_share(), 1, tolerance = 1e-12)

  expected_Xbar <- do.call(cbind, lapply(1:q, function(k) colMeans(X[, , k])))
  expect_equal(out$covar_means(), expected_Xbar, tolerance = 1e-12)
})


test_that("OutcomeMeans: Covariates_WithBootstrap_AndUnitSelection", {
  N <- 8L; T_c <- 3L; q <- 2L; B <- 4L
  Y <- matrix(0, nrow = N, ncol = T_c)
  for (j in 1:T_c) Y[, j] <- 5 * j + seq_len(N)
  X <- array(0, dim = c(N, T_c, q))
  for (k in 1:q) for (j in 1:T_c) X[, j, k] <- 100 * k + 2 * j + seq_len(N)

  wb <- get_weighted_bootstrap_draws(N, B, type = "multinomial", seed = 321)
  unit_idxs <- c(2L, 4L, 6L, 8L)

  est <- OutcomeMeanSuffStatEstimator$new(T_c, T = T_c, q = q, bootstrap = wb)
  est$add_data(unit_idxs, Y[unit_idxs, , drop = FALSE], X[unit_idxs, , , drop = FALSE])
  out <- est$estimate(total_units = length(unit_idxs))

  expect_true(out$has_bootstrap())
  expect_equal(out$num_bootstraps(), B)
  # Point share remains 1 when total_units equals number of provided units
  expect_equal(out$cohort_pop_share(), 1, tolerance = 1e-12)

  # Manual expected means for a chosen replicate
  W_rows <- wb$obs_rows(unit_idxs) # length(unit_idxs) x B
  for (b in c(1L, B)) {
    w <- as.numeric(W_rows[, b]); w <- w / sum(w)
    # outcome means
    expected_mb <- as.vector(t(Y[unit_idxs, , drop = FALSE]) %*% w)
    expect_equal(out$observed_outcome_means(b), expected_mb, tolerance = 1e-12)
    # covariate means
    expected_Xbar_b <- do.call(cbind, lapply(1:q, function(k) {
      Xk <- X[unit_idxs, , k, drop = TRUE]
      Xk <- matrix(Xk, nrow = length(unit_idxs), ncol = T_c)
      as.vector(t(Xk) %*% w)
    }))
    expect_equal(out$covar_means(b), expected_Xbar_b, tolerance = 1e-12)
    # Bootstrap share equals unnormalized column sum over the provided units
    expect_equal(out$cohort_pop_share(b), sum(W_rows[, b]), tolerance = 1e-12)
  }
})