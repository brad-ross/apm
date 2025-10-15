context("Testing TWFE estimator")

test_that("TWFEEstimator: SingleBatch_NoBootstrap_GOnes_MeanFE_NoCovars", {
  T_c <- 3L
  est <- TWFEEstimator$new(T_c)

  Y <- rbind(
    c(1, 2, 3),
    c(3, 2, 1),
    c(2, 2, 2)
  )
  unit_idxs <- 1:nrow(Y)

  est$add_data(unit_idxs, Y)
  out <- est$estimate()

  expect_equal(out$G(), matrix(1, nrow = T_c, ncol = 1))
  expect_true(out$has_g0())
  expect_equal(as.numeric(out$g0()), as.numeric(colMeans(Y)), tolerance = 1e-12)
  expect_false(out$has_a())
  expect_false(out$has_bootstrap())
})

test_that("TWFEEstimator: SplitBatch_And_AddDatum_Equivalence", {
  T_c <- 2L
  est_batch <- TWFEEstimator$new(T_c)
  est_datum <- TWFEEstimator$new(T_c)

  Y <- rbind(
    c(1, 2),
    c(3, 4),
    c(5, 6),
    c(7, 8)
  )

  est_batch$add_data(c(1L, 2L), Y[1:2, , drop = FALSE])
  est_batch$add_data(c(3L, 4L), Y[3:4, , drop = FALSE])

  for (i in 1:nrow(Y)) est_datum$add_datum(i, Y[i, ])

  out1 <- est_batch$estimate()
  out2 <- est_datum$estimate()

  expect_true(out1$has_g0())
  expect_true(out2$has_g0())
  expect_equal(as.numeric(out1$g0()), as.numeric(out2$g0()), tolerance = 1e-12)
  expect_equal(out1$G(), out2$G())
})

test_that("TWFEEstimator: Bootstrap_ReproducibleWithSeed_AndSizes", {
  T_c <- 2L; B <- 2L

  Y <- rbind(
    c(1, 0),
    c(1, 0),
    c(0, 2),
    c(0, 2)
  )
  unit_idxs <- 1:nrow(Y)

  wb1 <- WeightedBootstrap$new(N = nrow(Y), B = B, type = "multinomial", seed = 777)
  wb2 <- WeightedBootstrap$new(N = nrow(Y), B = B, type = "multinomial", seed = 777)

  est1 <- TWFEEstimator$new(T_c, bootstrap = wb1)
  est1$add_data(unit_idxs, Y)
  out1 <- est1$estimate()

  est2 <- TWFEEstimator$new(T_c, bootstrap = wb2)
  est2$add_data(unit_idxs, Y)
  out2 <- est2$estimate()

  expect_true(out1$has_bootstrap())
  expect_equal(out1$num_bootstraps(), B)
  expect_true(out1$has_g0())

  expect_equal(out1$g0(), out2$g0(), tolerance = 1e-12)
  expect_equal(out1$G(), out2$G())

  for (b in 1:B) {
    expect_equal(out1$g0(b), out2$g0(b), tolerance = 1e-12)
    expect_equal(out1$G(b), out2$G(b))
  }
})

test_that("TWFEEstimator: QHandling_ZeroCoeffsPresentWhenQPositive", {
  T_c <- 3L; q <- 2L
  Y <- rbind(c(1, 2, 3), c(2, 3, 4))
  unit_idxs <- 1:nrow(Y)
  X <- array(1, dim = c(nrow(Y), T_c, q))

  est <- TWFEEstimator$new(T_c, q = q)
  est$add_data(unit_idxs, Y, X)
  out <- est$estimate()

  expect_true(out$has_a())
  expect_equal(as.numeric(out$a()), as.numeric(rep(0, q)))
})