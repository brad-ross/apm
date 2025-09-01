context("Testing bootstrap data structures")

test_that("Multinomial bootstrap dims, sums, and accessors", {
  N <- 25L; B <- 100L
  wb <- WeightedBootstrap$new(N, B, type = "multinomial", seed = 123)

  expect_equal(wb$n_obs(), N)
  expect_equal(wb$n_bootstraps(), B)

  W <- wb$weights()
  expect_equal(dim(W), c(N, B))
  expect_true(all(abs(colSums(W) - 1) < 1e-12))
  expect_true(all(W >= 0))

  d1 <- wb$draw(1)
  expect_equal(length(d1), N)

  oi <- wb$obs(1)
  expect_equal(length(oi), B)

  idx <- c(1L, 3L, 3L, N)
  sub <- wb$obs_rows(idx)
  expect_equal(dim(sub), c(length(idx), B))
  expect_equal(sub[2, ], sub[3, ])

  # expect_error(wb$draw(B + 1L))
  # expect_error(wb$obs(N + 1L))
  # expect_error(wb$obs_rows(c(1L, N + 5L)))
})

test_that("Bayesian bootstrap dims and sums", {
  N <- 30L; B <- 80L
  wb <- WeightedBootstrap$new(N, B, type = "bayesian", seed = 321)

  expect_equal(wb$n_obs(), N)
  expect_equal(wb$n_bootstraps(), B)

  W <- wb$weights()
  expect_equal(dim(W), c(N, B))
  expect_true(all(abs(colSums(W) - 1) < 1e-12))
  expect_true(all(W >= 0))
})

test_that("get_weighted_bootstrap_draws returns WeightedBootstrap instance", {
  N <- 12L; B <- 7L
  wb <- get_weighted_bootstrap_draws(N, B, type = "multinomial", seed = 7)
  expect_true(inherits(wb, "WeightedBootstrap"))
  expect_equal(wb$n_obs(), N)
  expect_equal(wb$n_bootstraps(), B)
  W <- wb$weights()
  expect_equal(dim(W), c(N, B))
  expect_true(all(abs(colSums(W) - 1) < 1e-12))
  expect_true(all(W >= 0))

  expect_error(get_weighted_bootstrap_draws(N, B, type = "not-a-type", seed = 1))
})