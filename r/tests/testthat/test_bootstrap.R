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

test_that("get_bootstrap_inference returns well-formed results and sensible bands", {
  set.seed(123)
  p <- 4L; B <- 400L; N <- 200L
  point <- rnorm(p, 0, 1 / sqrt(N))
  boot <- matrix(rnorm(p * B, 0, 1 / sqrt(N)), nrow = p)
  boot <- sweep(boot, 1, point, FUN = "+")

  sir <- get_bootstrap_inference(point, boot, N, sig_level = 0.05)
  expect_true(inherits(sir, "SimultaneousInferenceResults"))

  est <- sir$point()
  t <- sir$t_stats()
  pvals <- sir$p_vals()
  pvals_fwer <- sir$fwer_control_p_vals()
  ci <- sir$ci()
  cb <- sir$cb()
  se <- sir$std_error()
  sqrtN <- sqrt(N)

  expect_equal(length(est), p)
  expect_equal(length(t), p)
  expect_equal(length(pvals), p)
  expect_equal(length(pvals_fwer), p)
  expect_equal(length(ci$lb), p)
  expect_equal(length(ci$ub), p)
  expect_equal(length(cb$lb), p)
  expect_equal(length(cb$ub), p)
  expect_equal(length(se), p)
  expect_true(all(is.finite(se)))
  expect_equal(sir$se(), se)
  # FWER p-values should be at least as large as pointwise p-values
  expect_true(all(pvals_fwer >= pvals - 1e-12))

  expect_true(all(ci$lb <= est & est <= ci$ub))
  expect_true(all(cb$lb <= est & est <= cb$ub))
  expect_true(all(abs(se - 1 / sqrtN) < 0.02))

  ci_half <- (ci$ub - ci$lb) / 2
  cb_half <- (cb$ub - cb$lb) / 2
  expect_true(all(cb_half + 1e-12 >= ci_half))

  expect_equal(sir$sig_level(), 0.05)

  df <- sir$as_data_frame()
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), p)
  expect_true(all(c("parameter","estimate","std_error","t_stat","p_value","p_value_fwer","ci_lb","ci_ub","cb_lb","cb_ub") %in% names(df)))
  expect_equal(df$std_error, se)
})

test_that("Bootstrap inference coverage and p-values behave under the null", {
  skip_on_cran()
  if (!isTRUE(getOption("apm_run_perf_tests", FALSE))) skip("Set options(apm_run_perf_tests = TRUE) to enable perf checks.")
  
  set.seed(12345)
  p <- 5L
  B <- 1000L
  N <- 100L
  sig_level <- 0.05
  n_sims <- 5000L

  sqrtN <- sqrt(N)
  simult_covered <- logical(n_sims)
  pointwise_counts <- integer(p)
  p_value_matrix <- matrix(NA_real_, nrow = p, ncol = n_sims)
  fwer_reject_count <- 0L

  for (sim in seq_len(n_sims)) {
    point <- rnorm(p) / sqrtN
    boot <- matrix(rnorm(p * B) / sqrtN, nrow = p)
    boot <- sweep(boot, 1, point, FUN = "+")
    sir <- get_bootstrap_inference(point, boot, N, sig_level)

    ci <- sir$ci()
    cb <- sir$cb()
    pvals <- sir$p_vals()

    covered_pointwise <- (ci$lb <= 0) & (0 <= ci$ub)
    pointwise_counts <- pointwise_counts + as.integer(covered_pointwise)

    covered_simult <- all((cb$lb <= 0) & (0 <= cb$ub))
    simult_covered[sim] <- covered_simult

    p_value_matrix[, sim] <- pvals

    # Family-wise error: any Romano-Wolf adjusted p-value below sig_level
    if (any(sir$fwer_control_p_vals() <= sig_level)) {
      fwer_reject_count <- fwer_reject_count + 1L
    }
  }

  simult_rate <- mean(simult_covered)
  expect_gt(simult_rate, 0.93)
  expect_lt(simult_rate, 0.97)

  pointwise_rates <- pointwise_counts / n_sims
  for (i in seq_len(p)) {
    expect_gt(pointwise_rates[i], 0.93)
    expect_lt(pointwise_rates[i], 0.97)
  }

  deciles <- seq(0.1, 0.9, by = 0.1)
  tol <- 0.03
  for (i in seq_len(p)) {
    pv <- p_value_matrix[i, ]
    for (q in deciles) {
      share <- mean(pv < q)
      expect_lt(abs(share - q), tol)
    }
  }

  # FWER should be controlled at sig_level (allow small tolerance)
  fwer <- fwer_reject_count / n_sims
  expect_lt(fwer, sig_level + 0.005)
})