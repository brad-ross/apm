context("Testing code that aggregates cohort-specific factor model estimates and estimates outcome means")

test_that("EstimateMeans_EstimatesWithBootstrap_AllComponents (R6)", {
  data <- setup_estimation_test_data_r()

  true_m <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  true_m[1,] <- (data$G %*% data$l_c[[1]] + data$X_c_vec[[1]] %*% data$a + data$g_0)
  true_m[2,] <- (data$G %*% data$l_c[[2]] + data$X_c_vec[[2]] %*% data$a + data$g_0)

  # Build OutcomeMeanSuffStatEstimates list (point + B replicates)
  B <- 2L
  suff_est_vec <- lapply(seq_len(data$Cval), function(c) {
    m_c <- true_m[c, data$observed_outcome_indices[[c]]]
    xp <- make_outcome_mean_suff_stat_estimates_cpp(m_c, data$X_c_vec[[c]], B)
    OutcomeMeanSuffStatEstimates$new(xp)
  })

  # FactorModelEstimates (point + B replicates)
  xp_fme <- make_factor_model_estimates_cpp(data$G, data$g_0, data$a, B)
  fme <- FactorModelEstimates$new(xp_fme)

  out <- estimate_outcome_means_across_cohorts(fme, data$observed_outcome_indices, suff_est_vec)
  expect_equal(out$mean_outcomes(), true_m, tolerance = 1e-9)
  expect_true(out$has_bootstrap())
  expect_equal(out$num_bootstraps(), B)
  for (b in seq_len(B)) {
    expect_equal(out$mean_outcomes(b), out$mean_outcomes(), tolerance = 1e-9)
  }
})

test_that("AggregateFactorModelParams_MapOverSpecs_Succeeds (R6 + map)", {
  # Build a simple scenario mirroring the C++ test
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

  C <- length(observed_outcome_indices)
  q <- 2L
  B <- 2L

  rotation_matrices <- generate_symmetric_rotation_matrices(C, ncol(true_factors))
  g0_true <- c(0.10, 0.20, 0.30, 0.40, 0.50)

  # Build per-cohort FMEs (point + B identical replicates)
  cohort_estimates <- lapply(seq_len(C), function(c) {
    G_c <- true_factors[observed_outcome_indices[[c]], ] %*% rotation_matrices[[c]]
    g0_c <- g0_true[observed_outcome_indices[[c]]]
    xp <- make_factor_model_estimates_cpp(G_c, g0_c, c(0.5 + 0.1 * (c - 1), 1.0 + 0.2 * (c - 1)), B)
    FactorModelEstimates$new(xp)
  })

  # specA equal weights; specB unequal
  wA <- CohortWeightEstimates$new(make_cohort_weight_estimates_cpp(rep(1, C), bootstrap_weights = replicate(B, rep(1, C), simplify = FALSE)))
  w_point <- c(1.0, 2.0, 3.0)
  wB <- CohortWeightEstimates$new(make_cohort_weight_estimates_cpp(w_point, bootstrap_weights = replicate(B, w_point, simplify = FALSE)))

  res <- aggregate_factor_model_params_by_spec(
    list(specA = cohort_estimates, specB = cohort_estimates),
    observed_outcome_indices,
    list(specA = wA, specB = wB)
  )

  expect_true(all(c("specA", "specB") %in% names(res)))
  # Validate structures and bootstrap presence
  expect_true(res$specA$has_bootstrap())
  expect_true(res$specB$has_bootstrap())
})

test_that("EstimateMeans_MapOverSpecs_Succeeds (R6 + map)", {
  data <- setup_estimation_test_data_r()
  true_m <- matrix(0, nrow = data$Cval, ncol = data$Tval)
  true_m[1,] <- (data$G %*% data$l_c[[1]] + data$X_c_vec[[1]] %*% data$a + data$g_0)
  true_m[2,] <- (data$G %*% data$l_c[[2]] + data$X_c_vec[[2]] %*% data$a + data$g_0)

  B <- 2L
  # Build FME map with two identical specs
  xp_fme <- make_factor_model_estimates_cpp(data$G, data$g_0, data$a, B)
  fmap <- list(specA = FactorModelEstimates$new(xp_fme), specB = FactorModelEstimates$new(xp_fme))

  # Build suff stats list
  suff_est_vec <- lapply(seq_len(data$Cval), function(c) {
    m_c <- true_m[c, data$observed_outcome_indices[[c]]]
    xp <- make_outcome_mean_suff_stat_estimates_cpp(m_c, data$X_c_vec[[c]], B)
    OutcomeMeanSuffStatEstimates$new(xp)
  })

  out_map <- estimate_outcome_means_across_cohorts_by_spec(fmap, data$observed_outcome_indices, suff_est_vec)
  expect_true(all(c("specA", "specB") %in% names(out_map)))
  expect_equal(out_map$specA$mean_outcomes(), true_m, tolerance = 1e-9)
  expect_equal(out_map$specB$mean_outcomes(), true_m, tolerance = 1e-9)
})