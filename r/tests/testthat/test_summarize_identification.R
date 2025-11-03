context("Testing identification summarization functionality")

test_that("summarize_identification returns named list with expected fields", {
  ooi <- list(c(1L, 3L), c(2L, 3L))
  sizes <- c(50L, 40L)
  out <- summarize_identification(ooi, sizes, r = 2L)
  expect_type(out, "list")
  expect_setequal(names(out), c(
    "largest_super_cohort_size",
    "largest_super_cohort_share",
    "min_cohort_size_in_largest_super",
    "num_outcomes_in_largest_super_cohort",
    "num_o3_iterations"
  ))
  expect_true(out$largest_super_cohort_size >= 0)
  expect_true(out$min_cohort_size_in_largest_super >= 0)
  expect_gte(out$largest_super_cohort_share, 0)
  expect_lte(out$largest_super_cohort_share, 1)
  # With r=2 and these cohorts, no merges occur; only the initial state is present
  expect_equal(out$num_o3_iterations, 1)
  expect_equal(out$largest_super_cohort_size, 50)
  expect_equal(out$min_cohort_size_in_largest_super, 50)
  expect_equal(out$largest_super_cohort_share, 50/90, tolerance = 1e-12)
  # Largest super cohort is the larger single cohort {1,3} (1-based); union outcomes count is 2
  expect_equal(out$num_outcomes_in_largest_super_cohort, 2)
})

test_that("summarize_identification returns list of named lists for many panels", {
  ooi_list <- list(list(c(1L), c(2L, 3L)))
  size_list <- list(c(10L, 25L))
  out <- summarize_identification(ooi_list, size_list, r = 2L)
  expect_true(is.list(out))
  expect_equal(length(out), 1)
  el <- out[[1]]
  expect_setequal(names(el), c(
    "largest_super_cohort_size",
    "largest_super_cohort_share",
    "min_cohort_size_in_largest_super",
    "num_outcomes_in_largest_super_cohort",
    "num_o3_iterations"
  ))
  # For this case, no merges occur; only initial state is present
  expect_equal(el$num_o3_iterations, 1)
  expect_equal(el$largest_super_cohort_size, 25)
  expect_equal(el$min_cohort_size_in_largest_super, 25)
  expect_equal(el$largest_super_cohort_share, 25/35, tolerance = 1e-12)
  # Largest super cohort is the larger single cohort {2,3} (1-based); union outcomes count is 2
  expect_equal(el$num_outcomes_in_largest_super_cohort, 2)
})

test_that("aligned_factors_identified returns TRUE for staircase pattern", {
  observed_outcome_indices <- list(
    c(1, 2, 3),
    c(2, 3, 4),
    c(3, 4, 5)
  )
  r <- 2L
  expect_true(aligned_factors_identified(observed_outcome_indices, r))
})

test_that("aligned_factors_identified returns TRUE for non-contiguous pattern", {
  observed_outcome_indices <- list(
    c(1, 2, 3), # Cohort 1
    c(2, 3, 4), # Cohort 2
    c(1, 4, 5)  # Cohort 3
  )
  r <- 2L
  expect_true(aligned_factors_identified(observed_outcome_indices, r))
})

test_that("aligned_factors_identified returns FALSE for unconnected cohorts", {
  observed_outcome_indices <- list(
    c(1, 2), # Cohort 1
    c(2, 3, 4), # Cohort 2
    c(3, 4, 5)     # This cohort has no overlap with the others
  )
  r <- 2L
  expect_false(aligned_factors_identified(observed_outcome_indices, r))
})

test_that("aligned_factors_identified returns FALSE when no merges occur", {
  observed_outcome_indices <- list(
    c(1, 2),
    c(2, 3),
    c(3, 4, 5)
  )
  r <- 2L
  expect_false(aligned_factors_identified(observed_outcome_indices, r))
})

test_that("count_outcomes_with_rank_overlap_per_cohort matches expected counts", {
  ooi <- list(
    c(1L, 2L, 3L),
    c(2L, 3L, 4L),
    c(5L)
  )
  counts <- count_outcomes_with_rank_overlap_per_cohort(ooi, rank = 2L)
  expect_equal(as.integer(counts), c(4L, 4L, 1L))

  counts_all <- count_outcomes_with_rank_overlap_per_cohort(ooi, rank = 0L)
  expect_equal(as.integer(counts_all), c(5L, 5L, 5L))
})

test_that("get_masked_observed_outcome_indices masks per cohort as expected", {
  ooi <- list(c(1L, 3L), c(2L, 3L))
  # Drop outcome 3 from cohort 1 only
  mask <- list(`1` = c(3L))
  res <- get_masked_observed_outcome_indices(ooi, mask)
  expect_equal(res[[1]], c(1L))
  expect_equal(res[[2]], c(2L, 3L))

  # NULL mask returns unchanged
  res2 <- get_masked_observed_outcome_indices(ooi, NULL)
  expect_identical(res2, ooi)

  # Empty mask returns unchanged
  res3 <- get_masked_observed_outcome_indices(ooi, list())
  expect_identical(res3, ooi)
})