test_that("summarize_identification returns named list with expected fields", {
  ooi <- list(c(1L, 3L), c(2L, 3L))
  sizes <- c(50L, 40L)
  out <- summarize_identification(ooi, sizes, r = 2L)
  expect_type(out, "list")
  expect_setequal(names(out), c(
    "largest_super_cohort_size",
    "largest_super_cohort_share",
    "min_cohort_size_in_largest_super",
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
})

test_that("summarize_identification_many returns list of named lists", {
  ooi_list <- list(list(c(1L), c(2L, 3L)))
  size_list <- list(c(10L, 25L))
  out <- summarize_identification_many(ooi_list, size_list, r = 2L)
  expect_true(is.list(out))
  expect_equal(length(out), 1)
  el <- out[[1]]
  expect_setequal(names(el), c(
    "largest_super_cohort_size",
    "largest_super_cohort_share",
    "min_cohort_size_in_largest_super",
    "num_o3_iterations"
  ))
  # For this case, no merges occur; only initial state is present
  expect_equal(el$num_o3_iterations, 1)
  expect_equal(el$largest_super_cohort_size, 25)
  expect_equal(el$min_cohort_size_in_largest_super, 25)
  expect_equal(el$largest_super_cohort_share, 25/35, tolerance = 1e-12)
})