test_that("get_version works", {
  version <- get_version()
  expect_type(version, "character")
  expect_equal(version, "0.1.0")
}) 