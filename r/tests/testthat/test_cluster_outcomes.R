testthat::test_that("comp_outcome_clustering: two clusters single-k vs range equality", {
  make_panel <- function(K = 5L) {
    T <- 4L
    df <- data.frame(
      unit_id = rep(seq_len(K), each = T),
      outcome_id = rep(seq_len(T), times = K),
      y = rep(c(0, 0, 10, 10), times = K)
    )
    UnbalancedPanel$new(
      df,
      unit_id_col = "unit_id",
      outcome_id_col = "outcome_id",
      outcome_value_col = "y"
    )
  }

  grid_size <- 3L
  panel <- make_panel()

  set.seed(123)
  labels_single <- comp_outcome_clustering(panel, grid_size = grid_size, k = 2L)
  testthat::expect_type(labels_single, "integer")
  testthat::expect_length(labels_single, 4L)
  testthat::expect_false(any(is.na(labels_single)))
  testthat::expect_true(all(labels_single %in% 1:2))

  # Grouping pattern: outcomes 1-2 together, 3-4 together, and groups differ
  testthat::expect_equal(unname(labels_single[1L]), unname(labels_single[2L]))
  testthat::expect_equal(unname(labels_single[3L]), unname(labels_single[4L]))
  testthat::expect_false(unname(labels_single[1L]) == unname(labels_single[3L]))

  set.seed(123)
  labels_range_mat <- comp_outcome_clusterings(panel, grid_size = grid_size, min_k = 2L, max_k = 2L)
  testthat::expect_true(is.matrix(labels_range_mat))
  testthat::expect_equal(nrow(labels_range_mat), 4L)
  testthat::expect_equal(ncol(labels_range_mat), 1L)
  # Prefer column by name if available; otherwise use the first column
  k2_col <- if (!is.null(colnames(labels_range_mat)) && any(colnames(labels_range_mat) == "2")) {
    labels_range_mat[, "2", drop = TRUE]
  } else {
    labels_range_mat[, 1L, drop = TRUE]
  }
  testthat::expect_type(k2_col, "integer")
  testthat::expect_identical(as.integer(labels_single), as.integer(k2_col))
})

testthat::test_that("comp_outcome_clusterings: k-range 1..3 and grouping at k=3", {
  make_panel <- function(K = 5L) {
    T <- 4L
    df <- data.frame(
      unit_id = rep(seq_len(K), each = T),
      outcome_id = rep(seq_len(T), times = K),
      y = rep(c(0, 0, 10, 10), times = K)
    )
    UnbalancedPanel$new(
      df,
      unit_id_col = "unit_id",
      outcome_id_col = "outcome_id",
      outcome_value_col = "y"
    )
  }

  grid_size <- 3L
  panel <- make_panel()

  set.seed(123)
  labels_range_mat <- comp_outcome_clusterings(panel, grid_size = grid_size, min_k = 1L, max_k = 3L)
  testthat::expect_true(is.matrix(labels_range_mat))
  testthat::expect_equal(nrow(labels_range_mat), 4L)
  testthat::expect_equal(ncol(labels_range_mat), 3L)
  if (!is.null(colnames(labels_range_mat))) {
    testthat::expect_true(all(as.character(1:3) %in% colnames(labels_range_mat)))
  }

  # k = 1: all same label
  k1_col <- if (!is.null(colnames(labels_range_mat)) && any(colnames(labels_range_mat) == "1")) {
    labels_range_mat[, "1", drop = TRUE]
  } else {
    labels_range_mat[, 1L, drop = TRUE]
  }
  testthat::expect_equal(length(unique(as.integer(k1_col))), 1L)

  # k = 3: only 2 groups present due to identical feature rows per group
  k3_col <- if (!is.null(colnames(labels_range_mat)) && any(colnames(labels_range_mat) == "3")) {
    labels_range_mat[, "3", drop = TRUE]
  } else {
    labels_range_mat[, 3L, drop = TRUE]
  }
  testthat::expect_gte(length(unique(as.integer(k3_col))), 1L)
  testthat::expect_equal(length(unique(as.integer(k3_col))), 2L)
  testthat::expect_equal(unname(k3_col[1L]), unname(k3_col[2L]))
  testthat::expect_equal(unname(k3_col[3L]), unname(k3_col[4L]))
  testthat::expect_false(unname(k3_col[1L]) == unname(k3_col[3L]))
})

testthat::test_that("comp_outcome_clusterings: k-range 1..3 with multi-threading", {
  make_panel <- function(K = 5L) {
    T <- 4L
    df <- data.frame(
      unit_id = rep(seq_len(K), each = T),
      outcome_id = rep(seq_len(T), times = K),
      y = rep(c(0, 0, 10, 10), times = K)
    )
    UnbalancedPanel$new(
      df,
      unit_id_col = "unit_id",
      outcome_id_col = "outcome_id",
      outcome_value_col = "y"
    )
  }

  grid_size <- 3L
  panel <- make_panel()

  set.seed(123)
  labels_range_mat <- comp_outcome_clusterings(
    panel,
    grid_size = grid_size,
    min_k = 1L,
    max_k = 3L,
    num_threads = 2L
  )
  testthat::expect_true(is.matrix(labels_range_mat))
  testthat::expect_equal(nrow(labels_range_mat), 4L)
  testthat::expect_equal(ncol(labels_range_mat), 3L)
  if (!is.null(colnames(labels_range_mat))) {
    testthat::expect_true(all(as.character(1:3) %in% colnames(labels_range_mat)))
  }

  # k = 1: all same label
  k1_col <- if (!is.null(colnames(labels_range_mat)) && any(colnames(labels_range_mat) == "1")) {
    labels_range_mat[, "1", drop = TRUE]
  } else {
    labels_range_mat[, 1L, drop = TRUE]
  }
  testthat::expect_equal(length(unique(as.integer(k1_col))), 1L)

  # k = 3: only 2 groups present due to identical feature rows per group
  k3_col <- if (!is.null(colnames(labels_range_mat)) && any(colnames(labels_range_mat) == "3")) {
    labels_range_mat[, "3", drop = TRUE]
  } else {
    labels_range_mat[, 3L, drop = TRUE]
  }
  testthat::expect_equal(length(unique(as.integer(k3_col))), 2L)
  testthat::expect_equal(unname(k3_col[1L]), unname(k3_col[2L]))
  testthat::expect_equal(unname(k3_col[3L]), unname(k3_col[4L]))
  testthat::expect_false(unname(k3_col[1L]) == unname(k3_col[3L]))
})

testthat::test_that("comp_outcome_clustering vs comp_outcome_clusterings: seed consistency for k=2,3", {
  make_panel <- function(K = 5L) {
    T <- 4L
    df <- data.frame(
      unit_id = rep(seq_len(K), each = T),
      outcome_id = rep(seq_len(T), times = K),
      y = rep(c(0, 0, 10, 10), times = K)
    )
    UnbalancedPanel$new(
      df,
      unit_id_col = "unit_id",
      outcome_id_col = "outcome_id",
      outcome_value_col = "y"
    )
  }

  grid_size <- 3L
  panel <- make_panel()

  set.seed(777)
  labels_range_mat <- comp_outcome_clusterings(panel, grid_size = grid_size, min_k = 2L, max_k = 3L)
  for (k in 2:3) {
    set.seed(777)
    single <- comp_outcome_clustering(panel, grid_size = grid_size, k = k)
    col <- if (!is.null(colnames(labels_range_mat)) && any(colnames(labels_range_mat) == as.character(k))) {
      labels_range_mat[, as.character(k), drop = TRUE]
    } else {
      labels_range_mat[, k - 1L, drop = TRUE]
    }
    testthat::expect_identical(as.integer(single), as.integer(col))
  }
})
testthat::test_that("clustering seeds isolate randomness and agree across thread counts", {
  df <- data.frame(
    unit_id = rep(1:8, each = 12),
    outcome_id = rep(1:12, times = 8),
    y = sin(seq_len(96))
  )
  panel <- UnbalancedPanel$new(df, "unit_id", "outcome_id", "y")
  cluster <- function(seed, threads, n_inits = 20L) {
    comp_outcome_clusterings(panel, grid_size = 10L, min_k = 2L,
                            max_k = 5L, n_inits = n_inits,
                            seed = seed, num_threads = threads)
  }
  for (seed in c(0, 9001)) {
    set.seed(11)
    rng_before <- .Random.seed
    testthat::expect_warning(serial <- cluster(seed, 1L), NA)
    testthat::expect_identical(.Random.seed, rng_before)
    set.seed(782)
    testthat::expect_identical(cluster(seed, 1L), serial)
    for (i in 1:3) testthat::expect_identical(cluster(seed, 4L), serial)
    for (k in 2:5) {
      single <- comp_outcome_clustering(panel, 10L, k, n_inits = 20L,
                                       seed = seed, num_threads = 4L)
      testthat::expect_identical(as.integer(single), as.integer(serial[, as.character(k)]))
    }
  }
  set.seed(123)
  testthat::expect_warning(first <- cluster(NULL, 1L), NA)
  set.seed(123)
  testthat::expect_identical(cluster(NULL, 4L), first)
  # A single initialization on these overlapping outcome distributions exposes
  # different random starts; a constant implementation would fail this check.
  starts <- lapply(0:10, function(seed) cluster(seed, 1L, n_inits = 1L))
  testthat::expect_gt(length(unique(starts)), 1L)
})

testthat::test_that("equal-SSE partitions select the earliest initialization", {
  # CDF columns are (1,1), (0,1), (0,0), with equal outcome weights.
  # Either adjacent pair can merge: both distinct partitions have SSE = 1/2.
  df <- data.frame(unit_id = 1L, outcome_id = 1:3, y = 0:2)
  panel <- UnbalancedPanel$new(df, "unit_id", "outcome_id", "y")
  cluster <- function(seed, n_inits, threads) {
    comp_outcome_clustering(panel, 2L, 2L, n_inits = n_inits,
                           seed = seed, num_threads = threads)
  }
  first_starts <- lapply(0:10, function(seed) cluster(seed, 1L, 1L))
  testthat::expect_equal(length(unique(first_starts)), 2L)
  for (seed in 0:10) {
    earliest <- first_starts[[seed + 1L]]
    testthat::expect_identical(cluster(seed, 100L, 1L), earliest)
    for (i in 1:3) testthat::expect_identical(cluster(seed, 100L, 4L), earliest)
  }
})
