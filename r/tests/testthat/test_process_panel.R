context("Testing functions used to process raw panel data")

library(data.table)

# Shared helpers are defined in helper-test-utils.R

test_that("staircase missingness with two units per cohort (matching core test)", {
    # Outcomes and units
    outcomes <- make_outcomes(5)

    # Define cohorts as staircase matching core test indices (1-based):
    cohort_indices <- make_staircase_observed_indices(5, 3)

    # Two units per cohort
    units_by_cohort <- make_units_by_cohort(3, 2)

    # Build panel with full outcome grid, NA y for unobserved, and covariates
    panel_dt <- build_panel_from_indices(outcomes, cohort_indices, units_by_cohort, include_covariates = TRUE)

    # Call function under test
    res <- construct_cohorts_from_panel(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        sort_cohorts_lexicographically = TRUE,
        cohort_observed_outcomes_as_df = FALSE
    )

    # 1) outcome_ids should be sorted unique outcomes
    expect_equal(res$outcome_ids, outcomes)

    # 2) observed_outcome_indices should match defined cohorts in cohort_id order
    expected_indices <- lapply(cohort_indices, as.integer)
    expect_equal(res$observed_outcome_indices, expected_indices)
    # Re-indexing sanity: mapping is a permutation of original cohort indices
    panel_idx_for_orig <- match_cohorts_panel_order(expected_indices, res$observed_outcome_indices)
    expect_true(!any(is.na(panel_idx_for_orig)))
    expect_equal(sort(panel_idx_for_orig), seq_len(length(expected_indices)))

    # 3) unit_cohorts should map two units per cohort correctly
    expect_true(is.data.table(res$unit_cohorts))
    expect_equal(sort(names(res$unit_cohorts)), sort(c("unit_id", "cohort_id")))

    # 4) cohort_sizes should reflect two units per cohort
    expect_equal(res$cohort_sizes, rep.int(2L, length(expected_indices)))

    # Build expected mapping
    expected_map <- build_expected_unit_map(units_by_cohort)

    # Compare after ordering by unit_id
    setorder(res$unit_cohorts, unit_id)
    setorder(expected_map, unit_id)
    expect_equal(res$unit_cohorts, expected_map)
})

test_that("UnbalancedPanel initializes and processes panel correctly", {
    # Outcomes and units (staircase pattern like existing tests)
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3)
    units_by_cohort <- make_units_by_cohort(3, 2)

    # Build panel with full outcome grid, NA y for unobserved, and covariates
    panel_dt <- build_panel_from_indices(outcomes, cohort_indices, units_by_cohort, include_covariates = TRUE)

    # Construct object
    obj <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        covar_cols = c("cov1", "cov2"),
        model_rank = 2,
        min_cohort_size = 2,
        sort_cohorts_lexicographically = TRUE
    )

    # outcome_ids should be sorted unique outcomes
    expect_equal(obj$get_outcome_ids(), outcomes)
    # unit_ids should be sorted unique unit ids
    expect_equal(obj$get_unit_ids(), sort(unique(unlist(units_by_cohort))))

    # observed_outcome_indices should match defined cohorts
    expected_indices <- lapply(cohort_indices, as.integer)
    expect_equal(obj$get_observed_outcome_indices(), expected_indices)
    # Re-indexing sanity: mapping is a permutation of original cohort indices
    panel_idx_for_orig <- match_cohorts_panel_order(expected_indices, obj$get_observed_outcome_indices())
    expect_true(!any(is.na(panel_idx_for_orig)))
    expect_equal(sort(panel_idx_for_orig), seq_len(length(expected_indices)))

    # unit_cohorts should map two units per cohort correctly and include unit_idx
    unit_cohorts <- obj$get_unit_cohorts()
    expect_true(is.data.table(unit_cohorts))
    expect_equal(sort(names(unit_cohorts)), sort(c("unit_id", "unit_idx", "cohort_id")))

    expected_map <- build_expected_unit_map(units_by_cohort)
    unit_names <- sort(unique(unlist(units_by_cohort)))
    expected_map[, unit_idx := match(unit_id, unit_names)]
    setorder(unit_cohorts, unit_id)
    setorder(expected_map, unit_id)
    expect_equal(unit_cohorts, expected_map)

    # processed_panel should be correctly joined, indexed, and sorted
    pp <- obj$get_processed_panel()
    expect_true(is.data.table(pp))
    expect_equal(names(pp), c("unit_idx", "cohort_id", "outcome_idx", "y", "cov1", "cov2"))

    # Check ordering: cohort_id, unit_idx, outcome_idx
    pp_copy <- copy(pp)
    setorder(pp_copy, cohort_id, unit_idx, outcome_idx)
    expect_equal(pp, pp_copy)

    # Build expected processed panel
    expected_pp0 <- build_expected_processed_panel(outcomes, cohort_indices, units_by_cohort, include_covariates = TRUE)
    unit_names <- sort(unique(unlist(units_by_cohort)))
    expected_pp0[, unit_idx := match(unit_id, unit_names)]
    expected_pp <- expected_pp0[, .(unit_idx, cohort_id, outcome_idx, y, cov1, cov2)]
    setorder(expected_pp, cohort_id, unit_idx, outcome_idx)

    expect_equal(pp, expected_pp)
})

test_that("UnbalancedPanel works without covariates provided", {
    # Outcomes and units (staircase pattern)
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3)
    units_by_cohort <- make_units_by_cohort(3, 2)

    # Build panel with full outcome grid, NA y for unobserved, and covariates (which we won't pass)
    panel_dt <- build_panel_from_indices(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE)

    # Construct object without covariates
    obj <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2,
        sort_cohorts_lexicographically = TRUE
    )

    # covar_cols should be empty
    expect_equal(obj$get_covar_cols(), character(0))

    # outcome_ids and observed_outcome_indices as expected
    expect_equal(obj$get_outcome_ids(), outcomes)
    expected_indices <- lapply(cohort_indices, as.integer)
    expect_equal(obj$get_observed_outcome_indices(), expected_indices)
    # Re-indexing sanity: mapping is a permutation of original cohort indices
    panel_idx_for_orig <- match_cohorts_panel_order(expected_indices, obj$get_observed_outcome_indices())
    expect_true(!any(is.na(panel_idx_for_orig)))
    expect_equal(sort(panel_idx_for_orig), seq_len(length(expected_indices)))

    # processed panel should not include covariate columns
    pp <- obj$get_processed_panel()
    expect_true(is.data.table(pp))
    expect_equal(names(pp), c("unit_idx", "cohort_id", "outcome_idx", "y"))

    # Build expected processed panel without covariates
    expected_pp0 <- build_expected_processed_panel(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE)
    unit_names <- sort(unique(unlist(units_by_cohort)))
    expected_pp0[, unit_idx := match(unit_id, unit_names)]
    expected_pp <- expected_pp0[, .(unit_idx, cohort_id, outcome_idx, y)]
    setorder(expected_pp, cohort_id, unit_idx, outcome_idx)

    expect_equal(pp, expected_pp)
})

test_that("drops cohort by size (min_cohort_size=2) and by rank (model_rank=2)", {

    outcomes <- sprintf("%03d", 1:3)

    # Two large cohorts and one small cohort with a single unit
    cohort_indices <- list(
        c(1, 2),  # Cohort 1
        c(2, 3),  # Cohort 2
        c(1)      # Small cohort to be dropped
    )

    units_by_cohort <- list(
        c("u1", "u2"), # two units
        c("u3", "u4"), # two units
        c("u5")         # single unit -> should be dropped
    )

    panel_dt <- build_panel_from_indices(outcomes, cohort_indices, units_by_cohort)

    # Case 1: Drop by size only (third cohort has 1 unit but 1 outcome; rank threshold met/not relevant)
    res <- construct_cohorts_from_panel(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 1,         # only require at least 1 outcome
        min_cohort_size = 2,    # require at least 2 units per cohort
        sort_cohorts_lexicographically = TRUE,
        cohort_observed_outcomes_as_df = FALSE
    )

    # all outcomes should be intact
    expect_equal(res$outcome_ids, outcomes)

    # The small cohort has outcome_count=1 < 2 and num_units=1 < 2, so it's dropped
    expected_indices_size <- list(as.integer(c(1, 2)), as.integer(c(2, 3)))
    expect_equal(res$observed_outcome_indices, expected_indices_size)

    # unit_cohorts should include only u1,u2,u3,u4 mapped to cohort_id 1 and 2
    expect_true(is.data.table(res$unit_cohorts))
    expect_equal(sort(names(res$unit_cohorts)), sort(c("unit_id", "cohort_id")))

    expected_map <- rbindlist(list(
        data.table(unit_id = c("u1", "u2"), cohort_id = 1L),
        data.table(unit_id = c("u3", "u4"), cohort_id = 2L)
    ))
    setkey(expected_map, unit_id)

    setorder(res$unit_cohorts, unit_id)
    setorder(expected_map, unit_id)
    expect_equal(res$unit_cohorts, expected_map)
    # cohort_sizes should reflect two units in each retained cohort
    expect_equal(res$cohort_sizes, c(2L, 2L))
    # Case 2: Drop by rank only (third cohort has 2 units but only 1 outcome)
    units_by_cohort_rank <- list(
        c("u1", "u2"),
        c("u3", "u4"),
        c("u5", "u6")  # now two units in last cohort
    )

    panel_dt_rank <- build_panel_from_indices(outcomes, cohort_indices, units_by_cohort_rank, include_covariates = FALSE)

    res_rank <- construct_cohorts_from_panel(
        panel_df = panel_dt_rank,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,         # now require at least 2 outcomes
        min_cohort_size = 1,    # size condition met for all cohorts
        sort_cohorts_lexicographically = TRUE,
        cohort_observed_outcomes_as_df = FALSE
    )

    expected_indices_rank <- list(as.integer(c(1, 2)), as.integer(c(2, 3)))
    expect_equal(res_rank$observed_outcome_indices, expected_indices_rank)

    # unit_cohorts should include only the four units from the retained cohorts
    expect_true(is.data.table(res_rank$unit_cohorts))
    expect_equal(sort(names(res_rank$unit_cohorts)), sort(c("unit_id", "cohort_id")))
    expect_equal(nrow(res_rank$unit_cohorts), 4L)
    expect_equal(length(unique(res_rank$unit_cohorts$cohort_id)), 2L)
    expected_map_rank <- rbindlist(list(
        data.table(unit_id = c("u1", "u2"), cohort_id = 1L),
        data.table(unit_id = c("u3", "u4"), cohort_id = 2L)
    ))
    setkey(expected_map_rank, unit_id)
    setorder(res_rank$unit_cohorts, unit_id)
    setorder(expected_map_rank, unit_id)
    expect_equal(res_rank$unit_cohorts, expected_map_rank)
    # cohort_sizes should reflect two units in each retained cohort
    expect_equal(res_rank$cohort_sizes, c(2L, 2L))
    
})

test_that("to_data_table converts base data.frame to data.table", {
    # Build a simple panel as data.table
    dt0 <- data.table(
        unit_id = c("u1", "u1", "u2", "u3"),
        outcome_id = c("001", "002", "002", "003")
    )

    # Convert to base data.frame
    df <- as.data.frame(dt0, stringsAsFactors = FALSE)

    # Use internal helper via triple-colon
    dt1 <- apm:::to_data_table(df)

    # Validate class and equality (order-insensitive)
    expect_true(is.data.table(dt1))
    setorder(dt0, unit_id, outcome_id)
    setorder(dt1, unit_id, outcome_id)
    expect_equal(dt1, dt0)
})

test_that("validate_required_panel_cols enforces required columns", {

    dt <- data.table(unit_id = c("u1", "u2"), outcome_id = c("001", "002"), y = 1:2)

    # Passes with required cols present
    expect_silent(apm:::validate_required_panel_cols(dt, "unit_id", "outcome_id"))
    expect_silent(apm:::validate_required_panel_cols(dt, "unit_id", "outcome_id", "y"))

    # Fails when missing unit_id
    expect_error(apm:::validate_required_panel_cols(dt[, .(outcome_id, y)], "unit_id", "outcome_id"),
                 regexp = "missing required column")

    # Fails when missing outcome_id
    expect_error(apm:::validate_required_panel_cols(dt[, .(unit_id, y)], "unit_id", "outcome_id"),
                 regexp = "missing required column")

    # Fails when missing optional outcome_value_col
    expect_error(apm:::validate_required_panel_cols(dt[, .(unit_id, outcome_id)], "unit_id", "outcome_id", "y"),
                 regexp = "missing required column")
})


test_that("construct_cohort_observed_outcomes_df builds long-form mapping", {
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3)

    res_df <- apm:::construct_cohort_observed_outcomes_df(
        outcome_ids = outcomes,
        observed_outcome_indices = cohort_indices
    )

    expect_true(is.data.frame(res_df))
    expect_equal(names(res_df), c("cohort_id", "outcome_idx", "outcome_name"))

    expected_df <- data.frame(
        cohort_id = c(rep(1L, 3), rep(2L, 3), rep(3L, 3)),
        outcome_idx = c(1L, 2L, 3L, 2L, 3L, 4L, 3L, 4L, 5L),
        outcome_name = outcomes[c(1L, 2L, 3L, 2L, 3L, 4L, 3L, 4L, 5L)],
        stringsAsFactors = FALSE
    )

    expect_equal(res_df, expected_df)
})

test_that("subset_to_largest_super_cohort retains all cohorts when model_rank equals adjacency overlap", {
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3) # cohorts: [1,2,3], [2,3,4], [3,4,5]
    units_by_cohort <- make_units_by_cohort(3, 2)            # two units per cohort

    panel_dt <- build_panel_from_indices(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE)

    # Baseline without subsetting
    base <- construct_cohorts_from_panel(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,                         # equals adjacency overlap for k=3
        min_cohort_size = 0,
        sort_cohorts_lexicographically = TRUE,
        cohort_observed_outcomes_as_df = FALSE
    )

    # With subsetting to largest super cohort (should keep the full set)
    subsetted <- construct_cohorts_from_panel(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 0,
        subset_to_largest_super_cohort = TRUE,
        sort_cohorts_lexicographically = TRUE,
        cohort_observed_outcomes_as_df = FALSE
    )

    expect_equal(subsetted$outcome_ids, base$outcome_ids)
    expect_equal(subsetted$observed_outcome_indices, base$observed_outcome_indices)

    setorder(subsetted$unit_cohorts, unit_id)
    setorder(base$unit_cohorts, unit_id)
    expect_equal(subsetted$unit_cohorts, base$unit_cohorts)

    expect_equal(subsetted$cohort_sizes, base$cohort_sizes)
})

test_that("subset_to_largest_super_cohort picks the largest cohort when model_rank exceeds adjacency overlap", {
    outcomes <- make_outcomes(5)
    cohort_indices <- make_staircase_observed_indices(5, 3) # overlap between adjacent cohorts is 2

    # Make cohorts of unequal sizes so the largest is unambiguous
    units_by_cohort <- list(
        c("u1", "u2", "u3"),  # 3 units (largest)
        c("u4", "u5"),        # 2 units
        c("u6")               # 1 unit
    )

    panel_dt <- build_panel_from_indices(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE)

    res <- construct_cohorts_from_panel(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 3,                         # > overlap (2) -> super cohort cannot span multiple cohorts
        min_cohort_size = 0,
        subset_to_largest_super_cohort = TRUE,
        sort_cohorts_lexicographically = TRUE,
        cohort_observed_outcomes_as_df = FALSE
    )

    # Only the largest cohort should remain; its outcomes are the first cohort's indices
    expect_equal(res$observed_outcome_indices, list(as.integer(cohort_indices[[1]])))

    # unit_cohorts should include only units from the largest cohort
    expect_true(is.data.table(res$unit_cohorts))
    setorder(res$unit_cohorts, unit_id)
    expected_uc <- data.table(unit_id = units_by_cohort[[1]], cohort_id = 1L)
    setkey(expected_uc, unit_id)
    expect_equal(res$unit_cohorts, expected_uc)

    # cohort_sizes should reflect the retained cohort's size
    expect_equal(res$cohort_sizes, 3L)

    # outcome_ids restricted to that cohort's outcomes (since unobserved rows are dropped)
    expect_equal(res$outcome_ids, outcomes[cohort_indices[[1]]])
})