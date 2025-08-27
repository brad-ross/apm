context("Testing functions used to process raw panel data")

library(data.table)

# Test helpers to avoid repetition ------------------------------------------------
build_panel_from_indices <- function(outcomes, cohort_indices, units_by_cohort) {
    rbindlist(lapply(seq_along(cohort_indices), function(k) {
        idx <- cohort_indices[[k]]
        unit_ids <- units_by_cohort[[k]]
        rbindlist(lapply(unit_ids, function(u) {
            data.table(unit_id = u, outcome_id = outcomes[idx])
        }))
    }))
}

build_expected_unit_map <- function(units_by_cohort) {
    rbindlist(mapply(function(units, cid) {
        data.table(unit_id = units, cohort_id = cid)
    }, units_by_cohort, seq_along(units_by_cohort), SIMPLIFY = FALSE))
}

build_expected_processed_panel <- function(cohort_indices, units_by_cohort) {
    rbindlist(lapply(seq_along(cohort_indices), function(cid) {
        idxs <- cohort_indices[[cid]]
        rbindlist(lapply(units_by_cohort[[cid]], function(u) {
            data.table(
                unit_id = u,
                cohort_id = cid,
                outcome_idx = idxs,
                y = idxs
            )
        }))
    }))
}

test_that("staircase missingness with two units per cohort (matching core test)", {
    # Outcomes and units
    outcomes <- c("A", "B", "C", "D", "E")

    # Define cohorts as staircase matching core test indices (1-based):
    # {1,2,3}, {2,3,4}, {3,4,5}
    cohort_indices <- list(
        1:3,
        2:4,
        3:5
    )

    # Two units per cohort
    units_by_cohort <- list(
        c("u1", "u2"),
        c("u3", "u4"),
        c("u5", "u6")
    )

    # Build panel as data.table of (unit_id, outcome_id)
    panel_dt <- build_panel_from_indices(outcomes, cohort_indices, units_by_cohort)

    # Call function under test
    res <- construct_cohorts_from_panel(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        model_rank = 2,
        min_cohort_size = 2,
        cohort_observed_outcomes_as_df = FALSE
    )

    # 1) outcome_names should be sorted unique outcomes
    expect_equal(res$outcome_names, outcomes)

    # 2) observed_outcome_indices should match defined cohorts in cohort_id order
    expected_indices <- lapply(cohort_indices, as.integer)
    expect_equal(res$observed_outcome_indices, expected_indices)

    # 3) unit_cohorts should map two units per cohort correctly
    expect_true(is.data.table(res$unit_cohorts))
    expect_equal(sort(names(res$unit_cohorts)), sort(c("unit_id", "cohort_id")))

    # Build expected mapping
    expected_map <- build_expected_unit_map(units_by_cohort)

    # Compare after ordering by unit_id
    setorder(res$unit_cohorts, unit_id)
    setorder(expected_map, unit_id)
    expect_equal(res$unit_cohorts, expected_map)
})

test_that("UnbalancedPanel initializes and processes panel correctly", {
    # Outcomes and units (staircase pattern like existing tests)
    outcomes <- c("A", "B", "C", "D", "E")
    cohort_indices <- list(1:3, 2:4, 3:5)
    units_by_cohort <- list(
        c("u1", "u2"),
        c("u3", "u4"),
        c("u5", "u6")
    )

    # Build panel with value column 'y' = match(outcome_id, outcomes)
    panel_dt <- build_panel_from_indices(outcomes, cohort_indices, units_by_cohort)
    panel_dt[, y := match(outcome_id, outcomes)]

    # Construct object
    obj <- UnbalancedPanel$new(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        outcome_value_col = "y",
        model_rank = 2,
        min_cohort_size = 2
    )

    # outcome_names should be sorted unique outcomes
    expect_equal(obj$get_outcome_names(), outcomes)

    # observed_outcome_indices should match defined cohorts
    expected_indices <- lapply(cohort_indices, as.integer)
    expect_equal(obj$get_observed_outcome_indices(), expected_indices)

    # unit_cohorts should map two units per cohort correctly
    unit_cohorts <- obj$get_unit_cohorts()
    expect_true(is.data.table(unit_cohorts))
    expect_equal(sort(names(unit_cohorts)), sort(c("unit_id", "cohort_id")))

    expected_map <- build_expected_unit_map(units_by_cohort)
    setorder(unit_cohorts, unit_id)
    setorder(expected_map, unit_id)
    expect_equal(unit_cohorts, expected_map)

    # processed_panel should be correctly joined, indexed, and sorted
    pp <- obj$get_processed_panel()
    expect_true(is.data.table(pp))
    expect_equal(names(pp), c("unit_id", "cohort_id", "outcome_idx", "y"))

    # Check ordering: cohort_id, unit_id, outcome_idx
    pp_copy <- copy(pp)
    setorder(pp_copy, cohort_id, unit_id, outcome_idx)
    expect_equal(pp, pp_copy)

    # Build expected processed panel
    expected_pp <- build_expected_processed_panel(cohort_indices, units_by_cohort)
    setorder(expected_pp, cohort_id, unit_id, outcome_idx)

    expect_equal(pp, expected_pp)
})

test_that("drops cohort by size (min_cohort_size=2) and by rank (model_rank=2)", {

    outcomes <- c("A", "B", "C")

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

    panel_dt <- rbindlist(lapply(seq_along(cohort_indices), function(k) {
        idx <- cohort_indices[[k]]
        unit_ids <- units_by_cohort[[k]]
        rbindlist(lapply(unit_ids, function(u) {
            data.table(unit_id = u, outcome_id = outcomes[idx])
        }))
    }))

    # Case 1: Drop by size only (third cohort has 1 unit but 1 outcome; rank threshold met/not relevant)
    res <- construct_cohorts_from_panel(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        model_rank = 1,         # only require at least 1 outcome
        min_cohort_size = 2,    # require at least 2 units per cohort
        cohort_observed_outcomes_as_df = FALSE
    )

    # all outcomes should be intact
    expect_equal(res$outcome_names, outcomes)

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

    setorder(res$unit_cohorts, unit_id)
    setorder(expected_map, unit_id)
    expect_equal(res$unit_cohorts, expected_map)
    # Case 2: Drop by rank only (third cohort has 2 units but only 1 outcome)
    units_by_cohort_rank <- list(
        c("u1", "u2"),
        c("u3", "u4"),
        c("u5", "u6")  # now two units in last cohort
    )

    panel_dt_rank <- rbindlist(lapply(seq_along(cohort_indices), function(k) {
        idx <- cohort_indices[[k]]
        unit_ids <- units_by_cohort_rank[[k]]
        rbindlist(lapply(unit_ids, function(u) {
            data.table(unit_id = u, outcome_id = outcomes[idx])
        }))
    }))

    res_rank <- construct_cohorts_from_panel(
        panel_df = panel_dt_rank,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        model_rank = 2,         # now require at least 2 outcomes
        min_cohort_size = 1,    # size condition met for all cohorts
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
    setorder(res_rank$unit_cohorts, unit_id)
    setorder(expected_map_rank, unit_id)
    expect_equal(res_rank$unit_cohorts, expected_map_rank)
    
})

test_that("to_data_table converts base data.frame to data.table", {
    # Build a simple panel as data.table
    dt0 <- data.table(
        unit_id = c("u1", "u1", "u2", "u3"),
        outcome_id = c("A", "B", "B", "C")
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

    dt <- data.table(unit_id = c("u1", "u2"), outcome_id = c("A", "B"), y = 1:2)

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
    outcomes <- c("A", "B", "C", "D", "E")
    cohort_indices <- list(1:3, 2:4, 3:5)

    res_df <- apm:::construct_cohort_observed_outcomes_df(
        outcome_names = outcomes,
        observed_outcome_indices = cohort_indices
    )

    expect_true(is.data.frame(res_df))
    expect_equal(names(res_df), c("cohort_id", "outcome_idx", "outcome_name"))

    expected_df <- data.frame(
        cohort_id = c(rep(1L, 3), rep(2L, 3), rep(3L, 3)),
        outcome_idx = c(1L, 2L, 3L, 2L, 3L, 4L, 3L, 4L, 5L),
        outcome_name = c("A", "B", "C", "B", "C", "D", "C", "D", "E"),
        stringsAsFactors = FALSE
    )

    expect_equal(res_df, expected_df)
})

