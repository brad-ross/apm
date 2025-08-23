context("process_panel: construct_cohorts_from_panel")

test_that("staircase missingness with two units per cohort (matching core test)", {
    library(data.table)

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
    panel_dt <- rbindlist(lapply(seq_along(cohort_indices), function(k) {
        idx <- cohort_indices[[k]]
        unit_ids <- units_by_cohort[[k]]
        rbindlist(lapply(unit_ids, function(u) {
            data.table(unit_id = u, outcome_id = outcomes[idx])
        }))
    }))

    # Call function under test
    res <- construct_cohorts_from_panel(
        panel_df = panel_dt,
        unit_id_col = "unit_id",
        outcome_id_col = "outcome_id",
        model_rank = 2,
        min_cohort_size = 2
    )

    # 1) all_outcomes should be sorted unique outcomes
    expect_equal(res$all_outcomes, outcomes)

    # 2) observed_outcome_indices should match defined cohorts in cohort_id order
    expected_indices <- lapply(cohort_indices, as.integer)
    expect_equal(res$observed_outcome_indices, expected_indices)

    # 3) unit_cohorts should map two units per cohort correctly
    expect_true(is.data.table(res$unit_cohorts))
    expect_equal(sort(names(res$unit_cohorts)), sort(c("unit_id", "cohort_id")))

    # Build expected mapping
    expected_map <- rbindlist(mapply(function(units, cid) {
        data.table(unit_id = units, cohort_id = cid)
    }, units_by_cohort, seq_along(units_by_cohort), SIMPLIFY = FALSE))

    # Compare after ordering by unit_id
    setorder(res$unit_cohorts, unit_id)
    setorder(expected_map, unit_id)
    expect_equal(res$unit_cohorts, expected_map)
})

test_that("drops cohort by size (min_cohort_size=2) and by rank (model_rank=2)", {
    library(data.table)

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
        min_cohort_size = 2     # require at least 2 units per cohort
    )

    # all outcomes should be intact
    expect_equal(res$all_outcomes, outcomes)

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
    print(res$unit_cohorts)
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
        min_cohort_size = 1     # size condition met for all cohorts
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


