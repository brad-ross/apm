#' Construct cohorts from raw panel data
#'
#' Takes raw panel data with at least two columns: a unit identifier
#' (`unit_id_col`) and an outcome identifier (`outcome_id_col`). It defines
#' cohorts as groups of units that share the same set of observed outcomes.
#' Outcomes are globally indexed in sorted order to produce stable indices.
#' The function returns: (1) the sorted list of outcomes (in index order),
#' (2) the list of cohorts with their observed outcome indices, and (3) a
#' data.table assigning each unit to its cohort.
#'
#' @param panel_df data.table containing panel data
#' @param unit_id_col Column name for unit identifier
#' @param outcome_id_col Column name for outcome identifier
#' @param model_rank Integer model rank
#' @param min_cohort_size Minimum units per cohort to keep (default: 0)
#' @return list with:
#'   - all_outcomes: sorted unique values of outcome_id_col
#'   - observed_outcome_indices: list of integer index vectors per cohort (ordered by cohort_id)
#'   - unit_cohorts: data.table with columns unit_id_col, cohort_id
#' @importFrom data.table setorder
#' @export
construct_cohorts_from_panel <- function(panel_df, unit_id_col, outcome_id_col, model_rank, min_cohort_size = 0) {
    panel_dt <- panel_df

    # Validate required columns exist
    required_cols <- c(unit_id_col, outcome_id_col)
    missing_cols <- setdiff(required_cols, names(panel_dt))
    if (length(missing_cols) > 0) {
        stop(sprintf(
            "construct_cohorts_from_panel(): missing required column(s): %s",
            paste(missing_cols, collapse = ", ")
        ))
    }

    # Build the globally sorted unique outcome vector and an inverse index map
    all_outcomes <- sort(unique(panel_dt[[outcome_id_col]]))
    outcome_to_index <- stats::setNames(seq_along(all_outcomes), as.character(all_outcomes))

    # For each unit, compute sorted indices of its unique outcomes using the map
    # Also compute a canonical string key for grouping cohorts
    unit_observed_outcomes <- panel_dt[, {
        idx <- sort(unique(outcome_to_index[as.character(get(outcome_id_col))]))
        list(
            observed_outcomes_list = list(idx),
            observed_key = paste(idx, collapse = ",")
        )
    }, by = c(unit_id_col)]

    # Build cohort summary by unique observed outcome sets in a single aggregation
    cohorts_dt <- unit_observed_outcomes[, list(
        observed_outcomes_list = observed_outcomes_list[1L],
        num_units = .N,
        outcome_count = length(observed_outcomes_list[[1L]])
    ), by = observed_key]

    # Filter out cohorts that are too small in either dimension
    cohorts_dt <- cohorts_dt[outcome_count >= model_rank & num_units >= min_cohort_size]

    # Sort by observed_key and assign cohort_id
    setorder(cohorts_dt, observed_key)
    cohorts_dt[, "cohort_id"] <- seq_len(nrow(cohorts_dt))

    # Join cohort ids back to units; keep only unit id and cohort id
    unit_cohorts <- unit_observed_outcomes[
        cohorts_dt[, list(observed_key, cohort_id)],
        on = "observed_key",
        nomatch = 0L
    ][, c(unit_id_col, "cohort_id"), with = FALSE]

    # Build list of observed outcome index vectors per cohort, ordered by cohort_id
    observed_outcome_indices <- cohorts_dt[order(cohort_id)][["observed_outcomes_list"]]

    return(list(
        all_outcomes = all_outcomes,
        observed_outcome_indices = observed_outcome_indices,
        unit_cohorts = unit_cohorts
    ))
}


