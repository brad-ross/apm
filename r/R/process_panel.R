## Internal helpers ------------------------------------------------------------

#' Convert various table types to a data.table
#'
#' Accepts a base data.frame, a tibble, Arrow objects (Table, RecordBatch,
#' RecordBatchReader, Dataset), or a data.table, and returns a data.table.
#' Arrow-backed inputs are materialized into R memory before conversion.
#'
#' @param x Table-like object to convert
#' @return A data.table with the same rows/columns as the input
#' @keywords internal
to_data_table <- function(x) {
    if (data.table::is.data.table(x)) return(x)
    if (inherits(x, c("Table", "RecordBatch", "RecordBatchReader", "Dataset"))) {
        return(data.table::as.data.table(as.data.frame(x)))
    }
    data.table::as.data.table(x)
}

#' Validate required panel columns exist
#'
#' Checks that `unit_id_col`, `outcome_id_col`, and optionally
#' `outcome_value_col` are present as columns in `panel_dt`.
#'
#' @param panel_dt A data.table representing the panel dataset
#' @param unit_id_col Column name for unit identifier
#' @param outcome_id_col Column name for outcome identifier
#' @param outcome_value_col Optional column name for outcome value
#' @return Invisibly returns TRUE if validation passes; otherwise errors
#' @keywords internal
validate_required_panel_cols <- function(panel_dt,
                                        unit_id_col,
                                        outcome_id_col,
                                        outcome_value_col = NULL) {
    required_cols <- c(unit_id_col, outcome_id_col)
    if (!is.null(outcome_value_col)) {
        required_cols <- c(required_cols, outcome_value_col)
    }
    missing_cols <- setdiff(required_cols, names(panel_dt))
    if (length(missing_cols) > 0) {
        stop(sprintf(
            "validate_required_panel_cols(): missing required column(s): %s",
            paste(missing_cols, collapse = ", ")
        ))
    }
    invisible(TRUE)
}

# ------------------------------------------------------------------------------

#' Construct long-form mapping of cohort observed outcomes
#'
#' Given the globally sorted `outcome_ids` vector and the
#' `observed_outcome_indices` list (each element is an integer vector of outcome
#' indices observed in a cohort, ordered by `cohort_id`), build a data.frame
#' where each row corresponds to one (cohort, outcome) pair.
#'
#' Columns:
#'   - cohort_id: integer id of the cohort (1-based index into the list)
#'   - outcome_idx: integer index of the outcome (position in `outcome_ids`)
#'   - outcome_name: character name of the outcome (`outcome_ids[outcome_idx]`)
#'
#' @param outcome_ids Character (or coercible) vector of outcome ids
#' @param observed_outcome_indices List of integer vectors, per cohort
#' @return A base R data.frame with columns `cohort_id`, `outcome_idx`, `outcome_name`
#' @export
construct_cohort_observed_outcomes_df <- function(outcome_ids, observed_outcome_indices) {
    outcome_ids_chr <- as.character(outcome_ids)

    if (!is.list(observed_outcome_indices)) {
        stop("construct_cohort_observed_outcomes_df(): observed_outcome_indices must be a list of integer vectors")
    }

    lens <- vapply(observed_outcome_indices, length, integer(1))
    if (length(lens) == 0L || sum(lens) == 0L) {
        return(data.frame(
            cohort_id = integer(0),
            outcome_idx = integer(0),
            outcome_name = character(0),
            stringsAsFactors = FALSE
        ))
    }

    outcome_idx <- as.integer(unlist(observed_outcome_indices, use.names = FALSE))

    if (any(is.na(outcome_idx)) || any(outcome_idx < 1L) || any(outcome_idx > length(outcome_ids_chr))) {
        stop("construct_cohort_observed_outcomes_df(): outcome index out of bounds for some cohort")
    }

    cohort_id <- rep.int(seq_along(observed_outcome_indices), times = lens)
    outcome_name <- outcome_ids_chr[outcome_idx]

    data.frame(
        cohort_id = as.integer(cohort_id),
        outcome_idx = as.integer(outcome_idx),
        outcome_name = as.character(outcome_name),
        stringsAsFactors = FALSE
    )
}

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
#' @param cohort_observed_outcomes_as_df Logical; if TRUE (default), return a
#'   single `cohort_observed_outcomes_df` (with columns `cohort_id`, `outcome_idx`,
#'   `outcome_name`) instead of the triplet (`outcome_ids`, `outcome_to_index`,
#'   `observed_outcome_indices`). If FALSE, return the original triplet.
#' @return list with either:
#'   - when cohort_observed_outcomes_as_df = TRUE:
#'       - cohort_observed_outcomes_df: long-form mapping of cohorts to outcomes
#'       - unit_cohorts: data.table with columns unit_id_col, cohort_id
#'   - when cohort_observed_outcomes_as_df = FALSE:
#'       - outcome_ids: sorted unique values of outcome_id_col
#'       - outcome_ids: sorted unique values of outcome_id_col
#'       - outcome_to_index: named integer vector mapping outcome value -> index
#'       - observed_outcome_indices: list of integer index vectors per cohort (ordered by cohort_id)
#'       - unit_cohorts: data.table with columns unit_id_col, cohort_id
#' @importFrom data.table setorder
#' @export
construct_cohorts_from_panel <- function(panel_df,
                                         unit_id_col,
                                         outcome_id_col,
                                         model_rank,
                                         min_cohort_size = 0,
                                         cohort_observed_outcomes_as_df = TRUE) {
    panel_dt <- to_data_table(panel_df)

    # Validate required columns exist using internal helper
    validate_required_panel_cols(panel_dt, unit_id_col, outcome_id_col)

    # Build the globally sorted unique outcome vector and an inverse index map
    outcome_ids <- sort(unique(panel_dt[[outcome_id_col]]))
    outcome_to_index <- setNames(seq_along(outcome_ids), as.character(outcome_ids))

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

    if (isTRUE(cohort_observed_outcomes_as_df)) {
        return(list(
            cohort_observed_outcomes_df = construct_cohort_observed_outcomes_df(
                outcome_ids = outcome_ids,
                observed_outcome_indices = observed_outcome_indices
            ),
            unit_cohorts = unit_cohorts
        ))
    } else {
        return(list(
            outcome_ids = outcome_ids,
            outcome_to_index = outcome_to_index,
            observed_outcome_indices = observed_outcome_indices,
            unit_cohorts = unit_cohorts
        ))
    }
}

# ------------------------------------------------------------------------------

#' Unbalanced panel container and processor (stub)
#'
#' R6 class that holds panel data and related cohort artifacts. Constructor
#' mirrors `construct_cohorts_from_panel` but adds `outcome_value_col` after
#' `outcome_id_col`.
#'
#' @export
UnbalancedPanel <- R6Class(
    classname = "UnbalancedPanel",
    public = list(
        # Getters
        get_original_panel = function() private$original_panel,
        get_unit_id_col = function() private$unit_id_col,
        get_outcome_id_col = function() private$outcome_id_col,
        get_outcome_value_col = function() private$outcome_value_col,
        get_model_rank = function() private$model_rank,
        get_min_cohort_size = function() private$min_cohort_size,
        get_unit_ids = function() private$unit_ids,
        get_outcome_ids = function() private$outcome_ids,
        get_outcome_to_index = function() private$outcome_to_index,
        get_observed_outcome_indices = function() private$observed_outcome_indices,
        get_unit_cohorts = function() private$unit_cohorts,
        get_processed_panel = function() private$processed_panel,

        initialize = function(panel_df,
                              unit_id_col,
                              outcome_id_col,
                              outcome_value_col,
                              model_rank,
                              min_cohort_size = 0) {
            private$original_panel <- to_data_table(panel_df)
            private$unit_id_col <- unit_id_col
            private$outcome_id_col <- outcome_id_col
            private$outcome_value_col <- outcome_value_col
            private$model_rank <- as.integer(model_rank)
            private$min_cohort_size <- as.integer(min_cohort_size)
            validate_required_panel_cols(private$original_panel, 
                private$unit_id_col, private$outcome_id_col, private$outcome_value_col)
            
            # Build sorted unit ids and index map
            unit_ids <- sort(unique(private$original_panel[[private$unit_id_col]]))
            private$unit_ids <- unit_ids
            private$unit_to_index <- setNames(seq_along(unit_ids), as.character(unit_ids))

            # Compute cohorts and save artifacts
            coh <- construct_cohorts_from_panel(
                private$original_panel,
                private$unit_id_col,
                private$outcome_id_col,
                private$model_rank,
                private$min_cohort_size,
                cohort_observed_outcomes_as_df = FALSE
            )
            private$outcome_ids <- coh$outcome_ids
            private$outcome_to_index <- coh$outcome_to_index
            private$observed_outcome_indices <- coh$observed_outcome_indices
            private$unit_cohorts <- coh$unit_cohorts
            
            # Augment unit_cohorts with unit_idx
            private$unit_cohorts[, unit_idx := private$unit_to_index[as.character(get(private$unit_id_col))]]

            # Select only relevant columns from the original panel
            orig_panel_only_relevant_cols <- private$original_panel[
                , c(private$unit_id_col, private$outcome_id_col, private$outcome_value_col)
                , with = FALSE
            ]

            # Inner join on unit id to attach cohort_id to each observation
            processed <- private$unit_cohorts[orig_panel_only_relevant_cols, on = private$unit_id_col, nomatch = 0L]

            # Map outcome ids to outcome indices and drop original outcome id column
            processed[, outcome_idx := private$outcome_to_index[as.character(get(private$outcome_id_col))]]
            processed[, (private$outcome_id_col) := NULL]
            
            # Map unit ids to unit indices and drop original unit id column
            processed[, unit_idx := private$unit_to_index[as.character(get(private$unit_id_col))]]
            processed[, (private$unit_id_col) := NULL]

            # Sort by cohort_id, unit_idx, then outcome index (explicit column names)
            setorderv(processed, c("cohort_id", "unit_idx", "outcome_idx"))

            # Ensure column order: unit_idx, cohort_id, outcome_idx, outcome value
            setcolorder(processed, c("unit_idx", "cohort_id", "outcome_idx", private$outcome_value_col))

            private$processed_panel <- processed

            invisible(self)
        }
    ),
    private = list(
        original_panel = NULL,
        unit_id_col = NULL,
        outcome_id_col = NULL,
        outcome_value_col = NULL,
        model_rank = NULL,
        min_cohort_size = 0L,

        unit_ids = NULL,
        unit_to_index = NULL,
        outcome_ids = NULL,
        outcome_to_index = NULL,
        observed_outcome_indices = NULL,
        unit_cohorts = NULL,
        processed_panel = NULL
    )
)

# ------------------------------------------------------------------------------
