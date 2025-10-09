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
#' @param outcome_value_col Column name for the outcome value; rows with missing
#'   values in this column are dropped prior to cohort construction
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
                                         outcome_value_col,
                                         model_rank,
                                         min_cohort_size = 0,
                                         sort_cohorts_lexicographically = FALSE,
                                         cohort_observed_outcomes_as_df = TRUE,
                                         verbose = FALSE) {
    default_datatable_options <- list(
        datatable.verbose=getOption("datatable.verbose"), 
        datatable.showProgress=getOption("datatable.showProgress")
    )
    if (isTRUE(verbose)) {
        # TODO: figure out how to set datatable.verbose to TRUE without causing a lot of noise
        options(datatable.verbose=FALSE, datatable.showProgress=TRUE)
    } else {
        options(datatable.verbose=FALSE, datatable.showProgress=FALSE)
    }
    on.exit(options(
        datatable.verbose=default_datatable_options$datatable.verbose, 
        datatable.showProgress=default_datatable_options$datatable.showProgress
    ))

    panel_dt <- to_data_table(panel_df)

    # Validate required columns exist and drop rows with missing outcome values
    validate_required_panel_cols(panel_dt, unit_id_col, outcome_id_col, outcome_value_col)
    panel_dt <- panel_dt[!is.na(get(outcome_value_col))]

    # Build globally sorted unique outcomes and index map (atomic types)
    outcome_ids <- sort(unique(panel_dt[[outcome_id_col]]))
    outcome_to_index <- setNames(seq_along(outcome_ids), as.character(outcome_ids))

    # Keep only unit and outcome; map outcome to integer index (atomic)
    dt <- panel_dt[, .(
        unit_id = get(unit_id_col),
        outcome_id = get(outcome_id_col)
    )]
    dt[, outcome_idx := outcome_to_index[as.character(outcome_id)]]
    dt[, outcome_id := NULL]

    # Unique unit-outcome pairs and sort by (unit_id, outcome_idx)
    uo <- unique(dt, by = c("unit_id", "outcome_idx"))
    setkey(uo, unit_id, outcome_idx)
    # setorder(uo, unit_id, outcome_idx)

    # Per-unit canonical key: prefer fast C-level hashing via xxhashlite; fallback to zero-padded string
    if (requireNamespace("xxhashlite", quietly = TRUE)) {
        unit_keys <- uo[, .(
            outcome_count = .N,
            cohort_key = as.character(xxhashlite::xxhash(outcome_idx, algo = "xxh64"))
        ), by = unit_id]
    } else {
        # Determine width for zero-padding based on the maximum outcome index
        pad_width <- nchar(as.character(length(outcome_ids)))
        pad_fmt <- paste0("%0", pad_width, "d")
        unit_keys <- uo[, .(
            outcome_count = .N,
            cohort_key = paste(sprintf(pad_fmt, outcome_idx), collapse = ",")
        ), by = unit_id]
    }

    # Cohort map: units sharing key; cohort-level counts (atomic-only ops)
    coh_map <- unit_keys[, .(
        num_units = .N,
        outcome_count = outcome_count[1L]
    ), by = cohort_key]

    # Filter cohorts by requirements
    coh_map <- coh_map[outcome_count >= model_rank & num_units >= min_cohort_size]

    # Stable ordering and cohort_id assignment
    if (isTRUE(sort_cohorts_lexicographically)) {
        pad_width <- nchar(as.character(length(outcome_ids)))
        pad_fmt <- paste0("%0", pad_width, "d")
        uo_with_key <- uo[unit_keys, on = .(unit_id), nomatch = 0L][, .(cohort_key, outcome_idx)]
        order_key_dt <- unique(uo_with_key, by = c("cohort_key", "outcome_idx"))[
            , .(cohort_order_key = paste(sprintf(pad_fmt, sort(outcome_idx)), collapse = ",")), by = cohort_key
        ]
        coh_map <- order_key_dt[coh_map, on = .(cohort_key)]
        setkey(coh_map, cohort_order_key)
    } else {
        setkey(coh_map, cohort_key)
    }
    coh_map[, cohort_id := seq_len(.N)]

    # Assign cohort_id to units (inner join keeps only filtered cohorts)
    unit_cohorts <- unit_keys[coh_map[, .(cohort_key, cohort_id)], on = .(cohort_key), nomatch = 0L][
        , .(unit_id, cohort_id)
    ]
    setkey(unit_cohorts, unit_id)

    # Drop key columns from coh_map after cohort_id has been defined
    if (isTRUE(sort_cohorts_lexicographically)) {
        coh_map[, c("cohort_key", "cohort_order_key") := NULL]
    } else {
        coh_map[, cohort_key := NULL]
    }

    # Long-form cohort-outcome mapping via atomic joins; dedupe at cohort level
    cohort_outcomes <- unique(
        uo[unit_cohorts, on = .(unit_id), nomatch = 0L][, .(cohort_id, outcome_idx)],
        by = c("cohort_id", "outcome_idx")
    )
    setorder(cohort_outcomes, cohort_id, outcome_idx)
    cohort_outcomes[, outcome_name := as.character(outcome_ids[outcome_idx])]

    options(
        datatable.verbose=default_datatable_options$datatable.verbose, 
        datatable.showProgress=default_datatable_options$datatable.showProgress
    )

    # Prepare return values
    if (isTRUE(cohort_observed_outcomes_as_df)) {
        return(list(
            cohort_observed_outcomes_df = cohort_outcomes,
            unit_cohorts = unit_cohorts
        ))
    } else {
        # Reconstruct list-of-indices per cohort (only at the very end)
        split_list <- split(cohort_outcomes$outcome_idx, cohort_outcomes$cohort_id)
        cohort_order <- as.integer(names(split_list))
        observed_outcome_indices <- unname(split_list[order(cohort_order)])
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
        get_covar_cols = function() private$covar_cols,
        get_auxiliary_cols = function() private$auxiliary_cols,
        get_panel_holder_xptr = function() private$panel_holder_xptr,

        initialize = function(panel_df,
                              unit_id_col,
                              outcome_id_col,
                              outcome_value_col,
                              model_rank,
                              min_cohort_size = 0,
                              sort_cohorts_lexicographically = FALSE,
                              covar_cols = character(0),
                              auxiliary_cols = character(0),
                              verbose = FALSE) {
            default_datatable_options <- list(
                datatable.verbose=getOption("datatable.verbose"), 
                datatable.showProgress=getOption("datatable.showProgress")
            )
            if (isTRUE(verbose)) {
                # TODO: figure out how to set datatable.verbose to TRUE without causing a lot of noise
                options(datatable.verbose=FALSE, datatable.showProgress=TRUE)
            } else {
                options(datatable.verbose=FALSE, datatable.showProgress=FALSE)
            }
            on.exit(options(
                datatable.verbose=default_datatable_options$datatable.verbose, 
                datatable.showProgress=default_datatable_options$datatable.showProgress
            ))
            
            private$original_panel <- to_data_table(panel_df)
            private$unit_id_col <- unit_id_col
            private$outcome_id_col <- outcome_id_col
            private$outcome_value_col <- outcome_value_col
            private$model_rank <- as.integer(model_rank)
            private$min_cohort_size <- as.integer(min_cohort_size)
            validate_required_panel_cols(private$original_panel, 
                private$unit_id_col, private$outcome_id_col, private$outcome_value_col)
            
            # Normalize and validate covariate columns
            covar_cols <- as.character(covar_cols)
            covar_cols <- unique(covar_cols)
            covar_cols <- setdiff(covar_cols, c(private$unit_id_col, private$outcome_id_col, private$outcome_value_col))
            missing_covar <- setdiff(covar_cols, names(private$original_panel))
            if (length(missing_covar) > 0L) {
                stop(sprintf("UnbalancedPanel: covar_cols not found in original_panel: %s",
                             paste(missing_covar, collapse = ", ")))
            }
            private$covar_cols <- covar_cols
            
            # Normalize and validate auxiliary columns
            auxiliary_cols <- as.character(auxiliary_cols)
            auxiliary_cols <- unique(auxiliary_cols)
            auxiliary_cols <- setdiff(auxiliary_cols, c(private$unit_id_col, private$outcome_id_col, private$outcome_value_col, private$covar_cols))
            missing_aux <- setdiff(auxiliary_cols, names(private$original_panel))
            if (length(missing_aux) > 0L) {
                stop(sprintf("UnbalancedPanel: auxiliary_cols not found in original_panel: %s",
                             paste(missing_aux, collapse = ", ")))
            }
            private$auxiliary_cols <- auxiliary_cols
            
            # Build sorted unit ids and index map
            unit_ids <- sort(unique(private$original_panel[[private$unit_id_col]]))
            private$unit_ids <- unit_ids
            private$unit_to_index <- setNames(seq_along(unit_ids), as.character(unit_ids))

            # Compute cohorts and save artifacts
            coh <- construct_cohorts_from_panel(
                panel_df = private$original_panel,
                unit_id_col = private$unit_id_col,
                outcome_id_col = private$outcome_id_col,
                outcome_value_col = private$outcome_value_col,
                model_rank = private$model_rank,
                min_cohort_size = private$min_cohort_size,
                sort_cohorts_lexicographically = sort_cohorts_lexicographically,
                cohort_observed_outcomes_as_df = FALSE
            )
            private$outcome_ids <- coh$outcome_ids
            private$outcome_to_index <- coh$outcome_to_index
            private$observed_outcome_indices <- coh$observed_outcome_indices
            private$unit_cohorts <- coh$unit_cohorts
            
            # Augment unit_cohorts with unit_idx
            private$unit_cohorts[, unit_idx := private$unit_to_index[as.character(get(private$unit_id_col))]]

            # Select only relevant columns from the original panel
            keep_cols <- c(private$unit_id_col, private$outcome_id_col, private$outcome_value_col, private$covar_cols, private$auxiliary_cols)
            orig_panel_only_relevant_cols <- private$original_panel[
                , keep_cols
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

            # Ensure column order: unit_idx, cohort_id, outcome_idx, outcome value, then covariates and auxiliary
            setcolorder(processed, c("unit_idx", "cohort_id", "outcome_idx", private$outcome_value_col, private$covar_cols, private$auxiliary_cols))

            private$processed_panel <- processed

            # Build and store a C++ panel holder (pins columns and builds internal panel)
            private$panel_holder_xptr <- build_R_panel_holder_cpp(
                processed_panel = private$processed_panel,
                observed_outcome_indices = private$observed_outcome_indices,
                outcome_value_col = private$outcome_value_col,
                covar_cols = private$covar_cols,
                auxiliary_cols = private$auxiliary_cols,
                num_units_in = length(private$unit_ids)
            )

            options(
                datatable.verbose=default_datatable_options$datatable.verbose, 
                datatable.showProgress=default_datatable_options$datatable.showProgress
            )

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
        processed_panel = NULL,
        covar_cols = character(0),
        auxiliary_cols = character(0),
        panel_holder_xptr = NULL
    )
)

# ------------------------------------------------------------------------------
