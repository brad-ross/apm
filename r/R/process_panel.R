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
#' Converts a list of observed outcome indices per cohort into a long-form
#' data.frame where each row corresponds to one (cohort, outcome) pair.
#'
#' @details
#' This is a utility function for converting the compact list-of-vectors
#' representation of cohort-outcome mappings into a tidy data.frame format
#' suitable for analysis and visualization.
#'
#' @param outcome_ids Character (or coercible to character) vector of outcome
#'   identifiers. The order defines the mapping from indices to names.
#' @param observed_outcome_indices List of integer vectors where element `c`
#'   contains the 1-based indices of outcomes observed by cohort `c`.
#'
#' @return A data.frame with columns:
#'   \describe{
#'     \item{cohort_id}{Integer; 1-based cohort identifier.}
#'     \item{outcome_idx}{Integer; 1-based outcome index (position in `outcome_ids`).}
#'     \item{outcome_name}{Character; the outcome identifier from `outcome_ids`.}
#'   }
#'
#' @examples
#' outcome_ids <- c("2020-01", "2020-02", "2020-03", "2020-04")
#' obs_indices <- list(
#'   c(1L, 2L, 3L),      # Cohort 1 observes outcomes 1, 2, 3
#'   c(2L, 3L, 4L),      # Cohort 2 observes outcomes 2, 3, 4
#'   c(1L, 2L, 3L, 4L)   # Cohort 3 observes all outcomes
#' )
#' df <- construct_cohort_observed_outcomes_df(outcome_ids, obs_indices)
#' print(df)
#'
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
#' @details
#' A **cohort** is defined as a group of units that share exactly the same set
#' of observed outcomes. This function:
#' \enumerate{
#'   \item Drops rows with missing outcome values.
#'   \item Assigns a global index to each unique outcome (sorted alphabetically/numerically).
#'   \item Groups units by the set of outcomes they observe.
#'   \item Filters cohorts to keep only those with at least `model_rank` outcomes
#'         and at least `min_cohort_size` units.
#'   \item Optionally subsets to the largest "super cohort" using the O^3 algorithm
#'         to ensure factor identification.
#' }
#'
#' The O^3 (Observed Outcome Overlap) algorithm iteratively merges cohorts whose
#' observed outcomes overlap by at least `model_rank` outcomes, ensuring that
#' factors can be identified across the panel.
#'
#' @param panel_df A data.frame, data.table, tibble, or Arrow table containing
#'   panel data in long format (one row per unit-outcome observation).
#' @param unit_id_col Character string; column name for the unit identifier.
#' @param outcome_id_col Character string; column name for the outcome identifier
#'   (e.g., time period, product, location).
#' @param outcome_value_col Character string; column name for the outcome value.
#'   Rows with missing values in this column are dropped prior to cohort construction.
#' @param model_rank Integer; the factor model rank. Cohorts must have at least
#'   this many observed outcomes to be retained (default: 1).
#' @param min_cohort_size Integer; minimum number of units per cohort to keep
#'   (default: 0, meaning no minimum).
#' @param subset_to_largest_super_cohort Logical; if `TRUE` (default), subset the
#'   panel to the largest super cohort at the final iteration of the O^3 algorithm.
#'   This ensures factor identification. If `FALSE`, subsequent estimation may fail
#'   if factors are not identified.
#' @param sort_cohorts_lexicographically Logical; if `TRUE`, sort cohorts
#'   lexicographically by their outcome indices. If `FALSE` (default), cohorts are
#'   ordered by their hash key (faster but less deterministic across runs).
#' @param cohort_observed_outcomes_as_df Logical; if `TRUE` (default), include a
#'   `cohort_observed_outcomes_df` data.frame in the output with columns `cohort_id`,
#'   `outcome_idx`, and `outcome_name`.
#' @param verbose Logical; if `TRUE`, print progress during construction
#'   (default: `FALSE`).
#'
#' @return A named list with the following elements:
#'   \describe{
#'     \item{outcome_ids}{Sorted unique values from `outcome_id_col` (vector).}
#'     \item{outcome_to_index}{Named integer vector mapping each outcome value to
#'           its 1-based index.}
#'     \item{observed_outcome_indices}{List of integer vectors; element `c` contains
#'           the 1-based outcome indices observed by cohort `c`.}
#'     \item{unit_cohorts}{data.table with columns `unit_id` and `cohort_id` mapping
#'           each unit to its cohort.}
#'     \item{cohort_sizes}{Integer vector; number of units in each cohort, ordered
#'           by `cohort_id`.}
#'     \item{cohort_observed_outcomes_df}{(When `cohort_observed_outcomes_as_df = TRUE`)
#'           A data.table with columns `cohort_id`, `outcome_idx`, `outcome_name`
#'           providing a long-form mapping of cohorts to their observed outcomes.}
#'   }
#'
#' @seealso \code{\link{UnbalancedPanel}} for an R6 class that wraps this function
#'   and provides additional functionality for estimation.
#' @seealso \code{\link{o3_algorithm}} for details on the Observed Outcome Overlap
#'   algorithm used for identification.
#' @seealso \code{\link{get_largest_super_cohort}} for extracting the largest
#'   super cohort.
#'
#' @examples
#' # Create synthetic panel data
#' set.seed(123)
#' panel <- data.frame(
#'   unit = rep(1:100, each = 5),
#'   time = rep(1:5, 100),
#'   outcome = rnorm(500)
#' )
#' # Remove some observations to create unbalanced structure
#' panel <- panel[sample(nrow(panel), 400), ]
#'
#' # Construct cohorts
#' result <- construct_cohorts_from_panel(
#'   panel_df = panel,
#'   unit_id_col = "unit",
#'   outcome_id_col = "time",
#'   outcome_value_col = "outcome",
#'   model_rank = 2
#' )
#'
#' # Examine cohort structure
#' print(result$cohort_sizes)
#' print(result$observed_outcome_indices)
#'
#' @importFrom data.table setorder
#' @export
construct_cohorts_from_panel <- function(panel_df,
                                         unit_id_col,
                                         outcome_id_col,
                                         outcome_value_col,
                                         model_rank = 1,
                                         min_cohort_size = 0,
                                         subset_to_largest_super_cohort = TRUE,
                                         sort_cohorts_lexicographically = FALSE,
                                         cohort_observed_outcomes_as_df = TRUE,
                                         verbose = FALSE) {
    full_panel_cohorts <- construct_cohorts_from_panel_core(
        panel_df, 
        unit_id_col, 
        outcome_id_col, 
        outcome_value_col, 
        model_rank, 
        min_cohort_size, 
        sort_cohorts_lexicographically, 
        cohort_observed_outcomes_as_df, verbose)

    if (isTRUE(subset_to_largest_super_cohort)) {
        largest_super_cohort <- get_largest_super_cohort(
            full_panel_cohorts$observed_outcome_indices, 
            full_panel_cohorts$cohort_sizes, 
            model_rank)
        uc <- to_data_table(full_panel_cohorts$unit_cohorts)
        data.table::setindexv(uc, "cohort_id")
        unit_cohorts_in_largest_super_cohort <- uc[
            .(largest_super_cohort), on = "cohort_id", nomatch = 0L, .(unit_id)
        ]
        data.table::setnames(unit_cohorts_in_largest_super_cohort, "unit_id", unit_id_col)
        panel_df <- to_data_table(panel_df)[
            unit_cohorts_in_largest_super_cohort, on = unit_id_col, nomatch = 0L
        ]
        return(construct_cohorts_from_panel_core(
            panel_df, unit_id_col, outcome_id_col, outcome_value_col, model_rank, min_cohort_size, sort_cohorts_lexicographically, cohort_observed_outcomes_as_df, verbose))
    }

    full_panel_cohorts
}

construct_cohorts_from_panel_core <- function(panel_df,
                                         unit_id_col,
                                         outcome_id_col,
                                         outcome_value_col,
                                         model_rank = 1,
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

    # Compute cohort sizes (number of units per cohort) in cohort_id order
    cohort_sizes <- unit_cohorts[, .N, by = cohort_id][order(cohort_id)][["N"]]
    cohort_sizes <- as.integer(cohort_sizes)

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

    # Reconstruct list-of-indices per cohort (only at the very end)
    split_list <- split(cohort_outcomes$outcome_idx, cohort_outcomes$cohort_id)
    cohort_order <- as.integer(names(split_list))
    observed_outcome_indices <- unname(split_list[order(cohort_order)])

    # Prepare return values
    if (isTRUE(cohort_observed_outcomes_as_df)) {
        return(list(
            outcome_ids = outcome_ids,
            outcome_to_index = outcome_to_index,
            observed_outcome_indices = observed_outcome_indices,
            cohort_observed_outcomes_df = cohort_outcomes,
            unit_cohorts = unit_cohorts,
            cohort_sizes = cohort_sizes
        ))
    } else {
        return(list(
            outcome_ids = outcome_ids,
            outcome_to_index = outcome_to_index,
            observed_outcome_indices = observed_outcome_indices,
            unit_cohorts = unit_cohorts,
            cohort_sizes = cohort_sizes
        ))
    }
}

# ------------------------------------------------------------------------------

#' Unbalanced Panel Data Container
#'
#' An R6 class that holds unbalanced panel data and provides efficient access
#' to cohort structure, outcome indexing, and processed data for estimation.
#' This is the primary data structure used throughout the apm package.
#'
#' @description
#' `UnbalancedPanel` encapsulates panel data where different units may observe
#' different subsets of outcomes. It automatically (based on `construct_cohorts_from_panel`):
#' \itemize{
#'   \item Constructs cohorts (groups of units with identical outcome patterns)
#'   \item Indexes outcomes and units for efficient lookup
#'   \item Processes the panel into a format suitable for C++ estimation routines
#'   \item Optionally subsets to the largest identifiable super cohort
#' }
#'
#' @details
#' The panel data should be in long format with one row per (unit, outcome)

#' observation. A **cohort** is defined as a group of units that observe exactly
#' the same set of outcomes.
#'
#' **Covariates vs. Auxiliary Columns:**
#' \itemize{
#'   \item `covar_cols`: Columns used in factor model estimation (e.g., for
#'         covariate-adjusted models). These affect the estimated parameters.
#'   \item `auxiliary_cols`: Additional data columns tracked alongside the panel
#'         but not used in estimation. Useful for post-estimation analysis.
#' }
#'
#' @section Constructor:
#' \preformatted{
#' UnbalancedPanel$new(
#'   panel_df,
#'   unit_id_col,
#'   outcome_id_col,
#'   outcome_value_col,
#'   model_rank = 1,
#'   min_cohort_size = 0,
#'   subset_to_largest_super_cohort = TRUE,
#'   sort_cohorts_lexicographically = FALSE,
#'   covar_cols = character(0),
#'   auxiliary_cols = character(0),
#'   verbose = FALSE
#' )
#' }
#'
#' @section Constructor Arguments:
#' \describe{
#'   \item{panel_df}{A data.frame, data.table, tibble, or Arrow table in long
#'         format (one row per unit-outcome observation).
#'   }
#'   \item{unit_id_col}{Character; column name for the unit identifier.}
#'   \item{outcome_id_col}{Character; column name for the outcome identifier
#'         (e.g., time period).}
#'   \item{outcome_value_col}{Character; column name for the outcome value.}
#'   \item{model_rank}{Integer; factor model rank. Cohorts with fewer than this
#'         many outcomes are dropped (default: 1).}
#'   \item{min_cohort_size}{Integer; minimum units per cohort (default: 0).}
#'   \item{subset_to_largest_super_cohort}{Logical; if TRUE (default), subset to
#'         the largest super cohort for identification. See \code{\link{o3_algorithm}}.}
#'   \item{sort_cohorts_lexicographically}{Logical; if TRUE, sort cohorts by
#'         outcome indices (default: FALSE).}
#'   \item{covar_cols}{Character vector; column names of covariates to include
#'         in estimation.}
#'   \item{auxiliary_cols}{Character vector; column names of auxiliary data to
#'         track (not used in estimation).}
#'   \item{verbose}{Logical; print progress messages (default: FALSE).}
#' }
#'
#' @section Public Methods:
#' \describe{
#'   \item{\code{get_original_panel()}}{Returns the original panel data.table.}
#'   \item{\code{get_unit_id_col()}}{Returns the unit ID column name.}
#'   \item{\code{get_outcome_id_col()}}{Returns the outcome ID column name.}
#'   \item{\code{get_outcome_value_col()}}{Returns the outcome value column name.}
#'   \item{\code{get_model_rank()}}{Returns the model rank.}
#'   \item{\code{get_min_cohort_size()}}{Returns the minimum cohort size.}
#'   \item{\code{get_unit_ids()}}{Returns sorted vector of unit IDs.}
#'   \item{\code{get_num_units()}}{Returns number of units (N).}
#'   \item{\code{get_outcome_ids()}}{Returns sorted vector of outcome IDs.}
#'   \item{\code{get_num_outcomes()}}{Returns number of outcomes (T).}
#'   \item{\code{get_outcome_to_index()}}{Returns named vector mapping outcome
#'         IDs to 1-based indices.}
#'   \item{\code{get_observed_outcome_indices()}}{Returns list of integer vectors;
#'         element c contains 1-based outcome indices for cohort c.}
#'   \item{\code{get_unit_cohorts()}}{Returns data.table mapping units to cohorts.}
#'   \item{\code{get_num_cohorts()}}{Returns number of cohorts (C).}
#'   \item{\code{get_cohort_sizes()}}{Returns integer vector of cohort sizes.}
#'   \item{\code{get_processed_panel()}}{Returns processed data.table used
#'         internally for estimation.}
#'   \item{\code{get_covar_cols()}}{Returns character vector of covariate column
#'         names.}
#'   \item{\code{get_auxiliary_cols()}}{Returns character vector of auxiliary
#'         column names.}
#'   \item{\code{get_panel_holder_xptr()}}{Returns external pointer to C++ panel
#'         holder (for internal use).}
#' }
#'
#' @seealso \code{\link{construct_cohorts_from_panel}} for the underlying cohort
#'   construction logic.
#' @seealso \code{\link{est_cohort_specific_params}} for estimating cohort-specific
#'   parameters from an UnbalancedPanel.
#' @seealso \code{\link{est_target_param_components}} for end-to-end estimation.
#'
#' @examples
#' # Create synthetic panel data
#' set.seed(42)
#' n_units <- 50
#' n_times <- 6
#' panel <- data.frame(
#'   unit = rep(1:n_units, each = n_times),
#'   time = rep(1:n_times, n_units),
#'   y = rnorm(n_units * n_times),
#'   x = rnorm(n_units * n_times)
#' )
#' # Create unbalanced structure by removing some observations
#' panel <- panel[sample(nrow(panel), 250), ]
#'
#' # Create UnbalancedPanel
#' up <- UnbalancedPanel$new(
#'   panel_df = panel,
#'   unit_id_col = "unit",
#'   outcome_id_col = "time",
#'   outcome_value_col = "y",
#'   model_rank = 2,
#'   covar_cols = "x"
#' )
#'
#' # Access panel properties
#' up$get_num_units()
#' up$get_num_cohorts()
#' up$get_cohort_sizes()
#' up$get_observed_outcome_indices()
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
        get_num_units = function() length(private$unit_ids),
        get_outcome_ids = function() private$outcome_ids,
        get_num_outcomes = function() length(private$outcome_ids),
        get_outcome_to_index = function() private$outcome_to_index,
        get_observed_outcome_indices = function() private$observed_outcome_indices,
        get_unit_cohorts = function() private$unit_cohorts,
        get_num_cohorts = function() length(private$cohort_sizes),
        get_cohort_sizes = function() private$cohort_sizes,
        get_processed_panel = function() private$processed_panel,
        get_covar_cols = function() private$covar_cols,
        get_auxiliary_cols = function() private$auxiliary_cols,
        get_panel_holder_xptr = function() private$panel_holder_xptr,

        initialize = function(panel_df,
                              unit_id_col,
                              outcome_id_col,
                              outcome_value_col,
                              model_rank = 1,
                              min_cohort_size = 0,
                              subset_to_largest_super_cohort = TRUE,
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
            
            # Compute cohorts and save artifacts
            coh <- construct_cohorts_from_panel(
                panel_df = private$original_panel,
                unit_id_col = private$unit_id_col,
                outcome_id_col = private$outcome_id_col,
                outcome_value_col = private$outcome_value_col,
                model_rank = private$model_rank,
                subset_to_largest_super_cohort = subset_to_largest_super_cohort,
                min_cohort_size = private$min_cohort_size,
                sort_cohorts_lexicographically = sort_cohorts_lexicographically,
                cohort_observed_outcomes_as_df = FALSE
            )
            
            private$outcome_ids <- coh$outcome_ids
            private$outcome_to_index <- coh$outcome_to_index
            private$observed_outcome_indices <- coh$observed_outcome_indices
            private$unit_cohorts <- coh$unit_cohorts
            private$cohort_sizes <- coh$cohort_sizes

            # Build sorted unit ids and index map
            unit_ids <- sort(unique(private$unit_cohorts$unit_id))
            private$unit_ids <- unit_ids
            private$unit_to_index <- setNames(seq_along(unit_ids), as.character(unit_ids))

            # Augment unit_cohorts with unit_idx
            private$unit_cohorts[, unit_idx := private$unit_to_index[as.character(unit_id)]]
            
            # Select only relevant columns from the original panel
            keep_cols <- c(private$unit_id_col, private$outcome_id_col, private$outcome_value_col, private$covar_cols, private$auxiliary_cols)
            orig_panel_only_relevant_cols <- private$original_panel[
                , keep_cols
                , with = FALSE
            ]

            # Inner join on unit id to attach cohort_id to each observation
            processed <- private$unit_cohorts[orig_panel_only_relevant_cols, on = c("unit_id" = private$unit_id_col), nomatch = 0L]

            # Map outcome ids to outcome indices and drop original outcome id column
            processed[, outcome_idx := private$outcome_to_index[as.character(get(private$outcome_id_col))]]
            processed[, (private$outcome_id_col) := NULL]
            processed[, unit_id := NULL]

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
        cohort_sizes = NULL,
        processed_panel = NULL,
        covar_cols = character(0),
        auxiliary_cols = character(0),
        panel_holder_xptr = NULL
    )
)

# ------------------------------------------------------------------------------
