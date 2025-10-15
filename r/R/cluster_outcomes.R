#' Outcome clustering utilities
#'
#' R wrappers for outcome clustering over `UnbalancedPanel`.
#'
#' @name outcome_clustering
NULL

#' Compute outcome clustering for a single k
#'
#' @param panel An `UnbalancedPanel` instance
#' @param grid_size Integer number of quantile grid points (> 0)
#' @param k Number of clusters (> 0)
#' @param n_inits Optional integer number of KMeans initializations
#' @param seed Optional numeric seed for reproducibility
#' @return Integer vector of length T with cluster ids in 1..k
#' @export
comp_outcome_clustering <- function(panel, grid_size, k, n_inits = NULL, seed = NULL) {
    stopifnot(inherits(panel, "UnbalancedPanel"))
    xp <- panel$get_panel_holder_xptr()
    res <- comp_outcome_clustering_cpp(
        panel_holder_xptr = xp,
        grid_size = as.integer(grid_size),
        k = as.integer(k),
        n_inits_in = if (is.null(n_inits)) NULL else as.integer(n_inits),
        seed_in = if (is.null(seed)) NULL else as.numeric(seed)
    )
    # Optionally set names to outcome ids if available
    outcome_ids <- try(panel$get_outcome_ids(), silent = TRUE)
    if (!inherits(outcome_ids, "try-error") && length(outcome_ids) == length(res)) {
        names(res) <- as.character(outcome_ids)
    }
    res
}

#' Compute outcome clusterings for a range of k
#'
#' @param panel An `UnbalancedPanel` instance
#' @param grid_size Integer number of quantile grid points (> 0)
#' @param min_k Minimum number of clusters (>= 1)
#' @param max_k Maximum number of clusters (>= min_k)
#' @param n_inits Optional integer number of initializations per k
#' @param seed Optional numeric seed for reproducibility
#' @return Integer matrix of dimension T x (#k), columns named by k
#' @export
comp_outcome_clusterings <- function(panel, grid_size, min_k, max_k, n_inits = NULL, seed = NULL) {
    stopifnot(inherits(panel, "UnbalancedPanel"))
    xp <- panel$get_panel_holder_xptr()
    res <- comp_outcome_clusterings_cpp(
        panel_holder_xptr = xp,
        grid_size = as.integer(grid_size),
        min_k = as.integer(min_k),
        max_k = as.integer(max_k),
        n_inits_in = if (is.null(n_inits)) NULL else as.integer(n_inits),
        seed_in = if (is.null(seed)) NULL else as.numeric(seed)
    )
    # Optionally set rownames to outcome ids if available
    outcome_ids <- try(panel$get_outcome_ids(), silent = TRUE)
    if (!inherits(outcome_ids, "try-error") && length(outcome_ids) == nrow(res)) {
        rownames(res) <- as.character(outcome_ids)
    }
    res
}

#' Combine outcomes into new cohorts
#'
#' Given a mapping from original outcomes (1..T) to new outcomes (1..T'),
#' return the resulting observed outcome indices per cohort and the new cohort sizes.
#'
#' @param panel An `UnbalancedPanel` instance
#' @param old_to_new_outcome Integer vector length T with values in 1..T'
#' @return List with `observed_outcome_indices` (list of integer vectors, 1-based)
#'   and `cohort_sizes` (integer vector)
#' @export
combine_outcomes_into_new_cohorts <- function(panel, old_to_new_outcome) {
    stopifnot(inherits(panel, "UnbalancedPanel"))
    xp <- panel$get_panel_holder_xptr()
    old_to_new_outcome <- as.integer(old_to_new_outcome)
    if (any(is.na(old_to_new_outcome))) stop("old_to_new_outcome contains NA")
    T <- length(panel$get_outcome_ids())
    if (length(old_to_new_outcome) != T) stop("old_to_new_outcome must have length equal to number of outcomes")
    comp <- get_new_cohorts_from_combining_outcomes_cpp(xp, old_to_new_outcome)
    comp
}