#' Outcome Clustering Utilities
#'
#' Functions for clustering outcomes based on their distributional characteristics
#' across units. Useful for grouping similar outcomes (e.g., time periods) before
#' analysis.
#'
#' @name outcome_clustering
NULL

#' Compute Outcome Clustering for a Single k
#'
#' Clusters outcomes into k groups based on their CDFs across
#' units, using k-means clustering on CDF features.
#'
#' @description
#' This function represents each outcome by a vector of CDF values computed
#' across all units that observe it, then applies k-means clustering to group
#' outcomes with similar distributions.
#'
#' @details
#' **Algorithm:**
#' \enumerate{
#'   \item For each outcome t, compute `grid_size` equally-spaced quantiles of
#'         the outcome values across all observing units.
#'   \item Run k-means clustering on the resulting T x grid_size feature matrix.
#'   \item Return the cluster assignment for each outcome.
#' }
#'
#' **Reproducibility:**
#' For deterministic results, set `num_threads = 1` and provide a fixed `seed`.
#' Parallel runs may produce minor differences.
#'
#' @param panel An \code{\link{UnbalancedPanel}} R6 object.
#' @param grid_size Integer; number of quantile grid points (> 0). More points
#'   provide finer distributional representation.
#' @param k Integer; number of clusters (> 0).
#' @param n_inits Optional integer; number of k-means initializations. More
#'   initializations improve chances of finding global optimum.
#' @param seed Optional numeric; random seed for reproducibility.
#' @param num_threads Optional integer; number of threads (default: library default).
#'
#' @return Integer vector of length T (number of outcomes) with cluster IDs in
#'   1..k. Names are set to outcome IDs if available.
#'
#' @seealso \code{\link{comp_outcome_clusterings}} for clustering with multiple k.
#' @seealso \code{\link{combine_outcomes_into_new_cohorts}} for using clustering
#'   results.
#'
#' @examples
#' \dontrun{
#' # Cluster outcomes into 3 groups
#' clusters <- comp_outcome_clustering(panel, grid_size = 20, k = 3)
#' table(clusters)  # Cluster sizes
#' }
#'
#' @export
comp_outcome_clustering <- function(panel, grid_size, k, n_inits = NULL, seed = NULL, num_threads = NULL) {
    stopifnot(inherits(panel, "UnbalancedPanel"))
    xp <- panel$get_panel_holder_xptr()
    res <- comp_outcome_clustering_cpp(
        panel_holder_xptr = xp,
        grid_size = as.integer(grid_size),
        k = as.integer(k),
        n_inits_in = if (is.null(n_inits)) NULL else as.integer(n_inits),
        seed_in = if (is.null(seed)) NULL else as.numeric(seed),
        num_threads_in = if (is.null(num_threads)) NULL else as.integer(num_threads)
    )
    # Optionally set names to outcome ids if available
    outcome_ids <- try(panel$get_outcome_ids(), silent = TRUE)
    if (!inherits(outcome_ids, "try-error") && length(outcome_ids) == length(res)) {
        names(res) <- as.character(outcome_ids)
    }
    res
}

#' Compute Outcome Clusterings for a Range of k Values
#'
#' Runs outcome clustering for multiple values of k, enabling model selection
#' (e.g., via elbow method or silhouette analysis).
#'
#' @description
#' This is a vectorized version of \code{\link{comp_outcome_clustering}} that
#' computes clusterings for k = min_k, min_k+1, ..., max_k in a single call.
#'
#' @details
#' See \code{\link{comp_outcome_clustering}} for algorithm details.
#'
#' **Reproducibility:**
#' For deterministic results, set `num_threads = 1` and provide a fixed `seed`.
#'
#' @param panel An \code{\link{UnbalancedPanel}} R6 object.
#' @param grid_size Integer; number of quantile grid points (> 0).
#' @param min_k Integer; minimum number of clusters (>= 1).
#' @param max_k Integer; maximum number of clusters (>= min_k).
#' @param n_inits Optional integer; number of k-means initializations per k
#'   (default: 10).
#' @param seed Optional numeric; random seed for reproducibility.
#' @param num_threads Optional integer; number of threads.
#'
#' @return Integer matrix of dimension T x (max_k - min_k + 1). Row t contains
#'   cluster assignments for outcome t across different k values. Columns are
#'   named by k. Row names are set to outcome IDs if available.
#'
#' @seealso \code{\link{comp_outcome_clustering}} for single k.
#'
#' @examples
#' \dontrun{
#' # Compare clusterings for k = 2, 3, 4, 5
#' results <- comp_outcome_clusterings(panel, grid_size = 20, min_k = 2, max_k = 5)
#' colnames(results)  # "2", "3", "4", "5"
#' }
#'
#' @export
comp_outcome_clusterings <- function(panel, grid_size, min_k, max_k, n_inits = NULL, seed = NULL, num_threads = NULL) {
    stopifnot(inherits(panel, "UnbalancedPanel"))
    xp <- panel$get_panel_holder_xptr()
    res <- comp_outcome_clusterings_cpp(
        panel_holder_xptr = xp,
        grid_size = as.integer(grid_size),
        min_k = as.integer(min_k),
        max_k = as.integer(max_k),
        n_inits_in = if (is.null(n_inits)) NULL else as.integer(n_inits),
        seed_in = if (is.null(seed)) NULL else as.numeric(seed),
        num_threads_in = if (is.null(num_threads)) NULL else as.integer(num_threads)
    )
    # Optionally set rownames to outcome ids if available
    outcome_ids <- try(panel$get_outcome_ids(), silent = TRUE)
    if (!inherits(outcome_ids, "try-error") && length(outcome_ids) == nrow(res)) {
        rownames(res) <- as.character(outcome_ids)
    }
    res
}

#' Combine Outcomes into New Cohorts
#'
#' Given a mapping from original outcomes to new (aggregated) outcomes, computes
#' the resulting cohort structure after outcome combination.
#'
#' @description
#' This function is useful after clustering outcomes: it takes a cluster
#' assignment for each outcome and computes what the cohort structure would
#' look like if outcomes within each cluster were treated as a single outcome.
#'
#' @details
#' After outcome combination:
#' \itemize{
#'   \item A unit observing any outcome in cluster k is considered to observe
#'         the new combined outcome k.
#'   \item Cohorts are redefined based on which combined outcomes each unit observes.
#'   \item The number of cohorts may decrease (units that differed only in which
#'         specific outcomes they observed within a cluster now belong to the
#'         same cohort).
#' }
#'
#' @param panel An \code{\link{UnbalancedPanel}} R6 object.
#' @param old_to_new_outcome Integer vector of length T mapping each original
#'   outcome (1..T) to a new outcome index (1..T'). Typically obtained from
#'   \code{\link{comp_outcome_clustering}}.
#'
#' @return A named list with:
#'   \describe{
#'     \item{observed_outcome_indices}{List of integer vectors; element c contains
#'           1-based new outcome indices for the new cohort c.}
#'     \item{cohort_sizes}{Integer vector of new cohort sizes.}
#'   }
#'
#' @seealso \code{\link{comp_outcome_clustering}} for generating outcome mappings.
#'
#' @examples
#' \dontrun{
#' # Cluster outcomes
#' clusters <- comp_outcome_clustering(panel, grid_size = 20, k = 3)
#'
#' # See resulting cohort structure
#' new_cohorts <- combine_outcomes_into_new_cohorts(panel, clusters)
#' length(new_cohorts$cohort_sizes)  # Number of new cohorts
#' }
#'
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