#' Summarize Factor Identification via O^3 Algorithm
#'
#' Computes summary diagnostics of factor model identification using the
#' O^3 (Observed Outcome Overlap) super-cohort merging algorithm.
#'
#' @description
#' This function analyzes whether factors are identified in an unbalanced panel
#' by examining the overlap structure of observed outcomes across cohorts. It
#' uses the O^3 algorithm to iteratively merge cohorts that share sufficient
#' outcome overlap, then reports summary statistics about the largest resulting
#' super cohort.
#'
#' @details
#' **The O^3 Algorithm:**
#'
#' The O^3 algorithm iteratively merges cohorts based on outcome overlap:
#' \enumerate{
#'   \item Start with each cohort as its own super cohort.
#'   \item Merge any two super cohorts that share at least r outcomes.
#'   \item Repeat until no more merges are possible.
#' }
#'
#' Factors are identified if and only if all cohorts end up in a single super
#' cohort at the final iteration.
#'
#' **Summary Statistics:**
#'
#' The function reports on the largest super cohort at the specified iteration,
#' including its size (in units), outcome coverage, and the number of algorithm
#' iterations.
#'
#' **Vectorized Interface:**
#'
#' For efficiency, you can analyze multiple panels simultaneously by passing
#' lists of lists for `observed_outcome_indices` and lists of vectors for
#' `cohort_sizes`.
#'
#' @param observed_outcome_indices For a single panel: a list of integer vectors
#'   where element c contains 1-based outcome indices for cohort c. For multiple
#'   panels: a list of such lists.
#' @param cohort_sizes For a single panel: an integer vector of cohort sizes.
#'   For multiple panels: a list of such vectors.
#' @param r Integer; the model rank (minimum overlap threshold for merging).
#' @param iter Integer; iteration to summarize. Use -1 (default) for the final
#'   iteration, -2 for second-to-last, etc. Positive values are 1-based and
#'   clamped to available iterations.
#' @param outcome_weights Optional numeric vector (single panel) or list of
#'   vectors (multiple panels) of per-outcome weights. Defaults to 1 for all.
#'
#' @return For a single panel: a named list with summary statistics.
#'
#'   For multiple panels: a list of such named lists.
#'
#'   Each summary contains:
#'   \describe{
#'     \item{largest_super_cohort_size}{Total units in the largest super cohort.}
#'     \item{largest_super_cohort_share}{Share of panel units in largest super cohort.}
#'     \item{min_cohort_size_in_largest_super}{Minimum cohort size within the
#'           largest super cohort.}
#'     \item{num_outcomes_in_largest_super_cohort}{Unique outcomes observed by
#'           cohorts in the largest super cohort.}
#'     \item{total_outcome_weight_in_largest_super_cohort}{Sum of outcome weights.}
#'     \item{share_outcomes_in_largest_super_cohort}{Fraction of outcomes covered.}
#'     \item{share_outcome_weight_in_largest_super_cohort}{Fraction of total weight.}
#'     \item{num_o3_iterations}{Number of O^3 iterations performed.}
#'   }
#'
#' @seealso \code{\link{o3_algorithm}} for the underlying algorithm.
#' @seealso \code{\link{aligned_factors_identified}} for a simple yes/no check.
#' @seealso \code{\link{get_largest_super_cohort}} for extracting cohort indices.
#'
#' @examples
#' # Single panel: 2 cohorts, cohort 1 sees outcomes 1,3; cohort 2 sees 2,3
#' summarize_identification(list(c(1L, 3L), c(2L, 3L)), c(50L, 40L), r = 2L)
#'
#' # Multiple panels
#' summarize_identification(
#'   list(list(c(1L), c(2L, 3L))),
#'   list(c(10L, 25L)),
#'   r = 2L
#' )
#'
#' @export
summarize_identification <- function(observed_outcome_indices, cohort_sizes, r, iter = -1L, outcome_weights = NULL) {
  if (is.list(observed_outcome_indices) && length(observed_outcome_indices) > 0 &&
      is.list(observed_outcome_indices[[1]]) && is.list(cohort_sizes)) {
    # Many panels
    stopifnot(length(observed_outcome_indices) == length(cohort_sizes))
    if (!is.null(outcome_weights)) {
      stopifnot(is.list(outcome_weights))
      stopifnot(length(outcome_weights) == length(observed_outcome_indices))
      outcome_weights <- lapply(outcome_weights, as.numeric)
    }
    return(summarize_identification_many_cpp(
      observed_outcome_indices,
      lapply(cohort_sizes, as.integer),
      as.integer(r),
      as.integer(iter),
      outcome_weights
    ))
  }

  # Single panel
  stopifnot(is.list(observed_outcome_indices))
  stopifnot(length(observed_outcome_indices) == length(cohort_sizes))
  if (!is.null(outcome_weights)) {
    outcome_weights <- as.numeric(outcome_weights)
  }
  return(summarize_identification_cpp(
    observed_outcome_indices,
    as.integer(cohort_sizes),
    as.integer(r),
    as.integer(iter),
    outcome_weights
  ))
}