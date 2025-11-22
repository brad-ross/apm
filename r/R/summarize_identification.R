#' Summarize identification for one or many panels
#'
#' This computes summary diagnostics of factor identification using the
#' O^3 super-cohort merging process for either a single panel or many panels.
#'
#' - If `observed_outcome_indices` is a list of integer vectors and
#'   `cohort_sizes` is an integer vector, dispatches to the singleton C++ binding.
#' - If `observed_outcome_indices` is a list of lists and `cohort_sizes` is a
#'   list of integer vectors, dispatches to the vectorized C++ binding.
#'
#' @param observed_outcome_indices Either a list of integer vectors (single panel)
#'   or a list of such lists (many panels). Indices are 1-based.
#' @param cohort_sizes Either an integer vector (single panel) or a list of
#'   integer vectors (many panels).
#' @param r The maximum model rank.
#' @param iter Iteration to summarize. Use -1 for the final super cohort(s)
#'   (default), -2 for second-to-last, etc. Positive values are 1-based and
#'   clamp to the last available iteration.
#' @param outcome_weights Optional numeric vector (single panel) or list of
#'   numeric vectors (many panels) giving per-outcome weights. Defaults to all
#'   ones.
#' @return Either a single named list or a list of named lists, each with:
#'   - `largest_super_cohort_size`
#'   - `largest_super_cohort_share`
#'   - `min_cohort_size_in_largest_super`
#'   - `num_outcomes_in_largest_super_cohort`
#'   - `total_outcome_weight_in_largest_super_cohort`
#'   - `share_outcomes_in_largest_super_cohort`
#'   - `share_outcome_weight_in_largest_super_cohort`
#'   - `num_o3_iterations`
#' @examples
#' summarize_identification(list(c(1,3), c(2,3)), c(50, 40), r = 2)
#' summarize_identification(list(list(c(1), c(2,3))), list(c(10, 25)), r = 2)
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