#' Summarize identification for one set of cohorts
#'
#' This computes summary diagnostics of factor identification using the
#' O^3 super-cohort merging process for a single panel.
#'
#' @param observed_outcome_indices A list where each element is an integer
#'   vector of 1-based indices indicating observed outcomes for a cohort.
#' @param cohort_sizes An integer vector giving the number of units in each
#'   cohort (same length as `observed_outcome_indices`).
#' @param r The maximum model rank.
#' @return A named list with elements:
#'   - `largest_super_cohort_size`
#'   - `largest_super_cohort_share`
#'   - `min_cohort_size_in_largest_super`
#'   - `num_o3_iterations`
#' @examples
#' summarize_identification(list(c(1,3), c(2,3)), c(50, 40), r = 2)
#' @export
summarize_identification <- function(observed_outcome_indices, cohort_sizes, r) {
  stopifnot(is.list(observed_outcome_indices))
  stopifnot(length(observed_outcome_indices) == length(cohort_sizes))
  summarize_identification_cpp(observed_outcome_indices, as.integer(cohort_sizes), as.integer(r))
}

#' Summarize identification for multiple panels
#'
#' Vectorized version of `summarize_identification` for many panels at once.
#'
#' @param observed_outcome_indices_list A list of `observed_outcome_indices` lists.
#' @param cohort_sizes_list A list of integer vectors of cohort sizes.
#' @param r The maximum model rank.
#' @return A list of named lists as returned by `summarize_identification`.
#' @examples
#' summarize_identification_many(list(list(c(1), c(2,3))), list(c(10, 25)), r = 2)
#' @export
summarize_identification_many <- function(observed_outcome_indices_list, cohort_sizes_list, r) {
  stopifnot(is.list(observed_outcome_indices_list))
  stopifnot(is.list(cohort_sizes_list))
  stopifnot(length(observed_outcome_indices_list) == length(cohort_sizes_list))
  summarize_identification_many_cpp(observed_outcome_indices_list, lapply(cohort_sizes_list, as.integer), as.integer(r))
}