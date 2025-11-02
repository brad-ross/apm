#' Error metrics for masked cohort-outcome means
#'
#' Computes bias, standard error, and RMSE for masked cohort–outcome mean estimators
#' using bootstrap draws, given the components returned by
#' `est_target_param_components`.
#'
#' @param components The list returned by `est_target_param_components`.
#'
#' @return A data.frame with columns:
#'  `cohort` (1-based), `outcome` (1-based), `spec`, `bias`, `se`, `rmse`, `cohort_pop_share`.
#' @export
est_masked_outcome_mean_err_metrics <- function(components) {
  # Assume the R-level wrapper output of est_target_param_components()
  stopifnot(is.list(components))
  if (is.null(components$outcome_means)) {
    stop("components$outcome_means is required")
  }

  # Rebuild a C++-compatible list shape with XPtrs where applicable
  ome_xptr_by_spec <- lapply(components$outcome_means, function(ome) {
    stopifnot(inherits(ome, "OutcomeMeansEstimates"))
    ome$.__enclos_env__$private$xp
  })

  # Optional fields
  stats_xptr <- NULL
  if (!is.null(components$cohort_outcome_mean_ests)) {
    stats_xptr <- lapply(components$cohort_outcome_mean_ests, function(e) {
      stopifnot(inherits(e, "OutcomeMeanSuffStatEstimates"))
      e$.__enclos_env__$private$xp
    })
  }

  aux_xptr <- NULL
  if (!is.null(components$cohort_auxiliary_means)) {
    aux_xptr <- lapply(components$cohort_auxiliary_means, function(e) {
      stopifnot(inherits(e, "CohortAuxiliaryDataMeanEstimates"))
      e$.__enclos_env__$private$xp
    })
  }

  `%||%` <- function(a, b) if (is.null(a)) b else a

  comps_cpp <- list(
    outcome_means = ome_xptr_by_spec,
    cohort_outcome_mean_ests = stats_xptr,
    cohort_auxiliary_means = aux_xptr,
    masked_cohort_outcome_means = components$masked_cohort_outcome_means %||% NULL,
    masked_observed_outcome_indices = components$masked_observed_outcome_indices %||% NULL,
    cohort_outcome_mask = components$cohort_outcome_mask %||% NULL
  )

  est_masked_outcome_mean_err_metrics_cpp(comps_cpp)
}