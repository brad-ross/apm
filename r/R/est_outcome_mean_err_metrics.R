#' Error Metrics for Masked Outcome Means
#'
#' Computes prediction error metrics (bias, standard error, RMSE) for masked
#' cohort-outcome means using bootstrap replicates.
#'
#' @description
#' This function evaluates the quality of outcome mean predictions by comparing
#' imputed values to held-out (masked) true values. It is used for cross-validation
#' and out-of-sample prediction assessment.
#'
#' @details
#' **Metrics Computed:**
#' \itemize{
#'   \item **Bias**: Mean prediction error (imputed - true).
#'   \item **SE**: Standard error of predictions across bootstrap replicates.
#'   \item **RMSE**: Root mean squared error.
#' }
#'
#' The function requires that `est_target_param_components()` was called with
#' `cohort_outcomes_to_mask` specified and bootstrap enabled.
#'
#' @param components The list returned by \code{\link{est_target_param_components}}
#'   when using outcome masking and bootstrap.
#'
#' @return A data.frame with columns:
#'   \describe{
#'     \item{cohort}{Integer; 1-based cohort index.}
#'     \item{outcome}{Integer; 1-based outcome index.}
#'     \item{spec}{Character; estimation specification name.}
#'     \item{bias}{Numeric; average prediction bias.}
#'     \item{se}{Numeric; bootstrap standard error.}
#'     \item{rmse}{Numeric; root mean squared error.}
#'     \item{cohort_pop_share}{Numeric; cohort's share of total population.}
#'   }
#'
#' @seealso \code{\link{est_target_param_components}} for running estimation with
#'   masking enabled.
#'
#' @examples
#' \dontrun{
#' # Run estimation with masking
#' components <- est_target_param_components(
#'   panel, specs, bootstrap = wb,
#'   cohort_outcomes_to_mask = list("1" = c(3L, 4L))
#' )
#'
#' # Compute error metrics
#' metrics <- est_masked_outcome_mean_err_metrics(components)
#' print(metrics)
#' }
#'
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