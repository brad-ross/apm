#' Cohort Weight Estimates Container
#'
#' An R6 class that holds cohort weights used for aggregating cohort-specific
#' estimates across cohorts, with optional bootstrap replicates.
#'
#' @description
#' `CohortWeightEstimates` stores the weights assigned to each cohort during
#' aggregation of cohort-specific factor model parameters. Weights determine
#' how cohorts contribute to the aggregated (panel-level) estimates.
#'
#' Two weighting schemes are typically used:
#' \itemize{
#'   \item **Equal weighting** (`"equal"`): Each cohort receives weight 1/C.
#'   \item **Size-based weighting** (`"by_size"`): Cohorts are weighted by their
#'         share of total units in the panel.
#' }
#'
#' @details
#' This class is typically not constructed directly. It is returned as part of
#' the output from \code{\link{est_cohort_specific_params}}.
#'
#' When bootstrap is enabled, each bootstrap replicate may have different cohort
#' weights (particularly under size-based weighting where bootstrap resampling
#' changes the effective cohort sizes).
#'
#' @section Public Methods:
#' \describe{
#'   \item{\code{has_bootstrap()}}{Logical; whether bootstrap replicates exist.}
#'   \item{\code{num_bootstraps()}}{Integer; number of bootstrap replicates (0 if none).}
#'   \item{\code{cohort_weights()}}{Numeric vector of length C; point estimate cohort
#'         weights (sum to 1).}
#'   \item{\code{bootstrap_cohort_weights(b)}}{Numeric vector of length C; cohort
#'         weights for bootstrap replicate `b` (1-indexed).}
#' }
#'
#' @seealso \code{\link{est_cohort_specific_params}} which returns this object.
#' @seealso \code{\link{aggregate_factor_model_params}} which uses cohort weights.
#'
#' @examples
#' # CohortWeightEstimates is typically obtained from est_cohort_specific_params
#' \dontrun{
#' results <- est_cohort_specific_params(panel, specs)
#' cw <- results$cohort_weights$my_spec
#' cw$cohort_weights()     # Point estimate weights
#' cw$has_bootstrap()      # Check for bootstrap
#' }
#'
#' @export
CohortWeightEstimates <- R6::R6Class(
  "CohortWeightEstimates",
  public = list(
    initialize = function(xptr) { private$xp <- xptr },

    has_bootstrap = function() cwe_has_bootstrap_cpp(private$xp),
    num_bootstraps = function() cwe_num_bootstrap_cpp(private$xp),

    cohort_weights = function() cwe_point_weights_cpp(private$xp),
    bootstrap_cohort_weights = function(b) cwe_boot_weights_cpp(private$xp, as.integer(b))
  ),
  private = list(xp = NULL)
)
