#' Cohort weight estimates holder
#'
#' R6 wrapper returned in `cohort_weights` from `est_cohort_specific_params_from_panel_cpp()`.
#' Exposes point cohort weights and optional bootstrap replicate weights.
#'
#' @format An R6 class with methods:
#' - has_bootstrap(), num_bootstraps()
#' - cohort_weights()
#' - bootstrap_cohort_weights(b)
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


