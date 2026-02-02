#' Cohort Auxiliary Data Mean Estimates Container
#'
#' An R6 class that holds cohort-level means of auxiliary data columns,
#' with optional bootstrap replicates.
#'
#' @description
#' `CohortAuxiliaryDataMeanEstimates` stores the mean values of auxiliary
#' data columns (specified via `auxiliary_cols` in \code{\link{UnbalancedPanel}})
#' computed at the cohort level for each outcome.
#'
#' Auxiliary data is additional information tracked alongside the panel that
#' is not used in factor model estimation but may be useful for post-estimation
#' analysis or target parameter computation.
#'
#' @details
#' The auxiliary means are stored as a T x d matrix where:
#' \itemize{
#'   \item T is the number of outcomes (rows correspond to outcome indices)
#'   \item d is the number of auxiliary columns
#' }
#'
#' This class is typically not constructed directly. It is returned as part of
#' the output from \code{\link{est_cohort_specific_params}} or
#' \code{\link{est_target_param_components}} when auxiliary columns are present.
#'
#' @section Public Methods:
#' \describe{
#'   \item{\code{has_bootstrap()}}{Logical; whether bootstrap replicates exist.}
#'   \item{\code{num_bootstraps()}}{Integer; number of bootstrap replicates (0 if none).}
#'   \item{\code{T()}}{Integer; number of outcomes (rows of auxiliary means matrix).}
#'   \item{\code{d()}}{Integer; number of auxiliary columns.}
#'   \item{\code{auxiliary_means(b = NULL)}}{Returns T x d matrix of auxiliary means.
#'         If `b` is NULL, returns point estimate; otherwise returns bootstrap
#'         replicate `b` (1-indexed).}
#' }
#'
#' @seealso \code{\link{UnbalancedPanel}} for specifying auxiliary columns.
#' @seealso \code{\link{est_cohort_specific_params}} which returns this object.
#' @seealso \code{\link{est_target_params}} for using auxiliary means in target
#'   parameter estimation.
#'
#' @examples
#' # CohortAuxiliaryDataMeanEstimates is obtained from estimation functions
#' \dontrun{
#' # Create panel with auxiliary column
#' panel <- UnbalancedPanel$new(
#'   panel_df = df,
#'   unit_id_col = "unit",
#'   outcome_id_col = "time",
#'   outcome_value_col = "y",
#'   auxiliary_cols = "treatment_indicator"
#' )
#'
#' results <- est_cohort_specific_params(panel, specs)
#' aux <- results$cohort_auxiliary_means[[1]]  # First cohort
#' aux$T()                # Number of outcomes
#' aux$d()                # Number of auxiliary columns
#' aux$auxiliary_means()  # T x d matrix
#' }
#'
#' @export
CohortAuxiliaryDataMeanEstimates <- R6::R6Class(
  "CohortAuxiliaryDataMeanEstimates",
  public = list(
    initialize = function(xptr) { private$xp <- xptr },

    has_bootstrap = function() caux_has_bootstrap_cpp(private$xp),
    num_bootstraps = function() caux_num_bootstrap_cpp(private$xp),

    T = function() caux_point_dims_cpp(private$xp)[1],
    d = function() caux_point_dims_cpp(private$xp)[2],

    auxiliary_means = function(b = NULL) {
      if (is.null(b)) caux_point_aux_means_cpp(private$xp)
      else caux_boot_aux_means_cpp(private$xp, as.integer(b))
    }
  ),
  private = list(xp = NULL)
)
