#' Two-way Fixed Effects estimator
#' @export
TWFEEstimator <- R6::R6Class(
  "TWFEEstimator",
  inherit = FactorModelEstimator,
  public = list(
    initialize = function(T_c, bootstrap = NULL, q = 0L) {
      xp <- if (is.null(bootstrap)) NULL else bootstrap$.__enclos_env__$private$xp
      private$xp <- twfe_estimator_new_cpp(as.integer(T_c), xp, as.integer(q))
    }
  )
)