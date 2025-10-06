#' Base class for PC estimators (mirrors C++ PCBase)
#' @export
PCBase <- R6::R6Class(
  "PCBase",
  inherit = FactorModelEstimator,
  public = list(
    initialize = function(...) {
      stop("abstract; use PCEstimator or PCEstimatorWithFEs")
    }
  )
)

#' Principal Components estimator
#' @export
PCEstimator <- R6::R6Class(
  "PCEstimator",
  inherit = PCBase,
  public = list(
    initialize = function(r, T_c, bootstrap = NULL, q = 0L) {
      xp <- if (is.null(bootstrap)) NULL else bootstrap$.__enclos_env__$private$xp
      private$xp <- pc_estimator_new_cpp(as.integer(r), as.integer(T_c), xp, as.integer(q))
    }
  )
)

#' PC with outcome fixed effects
#' @export
PCEstimatorWithFEs <- R6::R6Class(
  "PCEstimatorWithFEs",
  inherit = PCBase,
  public = list(
    initialize = function(r, T_c, bootstrap = NULL, q = 0L) {
      xp <- if (is.null(bootstrap)) NULL else bootstrap$.__enclos_env__$private$xp
      private$xp <- pc_fe_estimator_new_cpp(as.integer(r), as.integer(T_c), xp, as.integer(q))
    }
  )
)