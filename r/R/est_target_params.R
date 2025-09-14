#' Target parameter estimates holder
#'
#' R6 wrapper for target parameter estimates: a point p-vector and optional
#' bootstrap replicates.
#'
#' @export
TargetParameterEstimates <- R6::R6Class(
  "TargetParameterEstimates",
  public = list(
    initialize = function(xptr) { private$xp <- xptr },

    has_bootstrap = function() tpe_has_bootstrap_cpp(private$xp),
    num_bootstraps = function() tpe_num_bootstrap_cpp(private$xp),
    p = function() tpe_p_cpp(private$xp),

    target_params = function(b = NULL) {
      if (is.null(b)) tpe_point_params_cpp(private$xp)
      else tpe_boot_params_cpp(private$xp, as.integer(b))
    },
    boots_matrix = function() tpe_boot_params_matrix_cpp(private$xp)
  ),
  private = list(xp = NULL)
)

#' Estimate target parameters from estimated means (and optional aux means)
#'
#' @param outcome_means OutcomeMeansEstimates R6 object (C x T means with optional bootstrap).
#' @param fn function(Y, eta = NULL) returning numeric vector length p.
#' @param aux_means optional list of CohortAuxiliaryDataMeanEstimates (one per cohort)
#' @return TargetParameterEstimates R6 object.
#' @export
est_target_params <- function(outcome_means, fn, aux_means = NULL) {
  stopifnot(inherits(outcome_means, "OutcomeMeansEstimates"))
  if (!is.null(aux_means)) stopifnot(is.list(aux_means))
  if (!is.null(aux_means)) {
    aux_means <- lapply(aux_means, function(e) {
      if (is.null(e)) return(NULL)
      stopifnot(inherits(e, "CohortAuxiliaryDataMeanEstimates"))
      e$.__enclos_env__$private$xp
    })
  }
  xp <- est_target_params_cpp(
    outcome_means$.__enclos_env__$private$xp,
    aux_means,
    fn
  )
  TargetParameterEstimates$new(xp)
}

#' Estimate target parameters by spec (named lists)
#'
#' @param outcome_means_by_spec named list of OutcomeMeansEstimates
#' @param fn function(Y, eta = NULL)
#' @param aux_means_by_spec optional named list whose values are lists of CohortAuxiliaryDataMeanEstimates
#' @return named list of TargetParameterEstimates
#' @export
est_target_params_by_spec <- function(outcome_means_by_spec, fn, aux_means_by_spec = NULL) {
  stopifnot(is.list(outcome_means_by_spec))
  if (!is.null(aux_means_by_spec)) stopifnot(is.list(aux_means_by_spec))
  if (!is.null(aux_means_by_spec)) {
    aux_means_by_spec <- lapply(aux_means_by_spec, function(lst) {
      if (is.null(lst)) return(NULL)
      stopifnot(is.list(lst))
      lapply(lst, function(e) {
        if (is.null(e)) return(NULL)
        stopifnot(inherits(e, "CohortAuxiliaryDataMeanEstimates"))
        e$.__enclos_env__$private$xp
      })
    })
  }
  est_target_params_by_spec_cpp(outcome_means_by_spec, aux_means_by_spec, fn)
}