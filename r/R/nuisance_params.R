#' Cohort auxiliary data mean estimates holder
#'
#' R6 wrapper for cohort-level auxiliary data means (T x d matrix), with optional
#' bootstrap replicates.
#'
#' @format An R6 class with methods:
#' - has_bootstrap(), num_bootstraps()
#' - T(), d()
#' - auxiliary_means(b = NULL)
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


