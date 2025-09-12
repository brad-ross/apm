#' R6 wrapper for OutcomeMeansEstimates
#' @export
OutcomeMeansEstimates <- R6::R6Class(
  "OutcomeMeansEstimates",
  public = list(
    initialize = function(xptr) { private$xp <- xptr },
    has_bootstrap = function() ome_has_bootstrap_cpp(private$xp),
    num_bootstraps = function() ome_num_bootstrap_cpp(private$xp),
    C = function() ome_C_cpp(private$xp),
    T = function() ome_T_cpp(private$xp),
    mean_outcomes = function(b = NULL) {
      if (is.null(b)) ome_point_means_cpp(private$xp)
      else ome_boot_means_cpp(private$xp, as.integer(b))
    }
  ),
  private = list(xp = NULL)
)


