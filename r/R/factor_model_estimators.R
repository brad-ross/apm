#' R6 wrapper for FactorModelEstimates
#' @export
FactorModelEstimates <- R6::R6Class(
  "FactorModelEstimates",
  public = list(
    initialize = function(xptr) { private$xp <- xptr },
    has_bootstrap = function() fme_has_bootstrap_cpp(private$xp),
    num_bootstraps = function() fme_num_bootstrap_cpp(private$xp),
    has_g0 = function() fme_has_g0_cpp(private$xp),
    has_a = function() fme_has_a_cpp(private$xp),
    G = function(b = NULL) {
      if (is.null(b)) fme_point_G_cpp(private$xp)
      else fme_boot_G_cpp(private$xp, as.integer(b))
    },
    g0 = function(b = NULL) {
      if (is.null(b)) {
        if (!self$has_g0()) return(NULL)
        fme_point_g0_cpp(private$xp)
      } else {
        fme_boot_g0_cpp(private$xp, as.integer(b))
      }
    },
    a = function(b = NULL) {
      if (is.null(b)) {
        if (!self$has_a()) return(NULL)
        fme_point_a_cpp(private$xp)
      } else {
        fme_boot_a_cpp(private$xp, as.integer(b))
      }
    }
  ),
  private = list(xp = NULL)
)

#' Generic FactorModelEstimator R6 class
#' @export
FactorModelEstimator <- R6::R6Class(
  "FactorModelEstimator",
  public = list(
    initialize = function(...) {
      stop("abstract; construct a concrete estimator class")
    },
    add_data = function(unit_idxs, Y, X = NULL) {
      unit_idxs <- as.integer(unit_idxs)
      if (is.null(X)) {
        estimator_add_data_cpp(private$xp, unit_idxs, as.matrix(Y))
      } else {
        estimator_add_data_cpp(private$xp, unit_idxs, as.matrix(Y), array(X, dim = dim(X)))
      }
      invisible(self)
    },
    add_datum = function(unit_idx, Y, X = NULL) {
      unit_idx <- as.integer(unit_idx)
      if (is.null(X)) {
        estimator_add_datum_cpp(private$xp, unit_idx, as.numeric(Y))
      } else {
        estimator_add_datum_cpp(private$xp, unit_idx, as.numeric(Y), as.matrix(X))
      }
      invisible(self)
    },
    estimate = function() {
      fxp <- estimator_estimate_cpp(private$xp)
      FactorModelEstimates$new(fxp)
    },
    r = function() estimator_r_cpp(private$xp),
    T_c = function() estimator_Tc_cpp(private$xp),
    q = function() estimator_q_cpp(private$xp),
    B = function() estimator_B_cpp(private$xp)
  ),
  private = list(xp = NULL)
)


