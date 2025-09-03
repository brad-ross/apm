#' Base class for OutcomeMeanSuffStat estimator (streaming style)
#' @export
OutcomeMeanSuffStatEstimator <- R6::R6Class(
  "OutcomeMeanSuffStatEstimator",
  public = list(
    initialize = function(T_c, T = 0L, q = 0L, bootstrap = NULL) {
      xp <- if (is.null(bootstrap)) NULL else bootstrap$.__enclos_env__$private$xp
      private$xp <- outcome_mean_estimator_new_cpp(
        as.integer(T_c), xp, as.integer(T), as.integer(q)
      )
    },

    add_data = function(unit_idxs, Y, X = NULL) {
      unit_idxs <- as.integer(unit_idxs)
      if (is.null(X)) {
        omsse_add_data_cpp(private$xp, unit_idxs, as.matrix(Y))
      } else {
        omsse_add_data_cpp(private$xp, unit_idxs, as.matrix(Y), array(X, dim = dim(X)))
      }
      invisible(self)
    },

    add_datum = function(unit_idx, Y, X = NULL) {
      unit_idx <- as.integer(unit_idx)
      if (is.null(X)) {
        omsse_add_datum_cpp(private$xp, unit_idx, as.numeric(Y))
      } else {
        omsse_add_datum_cpp(private$xp, unit_idx, as.numeric(Y), as.matrix(X))
      }
      invisible(self)
    },

    estimate = function() {
      fxp <- omsse_estimate_cpp(private$xp)
      OutcomeMeanSuffStatEstimates$new(fxp)
    },

    T_c = function() omsse_Tc_cpp(private$xp),
    T   = function() omsse_T_cpp(private$xp),
    q   = function() omsse_q_cpp(private$xp),
    B   = function() omsse_B_cpp(private$xp)
  ),
  private = list(xp = NULL)
)

#' R6 wrapper for OutcomeMeanSuffStatEstimates (point + bootstrap reps)
#' @export
OutcomeMeanSuffStatEstimates <- R6::R6Class(
  "OutcomeMeanSuffStatEstimates",
  public = list(
    initialize = function(xptr) { private$xp <- xptr },

    has_bootstrap = function() omsse_has_bootstrap_cpp(private$xp),
    num_bootstraps = function() omsse_num_bootstrap_cpp(private$xp),

    T_c = function() omsse_point_Tc_cpp(private$xp),
    T   = function() omsse_point_T_cpp(private$xp),
    q   = function() omsse_point_q_cpp(private$xp),

    observed_outcome_means = function(b = NULL) {
      if (is.null(b)) as.numeric(omsse_point_observed_means_cpp(private$xp))
      else as.numeric(omsse_boot_observed_means_cpp(private$xp, as.integer(b)))
    },

    covar_means = function(b = NULL) {
      if (is.null(b)) {
        if (!omsse_point_has_covar_means_cpp(private$xp)) return(NULL)
        omsse_point_covar_means_cpp(private$xp)
      } else {
        omsse_boot_covar_means_cpp(private$xp, as.integer(b))
      }
    }
  ),
  private = list(xp = NULL)
)


