#' Outcome sufficient statistics R6 wrapper
#'
#' Constructs and wraps the C++ `apm::OutcomeMeanSufficientStatEstimates`, which
#' contains point sufficient statistics for outcome means (and optional
#' covariate means) and, when a bootstrap is supplied, bootstrap replicates of
#' those sufficient statistics.
#'
#' @param outcomes Numeric matrix with shape N x T_c containing the observed
#'   outcomes across N units for the cohort (columns correspond to the observed
#'   outcomes for this cohort).
#' @param bootstrap Optional [`WeightedBootstrap`] object. When provided,
#'   bootstrap replicates of sufficient statistics are computed using the
#'   bootstrap weights for the selected units at each draw.
#' @param covars Optional numeric array with shape N x T x q containing
#'   covariates across N units and T outcomes for q covariates. When provided,
#'   covariate means are computed and attached; otherwise `covar_means()`
#'   returns `NULL`.
#' @param unit_idxs Optional integer vector of 1-based unit indices to include
#'   when constructing bootstrap replicates. If `NULL` or empty, all rows of
#'   `outcomes` are used.
#'
#' @details The object stores:
#' - Point sufficient statistics: `observed_outcome_means()` and (optionally)
#'   `covar_means()`.
#' - If a bootstrap is supplied, `num_bootstraps()` > 0 and the same accessors
#'   accept a `b` argument to retrieve the b-th (1-based) bootstrap replicate.
#'
#' Dimension helpers:
#' - `T_c()` returns the number of observed outcomes for this cohort.
#' - `T()` returns the number of outcomes only when covariates are supplied;
#'   otherwise 0.
#' - `q()` returns the number of covariates when covariates are supplied;
#'   otherwise 0.
#'
#' @section Methods:
#' - `has_bootstrap()` -> logical
#' - `num_bootstraps()` -> integer
#' - `T_c()` -> integer
#' - `T()` -> integer
#' - `q()` -> integer
#' - `observed_outcome_means(b = NULL)` -> numeric vector length `T_c`
#' - `covar_means(b = NULL)` -> numeric matrix `T x q` or `NULL` if absent
#'
#' @return An R6 object of class `OutcomeMeanSufficientStatEstimates`.
#'
#' @examples
#' N <- 5; T_c <- 3; q <- 2
#' set.seed(1)
#' Y <- matrix(rnorm(N * T_c), N, T_c)
#' X <- array(rnorm(N * T_c * q), dim = c(N, T_c, q))
#' wb <- get_weighted_bootstrap_draws(N, B = 10, type = "multinomial", seed = 123)
#' oss <- OutcomeMeanSufficientStatEstimates$new(outcomes = Y, bootstrap = wb, covars = X)
#' oss$observed_outcome_means()
#' oss$covar_means()
#' oss$num_bootstraps()
#'
#' @seealso [`WeightedBootstrap`]
#' @export
OutcomeMeanSufficientStatEstimates <- R6::R6Class(
  "OutcomeMeanSufficientStatEstimates",
  public = list(
    initialize = function(outcomes, bootstrap = NULL, covars = NULL, unit_idxs = NULL) {
      xp <- if (is.null(bootstrap)) NULL else bootstrap$.__enclos_env__$private$xp
      private$xp <- outcome_suff_from_data_cpp(
        as.matrix(outcomes),
        xp,
        if (is.null(covars)) NULL else array(covars, dim = dim(covars)),
        if (is.null(unit_idxs)) NULL else as.integer(unit_idxs)
      )
    },

    has_bootstrap = function() omsse_has_bootstrap_cpp(private$xp),
    num_bootstraps = function() omsse_num_bootstrap_cpp(private$xp),

    T_c = function() omsse_point_Tc_cpp(private$xp),
    T = function() omsse_point_T_cpp(private$xp),
    q = function() omsse_point_q_cpp(private$xp),

    observed_outcome_means = function(b = NULL) {
      if (is.null(b)) omsse_point_observed_means_cpp(private$xp)
      else omsse_boot_observed_means_cpp(private$xp, as.integer(b))
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


