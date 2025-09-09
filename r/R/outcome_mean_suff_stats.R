#' Outcome mean sufficient statistics estimator (streaming)
#'
#' Incrementally computes cohort-level sufficient statistics for outcome means
#' and, optionally, covariate means. Data can be provided in batches or one unit
#' at a time. When a \link{WeightedBootstrap} is attached, bootstrap-weighted
#' aggregates are accumulated alongside the point estimates.
#'
#' Dimensions and expected shapes:
#' - T_c: number of observed outcomes in the cohort (outcomes used for identification)
#' - T:   total number of outcomes (required only if covariates are provided)
#' - q:   number of covariates (required only if covariates are provided)
#'
#' Input data shapes when adding data:
#' - Y: matrix of shape T_c x N_batch (columns correspond to units)
#' - X: optional array/cube of shape T x q x N_batch (aligned with Y's unit order)
#'
#' @format An R6 class with methods:
#' - initialize(T_c, T = 0L, q = 0L, bootstrap = NULL)
#' - add_data(unit_idxs, Y, X = NULL)
#' - add_datum(unit_idx, Y, X = NULL)
#' - estimate()
#' - T_c(), T(), q(), B()
#'
#' @section Methods:
#' \describe{
#'   \item{initialize(T_c, T = 0L, q = 0L, bootstrap = NULL)}{
#'     Construct an estimator. If `bootstrap` is a \link{WeightedBootstrap} instance,
#'     bootstrap replicates are tracked; otherwise only point estimates are tracked.
#'
#'     Arguments:
#'     - `T_c` Integer, number of observed outcomes in the cohort.
#'     - `T` Integer, total number of outcomes (required only if covariates are used).
#'     - `q` Integer, number of covariates (required only if covariates are used).
#'     - `bootstrap` Optional \link{WeightedBootstrap} controlling bootstrap weights.
#'   }
#'
#'   \item{add_data(unit_idxs, Y, X = NULL)}{
#'     Add a batch of units.
#'
#'     - `unit_idxs`: integer vector of unit indices used to derive bootstrap weights.
#'     - `Y`: numeric matrix with shape T_c x N_batch.
#'     - `X`: optional numeric array/cube with shape T x q x N_batch.
#'
#'     Returns the object itself (invisibly), enabling method chaining.
#'   }
#'
#'   \item{add_datum(unit_idx, Y, X = NULL)}{
#'     Add a single unit.
#'
#'     - `unit_idx`: integer index for this unit.
#'     - `Y`: numeric vector of length T_c.
#'     - `X`: optional numeric matrix with shape T x q.
#'
#'     Returns the object itself (invisibly).
#'   }
#'
#'   \item{estimate()}{{Finalize and return} an \link{OutcomeMeanSuffStatEstimates}
#'     object containing point estimates and, when available, bootstrap replicates.}
#'
#'   \item{T_c(), T(), q()}{{Accessors for} the configured dimensions.}
#'
#'   \item{B()}{{Number of} bootstrap replicates attached (0 if none).}
#' }
#'
#' @seealso \link{WeightedBootstrap}, \link{OutcomeMeanSuffStatEstimates}
#'
#' @examples
#' # Minimal example with point estimates only
#' set.seed(1)
#' T_c <- 3L; N <- 5L
#' Y <- matrix(rnorm(T_c * N), nrow = T_c, ncol = N)
#' est <- OutcomeMeanSuffStatEstimator$new(T_c = T_c)
#' est$add_data(unit_idxs = seq_len(N), Y = Y)
#' out <- est$estimate()
#' out$observed_outcome_means()
#'
#' # With covariates and bootstrap
#' T <- 4L; q <- 2L
#' X <- array(rnorm(T * q * N), dim = c(T, q, N))
#' wb <- get_weighted_bootstrap_draws(N = N, B = 10L, type = "bayesian")
#' est2 <- OutcomeMeanSuffStatEstimator$new(T_c = T_c, T = T, q = q, bootstrap = wb)
#' est2$add_data(unit_idxs = seq_len(N), Y = Y, X = X)
#' out2 <- est2$estimate()
#' out2$has_bootstrap()
#' out2$observed_outcome_means(b = 1L)
#'
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

#' OutcomeMeanSuffStat estimates holder
#'
#' R6 wrapper returned by \code{OutcomeMeanSuffStatEstimator$estimate()},
#' exposing point estimates and optional bootstrap replicates of the sufficient
#' statistics.
#'
#' Contents:
#' - Observed outcome means (length T_c) for the point estimate, and per-bootstrap
#'   replicates when requested.
#' - Optional covariate means (T x q) when covariates were supplied to the estimator.
#'
#' @format An R6 class with methods:
#' - has_bootstrap(), num_bootstraps()
#' - T_c(), T(), q()
#' - observed_outcome_means(b = NULL)
#' - covar_means(b = NULL)
#'
#' @section Methods:
#' \describe{
#'   \item{has_bootstrap()}{Logical, whether bootstrap replicates are available.}
#'   \item{num_bootstraps()}{Number of bootstrap replicates (0 if none).}
#'
#'   \item{T_c(), T(), q()}{Dimensions of the underlying estimates.}
#'
#'   \item{observed_outcome_means(b = NULL)}{
#'     Numeric vector (length T_c) of observed outcome means. If `b` is NULL,
#'     returns the point estimate; otherwise returns replicate `b` (1-indexed).
#'   }
#'
#'   \item{covar_means(b = NULL)}{
#'     Numeric matrix (T x q) of covariate means if covariates were tracked; otherwise NULL.
#'     If `b` is NULL, returns the point estimate; otherwise returns replicate `b` (1-indexed).
#'   }
#' }
#'
#' @seealso \link{OutcomeMeanSuffStatEstimator}
#'
#' @examples
#' T_c <- 2L; N <- 3L
#' Y <- matrix(rnorm(T_c * N), nrow = T_c)
#' est <- OutcomeMeanSuffStatEstimator$new(T_c = T_c)
#' out <- est$add_data(seq_len(N), Y)$estimate()
#' out$observed_outcome_means()
#'
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