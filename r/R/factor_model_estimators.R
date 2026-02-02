#' Factor Model Parameter Estimates Container
#'
#' An R6 class that holds estimated factor model parameters (factors G, optional
#' outcome fixed effects g0, optional covariate coefficients a, and optional
#' loadings L), along with optional bootstrap replicates.
#'
#' @description
#' `FactorModelEstimates` is the primary container for factor model estimation
#' results. It holds:
#' \itemize{
#'   \item **G**: T x r matrix of factor estimates (rows = outcomes, cols = factors)
#'   \item **g0**: Optional length-T vector of outcome fixed effects
#'   \item **a**: Optional length-q vector of covariate coefficients
#'   \item **L**: Optional N x r or C x r matrix of loadings (unit or cohort level)
#' }
#'
#' When bootstrap is enabled, each parameter has B replicates accessible via
#' the `b` argument to accessor methods.
#'
#' @details
#' This class is typically not constructed directly. Instead, it is returned by:
#' \itemize{
#'   \item \code{\link{FactorModelEstimator}$estimate()} and its subclasses
#'   \item \code{\link{est_cohort_specific_params}}
#'   \item \code{\link{aggregate_factor_model_params}}
#'   \item \code{\link{comp_imputation_components}}
#' }
#'
#' @section Public Methods:
#' \describe{
#'   \item{\code{has_bootstrap()}}{Logical; whether bootstrap replicates exist.}
#'   \item{\code{num_bootstraps()}}{Integer; number of bootstrap replicates (0 if none).}
#'   \item{\code{has_g0()}}{Logical; whether outcome fixed effects are present.}
#'   \item{\code{has_a()}}{Logical; whether covariate coefficients are present.}
#'   \item{\code{G(b = NULL)}}{Returns T x r factor matrix. If `b` is NULL, returns
#'         point estimate; otherwise returns bootstrap replicate `b` (1-indexed).}
#'   \item{\code{g0(b = NULL)}}{Returns length-T outcome fixed effects vector, or
#'         NULL if not present. If `b` is specified, returns bootstrap replicate.}
#'   \item{\code{a(b = NULL)}}{Returns length-q covariate coefficients vector, or
#'         NULL if not present. If `b` is specified, returns bootstrap replicate.}
#'   \item{\code{L(b = NULL)}}{Returns loadings matrix (N x r or C x r), or NULL 
#'        if not present. If `b` is specified, returns bootstrap replicate.}
#' }
#'
#' @seealso \code{\link{FactorModelEstimator}} for the abstract estimator class.
#' @seealso \code{\link{PCEstimator}}, \code{\link{PCEstimatorWithFEs}} for
#'   principal components estimators.
#' @seealso \code{\link{aggregate_factor_model_params}} for aggregating across cohorts.
#'
#' @examples
#' # FactorModelEstimates is typically obtained from estimation functions
#' # Example: create a PC estimator and get estimates
#' # (In practice, you would add real data)
#' \dontrun{
#' est <- PCEstimator$new(r = 2, T_c = 5)
#' # est$add_data(unit_idxs, Y_matrix)
#' # fme <- est$estimate()
#' # fme$G()      # T_c x r factor matrix
#' # fme$has_g0() # FALSE for PCEstimator
#' }
#'
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
    },
    L = function(b = NULL) {
      if (is.null(b)) {
        fme_point_L_cpp(private$xp)
      } else {
        fme_boot_L_cpp(private$xp, as.integer(b))
      }
    }
  ),
  private = list(xp = NULL)
)

#' Abstract Factor Model Estimator Base Class
#'
#' An abstract R6 class defining the interface for streaming factor model
#' estimators. Concrete implementations include \code{\link{PCEstimator}},
#' \code{\link{PCEstimatorWithFEs}}, and \code{\link{TWFEEstimator}}.
#'
#' @description
#' `FactorModelEstimator` defines a streaming interface for factor model
#' estimation where data is added incrementally via `add_data()` or `add_datum()`,
#' and final estimates are computed via `estimate()`.
#'
#' This class cannot be instantiated directly; use a concrete subclass.
#'
#' @details
#' The streaming interface allows efficient estimation for large panels by:
#' \itemize{
#'   \item Accumulating sufficient statistics incrementally
#'   \item Computing final estimates only when `estimate()` is called
#'   \item Supporting optional bootstrap weights via constructor arguments
#' }
#'
#' @section Public Methods:
#' \describe{
#'   \item{\code{add_data(unit_idxs, Y, X = NULL)}}{Add a batch of units.
#'     \itemize{
#'       \item `unit_idxs`: Integer vector of 1-based unit indices (for bootstrap weights).
#'       \item `Y`: Numeric matrix T_c x N_batch (outcomes x units).
#'       \item `X`: Optional array T x q x N_batch of covariates.
#'     }
#'     Returns `self` invisibly for method chaining.}
#'   \item{\code{add_datum(unit_idx, Y, X = NULL)}}{Add a single unit.
#'     \itemize{
#'       \item `unit_idx`: Integer 1-based unit index.
#'       \item `Y`: Numeric vector of length T_c.
#'       \item `X`: Optional T x q matrix of covariates.
#'     }
#'     Returns `self` invisibly.}
#'   \item{\code{estimate()}}{Finalize estimation and return a
#'     \code{\link{FactorModelEstimates}} object.}
#'   \item{\code{r()}}{Integer; the factor rank.}
#'   \item{\code{T_c()}}{Integer; number of observed outcomes.}
#'   \item{\code{q()}}{Integer; number of covariates (0 if none).}
#'   \item{\code{B()}}{Integer; number of bootstrap replicates (0 if none).}
#' }
#'
#' @seealso \code{\link{PCEstimator}} for principal components estimation.
#' @seealso \code{\link{PCEstimatorWithFEs}} for PC with outcome fixed effects.
#' @seealso \code{\link{TWFEEstimator}} for two-way fixed effects.
#' @seealso \code{\link{FactorModelEstimates}} for the returned estimates.
#'
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