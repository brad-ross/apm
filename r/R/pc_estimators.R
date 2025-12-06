#' Abstract Base Class for Principal Components Estimators
#'
#' An abstract R6 class that serves as the base for principal components-based
#' factor model estimators. Cannot be instantiated directly.
#'
#' @description
#' `PCBase` provides the common interface for PC-based estimators. Use
#' \code{\link{PCEstimator}} for standard PC or \code{\link{PCEstimatorWithFEs}}
#' for PC with outcome fixed effects.
#'
#' @inherit FactorModelEstimator
#'
#' @seealso \code{\link{PCEstimator}} for standard principal components.
#' @seealso \code{\link{PCEstimatorWithFEs}} for PC with fixed effects.
#' @seealso \code{\link{FactorModelEstimator}} for the parent class interface.
#'
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

#' Principal Components Factor Model Estimator
#'
#' An R6 class for estimating factor models using principal components (PC).
#' Computes the top `r` principal components of the outcome data as factor
#' estimates.
#'
#' @description
#' `PCEstimator` implements the classical principal components approach to
#' factor model estimation. Given outcome data Y (N_c x T_c), it computes the
#' singular value decomposition and extracts the top r left singular vectors
#' as factor estimates G.
#'
#' @details
#' The model assumes:
#' \deqn{Y_{it} = \gamma_t' \lambda_i + \epsilon_{it}}
#' where \gamma_t is an r-dimensional factor vector for outcome t, and \lambda_i
#' is the r-dimensional loading for unit i.
#'
#' @section Constructor:
#' \preformatted{
#' PCEstimator$new(r, T_c, bootstrap = NULL, q = 0L)
#' }
#'
#' @section Constructor Arguments:
#' \describe{
#'   \item{r}{Integer; the factor model rank (number of factors to extract).}
#'   \item{T_c}{Integer; number of observed outcomes for this cohort.}
#'   \item{bootstrap}{Optional \code{\link{WeightedBootstrap}} object for
#'         bootstrap inference.}
#'   \item{q}{Integer; number of covariates (default: 0).}
#' }
#'
#' @inherit FactorModelEstimator details
#'
#' @seealso \code{\link{PCEstimatorWithFEs}} for PC with outcome fixed effects.
#' @seealso \code{\link{FactorModelEstimator}} for the interface documentation.
#' @seealso \code{\link{FactorModelEstimates}} for the returned estimates.
#'
#' @examples
#' # Create estimator for rank-2 model with 5 outcomes
#' pc <- PCEstimator$new(r = 2, T_c = 5)
#'
#' # Add synthetic data (in practice, use real panel data)
#' set.seed(42)
#' Y <- matrix(rnorm(20 * 5), nrow = 20, ncol = 5)  # 20 units, 5 outcomes
#' pc$add_data(unit_idxs = 1:20, Y = Y)
#'
#' # Get estimates
#' fme <- pc$estimate()
#' G <- fme$G()  # 5 x 2 factor matrix
#' dim(G)
#'
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

#' Principal Components Estimator with Outcome Fixed Effects
#'
#' An R6 class for estimating factor models using principal components with
#' outcome-specific fixed effects (intercepts).
#'
#' @description
#' `PCEstimatorWithFEs` extends the standard PC estimator to jointly estimate
#' outcome fixed effects g0 alongside the factors G.
#'
#' @details
#' The model assumes:
#' \deqn{Y_{it} = \gamma_{0t} + \gamma_t' \lambda_i + \epsilon_{it}}
#' where \gamma_{0t} is an outcome-specific intercept, \gamma_t is an r-dimensional
#' factor vector, and \lambda_i is the unit loading.
#'
#' Estimation proceeds by first centering outcomes (absorbing means into g0),
#' then computing principal components on the centered data.
#'
#' @section Constructor:
#' \preformatted{
#' PCEstimatorWithFEs$new(r, T_c, bootstrap = NULL, q = 0L)
#' }
#'
#' @section Constructor Arguments:
#' \describe{
#'   \item{r}{Integer; the factor model rank (number of factors).}
#'   \item{T_c}{Integer; number of observed outcomes for this cohort.}
#'   \item{bootstrap}{Optional \code{\link{WeightedBootstrap}} object.}
#'   \item{q}{Integer; number of covariates (default: 0).}
#' }
#'
#' @inherit FactorModelEstimator details
#'
#' @seealso \code{\link{PCEstimator}} for PC without fixed effects.
#' @seealso \code{\link{FactorModelEstimates}} for the returned estimates.
#'
#' @examples
#' # Create estimator for rank-2 model with 5 outcomes and fixed effects
#' pc_fe <- PCEstimatorWithFEs$new(r = 2, T_c = 5)
#'
#' # Add data with outcome-specific means
#' set.seed(42)
#' outcome_means <- c(10, 20, 15, 25, 30)
#' Y <- matrix(rnorm(20 * 5), nrow = 20) + outcome_means
#' pc_fe$add_data(unit_idxs = 1:20, Y = Y)
#'
#' # Get estimates
#' fme <- pc_fe$estimate()
#' fme$has_g0()  # TRUE
#' fme$g0()      # Length-5 vector of fixed effects
#'
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
