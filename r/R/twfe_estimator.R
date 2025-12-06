#' Two-Way Fixed Effects Factor Model Estimator
#'
#' An R6 class for estimating factor models using two-way fixed effects (TWFE),
#' which corresponds to a rank-0 factor model with only outcome fixed effects.
#'
#' @description
#' `TWFEEstimator` implements a degenerate factor model where factors G are
#' absent (or equivalently, rank r = 0), and only outcome fixed effects g0
#' are estimated. This corresponds to the classical two-way fixed effects
#' specification common in panel data econometrics.
#'
#' @details
#' The model assumes:
#' \deqn{Y_{it} = \gamma_{0t} + \lambda_i + \epsilon_{it}}
#' where \gamma_{0t} is an outcome-specific fixed effect and \lambda_i is a unit-specific
#' effect. In the cohort framework, we estimate \gamma_{0t} as the cross-sectional
#' mean of outcomes within the cohort.
#'
#' TWFE is useful as a baseline comparison for factor models.
#'
#' @section Constructor:
#' \preformatted{
#' TWFEEstimator$new(T_c, bootstrap = NULL, q = 0L)
#' }
#'
#' @section Constructor Arguments:
#' \describe{
#'   \item{T_c}{Integer; number of observed outcomes for this cohort.}
#'   \item{bootstrap}{Optional \code{\link{WeightedBootstrap}} object for
#'         bootstrap inference.}
#'   \item{q}{Integer; number of covariates (default: 0).}
#' }
#'
#' @inherit FactorModelEstimator details
#'
#' @seealso \code{\link{PCEstimatorWithFEs}} for factor models with fixed effects.
#' @seealso \code{\link{FactorModelEstimator}} for the interface documentation.
#' @seealso \code{\link{FactorModelEstimates}} for the returned estimates.
#'
#' @examples
#' # Create TWFE estimator for 5 outcomes
#' twfe <- TWFEEstimator$new(T_c = 5)
#'
#' # Add data
#' set.seed(42)
#' Y <- matrix(rnorm(20 * 5), nrow = 20, ncol = 5)
#' twfe$add_data(unit_idxs = 1:20, Y = Y)
#'
#' # Get estimates
#' fme <- twfe$estimate()
#' fme$has_g0()  # TRUE
#' fme$g0()      # Length-5 vector of outcome means
#'
#' @export
TWFEEstimator <- R6::R6Class(
  "TWFEEstimator",
  inherit = FactorModelEstimator,
  public = list(
    initialize = function(T_c, bootstrap = NULL, q = 0L) {
      xp <- if (is.null(bootstrap)) NULL else bootstrap$.__enclos_env__$private$xp
      private$xp <- twfe_estimator_new_cpp(as.integer(T_c), xp, as.integer(q))
    }
  )
)
