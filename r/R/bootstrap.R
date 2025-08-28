#' Weighted bootstrap R6 wrapper and helper
#'
#' @name bootstrap
NULL

#' WeightedBootstrap R6 class (external pointer shell)
#'
#' @export
WeightedBootstrap <- R6::R6Class(
  classname = "WeightedBootstrap",
  public = list(
    initialize = function(N, B, type = c("multinomial", "bayesian"), seed = NULL) {
      type <- match.arg(type)
      if (!is.null(seed)) set.seed(as.integer(seed))
      private$xp <- get_weighted_bootstrap_ptr_cpp(N, B, type, seed)
    },
    n_obs = function() wb_n_obs_cpp(private$xp),
    n_bootstraps = function() wb_n_bootstraps_cpp(private$xp),
    draw = function(b) wb_draw_cpp(private$xp, as.integer(b)),
    obs = function(i) wb_obs_cpp(private$xp, as.integer(i)),
    obs_rows = function(idx) wb_obs_rows_cpp(private$xp, as.integer(idx)),
    weights = function() wb_weights_cpp(private$xp)
  ),
  private = list(
    xp = NULL
  )
)

#' Get weighted bootstrap draws (returns WeightedBootstrap R6 instance)
#'
#' @param N Integer, number of observations (rows)
#' @param B Integer, number of bootstrap draws (cols)
#' @param type Character, either "multinomial" or "bayesian"
#' @param seed Optional integer seed for reproducibility
#' @return WeightedBootstrap R6 object with methods: n_obs, n_bootstraps, draw, obs, obs_rows, weights
#' @export
get_weighted_bootstrap_draws <- function(N, B, type = c("multinomial", "bayesian"), seed = NULL) {
  if (!is.null(seed)) set.seed(as.integer(seed))
  WeightedBootstrap$new(N, B, type, seed)
}


