#' Weighted bootstrap R6 wrapper and helper
#'
#' @name bootstrap
NULL

#' WeightedBootstrap R6 class
#'
#' Immutable container of bootstrap weights with convenient accessors, backed by
#' efficient C++ implementations. Supports the classical multinomial (Efron)
#' bootstrap and the Bayesian (Rubin) bootstrap.
#'
#' Conventions and shapes:
#' - N: number of observations
#' - B: number of bootstrap draws
#' - We store a weights matrix W of shape N x B where columns correspond to
#'   bootstrap draws and each column sums to 1.
#'
#' Types:
#' - "multinomial": each draw samples N indices with replacement uniformly;
#'   weights are counts / N.
#' - "bayesian": each draw uses i.i.d. Exp(1) raw weights normalized to sum to 1, i.e. 
#'   weights drawn from a Dirichlet(1, 1, ..., 1) distribution.
#'
#' @format An R6 class with methods:
#' - initialize(N, B, type = c("multinomial", "bayesian"), seed = NULL)
#' - n_obs(), n_bootstraps()
#' - draw(b)
#' - obs(i), obs_rows(idx)
#' - weights()
#'
#' @section Methods:
#' \describe{
#'   \item{initialize(N, B, type, seed = NULL)}{Construct with N observations and
#'     B draws. `type` is either "multinomial" or "bayesian". If `seed` is supplied
#'     it is passed to the underlying RNG for reproducibility.}
#'
#'   \item{n_obs()}{Integer N, number of observations.}
#'
#'   \item{n_bootstraps()}{Integer B, number of bootstrap draws.}
#'
#'   \item{draw(b)}{Numeric vector of length N with weights for draw `b` (1-indexed).}
#'
#'   \item{obs(i)}{Numeric vector of length B with weights for observation `i`
#'     across all draws (1-indexed).}
#'
#'   \item{obs_rows(idx)}{Numeric matrix of shape length(idx) x B with rows of the
#'     weights matrix for the provided observation indices `idx` (1-indexed).}
#'
#'   \item{weights()}{Numeric matrix W of shape N x B containing all weights; each
#'     column sums to 1.}
#' }
#'
#' @seealso \link{get_weighted_bootstrap_draws}, \link{OutcomeMeanSuffStatEstimator}
#'
#' @examples
#' # Construct with multinomial bootstrap
#' N <- 5L; B <- 10L
#' wb <- WeightedBootstrap$new(N, B, type = "multinomial", seed = 1L)
#' wb$n_obs(); wb$n_bootstraps()
#' w1 <- wb$draw(1L)       # length N
#' r3 <- wb$obs(3L)        # length B
#' sub <- wb$obs_rows(c(1L, 5L))  # 2 x B
#' W <- wb$weights()       # N x B, cols sum to 1
#'
#' # Helper constructor (recommended)
#' wb2 <- get_weighted_bootstrap_draws(N, B, type = "bayesian", seed = 42L)
#' wb2$draw(2L)
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

#' Bootstrap and target-parameter inference results
#'
#' R6 wrapper around simultaneous inference results computed from bootstrap replicates.
#'
#' @export
SimultaneousInferenceResults <- R6::R6Class(
  classname = "SimultaneousInferenceResults",
  public = list(
    initialize = function(ptr) {
      private$xp <- ptr
    },
    point = function() sir_point_cpp(private$xp),
    t_stats = function() sir_pointwise_t_cpp(private$xp),
    p_vals = function() sir_pointwise_p_cpp(private$xp),
    sig_level = function() sir_sig_level_cpp(private$xp),
    ci = function() list(lb = sir_ci_lb_cpp(private$xp),
                         ub = sir_ci_ub_cpp(private$xp)),
    cb = function() list(lb = sir_cb_lb_cpp(private$xp),
                         ub = sir_cb_ub_cpp(private$xp)),
    as_data_frame = function(param_names = NULL) {
      est <- self$point()
      t <- self$t_stats()
      p <- self$p_vals()
      ci <- self$ci()
      cb <- self$cb()

      n <- length(est)
      if (!all(lengths(list(t, p, ci$lb, ci$ub, cb$lb, cb$ub)) == n)) {
        stop("Inconsistent lengths among inference components.")
      }

      df <- data.frame(
        estimate = est,
        t_stat = t,
        p_value = p,
        ci_lb = ci$lb,
        ci_ub = ci$ub,
        cb_lb = cb$lb,
        cb_ub = cb$ub,
        stringsAsFactors = FALSE
      )

      if (!is.null(param_names)) {
        if (length(param_names) != n) stop(paste0("param_names must have length ", n))
        df <- cbind(parameter = param_names, df)
      } else {
        df <- cbind(parameter = seq_len(n), df)
      }
      rownames(df) <- NULL
      df
    },
    as.data.frame = function(...) self$as_data_frame(...),
    print = function(...) {
      df <- self$as_data_frame()
      print(df)
      invisible(self)
    }
  ),
  private = list(
    xp = NULL
  )
)

#' Compute bootstrap-based inference from raw inputs
#' @param point numeric vector of point estimates
#' @param boot numeric matrix p x B of bootstrap estimates
#' @param N integer sample size
#' @param sig_level significance level in (0,1)
#' @return SimultaneousInferenceResults
#' @export
get_bootstrap_inference <- function(point, boot, N, sig_level = 0.05) {
  xp <- get_bootstrap_inference_cpp(as.numeric(point),
                                    as.matrix(boot),
                                    as.integer(N),
                                    sig_level)
  SimultaneousInferenceResults$new(xp)
}

#' Target-parameter inference (single or by spec)
#' @param tpe TargetParameterEstimates or named list of them
#' @param panel an UnbalancedPanel
#' @param sig_level significance level in (0,1)
#' @return SimultaneousInferenceResults or named list (by spec) of them
#' @export
target_param_inference <- function(tpe, panel, sig_level = 0.05) {
  stopifnot(inherits(panel, "UnbalancedPanel"))
  holder_xp <- panel$get_panel_holder_xptr()

  if (inherits(tpe, "TargetParameterEstimates")) {
    xp <- target_param_inference_cpp(
      tpe$.__enclos_env__$private$xp,
      holder_xp,
      sig_level
    )
    return(SimultaneousInferenceResults$new(xp))
  }

  if (is.list(tpe)) {
    .validate_tpe_by_spec(tpe)
    tpe_xp_by_spec <- lapply(tpe, function(e) {
      stopifnot(inherits(e, "TargetParameterEstimates"))
      e$.__enclos_env__$private$xp
    })
    res <- target_param_inference_by_spec_cpp(
      tpe_by_spec = tpe_xp_by_spec,
      panel_holder_xptr = holder_xp,
      sig_level = sig_level
    )
    return(.wrap_sir_xptr_list(res))
  }

  stop("Invalid 'tpe': expected a TargetParameterEstimates object or a named list of them.")
}

.validate_tpe_by_spec <- function(x) {
  if (!is.list(x) || length(x) == 0L) {
    stop("When providing by-spec inputs, 'tpe' must be a non-empty named list.")
  }
  if (is.null(names(x)) || any(!nzchar(names(x)))) {
    stop("When providing by-spec inputs, 'tpe' must be a named list.")
  }
  ok <- vapply(x, function(e) inherits(e, "TargetParameterEstimates"), logical(1))
  if (!all(ok)) stop("All elements of 'tpe' must inherit from 'TargetParameterEstimates'.")
  invisible(TRUE)
}

.wrap_sir_xptr_list <- function(res_named_xptr_list) {
  nms <- names(res_named_xptr_list)
  out <- setNames(vector("list", length(res_named_xptr_list)), nms)
  for (i in seq_along(res_named_xptr_list)) {
    out[[i]] <- SimultaneousInferenceResults$new(res_named_xptr_list[[i]])
  }
  out
}