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

#' Create Weighted Bootstrap Draws
#'
#' Convenience constructor for creating a \code{\link{WeightedBootstrap}} object.
#'
#' @description
#' This is the recommended way to create bootstrap weights for use throughout
#' the apm estimation pipeline. The resulting object can be passed to estimation
#' functions to enable bootstrap-based inference.
#'
#' @param N Integer; number of observations (units) in the sample.
#' @param B Integer; number of bootstrap draws to generate.
#' @param type Character; bootstrap type, either:
#'   \itemize{
#'     \item `"multinomial"` (default): Classical Efron bootstrap. Each draw
#'           samples N indices with replacement; weights are counts/N.
#'     \item `"bayesian"`: Rubin's Bayesian bootstrap. Weights are drawn from
#'           a Dirichlet(1,...,1) distribution (i.i.d. Exp(1), normalized).
#'   }
#' @param seed Optional integer; random seed for reproducibility. If provided,
#'   `set.seed()` is called before generating weights.
#'
#' @return A \code{\link{WeightedBootstrap}} R6 object with methods:
#'   `n_obs()`, `n_bootstraps()`, `draw(b)`, `obs(i)`, `obs_rows(idx)`, `weights()`.
#'
#' @seealso \code{\link{WeightedBootstrap}} for the returned class.
#' @seealso \code{\link{est_cohort_specific_params}} for using bootstrap in estimation.
#'
#' @examples
#' # Create 100 bootstrap draws for 50 observations
#' wb <- get_weighted_bootstrap_draws(N = 50, B = 100, type = "bayesian", seed = 123)
#' wb$n_obs()
#' wb$n_bootstraps()
#'
#' # Get weights for first bootstrap draw
#' w1 <- wb$draw(1)
#' sum(w1)  # Should equal 1
#'
#' @export
get_weighted_bootstrap_draws <- function(N, B, type = c("multinomial", "bayesian"), seed = NULL) {
  if (!is.null(seed)) set.seed(as.integer(seed))
  WeightedBootstrap$new(N, B, type, seed)
}

#' Simultaneous Inference Results from Bootstrap
#'
#' An R6 class that holds inference results computed from bootstrap replicates,
#' providing point estimates, standard errors, t-statistics, p-values, and
#' simultaneous confidence/credible bands.
#'
#' @description
#' `SimultaneousInferenceResults` wraps bootstrap-based inference computations
#' and provides both pointwise and simultaneous (FWER-controlling) inference.
#' Standard errors are computed using a robust IQR-based scale estimator.
#'
#' @details
#' The inference procedures implemented here follow the approach of using
#' bootstrap quantiles to construct:
#' \itemize{
#'   \item **Pointwise confidence intervals (CI)**: For each parameter individually.
#'   \item **Simultaneous confidence bands (CB)**: Controlling family-wise error rate
#'         across all parameters jointly.
#' }
#'
#' Standard errors are computed as `IQR / 1.349`, which is a robust estimator
#' of the standard deviation for normally distributed data (1.349 is the 
#' IQR of the standard normal distribution).
#'
#' @section Constructor:
#' This class is typically not constructed directly by users. Instead, use
#' \code{\link{get_bootstrap_inference}} or obtain results from
#' \code{\link{target_param_inference}}.
#'
#' @section Public Methods:
#' \describe{
#'   \item{\code{point()}}{Returns numeric vector of point estimates (length p).}
#'   \item{\code{t_stats()}}{Returns numeric vector of t-statistics (length p).}
#'   \item{\code{p_vals()}}{Returns numeric vector of pointwise two-sided p-values
#'         (length p).}
#'   \item{\code{std_error()}}{Returns numeric vector of robust standard errors
#'         (length p).}
#'   \item{\code{se()}}{Alias for `std_error()`.}
#'   \item{\code{fwer_control_p_vals()}}{Returns numeric vector of Romano-Wolf
#'         stepdown adjusted p-values (length p), controlling the family-wise
#'         error rate (FWER).}
#'   \item{\code{sig_level()}}{Returns the significance level used for inference.}
#'   \item{\code{ci()}}{Returns a list with `lb` (lower bound) and `ub` (upper bound)
#'         numeric vectors for pointwise confidence intervals.}
#'   \item{\code{cb()}}{Returns a list with `lb` and `ub` numeric vectors for
#'         simultaneous confidence bands.}
#'   \item{\code{as_data_frame(param_names = NULL)}}{Returns a data.frame with
#'         columns: `parameter`, `estimate`, `std_error`, `t_stat`, `p_value`,
#'         `p_value_fwer` (Romano-Wolf stepdown adjusted), `ci_lb`, `ci_ub`,
#'         `cb_lb`, `cb_ub`. If `param_names` is provided, it is used for the
#'         `parameter` column; otherwise 1-based indices are used.}
#'   \item{\code{as.data.frame(...)}}{Alias for `as_data_frame(...)`.}
#'   \item{\code{print(...)}}{Prints the results as a data.frame.}
#' }
#'
#' @seealso \code{\link{get_bootstrap_inference}} for constructing from raw inputs.
#' @seealso \code{\link{target_param_inference}} for inference on target parameters.
#' @seealso \code{\link{combine_inference_results_across_specs}} for combining
#'   results across multiple estimation specifications.
#'
#' @examples
#' # Typically obtained from target_param_inference or get_bootstrap_inference
#' # Example with synthetic data:
#' set.seed(123)
#' point <- c(1.0, 2.0, 3.0)
#' boot <- matrix(rnorm(300, mean = rep(point, each = 100), sd = 0.5),
#'                nrow = 3, ncol = 100, byrow = TRUE)
#' sir <- get_bootstrap_inference(point, boot, N = 50, sig_level = 0.05)
#'
#' # Access results
#' sir$point()
#' sir$std_error()
#' sir$ci()
#' sir$as_data_frame(param_names = c("alpha", "beta", "gamma"))
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
    std_error = function() sir_std_error_cpp(private$xp),
    se = function() self$std_error(),
    fwer_control_p_vals = function() sir_fwer_control_p_cpp(private$xp),
    sig_level = function() sir_sig_level_cpp(private$xp),
    ci = function() list(lb = sir_ci_lb_cpp(private$xp),
                         ub = sir_ci_ub_cpp(private$xp)),
    cb = function() list(lb = sir_cb_lb_cpp(private$xp),
                         ub = sir_cb_ub_cpp(private$xp)),
    as_data_frame = function(param_names = NULL) {
      est <- self$point()
      t <- self$t_stats()
      p <- self$p_vals()
      p_fwer <- self$fwer_control_p_vals()
      se <- self$std_error()
      ci <- self$ci()
      cb <- self$cb()

      n <- length(est)
      if (!all(lengths(list(t, p, p_fwer, se, ci$lb, ci$ub, cb$lb, cb$ub)) == n)) {
        stop("Inconsistent lengths among inference components.")
      }

      df <- data.frame(
        estimate = est,
        std_error = se,
        t_stat = t,
        p_value = p,
        p_value_fwer = p_fwer,
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

#' Compute Bootstrap-Based Inference from Raw Inputs
#'
#' Constructs a `SimultaneousInferenceResults` object from point estimates and
#' a matrix of bootstrap replicates.
#'
#' @description
#' This function computes pointwise and simultaneous inference (confidence
#' intervals, confidence bands, t-statistics, p-values) from user-provided
#' bootstrap replicates.
#'
#' @param point Numeric vector of point estimates (length p).
#' @param boot Numeric matrix of bootstrap estimates with dimensions p x B,
#'   where p is the number of parameters and B is the number of bootstrap draws.
#'   Each column represents one bootstrap replicate.
#' @param N Integer; the sample size used for computing standard errors.
#' @param sig_level Numeric; significance level for confidence intervals and
#'   bands, must be in (0, 1). Default is 0.05 for 95% intervals/bands.
#'
#' @return A \code{\link{SimultaneousInferenceResults}} R6 object with methods
#'   for accessing point estimates, standard errors, t-statistics, p-values,
#'   and confidence intervals/bands.
#'
#' @seealso \code{\link{SimultaneousInferenceResults}} for the returned object.
#' @seealso \code{\link{target_param_inference}} for inference on estimated
#'   target parameters.
#'
#' @examples
#' # Simulate bootstrap replicates
#' set.seed(42)
#' true_params <- c(1.5, -0.5)
#' point_est <- true_params + rnorm(2, sd = 0.1)
#' boot_reps <- matrix(
#'   rnorm(2 * 100, mean = rep(point_est, 100), sd = 0.2),
#'   nrow = 2, ncol = 100
#' )
#'
#' # Compute inference
#' results <- get_bootstrap_inference(point_est, boot_reps, N = 100, sig_level = 0.05)
#' results$point()
#' results$std_error()
#' results$ci()
#'
#' @export
get_bootstrap_inference <- function(point, boot, N, sig_level = 0.05) {
  xp <- get_bootstrap_inference_cpp(as.numeric(point),
                                    as.matrix(boot),
                                    as.integer(N),
                                    sig_level)
  SimultaneousInferenceResults$new(xp)
}

#' Combine Inference Results Across Estimation Specifications
#'
#' Stacks inference results from multiple estimation specifications into a single
#' data.frame, adding a `spec` column to identify which specification each row
#' belongs to.
#'
#' @description
#' When running estimation with multiple specifications (e.g., different model
#' ranks or weighting schemes), this function combines the inference results
#' into a single tidy data.frame for easy comparison and reporting.
#'
#' @param results_by_spec A named list of \code{\link{SimultaneousInferenceResults}}
#'   objects, one per estimation specification. Names become values in the `spec`
#'   column of the output.
#'
#' @return A data.frame with columns:
#'   \describe{
#'     \item{spec}{Character; the specification name (from list names).}
#'     \item{parameter}{Integer or character; parameter identifier.}
#'     \item{estimate}{Numeric; point estimate.}
#'     \item{std_error}{Numeric; robust standard error.}
#'     \item{t_stat}{Numeric; t-statistic.}
#'     \item{p_value}{Numeric; pointwise p-value.}
#'     \item{p_value_fwer}{Numeric; Romano-Wolf stepdown adjusted p-value
#'           (FWER-controlling).}
#'     \item{ci_lb}{Numeric; pointwise confidence interval lower bound.}
#'     \item{ci_ub}{Numeric; pointwise confidence interval upper bound.}
#'     \item{cb_lb}{Numeric; simultaneous confidence band lower bound.}
#'     \item{cb_ub}{Numeric; simultaneous confidence band upper bound.}
#'   }
#'
#' @seealso \code{\link{SimultaneousInferenceResults}} for individual results.
#' @seealso \code{\link{target_param_inference}} for computing inference.
#'
#' @examples
#' # Create mock inference results for two specifications
#' set.seed(1)
#' point1 <- c(1.0, 2.0)
#' boot1 <- matrix(rnorm(200, mean = rep(point1, 100)), nrow = 2)
#' sir1 <- get_bootstrap_inference(point1, boot1, N = 50)
#'
#' point2 <- c(1.1, 1.9)
#' boot2 <- matrix(rnorm(200, mean = rep(point2, 100)), nrow = 2)
#' sir2 <- get_bootstrap_inference(point2, boot2, N = 50)
#'
#' # Combine results
#' combined <- combine_inference_results_across_specs(
#'   list(spec_A = sir1, spec_B = sir2)
#' )
#' print(combined)
#'
#' @export
combine_inference_results_across_specs <- function(results_by_spec) {
  if (!is.list(results_by_spec) || length(results_by_spec) == 0L) {
    stop("results_by_spec must be a non-empty named list of SimultaneousInferenceResults")
  }
  if (is.null(names(results_by_spec)) || any(!nzchar(names(results_by_spec)))) {
    stop("results_by_spec must be a named list (names are spec identifiers)")
  }

  nms <- names(results_by_spec)
  dfs <- vector("list", length(results_by_spec))
  for (i in seq_along(results_by_spec)) {
    sir <- results_by_spec[[i]]
    if (!inherits(sir, "SimultaneousInferenceResults")) {
      stop("All elements of results_by_spec must inherit 'SimultaneousInferenceResults'")
    }
    df_i <- as.data.frame(sir)
    df_i <- cbind(data.frame(spec = nms[[i]], stringsAsFactors = FALSE), df_i)
    dfs[[i]] <- df_i
  }

  out <- do.call(rbind, dfs)
  rownames(out) <- NULL
  out
}