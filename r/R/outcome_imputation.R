#' Compute Imputation Components from Factor Estimates
#'
#' Updates factor model estimates with cohort mean loadings (L) and optional
#' outcome fixed effects g0 and covariate coefficients a (when present) computed via
#' regression imputation, enabling direct computation of cohort outcome means.
#'
#' @description
#' This function takes aggregated factor model estimates and computes the
#' cohort mean loadings L, optional outcome fixed effects g0, and covariate coefficients a 
#' by solving a regression problem with unit-varying slopes, outcome fixed effects, and
#' covariates. The resulting estimates can be used directly for outcome mean computation
#' without requiring the raw panel data.
#'
#' @details
#' **Imputation Process:**
#'
#' Given factors G (T x r) and optional fixed effects g0, covariate coefficients a, and optional
#' covariate means X_c (T x q) for each cohort c, the function estimates cohort mean 
#' loadings L (C x r), outcome fixed effects g0 (T), and covariate coefficients a (q) such that:
#'
#' \deqn{\hat{Y}_c = g_0 + G \lambda_c + X_c a}
#'
#' matches the observed outcome means for each cohort c on the outcomes that
#' cohort observes.
#'
#' **Imputation Options:**
#'
#' The `imputation_options` argument is a named list controlling the algorithm
#' that solves the regression problem. All fields are optional; missing fields
#' use defaults.
#'
#' *General options:*
#' \describe{
#'   \item{tol}{Convergence tolerance (default: 1e-12).}
#'   \item{max_iters}{Maximum iterations (default: unlimited).}
#'   \item{method}{Acceleration method: `"irons-tuck"` (default) or `"none"`.}
#'   \item{solver}{Solver choice: `"fixed-point"` (default) or `"lsmr"`.}
#' }
#'
#' *Advanced fixed-point options:*
#' \describe{
#'   \item{grand_period}{Grand acceleration period; 0 disables (default: 0).}
#'   \item{grand_k}{Exponent for grand acceleration (default: 4).}
#'   \item{stabilize_after}{Iteration to start stabilization; 0 disables (default: 0).}
#'   \item{extra_proj}{Extra projections before acceleration (default: 0).}
#' }
#'
#' *LSMR solver options (only used when `solver = "lsmr"`):*
#' \describe{
#'   \item{lsmr_diagonal_precond}{Enable diagonal preconditioning (default: FALSE).}
#'   \item{lsmr_num_diag_approx_draws}{Hutchinson draws for diagonal preconditioner;
#'         0 = exact computation (default: 0).}
#'   \item{lsmr_homotopy_iters}{Number of homotopy stages; 0 disables (default: 0).}
#'   \item{lsmr_atol}{Relative tolerance on \eqn{\|A^T r\|} (default: 1e-6).}
#'   \item{lsmr_btol}{Relative tolerance on \eqn{\|r\|} (default: 1e-6).}
#'   \item{lsmr_conlim}{Condition number limit (default: 1e+8).}
#'   \item{lsmr_max_iters}{Maximum LSMR iterations; 0 = auto (default: 0).}
#'   \item{lsmr_lambda}{Tikhonov damping parameter; 0 disables (default: 0).}
#' }
#'
#' @param panel An \code{\link{UnbalancedPanel}} R6 object.
#' @param factor_model_estimates A \code{\link{FactorModelEstimates}} R6 object
#'   containing aggregated factors G and optional g0, a. Or a named list of
#'   such objects (for by-spec computation).
#' @param cohort_outcome_mean_suff_stat_ests Optional list (one per cohort) of
#'   \code{\link{OutcomeMeanSuffStatEstimates}} objects.
#' @param weighted_bootstrap Optional \code{\link{WeightedBootstrap}} R6 object
#'   for computing bootstrap replicate loadings.
#' @param effective_observed_outcome_indices Optional list of 1-based outcome
#'   indices to override the panel's default. Useful when masking outcomes.
#' @param num_threads Optional integer; number of threads for parallel computation.
#' @param imputation_options Optional named list of algorithm options. See Details.
#'
#' @return For single-spec input: a \code{\link{FactorModelEstimates}} R6 object
#'   with L (cohort mean loadings) populated.
#'
#'   For by-spec input: a named list of such objects.
#'
#' @seealso \code{\link{impute_outcomes_across_cohorts}} for computing outcome
#'   means from the returned estimates.
#' @seealso \code{\link{estimate_outcome_means_across_cohorts}} for the complete
#'   outcome mean estimation pipeline.
#'
#' @examples
#' \dontrun{
#' # After aggregating factor estimates
#' fme_with_L <- comp_imputation_components(
#'   panel = panel,
#'   factor_model_estimates = agg_fme
#' )
#'
#' # The result now has L populated
#' L <- fme_with_L$L()  # C x r matrix
#' }
#'
#' @export
comp_imputation_components <- function(panel,
                                       factor_model_estimates,
                                       cohort_outcome_mean_suff_stat_ests = NULL,
                                       weighted_bootstrap = NULL,
                                       effective_observed_outcome_indices = NULL,
                                       num_threads = NULL,
                                       imputation_options = NULL) {
  stopifnot(inherits(panel, "UnbalancedPanel"))
  holder_xp <- panel$get_panel_holder_xptr()
  nt <- if (is.null(num_threads)) NULL else as.integer(num_threads)

  if (inherits(factor_model_estimates, "FactorModelEstimates")) {
    xplist <- if (is.null(cohort_outcome_mean_suff_stat_ests)) NULL else lapply(cohort_outcome_mean_suff_stat_ests, function(om) om$.__enclos_env__$private$xp)
    xp_res <- comp_imputation_components_cpp(
      panel_holder_xptr = holder_xp,
      factor_model_estimates_xptr = factor_model_estimates$.__enclos_env__$private$xp,
      cohort_outcome_mean_suff_stat_ests = xplist,
      weighted_bootstrap_xptr = if (is.null(weighted_bootstrap)) NULL else weighted_bootstrap$.__enclos_env__$private$xp,
      effective_observed_outcome_indices = effective_observed_outcome_indices,
      imputation_options_in = imputation_options,
      num_threads_in = nt
    )
    return(FactorModelEstimates$new(xp_res))
  }

  if (is.list(factor_model_estimates)) {
    if (is.null(names(factor_model_estimates)) || any(!nzchar(names(factor_model_estimates)))) {
      stop("When providing by-spec inputs, 'factor_model_estimates' must be a named list.")
    }
    if (!all(vapply(factor_model_estimates, function(x) inherits(x, "FactorModelEstimates"), logical(1)))) {
      stop("All elements of 'factor_model_estimates' must inherit from class 'FactorModelEstimates'.")
    }
    fmap <- lapply(factor_model_estimates, function(fme) fme$.__enclos_env__$private$xp)
    xplist <- if (is.null(cohort_outcome_mean_suff_stat_ests)) NULL else lapply(cohort_outcome_mean_suff_stat_ests, function(om) om$.__enclos_env__$private$xp)
    res <- comp_imputation_components_by_spec_cpp(
      panel_holder_xptr = holder_xp,
      factor_model_estimates_by_spec = fmap,
      cohort_outcome_mean_suff_stat_ests = xplist,
      weighted_bootstrap_xptr = if (is.null(weighted_bootstrap)) NULL else weighted_bootstrap$.__enclos_env__$private$xp,
      effective_observed_outcome_indices = effective_observed_outcome_indices,
      imputation_options_in = imputation_options,
      num_threads_in = nt
    )
    nms <- names(res)
    out <- setNames(vector("list", length(res)), nms)
    for (i in seq_along(res)) out[[i]] <- FactorModelEstimates$new(res[[i]])
    return(out)
  }

  stop("Invalid 'factor_model_estimates': expected a FactorModelEstimates object or a named list of them.")
}


