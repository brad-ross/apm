#' Outcome Means Estimates Container
#'
#' An R6 class that holds estimated mean outcomes for each cohort across all
#' outcomes, with optional bootstrap replicates.
#'
#' @description
#' `OutcomeMeansEstimates` stores a C x T matrix of estimated mean outcomes
#' where C is the number of cohorts and T is the total number of outcomes.
#' Each row represents a cohort's estimated mean outcome for each time period
#' or outcome index.
#'
#' These estimates are computed by imputing unobserved outcomes using the
#' factor model and combining with observed outcome means.
#'
#' @details
#' This class is typically not constructed directly. It is returned by:
#' \itemize{
#'   \item \code{\link{estimate_outcome_means_across_cohorts}}
#'   \item \code{\link{est_target_param_components}}
#' }
#'
#' The estimated means serve as inputs to target parameter estimation
#' (see \code{\link{est_target_params}}).
#'
#' @section Public Methods:
#' \describe{
#'   \item{\code{has_bootstrap()}}{Logical; whether bootstrap replicates exist.}
#'   \item{\code{num_bootstraps()}}{Integer; number of bootstrap replicates (0 if none).}
#'   \item{\code{C()}}{Integer; number of cohorts.}
#'   \item{\code{T()}}{Integer; number of outcomes.}
#'   \item{\code{mean_outcomes(b = NULL)}}{Returns C x T matrix of estimated mean
#'         outcomes. If `b` is NULL, returns point estimate; otherwise returns
#'         bootstrap replicate `b` (1-indexed).}
#' }
#'
#' @seealso \code{\link{estimate_outcome_means_across_cohorts}} for computing
#'   outcome means from factor estimates.
#' @seealso \code{\link{est_target_params}} for using outcome means in target
#'   parameter estimation.
#'
#' @examples
#' # OutcomeMeansEstimates is typically obtained from estimation functions
#' \dontrun{
#' ome <- estimate_outcome_means_across_cohorts(
#'   factor_model_estimates,
#'   observed_outcome_indices,
#'   suff_stat_estimates
#' )
#' ome$C()             # Number of cohorts
#' ome$T()             # Number of outcomes
#' ome$mean_outcomes() # C x T matrix
#' }
#'
#' @export
OutcomeMeansEstimates <- R6::R6Class(
  "OutcomeMeansEstimates",
  public = list(
    initialize = function(xptr) { private$xp <- xptr },
    has_bootstrap = function() ome_has_bootstrap_cpp(private$xp),
    num_bootstraps = function() ome_num_bootstrap_cpp(private$xp),
    C = function() ome_C_cpp(private$xp),
    T = function() ome_T_cpp(private$xp),
    mean_outcomes = function(b = NULL) {
      if (is.null(b)) ome_point_means_cpp(private$xp)
      else ome_boot_means_cpp(private$xp, as.integer(b))
    }
  ),
  private = list(xp = NULL)
)

#' Estimate Outcome Means Across Cohorts
#'
#' Computes cohort-by-outcome mean estimates by combining aggregated factor model
#' parameters with cohort-specific sufficient statistics. Unobserved outcomes
#' are imputed using the factor model.
#'
#' @description
#' This function takes aggregated factor model estimates and cohort-level sufficient
#' statistics to produce a C x T matrix of estimated mean outcomes for each cohort.
#' For outcomes that a cohort observes, the observed mean is used; for unobserved
#' outcomes, the factor model is used to impute the expected mean.
#'
#' @details
#' The imputation formula depends on the factor model specification:
#' \itemize{
#'   \item **Factors only**: \eqn{\hat{Y}_{ct} = G_t' \hat{\lambda}_c}
#'   \item **With fixed effects**: \eqn{\hat{Y}_{ct} = g_{0t} + G_t' \hat{\lambda}_c}
#'   \item **With covariates**: \eqn{\hat{Y}_{ct} = G_t' \hat{\lambda}_c + X_{ct}' a}
#' }
#'
#' where \eqn{\hat{\lambda}_c} is the estimated cohort mean loading, derived from
#' observed outcomes.
#'
#' When bootstrap replicates are present in both inputs, outcome means are computed
#' for each bootstrap draw, enabling subsequent bootstrap inference.
#'
#' @param factor_model_estimates A \code{\link{FactorModelEstimates}} R6 object
#'   containing aggregated factor estimates, or a named list of such objects
#'   (for by-spec computation).
#' @param observed_outcome_indices List of integer vectors; element c contains
#'   the 1-based outcome indices observed by cohort c.
#' @param suff_stat_estimates List of \code{\link{OutcomeMeanSuffStatEstimates}}
#'   R6 objects (one per cohort) containing cohort-level sufficient statistics.
#'
#' @return For a single `FactorModelEstimates` input: an
#'   \code{\link{OutcomeMeansEstimates}} R6 object.
#'
#'   For a named list input: a named list of `OutcomeMeansEstimates` objects
#'   (one per specification).
#'
#' @seealso \code{\link{OutcomeMeansEstimates}} for the returned object.
#' @seealso \code{\link{aggregate_factor_model_params}} for aggregating cohort
#'   estimates.
#' @seealso \code{\link{est_target_params}} for using outcome means in target
#'   parameter estimation.
#'
#' @examples
#' # This function is typically called as part of the estimation pipeline
#' \dontrun{
#' # After obtaining aggregated factor estimates and cohort stats
#' ome <- estimate_outcome_means_across_cohorts(
#'   factor_model_estimates = agg_params,
#'   observed_outcome_indices = panel$get_observed_outcome_indices(),
#'   suff_stat_estimates = cohort_stats
#' )
#'
#' # Access results
#' Y_hat <- ome$mean_outcomes()  # C x T matrix
#' dim(Y_hat)
#' }
#'
#' @export
estimate_outcome_means_across_cohorts <- function(factor_model_estimates, observed_outcome_indices, suff_stat_estimates) {
  .validate_suff_stat_list(suff_stat_estimates)
  if (inherits(factor_model_estimates, "FactorModelEstimates")) {
    return(.estimate_means_single(factor_model_estimates, observed_outcome_indices, suff_stat_estimates))
  }
  if (is.list(factor_model_estimates)) {
    return(.estimate_means_by_spec(factor_model_estimates, observed_outcome_indices, suff_stat_estimates))
  }
  stop("Invalid 'factor_model_estimates': expected a FactorModelEstimates object or a named list of them.")
}

# -----------------------------------------------------------------------------
# Internal helpers (not exported)
# -----------------------------------------------------------------------------

.validate_suff_stat_list <- function(x) {
  if (!is.list(x) || length(x) == 0L) {
    stop("'suff_stat_estimates' must be a non-empty list of OutcomeMeanSuffStatEstimates (one per cohort).")
  }
  if (!all(vapply(x, function(e) inherits(e, "OutcomeMeanSuffStatEstimates"), logical(1)))) {
    stop("All elements of 'suff_stat_estimates' must inherit from class 'OutcomeMeanSuffStatEstimates'.")
  }
  invisible(TRUE)
}

.estimate_means_single <- function(factor_model_estimates, observed_outcome_indices, suff_stat_estimates) {
  xplist <- lapply(suff_stat_estimates, function(om) om$.__enclos_env__$private$xp)
  xp_res <- estimate_outcome_means_across_cohorts_cpp(factor_model_estimates$.__enclos_env__$private$xp, observed_outcome_indices, xplist)
  OutcomeMeansEstimates$new(xp_res)
}

.estimate_means_by_spec <- function(factor_model_estimates, observed_outcome_indices, suff_stat_estimates) {
  if (is.null(names(factor_model_estimates)) || any(!nzchar(names(factor_model_estimates)))) {
    stop("When providing by-spec inputs, 'factor_model_estimates' must be a named list. Hint: use something like list(specA = FactorModelEstimates(...), specB = ...).")
  }
  if (!all(vapply(factor_model_estimates, function(x) inherits(x, "FactorModelEstimates"), logical(1)))) {
    stop("All elements of 'factor_model_estimates' must inherit from class 'FactorModelEstimates'.")
  }
  fmap <- lapply(factor_model_estimates, function(fme) fme$.__enclos_env__$private$xp)
  xplist <- lapply(suff_stat_estimates, function(om) om$.__enclos_env__$private$xp)
  res <- estimate_outcome_means_across_cohorts_by_spec_cpp(fmap, observed_outcome_indices, xplist)
  nms <- names(res)
  out <- setNames(vector("list", length(res)), nms)
  for (i in seq_along(res)) out[[i]] <- OutcomeMeansEstimates$new(res[[i]])
  out
}