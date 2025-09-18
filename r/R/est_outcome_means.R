#' R6 wrapper for OutcomeMeansEstimates
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

#' Estimate outcome means across cohorts (single spec; R6 estimates API)
#'
#' Computes cohort mean outcomes using a single `FactorModelEstimates` object and
#' per-cohort `OutcomeMeanSuffStatEstimates`. If bootstrap replicates are present
#' in both inputs, outcomes are computed per bootstrap draw.
#'
#' @param factor_model_estimates a `FactorModelEstimates` R6 object with point
#'   parameters and optional bootstrap replicates.
#' @param observed_outcome_indices list of integer vectors (1-based indices) of
#'   observed outcomes for each cohort.
#' @param suff_stat_estimates list of `OutcomeMeanSuffStatEstimates` R6 objects
#'   (one per cohort).
#'
#' @return An `OutcomeMeansEstimates` R6 object with fields:
#'   - `mean_outcomes()` → C x T matrix (point) or replicate b via `mean_outcomes(b)`
#'   - `has_bootstrap()`, `num_bootstraps()`
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
# End-to-end wrapper
# -----------------------------------------------------------------------------
#' End-to-end: estimate target parameter components across cohorts (by spec)
#'
#' This runs cohort-specific estimation and immediately aggregates/estimates cohort
#' mean outcomes across cohorts, returning a list with:
#' - outcome_means: named list (by spec) of `OutcomeMeansEstimates` objects
#' - auxiliary_means: list of `CohortAuxiliaryDataMeanEstimates` (one per cohort), or NULL if none
#'
#' @inheritParams est_cohort_specific_params
#' @return list with `outcome_means` and `auxiliary_means`.
#' @export
est_target_param_components <- function(panel, est_specs, bootstrap = NULL, num_threads = NULL,
                                       cohort_outcomes_to_mask = NULL) {
  stopifnot(inherits(panel, "UnbalancedPanel"))
  if (!is.null(bootstrap) && !inherits(bootstrap, "WeightedBootstrap")) stop("bootstrap must be a WeightedBootstrap or NULL")
  .validate_est_specs(est_specs)

  holder_xp <- panel$get_panel_holder_xptr()

  validate_mask_arg(cohort_outcomes_to_mask)

  xp <- if (is.null(bootstrap)) NULL else bootstrap$.__enclos_env__$private$xp
  nt <- if (is.null(num_threads)) NULL else as.integer(num_threads)

  res <- est_target_param_components_from_panel_cpp(
    panel_holder_xptr = holder_xp,
    est_specs = est_specs,
    bootstrap_xptr = xp,
    num_threads_in = nt,
    cohort_outcomes_to_mask_in = cohort_outcomes_to_mask
  )
  out <- list(
    outcome_means = .wrap_outcome_means_xptr_list(res$outcome_means),
    auxiliary_means = NULL
  )
  if (!is.null(res$auxiliary_means)) {
    out$auxiliary_means <- lapply(res$auxiliary_means, function(xp) CohortAuxiliaryDataMeanEstimates$new(xp))
  }
  if (!is.null(res$masked_cohort_outcome_means)) {
    out$masked_cohort_outcome_means <- res$masked_cohort_outcome_means
  }
  if (!is.null(res$masked_observed_outcome_indices)) {
    out$masked_observed_outcome_indices <- res$masked_observed_outcome_indices
  }
  out
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