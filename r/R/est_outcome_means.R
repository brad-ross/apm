#' Aggregate factor model parameters across cohorts
#'
#' Aggregates per-cohort factor model parameter estimates (factors `G`, optional
#' outcome fixed effects `g0`, and optional covariate coefficients `a`) into a
#' single set of parameters. Bootstrap replicates, when present in the inputs and
#' weights, are aggregated per draw.
#'
#' @param per_cohort_factor_estimates list of `FactorModelEstimates` R6 objects
#'   (length = number of cohorts). Each element must carry point estimates and
#'   optionally bootstrap replicates with consistent `B` across cohorts.
#' @param observed_outcome_indices list of integer vectors (1-based indices) of
#'   observed outcomes for each cohort.
#' @param cohort_weights a `CohortWeightEstimates` R6 object containing point
#'   cohort weights and, when applicable, per-bootstrap weight vectors.
#'
#' @return A `FactorModelEstimates` R6 object containing aggregated parameters.
#'
#' @examples
#' # See tests or vignette for an end-to-end example
#'
#' @export
aggregate_factor_model_params <- function(per_cohort_factor_estimates, observed_outcome_indices, cohort_weights) {
  stopifnot(is.list(per_cohort_factor_estimates))
  xplist <- lapply(per_cohort_factor_estimates, function(fme) fme$.__enclos_env__$private$xp)
  xp_res <- aggregate_factor_model_params_cpp(xplist, observed_outcome_indices, cohort_weights$.__enclos_env__$private$xp)
  FactorModelEstimates$new(xp_res)
}

#' Aggregate factor model parameters by estimator specification
#'
#' For each estimator specification, aggregates per-cohort parameter estimates
#' using that spec's weights. Both maps must have identical keys. Bootstrap
#' replicates are aggregated per draw.
#'
#' @param cohort_specific_factor_ests named list whose values are lists of
#'   `FactorModelEstimates` R6 objects (per-cohort for that spec).
#' @param observed_outcome_indices list of integer vectors (1-based indices) of
#'   observed outcomes for each cohort.
#' @param cohort_weights_by_spec named list of `CohortWeightEstimates` R6 objects
#'   (one per spec).
#'
#' @return Named list whose values are `FactorModelEstimates` R6 objects.
#'
#' @export
aggregate_factor_model_params_by_spec <- function(cohort_specific_factor_ests, observed_outcome_indices, cohort_weights_by_spec) {
  stopifnot(is.list(cohort_specific_factor_ests), is.list(cohort_weights_by_spec))
  # Convert R6 to XPtr structures while preserving names
  fact_map <- lapply(cohort_specific_factor_ests, function(vec) lapply(vec, function(fme) fme$.__enclos_env__$private$xp))
  w_map <- lapply(cohort_weights_by_spec, function(w) w$.__enclos_env__$private$xp)
  res <- aggregate_factor_model_params_by_spec_cpp(fact_map, observed_outcome_indices, w_map)
  # Wrap each result in R6
  nms <- names(res)
  out <- setNames(vector("list", length(res)), nms)
  for (i in seq_along(res)) out[[i]] <- FactorModelEstimates$new(res[[i]])
  out
}

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
  xplist <- lapply(suff_stat_estimates, function(om) om$.__enclos_env__$private$xp)
  xp_res <- estimate_outcome_means_across_cohorts_cpp(factor_model_estimates$.__enclos_env__$private$xp, observed_outcome_indices, xplist)
  OutcomeMeansEstimates$new(xp_res)
}

#' Estimate outcome means across cohorts by estimator specification (R6 estimates API)
#'
#' For each estimator specification, computes cohort mean outcomes using that
#' spec's `FactorModelEstimates`. Requires that the sufficient statistics
#' estimates list has one entry per cohort.
#'
#' @param factor_model_estimates_by_spec named list of `FactorModelEstimates` R6
#'   objects (one per spec).
#' @param observed_outcome_indices list of integer vectors (1-based indices) of
#'   observed outcomes for each cohort.
#' @param suff_stat_estimates list of `OutcomeMeanSuffStatEstimates` R6 objects
#'   (one per cohort).
#'
#' @return Named list of `OutcomeMeansEstimates` R6 objects.
#'
#' @export
estimate_outcome_means_across_cohorts_by_spec <- function(factor_model_estimates_by_spec, observed_outcome_indices, suff_stat_estimates) {
  fmap <- lapply(factor_model_estimates_by_spec, function(fme) fme$.__enclos_env__$private$xp)
  xplist <- lapply(suff_stat_estimates, function(om) om$.__enclos_env__$private$xp)
  res <- estimate_outcome_means_across_cohorts_by_spec_cpp(fmap, observed_outcome_indices, xplist)
  nms <- names(res)
  out <- setNames(vector("list", length(res)), nms)
  for (i in seq_along(res)) out[[i]] <- OutcomeMeansEstimates$new(res[[i]])
  out
}


