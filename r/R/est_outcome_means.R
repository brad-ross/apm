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
  if (inherits(cohort_weights, "CohortWeightEstimates")) {
    return(.agg_params_single(per_cohort_factor_estimates, observed_outcome_indices, cohort_weights))
  }
  if (is.list(cohort_weights) && is.list(per_cohort_factor_estimates)) {
    return(.agg_params_by_spec(per_cohort_factor_estimates, observed_outcome_indices, cohort_weights))
  }
  stop("Invalid inputs: expected either (1) a list of per-cohort FactorModelEstimates plus a CohortWeightEstimates, or (2) named lists for by-spec aggregation.")
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

.validate_fme_list <- function(x) {
  if (!is.list(x) || length(x) == 0L) {
    stop("'per_cohort_factor_estimates' must be a non-empty list of FactorModelEstimates (one per cohort).")
  }
  if (!all(vapply(x, function(e) inherits(e, "FactorModelEstimates"), logical(1)))) {
    stop("All elements of 'per_cohort_factor_estimates' must inherit from class 'FactorModelEstimates'.")
  }
  invisible(TRUE)
}

.validate_by_spec_fme_and_weights <- function(fme_by_spec, weights_by_spec) {
  if (is.null(names(fme_by_spec)) || any(!nzchar(names(fme_by_spec)))) {
    stop("When providing by-spec inputs, 'per_cohort_factor_estimates' must be a named list. Hint: use something like list(specA = list(...), specB = list(...)).")
  }
  if (is.null(names(weights_by_spec)) || any(!nzchar(names(weights_by_spec)))) {
    stop("When providing by-spec inputs, 'cohort_weights' must be a named list. Hint: use something like list(specA = CohortWeightEstimates(...), specB = ...).")
  }
  if (!all(vapply(fme_by_spec, is.list, logical(1)))) {
    stop("Each value of 'per_cohort_factor_estimates' must be a list of FactorModelEstimates (one per cohort for that spec).")
  }
  for (spec in names(fme_by_spec)) {
    vec <- fme_by_spec[[spec]]
    if (!all(vapply(vec, function(z) inherits(z, "FactorModelEstimates"), logical(1)))) {
      stop(sprintf("All elements under spec '%s' must inherit from class 'FactorModelEstimates'.", spec))
    }
  }
  if (!all(vapply(weights_by_spec, function(w) inherits(w, "CohortWeightEstimates"), logical(1)))) {
    stop("All elements of 'cohort_weights' must inherit from class 'CohortWeightEstimates'.")
  }
  spec_keys_a <- sort(names(fme_by_spec))
  spec_keys_b <- sort(names(weights_by_spec))
  if (!identical(spec_keys_a, spec_keys_b)) {
    missing_in_weights <- setdiff(spec_keys_a, spec_keys_b)
    missing_in_ests <- setdiff(spec_keys_b, spec_keys_a)
    msg <- c()
    if (length(missing_in_weights) > 0) msg <- c(msg, sprintf("missing in weights: %s", paste(missing_in_weights, collapse = ", ")))
    if (length(missing_in_ests) > 0) msg <- c(msg, sprintf("missing in estimates: %s", paste(missing_in_ests, collapse = ", ")))
    stop(sprintf("Spec key mismatch between 'per_cohort_factor_estimates' and 'cohort_weights' (%s).", paste(msg, collapse = "; ")))
  }
  invisible(TRUE)
}

.validate_suff_stat_list <- function(x) {
  if (!is.list(x) || length(x) == 0L) {
    stop("'suff_stat_estimates' must be a non-empty list of OutcomeMeanSuffStatEstimates (one per cohort).")
  }
  if (!all(vapply(x, function(e) inherits(e, "OutcomeMeanSuffStatEstimates"), logical(1)))) {
    stop("All elements of 'suff_stat_estimates' must inherit from class 'OutcomeMeanSuffStatEstimates'.")
  }
  invisible(TRUE)
}

.agg_params_single <- function(per_cohort_factor_estimates, observed_outcome_indices, cohort_weights) {
  .validate_fme_list(per_cohort_factor_estimates)
  xplist <- lapply(per_cohort_factor_estimates, function(fme) fme$.__enclos_env__$private$xp)
  xp_res <- aggregate_factor_model_params_cpp(xplist, observed_outcome_indices, cohort_weights$.__enclos_env__$private$xp)
  FactorModelEstimates$new(xp_res)
}

.agg_params_by_spec <- function(per_cohort_factor_estimates, observed_outcome_indices, cohort_weights) {
  .validate_by_spec_fme_and_weights(per_cohort_factor_estimates, cohort_weights)
  fact_map <- lapply(per_cohort_factor_estimates, function(vec) lapply(vec, function(fme) fme$.__enclos_env__$private$xp))
  w_map <- lapply(cohort_weights, function(w) w$.__enclos_env__$private$xp)
  res <- aggregate_factor_model_params_by_spec_cpp(fact_map, observed_outcome_indices, w_map)
  nms <- names(res)
  out <- setNames(vector("list", length(res)), nms)
  for (i in seq_along(res)) out[[i]] <- FactorModelEstimates$new(res[[i]])
  out
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


