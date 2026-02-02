#' Aggregate Factor Model Parameters Across Cohorts
#'
#' Aggregates per-cohort factor model parameter estimates into a single set of
#' panel-level parameters using the Aggregated Projection Matrix (APM) method.
#'
#' @description
#' This function combines cohort-specific factor estimates (G matrices, optional
#' outcome fixed effects g0, optional covariate coefficients a) into aggregated
#' panel-level parameters. The APM method aligns factors across cohorts that
#' observe different subsets of outcomes.
#'
#' @details
#' **Aggregation Method:**
#'
#' The APM method works as follows:
#' \enumerate{
#'   \item Construct a weighted aggregated projection matrix from cohort-specific
#'         factor matrices.
#'   \item Compute an orthonormal basis for the estimated null space of this matrix.
#'   \item This basis forms the aggregated factor matrix G.
#' }
#'
#' **Component Aggregation:**
#' \itemize{
#'   \item **Factors G**: Aligned using the APM method to produce a T x r matrix.
#'   \item **Fixed effects g0**: Weighted average across cohorts for each outcome.
#'   \item **Covariate coefficients a**: Weighted average across cohorts.
#' }
#'
#' When bootstrap replicates are present, aggregation is performed independently
#' for each bootstrap draw using the corresponding bootstrap weights.
#'
#' @param per_cohort_factor_estimates List of \code{\link{FactorModelEstimates}}
#'   R6 objects (one per cohort), or a named list (by-spec) of such lists.
#'   Each element must have consistent bootstrap counts (all have B replicates
#'   or none have bootstrap).
#' @param observed_outcome_indices List of integer vectors; element c contains
#'   the 1-based outcome indices observed by cohort c.
#' @param cohort_weights A \code{\link{CohortWeightEstimates}} R6 object for
#'   single-spec aggregation, or a named list of such objects (by-spec) matching
#'   the specification names in `per_cohort_factor_estimates`.
#'
#' @return For single-spec input: a \code{\link{FactorModelEstimates}} R6 object
#'   containing aggregated parameters.
#'
#'   For by-spec input: a named list of `FactorModelEstimates` objects (one per
#'   specification).
#'
#' @seealso \code{\link{align_factors_using_apm}} for the underlying alignment method.
#' @seealso \code{\link{compute_aggregated_projection_matrix}} for the APM computation.
#' @seealso \code{\link{est_cohort_specific_params}} for obtaining cohort estimates.
#' @seealso \code{\link{estimate_outcome_means_across_cohorts}} for using aggregated
#'   parameters.
#'
#' @examples
#' # This function is typically used in a pipeline
#' \dontrun{
#' # After cohort-specific estimation
#' results <- est_cohort_specific_params(panel, specs)
#'
#' # Aggregate parameters
#' agg <- aggregate_factor_model_params(
#'   per_cohort_factor_estimates = results$cohort_specific_factor_ests$my_spec,
#'   observed_outcome_indices = panel$get_observed_outcome_indices(),
#'   cohort_weights = results$cohort_weights$my_spec
#' )
#'
#' # Access aggregated factors
#' G <- agg$G()  # T x r matrix
#' }
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