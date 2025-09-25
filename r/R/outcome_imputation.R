#' Compute imputation components (G, optional a, g0, and L) given panel and estimates
#'
#' For a single spec, returns a `FactorModelEstimates` R6 object with updated
#' parameters (point and bootstrap). For a named-list of specs, returns a named
#' list of `FactorModelEstimates`.
#'
#' @param panel an `UnbalancedPanel` R6 object used to derive cohort and unit structure
#' @param factor_model_estimates a `FactorModelEstimates` or a named list of them (by-spec)
#' @param cohort_outcome_mean_suff_stat_ests optional list of `OutcomeMeanSuffStatEstimates`
#' @param unit_weights optional numeric vector of unit weights
#' @param effective_observed_outcome_indices optional list of 1-based outcome indices to override panel
#' @param num_threads optional integer, number of threads for internal parallel sections
#' @param tol numeric tolerance for fixed-point iterations
#' @param max_iters integer max iterations for fixed-point iterations
#' @param fixed_point_method string, e.g. "irons-tuck"
#' @export
comp_imputation_components <- function(panel,
                                       factor_model_estimates,
                                       cohort_outcome_mean_suff_stat_ests = NULL,
                                       unit_weights = NULL,
                                       effective_observed_outcome_indices = NULL,
                                       num_threads = NULL,
                                       tol = 1e-10,
                                       max_iters = 1000L,
                                       fixed_point_method = "irons-tuck") {
  stopifnot(inherits(panel, "UnbalancedPanel"))
  holder_xp <- panel$get_panel_holder_xptr()
  nt <- if (is.null(num_threads)) NULL else as.integer(num_threads)

  if (inherits(factor_model_estimates, "FactorModelEstimates")) {
    xplist <- if (is.null(cohort_outcome_mean_suff_stat_ests)) NULL else lapply(cohort_outcome_mean_suff_stat_ests, function(om) om$.__enclos_env__$private$xp)
    xp_res <- comp_imputation_components_cpp(
      panel_holder_xptr = holder_xp,
      factor_model_estimates_xptr = factor_model_estimates$.__enclos_env__$private$xp,
      cohort_outcome_mean_suff_stat_ests = xplist,
      unit_weights = unit_weights,
      effective_observed_outcome_indices = effective_observed_outcome_indices,
      tol = tol,
      max_iters = as.integer(max_iters),
      fixed_point_method = fixed_point_method,
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
      unit_weights = unit_weights,
      effective_observed_outcome_indices = effective_observed_outcome_indices,
      tol = tol,
      max_iters = as.integer(max_iters),
      fixed_point_method = fixed_point_method,
      num_threads_in = nt
    )
    nms <- names(res)
    out <- setNames(vector("list", length(res)), nms)
    for (i in seq_along(res)) out[[i]] <- FactorModelEstimates$new(res[[i]])
    return(out)
  }

  stop("Invalid 'factor_model_estimates': expected a FactorModelEstimates object or a named list of them.")
}


