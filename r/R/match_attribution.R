#' FGW bipartite match outcome difference parameters
#'
#' Convenience wrapper that computes the FGW bipartite match attribution parameters
#' for two outcome indices and returns a `TargetParameterEstimates` object (or list).
#'
#' @param outcome_means `OutcomeMeansEstimates` or named list of them (by spec).
#' @param outcome_idx_1 First (1-based) outcome index.
#' @param outcome_idx_2 Second (1-based) outcome index.
#' @param observed_outcome_indices List of integer vectors (1-based) indicating outcomes observed per cohort.
#' @param aux_means Optional list (per cohort, or named list by spec) of `CohortAuxiliaryDataMeanEstimates`.
#' @param suff_stats Optional list (per cohort, or named list by spec) of `OutcomeMeanSuffStatEstimates`.
#' @return `TargetParameterEstimates` object or named list of them (by spec).
#' @export
est_fgw_bipartite_match_outcome_diff_params <- function(outcome_means,
                                                        outcome_idx_1,
                                                        outcome_idx_2,
                                                        observed_outcome_indices,
                                                        aux_means = NULL,
                                                        suff_stats = NULL) {
  stopifnot(is.list(observed_outcome_indices))
  if (inherits(outcome_means, "OutcomeMeansEstimates")) {
    return(.fgw_single(
      outcome_means,
      outcome_idx_1,
      outcome_idx_2,
      observed_outcome_indices,
      aux_means,
      suff_stats
    ))
  }
  if (is.list(outcome_means)) {
    return(.fgw_by_spec(
      outcome_means,
      outcome_idx_1,
      outcome_idx_2,
      observed_outcome_indices,
      aux_means,
      suff_stats
    ))
  }
  stop("Invalid 'outcome_means': expected an OutcomeMeansEstimates object or a named list of them.")
}

.fgw_single <- function(outcome_means,
                        outcome_idx_1,
                        outcome_idx_2,
                        observed_outcome_indices,
                        aux_means,
                        suff_stats) {
  if (!is.null(aux_means)) stopifnot(is.list(aux_means))
  if (!is.null(suff_stats)) stopifnot(is.list(suff_stats))
  if (!is.null(aux_means)) {
    aux_means <- lapply(aux_means, function(e) {
      if (is.null(e)) return(NULL)
      stopifnot(inherits(e, "CohortAuxiliaryDataMeanEstimates"))
      e$.__enclos_env__$private$xp
    })
  }
  if (!is.null(suff_stats)) {
    suff_stats <- lapply(suff_stats, function(e) {
      if (is.null(e)) return(NULL)
      stopifnot(inherits(e, "OutcomeMeanSuffStatEstimates"))
      e$.__enclos_env__$private$xp
    })
  }
  idx1 <- as.integer(outcome_idx_1)
  idx2 <- as.integer(outcome_idx_2)
  xp <- est_fgw_bipartite_match_outcome_diff_params_cpp(
    ome_xptr = outcome_means$.__enclos_env__$private$xp,
    stats_xptrs_by_cohort = suff_stats,
    eta_xptrs_by_cohort = aux_means,
    outcome_idx_1 = idx1,
    outcome_idx_2 = idx2,
    observed_outcome_indices_list = observed_outcome_indices
  )
  TargetParameterEstimates$new(xp)
}

.fgw_by_spec <- function(outcome_means_by_spec,
                         outcome_idx_1,
                         outcome_idx_2,
                         observed_outcome_indices,
                         aux_means_by_spec,
                         suff_stats_by_spec) {
  .validate_ome_by_spec(outcome_means_by_spec)
  if (!is.null(aux_means_by_spec)) .validate_eta_by_spec(aux_means_by_spec)
  if (!is.null(suff_stats_by_spec)) .validate_stats_by_spec(suff_stats_by_spec)

  ome_xp_by_spec <- lapply(outcome_means_by_spec, function(ome) ome$.__enclos_env__$private$xp)
  if (!is.null(aux_means_by_spec)) {
    eta_xp_by_spec <- lapply(aux_means_by_spec, function(lst) {
      if (is.null(lst)) return(NULL)
      stopifnot(is.list(lst))
      lapply(lst, function(e) {
        if (is.null(e)) return(NULL)
        stopifnot(inherits(e, "CohortAuxiliaryDataMeanEstimates"))
        e$.__enclos_env__$private$xp
      })
    })
  } else {
    eta_xp_by_spec <- NULL
  }
  if (!is.null(suff_stats_by_spec)) {
    stats_xp_by_spec <- lapply(suff_stats_by_spec, function(lst) {
      if (is.null(lst)) return(NULL)
      stopifnot(is.list(lst))
      lapply(lst, function(e) {
        if (is.null(e)) return(NULL)
        stopifnot(inherits(e, "OutcomeMeanSuffStatEstimates"))
        e$.__enclos_env__$private$xp
      })
    })
  } else {
    stats_xp_by_spec <- NULL
  }
  idx1 <- as.integer(outcome_idx_1)
  idx2 <- as.integer(outcome_idx_2)
  res <- est_fgw_bipartite_match_outcome_diff_params_by_spec_cpp(
    ome_by_spec = ome_xp_by_spec,
    stats_by_spec = stats_xp_by_spec,
    eta_by_spec = eta_xp_by_spec,
    outcome_idx_1 = idx1,
    outcome_idx_2 = idx2,
    observed_outcome_indices_list = observed_outcome_indices
  )
  .wrap_target_params_xptr_list(res)
}


