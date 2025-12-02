#' FGW bipartite match outcome difference parameters
#'
#' Convenience wrapper that computes the FGW bipartite match attribution parameters
#' for two outcome indices and returns a `TargetParameterEstimates` object (or list).
#'
#' @param outcome_means `OutcomeMeansEstimates` or named list of them (by spec).
#' @param outcome_idx_1 First (1-based) outcome index. Ignored when
#'   `outcome_indices_1` is supplied.
#' @param outcome_idx_2 Second (1-based) outcome index. Ignored when
#'   `outcome_indices_2` is supplied.
#' @param observed_outcome_indices List of integer vectors (1-based) indicating outcomes observed per cohort.
#' @param aux_means Optional list (per cohort, or named list by spec) of `CohortAuxiliaryDataMeanEstimates`.
#' @param suff_stats Optional list (per cohort, or named list by spec) of `OutcomeMeanSuffStatEstimates`.
#' @param outcome_indices_1 Optional integer vector (1-based) of outcomes whose weighted average
#'   defines the first contrast. Defaults to `outcome_idx_1` when unspecified.
#' @param outcome_indices_2 Optional integer vector (1-based) of outcomes whose weighted average
#'   defines the second contrast. Defaults to `outcome_idx_2` when unspecified.
#' @param outcome_weights Optional numeric vector of nonnegative weights (length equals the total
#'   number of observed outcomes). When omitted, all outcomes receive equal weight.
#' @return `TargetParameterEstimates` object or named list of them (by spec).
#' @export
est_fgw_bipartite_match_outcome_diff_params <- function(outcome_means,
                                                        outcome_idx_1 = NULL,
                                                        outcome_idx_2 = NULL,
                                                        observed_outcome_indices,
                                                        aux_means = NULL,
                                                        suff_stats = NULL,
                                                        outcome_indices_1 = NULL,
                                                        outcome_indices_2 = NULL,
                                                        outcome_weights = NULL) {
  stopifnot(is.list(observed_outcome_indices))
  idx_resolved <- .resolve_outcome_indices(
    outcome_idx_1,
    outcome_idx_2,
    outcome_indices_1,
    outcome_indices_2
  )
  if (!is.null(outcome_weights)) {
    outcome_weights <- as.numeric(outcome_weights)
  }
  if (inherits(outcome_means, "OutcomeMeansEstimates")) {
    return(.fgw_single(
      outcome_means,
      idx_resolved$outcome_indices_1,
      idx_resolved$outcome_indices_2,
      observed_outcome_indices,
      aux_means,
      suff_stats,
      outcome_weights
    ))
  }
  if (is.list(outcome_means)) {
    return(.fgw_by_spec(
      outcome_means,
      idx_resolved$outcome_indices_1,
      idx_resolved$outcome_indices_2,
      observed_outcome_indices,
      aux_means,
      suff_stats,
      outcome_weights
    ))
  }
  stop("Invalid 'outcome_means': expected an OutcomeMeansEstimates object or a named list of them.")
}

.fgw_single <- function(outcome_means,
                        outcome_indices_1,
                        outcome_indices_2,
                        observed_outcome_indices,
                        aux_means,
                        suff_stats,
                        outcome_weights) {
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
  idx1 <- as.integer(outcome_indices_1)
  idx2 <- as.integer(outcome_indices_2)
  xp <- est_fgw_bipartite_match_outcome_diff_params_multi_cpp(
    ome_xptr = outcome_means$.__enclos_env__$private$xp,
    stats_xptrs_by_cohort = suff_stats,
    eta_xptrs_by_cohort = aux_means,
    outcome_indices_1 = idx1,
    outcome_indices_2 = idx2,
    observed_outcome_indices_list = observed_outcome_indices,
    outcome_weights = outcome_weights
  )
  TargetParameterEstimates$new(xp)
}

.fgw_by_spec <- function(outcome_means_by_spec,
                         outcome_indices_1,
                         outcome_indices_2,
                         observed_outcome_indices,
                         aux_means_by_spec,
                         suff_stats_by_spec,
                         outcome_weights) {
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
  idx1 <- as.integer(outcome_indices_1)
  idx2 <- as.integer(outcome_indices_2)
  res <- est_fgw_bipartite_match_outcome_diff_params_by_spec_multi_cpp(
    ome_by_spec = ome_xp_by_spec,
    stats_by_spec = stats_xp_by_spec,
    eta_by_spec = eta_xp_by_spec,
    outcome_indices_1 = idx1,
    outcome_indices_2 = idx2,
    observed_outcome_indices_list = observed_outcome_indices,
    outcome_weights = outcome_weights
  )
  .wrap_target_params_xptr_list(res)
}

.resolve_outcome_indices <- function(outcome_idx_1,
                                     outcome_idx_2,
                                     outcome_indices_1,
                                     outcome_indices_2) {
  using_vectors <- !is.null(outcome_indices_1) || !is.null(outcome_indices_2)
  if (using_vectors) {
    if (is.null(outcome_indices_1) || is.null(outcome_indices_2)) {
      stop("Both outcome_indices_1 and outcome_indices_2 must be provided when specifying vectors.")
    }
    idx1 <- as.integer(outcome_indices_1)
    idx2 <- as.integer(outcome_indices_2)
    if (length(idx1) == 0L || length(idx2) == 0L) {
      stop("Outcome index vectors must have positive length.")
    }
    if (any(is.na(idx1)) || any(is.na(idx2))) {
      stop("Outcome index vectors must not contain NA values.")
    }
    return(list(outcome_indices_1 = idx1, outcome_indices_2 = idx2))
  }
  if (is.null(outcome_idx_1) || is.null(outcome_idx_2)) {
    stop("Either scalar outcome_idx_* or vector outcome_indices_* arguments must be supplied.")
  }
  idx1 <- as.integer(outcome_idx_1)
  idx2 <- as.integer(outcome_idx_2)
  if (any(is.na(idx1)) || any(is.na(idx2))) {
    stop("Outcome indices must not be NA.")
  }
  list(outcome_indices_1 = idx1, outcome_indices_2 = idx2)
}


