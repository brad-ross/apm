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
#' @param suff_stats Optional list (one entry per cohort) of `OutcomeMeanSuffStatEstimates`.
#'   NULL indicates that sufficient statistics are unavailable.
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
                        suff_stats,
                        outcome_weights) {
  suff_stats <- .extract_suff_stats_xptrs(suff_stats)
  idx1 <- as.integer(outcome_indices_1)
  idx2 <- as.integer(outcome_indices_2)
  xp <- est_fgw_bipartite_match_outcome_diff_params_multi_cpp(
    ome_xptr = outcome_means$.__enclos_env__$private$xp,
    stats_xptrs_by_cohort = suff_stats,
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
                         suff_stats_input,
                         outcome_weights) {
  .validate_ome_by_spec(outcome_means_by_spec)

  ome_xp_by_spec <- lapply(outcome_means_by_spec, function(ome) ome$.__enclos_env__$private$xp)
  stats_xp_shared <- .resolve_shared_suff_stats_input(suff_stats_input)
  idx1 <- as.integer(outcome_indices_1)
  idx2 <- as.integer(outcome_indices_2)
  res <- est_fgw_bipartite_match_outcome_diff_params_by_spec_multi_cpp(
    ome_by_spec = ome_xp_by_spec,
    stats_xptrs_by_cohort = stats_xp_shared,
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

#' Averaged FGW bipartite match outcome difference parameters
#'
#' Computes the FGW bipartite match attribution parameters averaged across all unordered
#' pairs of outcome groupings. Each grouping is averaged internally (using outcome weights)
#' before forming pairwise contrasts, and the resulting parameter vectors are averaged with
#' weights proportional to the total outcome weight of both groups in the pair.
#'
#' @param outcome_means `OutcomeMeansEstimates` or named list of them (by spec).
#' @param observed_outcome_indices List of integer vectors (1-based) indicating outcomes observed per cohort.
#' @param suff_stats Optional list (one entry per cohort) of `OutcomeMeanSuffStatEstimates`.
#'   NULL indicates that sufficient statistics are unavailable.
#' @param outcome_groupings Optional list of integer vectors (1-based) defining outcome groups. Must contain at least two groups.
#' @param outcome_indices Optional integer vector (1-based) of outcomes; treated as singleton groups when `outcome_groupings` is omitted.
#' @param outcome_weights Optional numeric vector of nonnegative weights (length equals the total number of observed outcomes). Defaults to equal weights.
#' @param num_threads Positive integer specifying threads passed to the underlying estimator. Defaults to 1.
#' @return `TargetParameterEstimates` object or named list of them (by spec).
#' @export
est_avg_fgw_bipartite_match_outcome_diff_params <- function(outcome_means,
                                                            observed_outcome_indices,
                                                            suff_stats = NULL,
                                                            outcome_groupings = NULL,
                                                            outcome_indices = NULL,
                                                            outcome_weights = NULL,
                                                            num_threads = 1) {
  stopifnot(is.list(observed_outcome_indices))
  resolved <- .resolve_avg_outcome_inputs(outcome_groupings, outcome_indices)
  if (!is.null(outcome_weights)) {
    outcome_weights <- as.numeric(outcome_weights)
  }
  num_threads <- .validate_num_threads(num_threads)
  if (inherits(outcome_means, "OutcomeMeansEstimates")) {
    return(.avg_fgw_single(
      outcome_means,
      observed_outcome_indices,
      suff_stats,
      resolved$outcome_groupings,
      resolved$outcome_indices,
      outcome_weights,
      num_threads
    ))
  }
  if (is.list(outcome_means)) {
    return(.avg_fgw_by_spec(
      outcome_means,
      observed_outcome_indices,
      suff_stats,
      resolved$outcome_groupings,
      resolved$outcome_indices,
      outcome_weights,
      num_threads
    ))
  }
  stop("Invalid 'outcome_means': expected an OutcomeMeansEstimates object or a named list of them.")
}

.avg_fgw_single <- function(outcome_means,
                            observed_outcome_indices,
                            suff_stats,
                            outcome_groupings,
                            outcome_indices,
                            outcome_weights,
                            num_threads) {
  suff_stats <- .extract_suff_stats_xptrs(suff_stats)
  if (!is.null(outcome_groupings)) {
    groupings <- lapply(outcome_groupings, as.integer)
    xp <- est_avg_fgw_bipartite_match_outcome_diff_params_multi_cpp(
      ome_xptr = outcome_means$.__enclos_env__$private$xp,
      stats_xptrs_by_cohort = suff_stats,
      outcome_groupings = groupings,
      observed_outcome_indices_list = observed_outcome_indices,
      outcome_weights = outcome_weights,
      num_threads = num_threads
    )
  } else {
    idx <- as.integer(outcome_indices)
    xp <- est_avg_fgw_bipartite_match_outcome_diff_params_cpp(
      ome_xptr = outcome_means$.__enclos_env__$private$xp,
      stats_xptrs_by_cohort = suff_stats,
      outcome_indices = idx,
      observed_outcome_indices_list = observed_outcome_indices,
      outcome_weights = outcome_weights,
      num_threads = num_threads
    )
  }
  TargetParameterEstimates$new(xp)
}

.avg_fgw_by_spec <- function(outcome_means_by_spec,
                             observed_outcome_indices,
                             suff_stats_input,
                             outcome_groupings,
                             outcome_indices,
                             outcome_weights,
                             num_threads) {
  .validate_ome_by_spec(outcome_means_by_spec)

  ome_xp_by_spec <- lapply(outcome_means_by_spec, function(ome) ome$.__enclos_env__$private$xp)
  stats_xp_shared <- .resolve_shared_suff_stats_input(suff_stats_input)
  if (!is.null(outcome_groupings)) {
    groupings <- lapply(outcome_groupings, as.integer)
    res <- est_avg_fgw_bipartite_match_outcome_diff_params_by_spec_multi_cpp(
      ome_by_spec = ome_xp_by_spec,
      stats_xptrs_by_cohort = stats_xp_shared,
      outcome_groupings = groupings,
      observed_outcome_indices_list = observed_outcome_indices,
      outcome_weights = outcome_weights,
      num_threads = num_threads
    )
  } else {
    idx <- as.integer(outcome_indices)
    res <- est_avg_fgw_bipartite_match_outcome_diff_params_by_spec_cpp(
      ome_by_spec = ome_xp_by_spec,
      stats_xptrs_by_cohort = stats_xp_shared,
      outcome_indices = idx,
      observed_outcome_indices_list = observed_outcome_indices,
      outcome_weights = outcome_weights,
      num_threads = num_threads
    )
  }
  .wrap_target_params_xptr_list(res)
}

.resolve_avg_outcome_inputs <- function(outcome_groupings,
                                        outcome_indices) {
  using_groupings <- !is.null(outcome_groupings)
  using_indices <- !is.null(outcome_indices)
  if (using_groupings && using_indices) {
    stop("Provide either outcome_groupings or outcome_indices, not both.")
  }
  if (!using_groupings && !using_indices) {
    stop("Either outcome_groupings or outcome_indices must be supplied.")
  }
  if (using_groupings) {
    if (!is.list(outcome_groupings) || length(outcome_groupings) < 2L) {
      stop("outcome_groupings must be a list with at least two elements.")
    }
    validated <- lapply(seq_along(outcome_groupings), function(i) {
      grp <- outcome_groupings[[i]]
      if (is.null(grp)) stop("Outcome groupings must not contain NULL entries.")
      grp_int <- as.integer(grp)
      if (length(grp_int) == 0L) stop("Outcome groupings must have positive length.")
      if (any(is.na(grp_int))) stop("Outcome groupings must not contain NA values.")
      grp_int
    })
    return(list(outcome_groupings = validated, outcome_indices = NULL))
  }
  idx <- as.integer(outcome_indices)
  if (length(idx) < 2L) {
    stop("outcome_indices must contain at least two entries.")
  }
  if (any(is.na(idx))) {
    stop("outcome_indices must not contain NA values.")
  }
  list(outcome_groupings = NULL, outcome_indices = idx)
}

.validate_num_threads <- function(num_threads) {
  nt <- as.integer(num_threads)
  if (length(nt) != 1L || is.na(nt) || nt < 1L) {
    stop("num_threads must be a positive integer.")
  }
  nt
}