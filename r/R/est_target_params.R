#' Target parameter estimates holder
#'
#' R6 wrapper for target parameter estimates: a point p-vector and optional
#' bootstrap replicates.
#'
#' @export
TargetParameterEstimates <- R6::R6Class(
  "TargetParameterEstimates",
  public = list(
    initialize = function(xptr) { private$xp <- xptr },

    has_bootstrap = function() tpe_has_bootstrap_cpp(private$xp),
    num_bootstraps = function() tpe_num_bootstrap_cpp(private$xp),
    p = function() tpe_p_cpp(private$xp),

    target_params = function(b = NULL) {
      if (is.null(b)) tpe_point_params_cpp(private$xp)
      else tpe_boot_params_cpp(private$xp, as.integer(b))
    },
    boots_matrix = function() tpe_boot_params_matrix_cpp(private$xp),
    subset = function(indices) {
      stopifnot(!missing(indices))
      indices <- as.integer(indices)
      if (length(indices) == 0L) stop("indices must be non-empty")
      if (any(is.na(indices))) stop("indices must not be NA")
      if (any(indices < 1L) || any(indices > self$p())) {
        stop("indices must be in range [1, p()]")
      }
      xp <- subset_target_param_ests_cpp(private$xp, indices - 1L)
      TargetParameterEstimates$new(xp)
    }
  ),
  private = list(xp = NULL)
)

#' Estimate target parameters from estimated means (and optional aux means)
#'
#' Dispatch based on the type of `outcome_means`.
#'
#' @param outcome_means `OutcomeMeansEstimates` or named list of them (by spec).
#' @param fn function(Y, eta = NULL) returning numeric vector length p.
#' @param aux_means optional list (one entry per cohort) of `CohortAuxiliaryDataMeanEstimates`.
#'   NULL is allowed when no auxiliary statistics are needed.
#' @param suff_stats optional list (one entry per cohort) of `OutcomeMeanSuffStatEstimates`.
#'   NULL indicates that sufficient statistics are unavailable.
#' @return `TargetParameterEstimates` R6 object or named list of them (by spec).
#' @export
est_target_params <- function(outcome_means, fn, aux_means = NULL, suff_stats = NULL) {
  if (inherits(outcome_means, "OutcomeMeansEstimates")) {
    return(.tpe_single(outcome_means, fn, aux_means, suff_stats))
  }
  if (is.list(outcome_means)) {
    return(.tpe_by_spec(outcome_means, fn, aux_means, suff_stats))
  }
  stop("Invalid 'outcome_means': expected an OutcomeMeansEstimates object or a named list of them.")
}

#' Difference between two target parameter estimates
#'
#' Computes `target_params_1 - target_params_2` entrywise for the point estimates
#' and, when available, for each bootstrap replicate.
#'
#' @param target_params_1 `TargetParameterEstimates`
#' @param target_params_2 `TargetParameterEstimates`
#' @return `TargetParameterEstimates`
#' @export
get_target_param_diff_ests <- function(target_params_1, target_params_2) {
  stopifnot(inherits(target_params_1, "TargetParameterEstimates"))
  stopifnot(inherits(target_params_2, "TargetParameterEstimates"))
  xp <- get_target_param_diff_ests_cpp(
    target_params_1$.__enclos_env__$private$xp,
    target_params_2$.__enclos_env__$private$xp
  )
  TargetParameterEstimates$new(xp)
}

#' Combine target parameter estimates
#'
#' If `target_params_1` is a list of `TargetParameterEstimates`, combines all
#' elements by concatenating their point vectors and stacking bootstrap
#' replicate matrices. Otherwise, combines two `TargetParameterEstimates`
#' provided in `target_params_1` and `target_params_2`.
#'
#' @param target_params_1 `TargetParameterEstimates` or a non-empty list of them
#' @param target_params_2 optional `TargetParameterEstimates` when combining two
#' @return `TargetParameterEstimates`
#' @export
combine_target_param_ests <- function(target_params_1, target_params_2 = NULL) {
  if (is.list(target_params_1)) {
    if (!is.null(target_params_2)) {
      stop("When combining a list, 'target_params_2' must be NULL.")
    }
    if (length(target_params_1) == 0L) stop("'target_params_1' list must be non-empty.")
    ok <- vapply(target_params_1, function(e) inherits(e, "TargetParameterEstimates"), logical(1))
    if (!all(ok)) stop("All elements of 'target_params_1' must be TargetParameterEstimates when providing a list.")
    xp_list <- lapply(target_params_1, function(e) e$.__enclos_env__$private$xp)
    xp <- combine_target_param_ests_multi_cpp(xp_list)
    return(TargetParameterEstimates$new(xp))
  }

  if (is.null(target_params_2)) {
    stop("When providing a single 'target_params_1', you must also supply 'target_params_2'.")
  }
  stopifnot(inherits(target_params_1, "TargetParameterEstimates"))
  stopifnot(inherits(target_params_2, "TargetParameterEstimates"))
  xp <- combine_target_param_ests_cpp(
    target_params_1$.__enclos_env__$private$xp,
    target_params_2$.__enclos_env__$private$xp
  )
  TargetParameterEstimates$new(xp)
}

# -----------------------------------------------------------------------------
# End-to-end wrapper
# -----------------------------------------------------------------------------
#' End-to-end: estimate target parameter components across cohorts (by spec)
#'
#' This runs cohort-specific estimation and immediately aggregates/estimates cohort
#' mean outcomes across cohorts, returning a list with:
#' - outcome_means: named list (by spec) of `OutcomeMeansEstimates` objects
#' - cohort_outcome_mean_ests: list of `OutcomeMeanSuffStatEstimates` (one per cohort)
#' - cohort_auxiliary_means: list of `CohortAuxiliaryDataMeanEstimates` (one per cohort), or NULL if none
#' - masked_cohort_outcome_means / masked_observed_outcome_indices: present when masking is used
#' - cohort_outcome_mask: named list of masked outcome indices when masking is used
#'
#' @inheritParams est_cohort_specific_params
#' @return list with the components described above.
#' @export
est_target_param_components <- function(panel, est_specs, bootstrap = NULL, num_threads = NULL,
                                       cohort_outcomes_to_mask = NULL,
                                       est_outcome_means_via_imputation = TRUE,
                                       imputation_options = NULL) {
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
    cohort_outcomes_to_mask_in = cohort_outcomes_to_mask,
    est_outcome_means_via_imputation = est_outcome_means_via_imputation,
    imputation_options_in = imputation_options
  )
  out <- list(
    outcome_means = .wrap_outcome_means_xptr_list(res$outcome_means),
    cohort_outcome_mean_ests = lapply(res$cohort_outcome_mean_ests, function(xp) OutcomeMeanSuffStatEstimates$new(xp)),
    cohort_auxiliary_means = NULL,
    masked_cohort_outcome_means = NULL,
    masked_observed_outcome_indices = NULL,
    cohort_outcome_mask = NULL
  )
  if (!is.null(res$cohort_auxiliary_means)) {
    out$cohort_auxiliary_means <- lapply(res$cohort_auxiliary_means, function(xp) CohortAuxiliaryDataMeanEstimates$new(xp))
  }
  if (!is.null(res$masked_cohort_outcome_means)) {
    out$masked_cohort_outcome_means <- res$masked_cohort_outcome_means
  }
  if (!is.null(res$masked_observed_outcome_indices)) {
    out$masked_observed_outcome_indices <- res$masked_observed_outcome_indices
  }
  if (!is.null(res$cohort_outcome_mask)) {
    out$cohort_outcome_mask <- res$cohort_outcome_mask
  }
  out
}

# -----------------------------------------------------------------------------
# Internal helpers (not exported)
# -----------------------------------------------------------------------------

.tpe_single <- function(outcome_means, fn, aux_means, suff_stats) {
  aux_means <- .extract_aux_means_xptrs(aux_means)
  suff_stats <- .extract_suff_stats_xptrs(suff_stats)
  xp <- est_target_params_cpp(
    outcome_means$.__enclos_env__$private$xp,
    suff_stats,
    aux_means,
    fn
  )
  TargetParameterEstimates$new(xp)
}

.tpe_by_spec <- function(outcome_means_by_spec, fn, aux_means_input, suff_stats_input) {
  .validate_ome_by_spec(outcome_means_by_spec)

  ome_xp_by_spec <- lapply(outcome_means_by_spec, function(ome) ome$.__enclos_env__$private$xp)
  eta_xp_shared <- .resolve_shared_aux_means_input(aux_means_input)
  stats_xp_shared <- .resolve_shared_suff_stats_input(suff_stats_input)

  res <- est_target_params_by_spec_cpp(
    ome_xp_by_spec,
    stats_xp_shared,
    eta_xp_shared,
    fn
  )
  .wrap_target_params_xptr_list(res)
}

.validate_ome_by_spec <- function(x) {
  if (!is.list(x) || length(x) == 0L) {
    stop("When providing by-spec inputs, 'outcome_means' must be a non-empty named list.")
  }
  if (is.null(names(x)) || any(!nzchar(names(x)))) {
    stop("When providing by-spec inputs, 'outcome_means' must be a named list. Hint: use something like list(specA = ..., specB = ...).")
  }
  if (!all(vapply(x, function(e) inherits(e, "OutcomeMeansEstimates"), logical(1)))) {
    stop("All elements of 'outcome_means' must inherit from class 'OutcomeMeansEstimates'.")
  }
  invisible(TRUE)
}

.validate_eta_by_spec <- function(x) {
  if (is.null(x)) return(invisible(TRUE))
  if (!is.list(x)) stop("'aux_means' must be a named list when using by-spec inputs.")
  if (length(x) == 0L) return(invisible(TRUE))
  if (is.null(names(x)) || any(!nzchar(names(x)))) {
    stop("When providing by-spec inputs, 'aux_means' must be a named list matching the spec keys.")
  }
  for (spec in names(x)) {
    lst <- x[[spec]]
    if (is.null(lst)) next
    if (!is.list(lst)) stop(sprintf("aux_means[['%s']] must be a list (one per cohort) or NULL.", spec))
    ok <- vapply(lst, function(e) is.null(e) || inherits(e, "CohortAuxiliaryDataMeanEstimates"), logical(1))
    if (!all(ok)) stop(sprintf("All elements under spec '%s' must be CohortAuxiliaryDataMeanEstimates or NULL.", spec))
  }
  invisible(TRUE)
}

.validate_stats_by_spec <- function(x) {
  if (is.null(x)) return(invisible(TRUE))
  if (!is.list(x)) stop("'suff_stats' must be a named list when using by-spec inputs.")
  if (length(x) == 0L) return(invisible(TRUE))
  if (is.null(names(x)) || any(!nzchar(names(x)))) {
    stop("When providing by-spec inputs, 'suff_stats' must be a named list matching the spec keys.")
  }
  for (spec in names(x)) {
    lst <- x[[spec]]
    if (is.null(lst)) next
    if (!is.list(lst)) stop(sprintf("suff_stats[['%s']] must be a list (one per cohort) or NULL.", spec))
    ok <- vapply(lst, function(e) is.null(e) || inherits(e, "OutcomeMeanSuffStatEstimates"), logical(1))
    if (!all(ok)) stop(sprintf("All elements under spec '%s' must be OutcomeMeanSuffStatEstimates or NULL.", spec))
  }
  invisible(TRUE)
}

.wrap_target_params_xptr_list <- function(res_named_xptr_list) {
  nms <- names(res_named_xptr_list)
  out <- setNames(vector("list", length(res_named_xptr_list)), nms)
  for (i in seq_along(res_named_xptr_list)) {
    out[[i]] <- TargetParameterEstimates$new(res_named_xptr_list[[i]])
  }
  out
}

.extract_aux_means_xptrs <- function(aux_means) {
  if (is.null(aux_means)) return(NULL)
  stopifnot(is.list(aux_means))
  lapply(aux_means, function(e) {
    if (is.null(e)) return(NULL)
    stopifnot(inherits(e, "CohortAuxiliaryDataMeanEstimates"))
    e$.__enclos_env__$private$xp
  })
}

.extract_suff_stats_xptrs <- function(suff_stats) {
  if (is.null(suff_stats)) return(NULL)
  stopifnot(is.list(suff_stats))
  lapply(suff_stats, function(e) {
    if (is.null(e)) return(NULL)
    stopifnot(inherits(e, "OutcomeMeanSuffStatEstimates"))
    e$.__enclos_env__$private$xp
  })
}

.resolve_shared_aux_means_input <- function(aux_means) {
  if (is.null(aux_means)) return(NULL)
  if (!is.list(aux_means)) {
    stop("When provided, 'aux_means' must be a list.")
  }
  if (.is_cohort_aux_list(aux_means)) {
    return(.extract_aux_means_xptrs(aux_means))
  }
  .validate_eta_by_spec(aux_means)
  .resolve_shared_by_spec(aux_means, .extract_aux_means_xptrs, "aux_means")
}

.resolve_shared_suff_stats_input <- function(suff_stats) {
  if (is.null(suff_stats)) return(NULL)
  if (!is.list(suff_stats)) {
    stop("When provided, 'suff_stats' must be a list.")
  }
  if (.is_cohort_suff_stats_list(suff_stats)) {
    return(.extract_suff_stats_xptrs(suff_stats))
  }
  .validate_stats_by_spec(suff_stats)
  .resolve_shared_by_spec(suff_stats, .extract_suff_stats_xptrs, "suff_stats")
}

.is_cohort_aux_list <- function(x) {
  if (length(x) == 0L) return(TRUE)
  all(vapply(x, function(e) is.null(e) || inherits(e, "CohortAuxiliaryDataMeanEstimates"), logical(1)))
}

.is_cohort_suff_stats_list <- function(x) {
  if (length(x) == 0L) return(TRUE)
  all(vapply(x, function(e) is.null(e) || inherits(e, "OutcomeMeanSuffStatEstimates"), logical(1)))
}

.resolve_shared_by_spec <- function(by_spec, extractor_fn, label) {
  if (is.null(by_spec)) return(NULL)
  stopifnot(is.list(by_spec))
  if (length(by_spec) == 0L) return(NULL)
  if (is.null(names(by_spec)) || any(!nzchar(names(by_spec)))) {
    stop(sprintf("When providing by-spec inputs, '%s' must be a named list matching the spec keys.", label))
  }
  shared <- NULL
  shared_set <- FALSE
  for (spec in names(by_spec)) {
    entry <- by_spec[[spec]]
    current <- if (is.null(entry)) NULL else extractor_fn(entry)
    if (!shared_set) {
      shared <- current
      shared_set <- TRUE
      next
    }
    if (!identical(shared, current)) {
      stop(sprintf("All specs must share identical %s; mismatch detected for spec '%s'.", label, spec))
    }
  }
  shared
}

#' Target-parameter inference (single or by spec)
#' @param tpe TargetParameterEstimates or named list of them
#' @param panel an UnbalancedPanel
#' @param sig_level significance level in (0,1)
#' @return SimultaneousInferenceResults or named list (by spec) of them
#' @export
target_param_inference <- function(tpe, panel, sig_level = 0.05) {
  stopifnot(inherits(panel, "UnbalancedPanel"))
  holder_xp <- panel$get_panel_holder_xptr()

  if (inherits(tpe, "TargetParameterEstimates")) {
    xp <- target_param_inference_cpp(
      tpe$.__enclos_env__$private$xp,
      holder_xp,
      sig_level
    )
    return(SimultaneousInferenceResults$new(xp))
  }

  if (is.list(tpe)) {
    .validate_tpe_by_spec(tpe)
    tpe_xp_by_spec <- lapply(tpe, function(e) {
      stopifnot(inherits(e, "TargetParameterEstimates"))
      e$.__enclos_env__$private$xp
    })
    res <- target_param_inference_by_spec_cpp(
      tpe_by_spec = tpe_xp_by_spec,
      panel_holder_xptr = holder_xp,
      sig_level = sig_level
    )
    return(.wrap_sir_xptr_list(res))
  }

  stop("Invalid 'tpe': expected a TargetParameterEstimates object or a named list of them.")
}

.validate_tpe_by_spec <- function(x) {
  if (!is.list(x) || length(x) == 0L) {
    stop("When providing by-spec inputs, 'tpe' must be a non-empty named list.")
  }
  if (is.null(names(x)) || any(!nzchar(names(x)))) {
    stop("When providing by-spec inputs, 'tpe' must be a named list.")
  }
  ok <- vapply(x, function(e) inherits(e, "TargetParameterEstimates"), logical(1))
  if (!all(ok)) stop("All elements of 'tpe' must inherit from 'TargetParameterEstimates'.")
  invisible(TRUE)
}

.wrap_sir_xptr_list <- function(res_named_xptr_list) {
  nms <- names(res_named_xptr_list)
  out <- setNames(vector("list", length(res_named_xptr_list)), nms)
  for (i in seq_along(res_named_xptr_list)) {
    out[[i]] <- SimultaneousInferenceResults$new(res_named_xptr_list[[i]])
  }
  out
}