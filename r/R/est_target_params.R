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
    boots_matrix = function() tpe_boot_params_matrix_cpp(private$xp)
  ),
  private = list(xp = NULL)
)

#' Estimate target parameters from estimated means (and optional aux means)
#'
#' Dispatch based on the type of `outcome_means`.
#'
#' @param outcome_means `OutcomeMeansEstimates` or named list of them (by spec).
#' @param fn function(Y, eta = NULL) returning numeric vector length p.
#' @param aux_means optional list of `CohortAuxiliaryDataMeanEstimates` (one per cohort) or
#'   a named list whose values are lists (by spec, then per cohort). NULL is allowed.
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
                                       cohort_outcomes_to_mask = NULL,
                                       est_outcome_means_via_imputation = TRUE) {
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
    est_outcome_means_via_imputation = est_outcome_means_via_imputation
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

.tpe_single <- function(outcome_means, fn, aux_means, suff_stats) {
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
  xp <- est_target_params_cpp(
    outcome_means$.__enclos_env__$private$xp,
    suff_stats,
    aux_means,
    fn
  )
  TargetParameterEstimates$new(xp)
}

.tpe_by_spec <- function(outcome_means_by_spec, fn, aux_means_by_spec, suff_stats_by_spec) {
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

  res <- est_target_params_by_spec_cpp(ome_xp_by_spec, stats_xp_by_spec, eta_xp_by_spec, fn)
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