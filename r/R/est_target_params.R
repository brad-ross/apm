#' Target Parameter Estimates Container
#'
#' An R6 class that holds estimated target parameters (a p-dimensional vector)
#' with optional bootstrap replicates for inference.
#'
#' @description
#' `TargetParameterEstimates` stores the output of target parameter estimation,
#' where target parameters are user-defined functions of cohort outcome means.
#' Common examples include treatment effects, average differences across time
#' periods, or policy-relevant aggregations.
#'
#' @details
#' Target parameters are computed by applying \code{\link{est_target_params}} or
#' the match attribution functions like \code{\link{est_fgw_bipartite_match_outcome_diff_params}}.
#'
#' @section Public Methods:
#' \describe{
#'   \item{\code{has_bootstrap()}}{Logical; whether bootstrap replicates exist.}
#'   \item{\code{num_bootstraps()}}{Integer; number of bootstrap replicates (0 if none).}
#'   \item{\code{p()}}{Integer; dimension of the target parameter vector.}
#'   \item{\code{target_params(b = NULL)}}{Returns numeric vector of length p.
#'         If `b` is NULL, returns point estimate; otherwise returns bootstrap
#'         replicate `b` (1-indexed).}
#'   \item{\code{boots_matrix()}}{Returns p x B matrix of all bootstrap replicates.}
#' }
#'
#' @seealso \code{\link{est_target_params}} for computing target parameters.
#' @seealso \code{\link{target_param_inference}} for inference on target parameters.
#' @seealso \code{\link{est_fgw_bipartite_match_outcome_diff_params}} for match
#'   attribution parameters.
#'
#' @examples
#' # TargetParameterEstimates is typically obtained from estimation functions
#' \dontrun{
#' tpe <- est_target_params(outcome_means, my_target_fn)
#' tpe$p()              # Dimension of target
#' tpe$target_params()  # Point estimate
#' tpe$has_bootstrap()  # Check for bootstrap
#' }
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

#' Estimate Target Parameters from Outcome Means
#'
#' Applies a user-defined target function to estimated outcome means to compute
#' target parameters of interest, with optional bootstrap replicates.
#'
#' @description
#' This function takes estimated cohort-by-outcome means and applies a custom
#' function to compute target parameters.
#'
#' @details
#' **Target Function Signature:**
#'
#' The `fn` argument must be a function with signature:
#'
#' `fn(Y, shares, observed_means, covar_means, eta)`
#'
#' where:
#' \itemize{
#'   \item `Y`: C x T numeric matrix of (imputed) cohort-by-outcome means
#'   \item `shares`: length-C numeric vector of cohort population shares
#'   \item `observed_means`: length-C list of observed outcome means per cohort
#'   \item `covar_means`: length-C list of covariate means matrices (or NULL)
#'   \item `eta`: length-C list of auxiliary data matrices per cohort
#' }
#'
#' The function must return a numeric vector of length p (the target dimension).
#'
#' When bootstrap replicates are present in `outcome_means`, the function is
#' applied to each bootstrap replicate to enable inference.
#'
#' **Performance Note:**
#'
#' Custom R functions passed to `fn` are executed single-threaded across
#' bootstrap replicates because calling R from multiple C++ threads is unsafe.
#' For maximum performance with bootstrap inference, use the built-in target
#' functions (e.g., match attribution functions like
#' \code{\link{est_fgw_bipartite_match_outcome_diff_params}}) which are
#' implemented in C++ and fully parallelized across bootstrap draws.
#'
#' @param outcome_means An \code{\link{OutcomeMeansEstimates}} R6 object, or a
#'   named list of such objects (for by-spec estimation).
#' @param fn A function with signature `fn(Y, shares, observed_means, covar_means, eta)`
#'   that takes outcome means and cohort statistics, returning a numeric vector
#'   of target parameters. See Details for the full specification of each argument.
#' @param aux_means Optional list (one entry per cohort) of
#'   \code{\link{CohortAuxiliaryDataMeanEstimates}} objects. Passed to `fn` as
#'   the `eta` argument. Use `NULL` when no auxiliary data is needed.
#' @param suff_stats Optional list (one entry per cohort) of
#'   \code{\link{OutcomeMeanSuffStatEstimates}} objects. Provides additional
#'   cohort-level statistics if needed.
#'
#' @return For single-spec input: a \code{\link{TargetParameterEstimates}} R6 object.
#'
#'   For by-spec input: a named list of `TargetParameterEstimates` objects.
#'
#' @seealso \code{\link{TargetParameterEstimates}} for the returned object.
#' @seealso \code{\link{target_param_inference}} for computing inference.
#' @seealso \code{\link{est_target_param_components}} for end-to-end estimation.
#'
#' @examples
#' # Define a simple target function: average outcome across all cohorts and times
#' # Note: fn receives 5 arguments but you can ignore unused ones
#' avg_outcome_fn <- function(Y, shares, observed_means, covar_means, eta) {
#'   c(grand_mean = mean(Y))
#' }
#'
#' # Weighted average using cohort shares
#' weighted_avg_fn <- function(Y, shares, observed_means, covar_means, eta) {
#'   cohort_means <- rowMeans(Y)  # Mean outcome per cohort
#'   c(weighted_mean = sum(shares * cohort_means))
#' }
#'
#' \dontrun{
#' # Apply to outcome means
#' tpe <- est_target_params(outcome_means, avg_outcome_fn)
#' tpe$target_params()  # The estimated grand mean
#' }
#'
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

# -----------------------------------------------------------------------------
# End-to-end wrapper
# -----------------------------------------------------------------------------
#' Estimate Target Parameter Components (End-to-End)
#'
#' Performs complete end-to-end estimation from panel data to outcome means,
#' combining cohort-specific estimation, aggregation, and outcome mean computation
#' in a single efficient pipeline.
#'
#' @description
#' This is a convenience function that wraps the entire estimation pipeline:
#' \enumerate{
#'   \item Cohort-specific factor model estimation
#'   \item Factor aggregation across cohorts
#'   \item Outcome mean estimation via imputation
#' }
#'
#' The output contains all intermediate results needed for target parameter
#' estimation and inference.
#'
#' @details
#' **Imputation Options:**
#'
#' When `est_outcome_means_via_imputation = TRUE` (default), unobserved outcomes
#' are imputed using the estimated factor model. The `imputation_options` argument
#' controls the imputation algorithm. See \code{\link{comp_imputation_components}}
#' for the full list of available options including solver choice, convergence
#' tolerances, and LSMR-specific settings.
#'
#' **Outcome Masking:**
#'
#' Use `cohort_outcomes_to_mask` for cross-validation or out-of-sample evaluation.
#' Masked outcomes are excluded from estimation but their true values are preserved.
#' See \code{\link{est_cohort_specific_params}} for details on the masking format.
#'
#' @inheritParams est_cohort_specific_params
#' @param est_outcome_means_via_imputation Logical; if `TRUE` (default), compute
#'   outcome means by imputing unobserved outcomes. If `FALSE`, use only observed
#'   outcome means (no imputation).
#' @param imputation_options Optional named list of imputation algorithm options.
#'   See \code{\link{comp_imputation_components}} for the full list of options.
#'
#' @return A named list with:
#'   \describe{
#'     \item{outcome_means}{Named list (by spec) of \code{\link{OutcomeMeansEstimates}}
#'           objects containing C x T cohort-by-outcome mean matrices.}
#'     \item{cohort_outcome_mean_ests}{List (by cohort) of
#'           \code{\link{OutcomeMeanSuffStatEstimates}} objects.}
#'     \item{cohort_auxiliary_means}{List (by cohort) of
#'           \code{\link{CohortAuxiliaryDataMeanEstimates}}, or NULL.}
#'     \item{masked_cohort_outcome_means}{(When masking) True means for masked outcomes.}
#'     \item{masked_observed_outcome_indices}{(When masking) Outcome indices after masking.}
#'     \item{cohort_outcome_mask}{(When masking) Named list of masked outcome indices.}
#'   }
#'
#' @seealso \code{\link{est_cohort_specific_params}} for cohort-specific estimation.
#' @seealso \code{\link{comp_imputation_components}} for imputation options details.
#' @seealso \code{\link{est_target_params}} for computing target parameters from
#'   the returned outcome means.
#' @seealso \code{\link{target_param_inference}} for inference.
#'
#' @examples
#' \dontrun{
#' # End-to-end estimation
#' components <- est_target_param_components(
#'   panel = panel,
#'   est_specs = list(
#'     pc_r2 = list(
#'       factor_model_estimator = "principal_components",
#'       include_outcome_fes = TRUE,
#'       r = 2L
#'     )
#'   ),
#'   bootstrap = wb
#' )
#'
#' # Compute target parameters
#' tpe <- est_target_params(components$outcome_means, my_target_fn)
#' }
#'
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

#' Compute Inference for Target Parameters
#'
#' Computes bootstrap-based inference (standard errors, confidence intervals,
#' confidence bands, p-values) for estimated target parameters.
#'
#' @description
#' This function takes target parameter estimates with bootstrap replicates
#' and computes:
#' \itemize{
#'   \item Robust standard errors (IQR-based)
#'   \item Pointwise t-statistics and p-values
#'   \item Pointwise confidence intervals
#'   \item Romano–Wolf stepdown adjusted p-values (FWER-controlling)
#'   \item Simultaneous confidence bands (FWER-controlling)
#' }
#'
#' @param tpe A \code{\link{TargetParameterEstimates}} R6 object with bootstrap
#'   replicates, or a named list of such objects (for by-spec inference).
#' @param panel An \code{\link{UnbalancedPanel}} R6 object (used to determine
#'   sample size for standard error computation).
#' @param sig_level Numeric; significance level for confidence intervals and
#'   bands, must be in (0, 1). Default is 0.05 for 95% intervals.
#'
#' @return For single-spec input: a \code{\link{SimultaneousInferenceResults}}
#'   R6 object.
#'
#'   For by-spec input: a named list of `SimultaneousInferenceResults` objects.
#'
#' @seealso \code{\link{SimultaneousInferenceResults}} for accessing inference results.
#' @seealso \code{\link{est_target_params}} for computing target parameters.
#' @seealso \code{\link{combine_inference_results_across_specs}} for combining
#'   results across specifications.
#'
#' @examples
#' \dontrun{
#' # After target parameter estimation
#' tpe <- est_target_params(outcome_means, my_fn)
#'
#' # Compute inference
#' sir <- target_param_inference(tpe, panel, sig_level = 0.05)
#'
#' # Access results
#' sir$point()       # Point estimates
#' sir$std_error()   # Standard errors
#' sir$ci()          # Confidence intervals
#' sir$as_data_frame()
#' }
#'
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