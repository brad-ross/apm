validate_mask_arg <- function(mask) {
    if (is.null(mask)) return(invisible(NULL))
    if (!is.list(mask)) stop("cohort_outcomes_to_mask must be a named list")
    if (length(mask) == 0L) return(invisible(NULL))
    nms <- names(mask)
    if (is.null(nms) || any(nms == "")) stop("cohort_outcomes_to_mask must have names = cohort ids (1-based)")
    for (nm in nms) {
        v <- mask[[nm]]
        if (!is.integer(v)) stop("each mask entry must be an integer vector (1-based outcome ids)")
        if (length(v) > 0L && any(is.na(v) | v <= 0L)) stop("mask outcome ids must be positive integers")
    }
    invisible(NULL)
}

#' Estimate Cohort-Specific Factor Model Parameters
#'
#' Estimates factor model parameters (factors G, optional fixed effects g0,
#' optional covariate coefficients a) separately for each cohort in an unbalanced
#' panel, along with cohort weights and outcome mean sufficient statistics.
#'
#' @description
#' This is a core estimation function that processes an \code{\link{UnbalancedPanel}}
#' to produce:
#' \itemize{
#'   \item Per-cohort factor model estimates for each specification
#'   \item Cohort-level sufficient statistics for outcome means
#'   \item Cohort weights for subsequent aggregation
#' }
#'
#' Multiple estimation specifications can be run simultaneously (e.g., different
#' ranks or with/without fixed effects), improving efficiency through shared
#' data processing and parallelization across cohorts.
#'
#' @details
#' **Estimation Specifications:**
#'
#' Each element of `est_specs` is a named list specifying one estimation approach:
#' \describe{
#'   \item{factor_model_estimator}{Character; currently only `"principal_components"`
#'         and "twfe" are supported. If "principal_components", uses PCA on cohort outcome data to extract factors.
#'         If "twfe", estimates a rank-0 factor model with only outcome fixed effects.}
#'   \item{include_outcome_fes}{Logical; if `TRUE`, jointly estimate outcome fixed
#'         effects g0 alongside factors G.}
#'   \item{r}{Integer; the factor model rank (number of latent factors).}
#'   \item{cohort_weighting}{Character; either `"equal"` (default) for uniform
#'         weights 1/C, or `"by_size"` to weight cohorts by their share of units.}
#' }
#'
#' **Outcome Masking:**
#'
#' The `cohort_outcomes_to_mask` argument allows holding out specific outcomes 
#' for specific cohorts for cross-validation or out-of-sample prediction. 
#' Masked outcomes are excluded from factor estimation but their true values are 
#' preserved for evaluation.
#'
#' @param panel An \code{\link{UnbalancedPanel}} R6 object containing the panel data.
#' @param est_specs A named list of estimation specifications. Each element is a
#'   list with fields: `factor_model_estimator`, `include_outcome_fes`, `r`, and
#'   optionally `cohort_weighting`. See Details.
#' @param bootstrap Optional \code{\link{WeightedBootstrap}} object for bootstrap
#'   inference. If provided, all estimates include bootstrap replicates.
#' @param num_threads Integer; number of threads for parallel computation
#'   (default: 1). Set to `NULL` to use the library default.
#' @param cohort_outcomes_to_mask Optional named list for outcome masking. Names
#'   are cohort IDs (as character strings of 1-based integers); values are integer
#'   vectors of 1-based outcome indices to mask in that cohort.
#'
#' @return A named list with:
#'   \describe{
#'     \item{cohort_specific_factor_ests}{Named list (by spec) of lists (by cohort)
#'           of \code{\link{FactorModelEstimates}} objects.}
#'     \item{cohort_outcome_means}{List (by cohort) of
#'           \code{\link{OutcomeMeanSuffStatEstimates}} objects.}
#'     \item{cohort_weights}{Named list (by spec) of
#'           \code{\link{CohortWeightEstimates}} objects.}
#'     \item{cohort_auxiliary_means}{(When auxiliary columns exist) List (by cohort)
#'           of \code{\link{CohortAuxiliaryDataMeanEstimates}} objects.}
#'     \item{masked_observed_outcome_indices}{(When masking) List of remaining observed outcome indices
#'           after masking for each cohort.}
#'     \item{masked_cohort_outcome_means}{(When masking) True means for masked cohort outcomes.}
#'     \item{cohort_outcome_mask}{(When masking) Named list of masked outcome indices.}
#'   }
#'
#' @seealso \code{\link{UnbalancedPanel}} for creating the input panel.
#' @seealso \code{\link{aggregate_factor_model_params}} for aggregating across cohorts.
#' @seealso \code{\link{est_target_param_components}} for end-to-end estimation.
#'
#' @examples
#' # Create synthetic panel
#' set.seed(123)
#' panel_df <- data.frame(
#'   unit = rep(1:50, each = 4),
#'   time = rep(1:4, 50),
#'   y = rnorm(200)
#' )
#' panel_df <- panel_df[sample(nrow(panel_df), 180), ]  # Make unbalanced
#'
#' # Create UnbalancedPanel
#' panel <- UnbalancedPanel$new(
#'   panel_df = panel_df,
#'   unit_id_col = "unit",
#'   outcome_id_col = "time",
#'   outcome_value_col = "y",
#'   model_rank = 2
#' )
#'
#' # Define estimation specifications
#' specs <- list(
#'   pc_r2 = list(
#'     factor_model_estimator = "principal_components",
#'     include_outcome_fes = FALSE,
#'     r = 2L
#'   ),
#'   pc_fe_r2 = list(
#'     factor_model_estimator = "principal_components",
#'     include_outcome_fes = TRUE,
#'     r = 2L
#'   )
#' )
#'
#' # Estimate
#' results <- est_cohort_specific_params(panel, specs)
#'
#' # Access results
#' names(results$cohort_specific_factor_ests)  # Spec names
#' length(results$cohort_outcome_means)        # Number of cohorts
#'
#' @export
est_cohort_specific_params <- function(panel, est_specs, bootstrap = NULL, num_threads = 1L,
                                       cohort_outcomes_to_mask = NULL) {
    stopifnot(inherits(panel, "UnbalancedPanel"))
    if (!is.list(est_specs) || length(est_specs) == 0L) stop("est_specs must be a non-empty list")
    if (!is.null(bootstrap) && !inherits(bootstrap, "WeightedBootstrap")) stop("bootstrap must be a WeightedBootstrap or NULL")

    .validate_est_specs(est_specs)

    holder_xp <- panel$get_panel_holder_xptr()

    validate_mask_arg(cohort_outcomes_to_mask)

    xp <- if (is.null(bootstrap)) NULL else bootstrap$.__enclos_env__$private$xp
    nt <- if (is.null(num_threads)) NULL else as.integer(num_threads)

    res <- est_cohort_specific_params_from_panel_cpp(
        panel_holder_xptr = holder_xp,
        est_specs = est_specs,
        bootstrap_xptr = xp,
        num_threads_in = nt,
        cohort_outcomes_to_mask = cohort_outcomes_to_mask
    )

    wrapped_factor <- lapply(res$cohort_specific_factor_ests, function(lst) {
        lapply(lst, function(xp) FactorModelEstimates$new(xp))
    })
    wrapped_oms <- lapply(res$cohort_outcome_means, function(xp) OutcomeMeanSuffStatEstimates$new(xp))
    wrapped_weights <- lapply(res$cohort_weights, function(xp) CohortWeightEstimates$new(xp))

    out <- list(
        cohort_specific_factor_ests = wrapped_factor,
        cohort_outcome_means = wrapped_oms,
        cohort_weights = wrapped_weights
    )
    if ("masked_observed_outcome_indices" %in% names(res)) {
        out$masked_observed_outcome_indices <- res$masked_observed_outcome_indices
    }
    if ("masked_cohort_outcome_means" %in% names(res)) {
        out$masked_cohort_outcome_means <- res$masked_cohort_outcome_means
    }
    if ("cohort_outcome_mask" %in% names(res)) {
        out$cohort_outcome_mask <- res$cohort_outcome_mask
    }
    if ("cohort_auxiliary_means" %in% names(res)) {
        wrapped_aux <- lapply(res$cohort_auxiliary_means, function(xp) CohortAuxiliaryDataMeanEstimates$new(xp))
        out$cohort_auxiliary_means <- wrapped_aux
    }
    out
}