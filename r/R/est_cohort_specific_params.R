"_PACKAGE"

#' Cohort-specific parameter estimation (thin wrapper to Rcpp)
#'
#' @param panel UnbalancedPanel instance
#' @param est_specs named list of specs; each spec is a list with fields:
#'   - factor_model_estimator: character(1), currently only "principal_components" is supported.
#'       Uses a principal components estimator for cohort-specific factor matrices; when
#'       `include_outcome_fes = TRUE`, the estimator jointly recovers outcome fixed effects.
#'   - include_outcome_fes: logical(1). If TRUE, include cohort-specific outcome fixed effects
#'       (g_0) in estimation and results; if FALSE, estimate factors only.
#'   - r: integer(1). The factor model rank (number of latent factors) to estimate
#'       within each cohort.
#'   - cohort_weighting: character(1), one of "equal" or "by_size" (default "equal").
#'       If "equal", cohorts are weighted equally with 1/C both for the point estimate and,
#'       when bootstrap is present, for each bootstrap draw. If "by_size", the point cohort
#'       weights equal the average across bootstrap draws of the per-draw cohort shares of total
#'       unit weight; each bootstrap replicate weight vector equals the per-draw cohort shares.
#'       When no bootstrap is provided, "by_size" uses cohort unit-count shares (n_c / sum n_c).
#' @param bootstrap optional WeightedBootstrap
#' @param num_threads integer number of threads (default 1L). If NULL, uses the
#'   core default (serial or TBB default, depending on build).
#' @return list with:
#'   - cohort_specific_factor_ests: named list over spec keys; each is a list over cohorts of FactorModelEstimates
#'   - cohort_outcome_means: list over cohorts of OutcomeMeanSuffStatEstimates
#'   - cohort_weights: named list over spec keys of CohortWeightEstimates
#' @export
est_cohort_specific_params <- function(panel, est_specs, bootstrap = NULL, num_threads = 1L) {
    stopifnot(inherits(panel, "UnbalancedPanel"))
    if (!is.list(est_specs) || length(est_specs) == 0L) stop("est_specs must be a non-empty list")
    if (!is.null(bootstrap) && !inherits(bootstrap, "WeightedBootstrap")) stop("bootstrap must be a WeightedBootstrap or NULL")

    # Minimal validation of specs
    for (i in seq_along(est_specs)) {
        sp <- est_specs[[i]]
        if (!is.list(sp)) stop(sprintf("est_specs[[%d]] must be a list", i))
        req <- c("factor_model_estimator", "include_outcome_fes", "r")
        miss <- setdiff(req, names(sp))
        if (length(miss) > 0L) stop(sprintf("spec %d missing fields: %s", i, paste(miss, collapse = ", ")))
    }

    pp <- panel$get_processed_panel()
    obs_idx <- panel$get_observed_outcome_indices()
    y_col <- panel$get_outcome_value_col()
    covar_cols <- panel$get_covar_cols()

    xp <- if (is.null(bootstrap)) NULL else bootstrap$.__enclos_env__$private$xp

    # If num_threads is NULL, pass NULL so that the C++ binding omits the argument
    nt <- if (is.null(num_threads)) NULL else as.integer(num_threads)

    res <- est_cohort_specific_params_from_panel_cpp(
        processed_panel = pp,
        observed_outcome_indices = obs_idx,
        outcome_value_col = y_col,
        covar_cols = covar_cols,
        est_specs = est_specs,
        bootstrap_xptr = xp,
        num_threads_in = nt
    )

    wrapped_factor <- lapply(res$cohort_specific_factor_ests, function(lst) {
        lapply(lst, function(xp) FactorModelEstimates$new(xp))
    })
    wrapped_oms <- lapply(res$cohort_outcome_means, function(xp) OutcomeMeanSuffStatEstimates$new(xp))
    wrapped_weights <- lapply(res$cohort_weights, function(xp) CohortWeightEstimates$new(xp))

    list(
        cohort_specific_factor_ests = wrapped_factor,
        cohort_outcome_means = wrapped_oms,
        cohort_weights = wrapped_weights
    )
}