#' Cohort-specific parameter estimation (grouped apply using data.table)
#'
#' @param panel UnbalancedPanel instance
#' @param est_specs list of estimator specs; if named, names are used as keys in the output
#'   Each spec is a list with fields:
#'     - name: "principal_components"
#'     - include_outcome_fes: logical
#'     - r: integer
#' @param bootstrap optional WeightedBootstrap
#' @return list with elements:
#'   - cohort_specific_factor_ests: list over specs (named if est_specs is named); each element is a named list over cohort_id of FactorModelEstimates
#'   - cohort_outcome_means: named list over cohort_id of OutcomeMeanSuffStatEstimates
#' @export
utils::globalVariables(c(
    ".BY", ".SD", ".",
    "cohort_id", "outcome_idx", "unit_idx",
    "PCEstimator", "PCEstimatorWithFEs", "OutcomeMeanSuffStatEstimator"
))

est_cohort_specific_params <- function(panel, est_specs, bootstrap = NULL) {
    stopifnot(inherits(panel, "UnbalancedPanel"))
    if (!is.list(est_specs) || length(est_specs) == 0L) stop("est_specs must be a non-empty list")
    if (!is.null(bootstrap) && !inherits(bootstrap, "WeightedBootstrap")) stop("bootstrap must be a WeightedBootstrap or NULL")
    validate_est_specs_(est_specs)

    pp <- panel$get_processed_panel()
    covar_cols <- panel$get_covar_cols()
    obs_idx_list <- panel$get_observed_outcome_indices()

    grp <- pp[, {
        cid <- .BY$cohort_id # nolint
        T_idx <- obs_idx_list[[cid]]

        yx <- build_Y_X_for_group_(.SD, T_idx, covar_cols) # nolint
        T_c <- ncol(yx$Y)
        q <- length(covar_cols)

        spec_res <- run_factor_estimators_for_group_(
            yx = yx, est_specs = est_specs, bootstrap = bootstrap,
            q = q, T_c = T_c, T_idx = T_idx
        )

        omsse <- compute_outcome_means_for_group_(
            yx = yx, q = q, T_c = T_c, bootstrap = bootstrap
        )

        list(factor_model_params = list(spec_res), outcome_means = list(omsse))
    }, by = list(cohort_id)] # nolint

    out_factor <- assemble_factor_results_(grp, est_specs)

    spec_names <- names(est_specs)
    if (!is.null(spec_names) && all(nzchar(spec_names))) {
        names(out_factor) <- spec_names
    } else {
        names(out_factor) <- paste0("spec_", seq_len(length(est_specs)))
    }

    out_outcome_means <- assemble_outcome_means_(grp)

    list(
        cohort_specific_factor_ests = out_factor,
        cohort_outcome_means = out_outcome_means
    )
}

# Internal: validate spec list
validate_est_specs_ <- function(est_specs) {
    for (i in seq_along(est_specs)) {
        sp <- est_specs[[i]]
        if (!is.list(sp)) stop(sprintf("spec %d must be a list", i))
        req <- c("name", "include_outcome_fes", "r")
        miss <- setdiff(req, names(sp))
        if (length(miss) > 0L) stop(sprintf("spec %d missing fields: %s", i, paste(miss, collapse = ", ")))
        if (!identical(sp$name, "principal_components")) stop("Only 'principal_components' is supported currently")
        if (!is.logical(sp$include_outcome_fes) || length(sp$include_outcome_fes) != 1L) stop("include_outcome_fes must be logical(1)")
        if (!is.numeric(sp$r) || length(sp$r) != 1L || sp$r < 1L) stop("r must be a positive integer")
    }
    invisible(TRUE)
}

# Internal: instantiate estimator from spec
new_estimator_from_spec_ <- function(spec, T_c, q, bootstrap) {
    r <- as.integer(spec$r)
    T_c <- as.integer(T_c)
    q <- as.integer(q)
    if (isTRUE(spec$include_outcome_fes)) {
        PCEstimatorWithFEs$new(r = r, T_c = T_c, bootstrap = bootstrap, q = q) # nolint
    } else {
        PCEstimator$new(r = r, T_c = T_c, bootstrap = bootstrap, q = q) # nolint
    }
}

# Internal: build Y and X using .SD for the current cohort
# .SD columns: unit_idx, outcome_idx, y, [covars...]
build_Y_X_for_group_ <- function(sd, T_idx, covar_cols) {
    dt <- data.table::as.data.table(sd)
    # Keep only observed outcomes for this cohort to avoid duplicates / NA columns (for Y)
    dt_obs <- dt[dt[["outcome_idx"]] %in% T_idx]
    # Build an explicit reshaping table to avoid in-place modifications (no :=)
    dt_y <- data.table::data.table(
        unit_idx = dt_obs[["unit_idx"]],
        outcome_idx_f = factor(dt_obs[["outcome_idx"]], levels = T_idx),
        y = dt_obs[["y"]]
    )

    y_wide <- data.table::dcast(
        dt_y,
        unit_idx ~ outcome_idx_f,
        value.var = "y",
        fill = NA_real_
    )
    unit_idxs <- as.integer(y_wide[["unit_idx"]])
    Y <- as.matrix(y_wide[, -"unit_idx"])

    q <- length(covar_cols)
    if (q == 0L) {
        # Also carry the full outcome index order for consumers
        T_all <- sort(unique(dt[["outcome_idx"]]))
        return(list(Y = Y, X = NULL, unit_idxs = unit_idxs, T_all = T_all))
    }

    # Build covariates over all outcomes (N x T x q), aligned to Y's units
    T_all <- sort(unique(dt[["outcome_idx"]]))
    units_keep <- y_wide[["unit_idx"]]
    N_c <- nrow(Y); T_full <- length(T_all)
    X <- array(NA_real_, dim = c(N_c, T_full, q))
    for (j in seq_along(covar_cols)) {
        colj <- covar_cols[[j]]
        dt_x <- data.table::data.table(
            unit_idx = dt[["unit_idx"]],
            outcome_idx_f = factor(dt[["outcome_idx"]], levels = T_all),
            value = dt[[colj]]
        )
        x_wide <- data.table::dcast(
            dt_x,
            unit_idx ~ outcome_idx_f,
            value.var = "value",
            fill = NA_real_
        )
        # Align rows to units_keep
        x_wide <- merge(
            data.table::data.table(unit_idx = units_keep),
            x_wide,
            by = "unit_idx",
            all.x = TRUE,
            sort = FALSE
        )
        X[, , j] <- as.matrix(x_wide[, -"unit_idx"])
    }
    list(Y = Y, X = X, unit_idxs = unit_idxs, T_all = T_all)
}

# Internal: run factor estimators for a cohort group
run_factor_estimators_for_group_ <- function(yx, est_specs, bootstrap, q, T_c, T_idx) {
    X_obs <- if (!is.null(yx$X)) yx$X[, match(T_idx, yx$T_all), , drop = FALSE] else NULL
    lapply(est_specs, function(spec) {
        est <- new_estimator_from_spec_(spec, T_c = T_c, q = q, bootstrap = bootstrap)
        if (q > 0L) {
            est$add_data(unit_idxs = yx$unit_idxs, Y = yx$Y, X = X_obs)
        } else {
            est$add_data(unit_idxs = yx$unit_idxs, Y = yx$Y)
        }
        est$estimate()
    })
}

# Internal: compute outcome mean sufficient statistics for a cohort group
compute_outcome_means_for_group_ <- function(yx, q, T_c, bootstrap) {
    T_all_len <- length(yx$T_all)
    if (q > 0L) {
        omsse_est <- OutcomeMeanSuffStatEstimator$new(T_c = T_c, T = T_all_len, q = q, bootstrap = bootstrap)
        omsse_est$add_data(unit_idxs = yx$unit_idxs, Y = yx$Y, X = yx$X)
        omsse_est$estimate()
    } else {
        omsse_est <- OutcomeMeanSuffStatEstimator$new(T_c = T_c, T = T_all_len, q = 0L, bootstrap = bootstrap)
        omsse_est$add_data(unit_idxs = yx$unit_idxs, Y = yx$Y)
        omsse_est$estimate()
    }
}

# Internal: assemble factor estimator results into spec-named lists indexed by cohort_id
assemble_factor_results_ <- function(grp, est_specs) {
    S <- length(est_specs)
    cohort_ids_vec <- as.integer(grp$cohort_id)
    max_cohort_id <- if (length(cohort_ids_vec) > 0L) max(cohort_ids_vec) else 0L
    out_factor <- lapply(seq_len(S), function(s) {
        lst <- vector("list", max_cohort_id)
        for (i in seq_len(nrow(grp))) {
            cid <- cohort_ids_vec[[i]]
            lst[[cid]] <- grp$factor_model_params[[i]][[s]]
        }
        lst
    })
    spec_names <- names(est_specs)
    if (!is.null(spec_names) && all(nzchar(spec_names))) {
        names(out_factor) <- spec_names
    } else {
        names(out_factor) <- paste0("spec_", seq_len(S))
    }
    out_factor
}

# Internal: assemble outcome means list indexed by cohort_id
assemble_outcome_means_ <- function(grp) {
    cohort_ids_vec <- as.integer(grp$cohort_id)
    max_cohort_id <- if (length(cohort_ids_vec) > 0L) max(cohort_ids_vec) else 0L
    lst <- vector("list", max_cohort_id)
    for (i in seq_len(nrow(grp))) {
        cid <- cohort_ids_vec[[i]]
        lst[[cid]] <- grp$outcome_means[[i]]
    }
    lst
}


