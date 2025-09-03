#' Cohort-specific parameter estimation (grouped apply using data.table)
#'
#' @param panel UnbalancedPanel instance
#' @param est_specs list of estimator specs; if named, names are used as keys in the output
#'   Each spec is a list with fields:
#'     - name: "principal_components"
#'     - include_outcome_fes: logical
#'     - r: integer
#' @param bootstrap optional WeightedBootstrap
#' @return list over specs (named if est_specs is named); each element is a named list over cohort_id of FactorModelEstimates
#' @export
utils::globalVariables(c(".BY", ".SD", "cohort_id", "PCEstimator", "PCEstimatorWithFEs"))

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

        spec_res <- lapply(est_specs, function(spec) {
            est <- new_estimator_from_spec_(spec, T_c = T_c, q = q, bootstrap = bootstrap)
            if (q > 0L) {
                est$add_data(unit_idxs = yx$unit_idxs, Y = yx$Y, X = yx$X)
            } else {
                est$add_data(unit_idxs = yx$unit_idxs, Y = yx$Y)
            }
            est$estimate()
        })

        list(results = list(spec_res))
    }, by = list(cohort_id)] # nolint

    S <- length(est_specs)
    out <- lapply(seq_len(S), function(s) {
        setNames(
            lapply(seq_len(nrow(grp)), function(i) grp$results[[i]][[s]]),
            as.character(grp$cohort_id)
        )
    })

    spec_names <- names(est_specs)
    if (!is.null(spec_names) && all(nzchar(spec_names))) {
        names(out) <- spec_names
    } else {
        names(out) <- paste0("spec_", seq_len(S))
    }
    out
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
    # Keep only observed outcomes for this cohort to avoid duplicates / NA columns
    dt_obs <- dt[outcome_idx %in% T_idx]
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
        return(list(Y = Y, X = NULL, unit_idxs = unit_idxs))
    }
    N_c <- nrow(Y); T_c <- ncol(Y)
    X <- array(0.0, dim = c(N_c, T_c, q))
    for (j in seq_along(covar_cols)) {
        colj <- covar_cols[[j]]
        dt_x <- data.table::data.table(
            unit_idx = dt_obs[["unit_idx"]],
            outcome_idx_f = factor(dt_obs[["outcome_idx"]], levels = T_idx),
            value = dt_obs[[colj]]
        )
        x_wide <- data.table::dcast(
            dt_x,
            unit_idx ~ outcome_idx_f,
            value.var = "value",
            fill = NA_real_
        )
        if (!identical(x_wide[["unit_idx"]], y_wide[["unit_idx"]])) stop("Row misalignment between Y and X during dcast")
        X[, , j] <- as.matrix(x_wide[, -"unit_idx"])
    }
    list(Y = Y, X = X, unit_idxs = unit_idxs)
}


