library(data.table)
utils::globalVariables(c(
    "y", "outcome_id", "cov1", "cov2", "outcome_idx",
    "unit_idx", "cohort_id", "outcome_idx_f", "aux1", "aux2"
))

# Outcome helpers
make_outcomes <- function(T) {
    LETTERS[seq_len(T)]
}

make_staircase_observed_indices <- function(T, window) {
    lapply(seq_len(T - window + 1L), function(start) as.integer(seq.int(start, start + window - 1L)))
}

make_units_by_cohort <- function(n_cohorts, units_per_cohort, prefix = "u", start_index = 1L) {
    idx <- seq.int(start_index, length.out = n_cohorts * units_per_cohort)
    split(sprintf("%s%d", prefix, idx), rep(seq_len(n_cohorts), each = units_per_cohort))
}

# Panel builders
build_panel_from_indices <- function(outcomes, cohort_indices, units_by_cohort, include_covariates = FALSE, include_auxiliary = FALSE) {
    all_units <- sort(unique(unlist(units_by_cohort)))
    data.table::rbindlist(lapply(seq_along(cohort_indices), function(k) {
        observed_idxs <- cohort_indices[[k]]
        observed_outcomes <- outcomes[observed_idxs]
        unit_ids <- units_by_cohort[[k]]
        data.table::rbindlist(lapply(unit_ids, function(u) {
            if (isTRUE(include_covariates) || isTRUE(include_auxiliary)) {
                dt <- data.table::data.table(
                    unit_id = u,
                    outcome_id = outcomes
                )
                dt[, ("y") := match(get("outcome_id"), outcomes)]
                dt[!get("outcome_id") %in% observed_outcomes, ("y") := NA_real_]
                if (isTRUE(include_covariates)) {
                    dt[, ("cov1") := as.integer(match(u, all_units))]
                    dt[, ("cov2") := as.integer(k)]
                }
                if (isTRUE(include_auxiliary)) {
                    dt[, ("aux1") := as.integer(match(u, all_units))]
                    dt[, ("aux2") := as.integer(k)]
                }
                dt
            } else {
                dt <- data.table::data.table(
                    unit_id = u,
                    outcome_id = observed_outcomes
                )
                dt[, ("y") := match(get("outcome_id"), outcomes)]
                dt
            }
        }))
    }))
}

# Factor model helpers ---------------------------------------------------------

# Deterministic general true factor generator used in tests
make_true_factors_general <- function(T, r) {
    if (T == 5L && r == 2L) return(make_true_factors_5x2())
    # Build a deterministic, full-rank, well-conditioned T x r matrix
    # using orthonormal columns from a smooth basis, to avoid near-collinearity.
    t <- seq_len(T)
    base <- sapply(seq_len(r), function(j) {
        # Mix sine and cosine at different frequencies deterministically
        sin(2 * pi * j * t / (T + 1)) + 0.5 * cos(2 * pi * (j + 1) * t / (T + 1))
    })
    # Orthonormalize columns via QR
    qr_fac <- qr(base)
    Q <- qr.Q(qr_fac)
    Q[, seq_len(r), drop = FALSE]
}

make_rotations <- function(C, r, rotate = TRUE) {
    if (isTRUE(rotate)) generate_symmetric_rotation_matrices(C, r) else rep(list(diag(r)), C)
}

unit_loading_from_all_units <- function(u_name, all_units, r) {
    u_idx <- match(u_name, all_units)
    as.numeric(0.3 * u_idx + 0.1 * seq_len(r))
}

build_factor_model_context <- function(outcomes, cohort_indices, units_by_cohort, r = 2L, rotate = TRUE) {
    all_units <- sort(unique(unlist(units_by_cohort)))
    T <- length(outcomes)
    r <- as.integer(r)
    true_factors <- make_true_factors_general(T, r)
    rotations <- make_rotations(length(cohort_indices), r, rotate)
    cohort_G_list <- lapply(seq_along(cohort_indices), function(cid) {
        idx <- as.integer(cohort_indices[[cid]])
        as.matrix(true_factors[idx, , drop = FALSE] %*% rotations[[cid]])
    })
    list(
        outcomes = outcomes,
        cohort_indices = cohort_indices,
        units_by_cohort = units_by_cohort,
        all_units = all_units,
        r = r,
        true_factors = true_factors,
        rotations = rotations,
        cohort_G_list = cohort_G_list
    )
}

expected_Y_for_units_ctx <- function(ctx, cohort_id, unit_ids, T_idx) {
    G_c <- ctx$true_factors[T_idx, ]
    do.call(rbind, lapply(unit_ids, function(u) {
        l_u <- unit_loading_from_all_units(u, ctx$all_units, ctx$r)
        if (is.null(dim(G_c))) {
            # G_c is a vector (length T), l_u is scalar
            as.numeric(G_c * l_u)
        } else {
            # G_c is a matrix (T x r), l_u is a vector (r)
            as.numeric(G_c %*% l_u)
        }
    }))
}

expected_covariates_for_units_ctx <- function(ctx, cohort_id, unit_ids, T_idx) {
    N <- length(unit_ids)
    TT <- length(T_idx)
    cov1 <- matrix(match(unit_ids, ctx$all_units), nrow = N, ncol = TT)
    cov2 <- matrix(as.integer(cohort_id), nrow = N, ncol = TT)
    array(c(cov1, cov2), dim = c(N, TT, 2L))
}

# Factor-based outcome generator to avoid signature conflicts with older helpers
build_panel_from_indices_factor <- function(outcomes, cohort_indices, units_by_cohort,
                                            include_covariates = FALSE,
                                            include_auxiliary = FALSE,
                                            r = 2L,
                                            rotate = TRUE,
                                            ctx = NULL) {
    if (is.null(ctx)) {
        ctx <- build_factor_model_context(outcomes, cohort_indices, units_by_cohort, r = r, rotate = rotate)
    }

    data.table::rbindlist(lapply(seq_along(cohort_indices), function(k) {
        observed_idxs <- cohort_indices[[k]]
        observed_outcomes <- outcomes[observed_idxs]
        unit_ids <- units_by_cohort[[k]]
        data.table::rbindlist(lapply(unit_ids, function(u) {
            if (isTRUE(include_covariates) || isTRUE(include_auxiliary)) {
                dt <- data.table::data.table(
                    unit_id = u,
                    outcome_id = outcomes
                )
                dt[, ("y") := NA_real_]
                y_obs <- expected_Y_for_units_ctx(ctx, k, unit_ids = c(u), T_idx = observed_idxs)[1, ]
                dt[get("outcome_id") %in% observed_outcomes, ("y") := y_obs]
                if (isTRUE(include_covariates)) {
                    dt[, ("cov1") := as.integer(match(u, ctx$all_units))]
                    dt[, ("cov2") := as.integer(k)]
                }
                if (isTRUE(include_auxiliary)) {
                    dt[, ("aux1") := as.integer(match(u, ctx$all_units))]
                    dt[, ("aux2") := as.integer(k)]
                }
                dt
            } else {
                dt <- data.table::data.table(
                    unit_id = u,
                    outcome_id = observed_outcomes
                )
                y_obs <- expected_Y_for_units_ctx(ctx, k, unit_ids = c(u), T_idx = observed_idxs)[1, ]
                dt[, ("y") := y_obs]
                dt
            }
        }))
    }))
}

build_expected_unit_map <- function(units_by_cohort) {
    data.table::rbindlist(mapply(function(units, cid) {
        data.table::data.table(unit_id = units, cohort_id = cid)
    }, units_by_cohort, seq_along(units_by_cohort), SIMPLIFY = FALSE))
}

build_expected_processed_panel <- function(outcomes, cohort_indices, units_by_cohort, include_covariates = TRUE) {
    all_units <- sort(unique(unlist(units_by_cohort)))
    data.table::rbindlist(lapply(seq_along(cohort_indices), function(cid) {
        observed_idxs <- cohort_indices[[cid]]
        data.table::rbindlist(lapply(units_by_cohort[[cid]], function(u) {
            if (isTRUE(include_covariates)) {
                dt <- data.table::data.table(
                    unit_id = u,
                    cohort_id = cid,
                    outcome_idx = seq_along(outcomes)
                )
                dt[, ("y") := as.integer(get("outcome_idx"))]
                dt[!get("outcome_idx") %in% observed_idxs, ("y") := NA_integer_]
                dt[, ("cov1") := as.integer(match(u, all_units))]
                dt[, ("cov2") := as.integer(cid)]
                dt
            } else {
                data.table::data.table(
                    unit_id = u,
                    cohort_id = cid,
                    outcome_idx = observed_idxs,
                    y = as.integer(observed_idxs)
                )
            }
        }))
    }))
}

# Linear algebra helpers
projection_matrix_r <- function(X) {
    if (ncol(X) == 0) {
        return(matrix(0, nrow = nrow(X), ncol = nrow(X)))
    }
    svd_X <- svd(X)
    tol <- max(dim(X)) * max(svd_X$d) * .Machine$double.eps
    rank <- sum(svd_X$d > tol)
    if (rank == 0) {
        return(matrix(0, nrow = nrow(X), ncol = nrow(X)))
    }
    U <- svd_X$u[, 1:rank, drop = FALSE]
    U %*% t(U)
}

generate_symmetric_rotation_matrices <- function(C, r) {
    lapply(1:C, function(c) {
        R <- matrix(0, nrow = r, ncol = r)
        for (i in 1:r) {
            for (j in 1:r) {
                R[i, j] <- 0.1 * c * i + 0.1 * j
            }
        }
        R <- (R + t(R)) / 2
        diag(R) <- diag(R) + r
        R
    })
}

run_alignment_test_r <- function(true_factors,
                                 observed_outcome_indices,
                                 cohort_weights = NULL) {
    C <- length(observed_outcome_indices)
    r <- ncol(true_factors)
    rotation_matrices <- generate_symmetric_rotation_matrices(C, r)

    cohort_factor_matrices <- lapply(1:C, function(c) {
        true_factors[observed_outcome_indices[[c]], ] %*% rotation_matrices[[c]]
    })

    if (is.null(cohort_weights)) {
        aligned_factors <- align_factors_using_apm(
            cohort_factor_matrices,
            observed_outcome_indices
        )
    } else {
        args <- list(cohort_factor_matrices, observed_outcome_indices, cohort_weights = cohort_weights)
        aligned_factors <- do.call(align_factors_using_apm, args)
    }

    proj_aligned <- projection_matrix_r(aligned_factors)
    proj_true <- projection_matrix_r(true_factors)

    testthat::expect_equal(proj_aligned, proj_true, tolerance = 1e-9)
}

# Estimation data
setup_estimation_test_data_r <- function() {
    list(
        Tval = 4, r = 2, Cval = 2, q = 2,
        G = matrix(c(1, 1, 2, 4, 3, 9, 4, 16), nrow = 4, ncol = 2, byrow = TRUE),
        a = c(0.5, 1.0),
        g_0 = seq(0.1, 0.4, by = 0.1),
        observed_outcome_indices = list(c(1, 2, 3), c(2, 3, 4)),
        l_c = list(c(1.0, 2.0), c(3.0, 4.0)),
        X_c_vec = list(
            matrix(c(0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.4, 0.8), nrow = 4, ncol = 2, byrow = TRUE),
            matrix(c(1.1, 1.5, 1.2, 1.6, 1.3, 1.7, 1.4, 1.8), nrow = 4, ncol = 2, byrow = TRUE)
        )
    )
}

# O3 canonicalizer
canonicalize_o3_output <- function(output) {
    lapply(output, function(iteration) {
        sorted_scs <- lapply(iteration, sort)
        sorted_scs[order(sapply(sorted_scs, function(x) paste(x, collapse = "_")))]
    })
}

# Specific true factors used across core tests
make_true_factors_5x2 <- function() {
    matrix(c(
        0.1, 0.6,
        0.2, 0.7,
        0.3, 0.8,
        0.4, 0.9,
        0.5, 1.0
    ), nrow = 5, ncol = 2, byrow = TRUE)
}