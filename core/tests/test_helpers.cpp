#include "test_helpers.h"
#include <gtest/gtest.h>
#include "linear_algebra_utils.h"

#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cmath>

// -------- Utilities and assertions --------

std::vector<arma::mat> generate_rotation_matrices(size_t C, arma::uword size) {
    std::vector<arma::mat> matrices;
    for (size_t c = 0; c < C; ++c) {
        arma::mat R(size, size);
        for (arma::uword i = 0; i < size; ++i) {
            for (arma::uword j = 0; j < size; ++j) {
                R(i, j) = 0.1 * (c + 1) * (i + 1) + 0.1 * (j + 1);
            }
        }
        R.diag() += size; // Ensure diagonal dominance for full rank
        if (arma::rank(R) != size) {
            throw std::runtime_error("Generated rotation matrix is not full rank.");
        }
        matrices.push_back(R);
    }
    return matrices;
}

std::vector<std::vector<std::set<arma::uword>>> canonicalize_o3_output(
    std::vector<std::vector<std::set<arma::uword>>> output) {
    for (auto& iteration : output) {
        std::sort(iteration.begin(), iteration.end());
    }
    return output;
}

void expect_same_subspace(const arma::mat& G1, const arma::mat& G2, double tol) {
    arma::mat P1 = apm::internal::projection_matrix(G1);
    arma::mat P2 = apm::internal::projection_matrix(G2);
    double rel_tol = std::max(arma::abs(P1).max(), arma::abs(P2).max());
    ASSERT_TRUE(arma::approx_equal(P1, P2, "absdiff", rel_tol));
}

// -------- Estimation fixtures --------

StaircaseData make_staircase_data(arma::uword T, arma::uword r, arma::uword C) {
    // Unify with panel context builder: infer T_c from T and C
    arma::uword T_c = T - C + 1;
    StaircasePanelContext ctx = make_staircase_panel_context(T, r, T_c);
    StaircaseData d;
    d.true_factors = ctx.G_true;
    d.observed_outcome_indices = ctx.observed_outcome_indices;
    d.g0_true = ctx.g0_true;
    return d;
}

std::vector<apm::OutcomeMeanSufficientStatistics> make_suff_stats_vec(
    const arma::mat& true_m,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::mat>& X_c_vec) {
    const arma::uword C = observed_outcome_indices.size();
    std::vector<apm::OutcomeMeanSufficientStatistics> out;
    out.reserve(C);
    for (arma::uword c = 0; c < C; ++c) {
        arma::vec m_c = arma::vec(true_m.row(c).t()).elem(observed_outcome_indices[c]);
        out.emplace_back(m_c, X_c_vec[c]);
    }
    return out;
}

apm::FactorModelEstimates duplicate_bootstrap(const apm::FactorModelParameters& point, std::size_t B) {
    return apm::FactorModelEstimates(point, std::vector<apm::FactorModelParameters>(B, point));
}

std::vector<apm::OutcomeMeanSuffStatEstimates> duplicate_bootstrap_suff(
    const std::vector<apm::OutcomeMeanSufficientStatistics>& suff_stats_point,
    std::size_t B) {
    std::vector<apm::OutcomeMeanSuffStatEstimates> out;
    out.reserve(suff_stats_point.size());
    for (const auto& stats_point : suff_stats_point) {
        std::vector<apm::OutcomeMeanSufficientStatistics> boot_stats(B, stats_point);
        out.emplace_back(stats_point, std::move(boot_stats));
    }
    return out;
}

std::vector<apm::FactorModelEstimates> build_cohort_estimates_all_with_bootstrap(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::mat>& rotation_matrices,
    const arma::vec& g0_true,
    arma::uword q,
    std::size_t B,
    std::vector<arma::vec>& out_a_c_vec) {
    const std::size_t C = observed_outcome_indices.size();
    std::vector<apm::FactorModelEstimates> cohort_estimates;
    cohort_estimates.reserve(C);
    out_a_c_vec.clear();
    out_a_c_vec.reserve(C);
    for (std::size_t c = 0; c < C; ++c) {
        arma::mat G_c = true_factors.rows(observed_outcome_indices[c]) * rotation_matrices[c];
        arma::vec g0_c = g0_true.elem(observed_outcome_indices[c]);
        arma::vec a_c(q);
        a_c(0) = 0.5 + 0.1 * static_cast<double>(c);
        a_c(1) = 1.0 + 0.2 * static_cast<double>(c);
        out_a_c_vec.push_back(a_c);
        apm::FactorModelParameters point(G_c, g0_c, a_c);
        cohort_estimates.push_back(duplicate_bootstrap(point, B));
    }
    return cohort_estimates;
}

void run_alignment_test(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const arma::vec& cohort_weights) {
    const size_t C = observed_outcome_indices.size();
    const arma::uword r = true_factors.n_cols;
    std::vector<arma::mat> rotation_matrices = generate_rotation_matrices(C, r);

    std::vector<arma::mat> cohort_factor_matrices;
    for (size_t c = 0; c < C; ++c) {
        cohort_factor_matrices.push_back(
            true_factors.rows(observed_outcome_indices[c]) * rotation_matrices[c]);
    }

    arma::mat aligned_factors = apm::align_factors_using_apm(cohort_factor_matrices, observed_outcome_indices, cohort_weights);

    arma::mat proj_aligned = apm::internal::projection_matrix(aligned_factors);
    arma::mat proj_true = apm::internal::projection_matrix(true_factors);

    ASSERT_TRUE(arma::approx_equal(proj_aligned, proj_true, "absdiff", 1e-9));
}

void run_alignment_test(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices) {
    const size_t C = observed_outcome_indices.size();
    const arma::uword r = true_factors.n_cols;
    std::vector<arma::mat> rotation_matrices = generate_rotation_matrices(C, r);

    std::vector<arma::mat> cohort_factor_matrices;
    for (size_t c = 0; c < C; ++c) {
        cohort_factor_matrices.push_back(
            true_factors.rows(observed_outcome_indices[c]) * rotation_matrices[c]);
    }

    arma::mat aligned_factors = apm::align_factors_using_apm(cohort_factor_matrices, observed_outcome_indices);

    arma::mat proj_aligned = apm::internal::projection_matrix(aligned_factors);
    arma::mat proj_true = apm::internal::projection_matrix(true_factors);

    ASSERT_TRUE(arma::approx_equal(proj_aligned, proj_true, "absdiff", 1e-9));
}

EstimationTestData setup_estimation_test_data() {
    EstimationTestData data;
    data.G = {{1, 1}, {2, 4}, {3, 9}, {4, 16}};
    data.a = {0.5, 1.0};
    data.g_0 = arma::linspace(0.1, 0.4, data.T);
    data.observed_outcome_indices = {{0, 1, 2}, {1, 2, 3}};
    data.l_c = {{1.0, 2.0}, {3.0, 4.0}};

    arma::mat X1 = {{0.1, 0.5}, {0.2, 0.6}, {0.3, 0.7}, {0.4, 0.8}};
    arma::mat X2 = {{1.1, 1.5}, {1.2, 1.6}, {1.3, 1.7}, {1.4, 1.8}};
    data.X_c_vec = {X1, X2};
    
    return data;
}

// -------- Staircase panel helpers --------

std::vector<arma::uvec> make_staircase_observed_indices(arma::uword T, arma::uword T_c) {
    if (T_c > T) {
        throw std::invalid_argument("Staircase window length T_c exceeds T");
    }
    const arma::uword C = T - T_c + 1;
    std::vector<arma::uvec> observed;
    observed.resize(static_cast<std::size_t>(C));
    for (arma::uword c = 0; c < C; ++c) {
        observed[static_cast<std::size_t>(c)] = arma::regspace<arma::uvec>(c, c + T_c - 1);
    }
    return observed;
}

StaircasePanelContext make_staircase_panel_context(
    arma::uword T,
    arma::uword r,
    arma::uword T_c,
    arma::uword units_per,
    arma::uword q) {
    StaircasePanelContext ctx;
    ctx.T = T;
    ctx.r = r;
    ctx.T_c = std::max(r, T_c);
    ctx.units_per = std::max(units_per, ctx.T_c);
    ctx.q = q;

    // Construct a smooth, full-column-rank basis and orthonormalize via thin QR
    arma::mat B(ctx.T, r, arma::fill::zeros);
    for (arma::uword t = 0; t < ctx.T; ++t) {
        double x = (static_cast<double>(t) + 1.0) / (static_cast<double>(ctx.T) + 1.0);
        for (arma::uword j = 0; j < ctx.r; ++j) {
            B(t, j) = std::pow(x, static_cast<int>(j + 1));
        }
    }
    arma::mat Q, R;
    arma::qr_econ(Q, R, B);

    // Set seed to 42 and shuffle the rows of Q for ctx.G_true
    arma::arma_rng::set_seed(42);
    arma::uvec idx = arma::randperm(ctx.T);
    ctx.G_true = Q.rows(idx);

    ctx.observed_outcome_indices = make_staircase_observed_indices(ctx.T, ctx.T_c);
    ctx.C = static_cast<arma::uword>(ctx.observed_outcome_indices.size());
    ctx.g0_true = arma::linspace(0.1, 0.5, ctx.T);
    ctx.a_true.set_size(ctx.q);
    for (arma::uword j = 0; j < ctx.q; ++j) ctx.a_true(j) = 0.5 + 0.5 * static_cast<double>(j)/static_cast<double>(ctx.q);

    ctx.l_unit.clear();
    std::size_t total_units = static_cast<std::size_t>(ctx.C * ctx.units_per);
    ctx.l_unit.reserve(total_units);
    for (arma::uword u = 0; u < total_units; ++u) {
        arma::vec l(ctx.r);
        for (arma::uword j = 0; j < ctx.r; ++j) l(j) = 1.0 + static_cast<double>(u)/static_cast<double>(total_units) + static_cast<double>(j)/static_cast<double>(ctx.r);
        ctx.l_unit.push_back(std::move(l));
    }
    return ctx;
}

RawPanelData make_raw_panel(
    const StaircasePanelContext& ctx,
    bool with_covariates,
    bool with_auxiliary) {
    RawPanelData rp;
    int global_unit = 0;
    for (int c = 0; c < static_cast<int>(ctx.C); ++c) {
        std::vector<char> observed(static_cast<std::size_t>(ctx.T), 0);
        std::vector<int> pos(static_cast<std::size_t>(ctx.T), -1);
        for (arma::uword k = 0; k < ctx.observed_outcome_indices[static_cast<std::size_t>(c)].n_elem; ++k) {
            int t_obs = static_cast<int>(ctx.observed_outcome_indices[static_cast<std::size_t>(c)][k]);
            observed[static_cast<std::size_t>(t_obs)] = 1;
            pos[static_cast<std::size_t>(t_obs)] = static_cast<int>(k);
        }

        for (int u = 0; u < static_cast<int>(ctx.units_per); ++u, ++global_unit) {
            if (with_covariates || with_auxiliary) {
                for (int t = 0; t < static_cast<int>(ctx.T); ++t) {
                    rp.unit_idx.push_back(global_unit);
                    rp.cohort_id.push_back(c);
                    rp.outcome_idx.push_back(t);
                    if (observed[static_cast<std::size_t>(t)]) {
                        double y_val = arma::as_scalar(
                            ctx.G_true.row(static_cast<arma::uword>(t)) * ctx.l_unit[static_cast<std::size_t>(global_unit)]
                        );
                        rp.y.push_back(y_val);
                    } else {
                        rp.y.push_back(std::numeric_limits<double>::quiet_NaN());
                    }
                    if (with_covariates) {
                        rp.cov1.push_back(static_cast<double>(global_unit + 1));
                        rp.cov2.push_back(static_cast<double>(c + 1));
                    }
                    if (with_auxiliary) {
                        rp.aux1.push_back(static_cast<double>(global_unit + 1));
                        rp.aux2.push_back(static_cast<double>(10 * c + t + 1));
                    }
                }
            } else {
                for (arma::uword k = 0; k < ctx.observed_outcome_indices[static_cast<std::size_t>(c)].n_elem; ++k) {
                    int t = static_cast<int>(ctx.observed_outcome_indices[static_cast<std::size_t>(c)][k]);
                    double y_val = arma::as_scalar(
                        ctx.G_true.row(static_cast<arma::uword>(t)) * ctx.l_unit[static_cast<std::size_t>(global_unit)]
                    );
                    rp.unit_idx.push_back(global_unit);
                    rp.cohort_id.push_back(c);
                    rp.outcome_idx.push_back(t);
                    rp.y.push_back(y_val);
                }
            }
        }
    }
    return rp;
}
