#include "test_helpers.h"
#include <gtest/gtest.h>
#include "linear_algebra_utils.h"

#include <stdexcept>
#include <algorithm>

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

StaircaseData make_staircase_data(arma::uword T, arma::uword r, arma::uword C) {
    StaircaseData d;
    d.true_factors.set_size(T, r);
    for (arma::uword t = 0; t < T; ++t) {
        double v = 0.1 * static_cast<double>(t + 1);
        d.true_factors(t, 0) = v;
        if (r >= 2) d.true_factors(t, 1) = v + 0.5;
        for (arma::uword j = 2; j < r; ++j) {
            d.true_factors(t, j) = v + 0.1 * static_cast<double>(j);
        }
    }

    const arma::uword win = T - C + 1;
    d.observed_outcome_indices.resize(C);
    for (arma::uword c = 0; c < C; ++c) {
        d.observed_outcome_indices[c] = arma::regspace<arma::uvec>(c, c + win - 1);
    }

    d.g0_true = arma::linspace(0.1, 0.1 * static_cast<double>(T), T);
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

std::vector<std::vector<std::set<arma::uword>>> canonicalize_o3_output(
    std::vector<std::vector<std::set<arma::uword>>> output) {
    for (auto& iteration : output) {
        std::sort(iteration.begin(), iteration.end());
    }
    return output;
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


