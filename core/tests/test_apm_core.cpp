#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include "apm_core.h"
#include "linear_algebra_utils.h"

namespace { // Anonymous namespace for test helpers

/**
 * @brief Generates a vector of deterministic, full-rank square matrices.
 *
 * This function creates C square matrices of size `size x size`. The matrices
 * are generated deterministically and are made to be strongly diagonally
 * dominant to ensure they are full-rank and well-conditioned.
 *
 * @param C The number of matrices to generate.
 * @param size The dimension of each square matrix.
 * @return A std::vector containing the generated matrices.
 */
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

struct StaircaseData {
    arma::mat true_factors;                       // T x r
    std::vector<arma::uvec> observed_outcome_indices; // size C, sliding windows
    arma::vec g0_true;                            // length T
};

// Generate a canonical "staircase" pattern used across tests.
// Defaults: T=5, r=2, C=3 with sliding window length = T - C + 1.
StaircaseData make_staircase_data(arma::uword T = 5, arma::uword r = 2, arma::uword C = 3) {
    StaircaseData d;
    // Construct simple true factors for r=2: col1 = 0.1..0.1*T, col2 = col1 + 0.5
    d.true_factors.set_size(T, r);
    for (arma::uword t = 0; t < T; ++t) {
        double v = 0.1 * static_cast<double>(t + 1);
        d.true_factors(t, 0) = v;
        if (r >= 2) d.true_factors(t, 1) = v + 0.5;
        for (arma::uword j = 2; j < r; ++j) {
            d.true_factors(t, j) = v + 0.1 * static_cast<double>(j);
        }
    }

    // Sliding window observed outcomes
    const arma::uword win = T - C + 1;
    d.observed_outcome_indices.resize(C);
    for (arma::uword c = 0; c < C; ++c) {
        d.observed_outcome_indices[c] = arma::regspace<arma::uvec>(c, c + win - 1);
    }

    // True g_0: 0.1, 0.2, ..., 0.1*T
    d.g0_true = arma::linspace(0.1, 0.1 * static_cast<double>(T), T);
    return d;
}

// Build OutcomeMeanSufficientStatistics per cohort from true mean matrix and X_c
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

// Duplicate bootstrap replicates for parameters
apm::FactorModelEstimates duplicate_bootstrap(const apm::FactorModelParameters& point, std::size_t B) {
    return apm::FactorModelEstimates(point, std::vector<apm::FactorModelParameters>(B, point));
}

// Duplicate bootstrap replicates for sufficient statistics per cohort
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

// Build cohort estimates (G, g_0, a) with B identical bootstrap replicates
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

/**
 * @brief Canonicalizes the nested list output from o3_algorithm for stable comparison.
 *
 * This function sorts the inner vectors of super-cohorts. The std::set
 * within each super-cohort already guarantees sorted order of cohort indices.
 * Sorting the vector of sets provides a canonical representation for each iteration's state.
 *
 * @param output The nested vector structure from o3_algorithm.
 * @return A canonicalized copy of the input.
 */
std::vector<std::vector<std::set<arma::uword>>> canonicalize_o3_output(
    std::vector<std::vector<std::set<arma::uword>>> output) {
    for (auto& iteration : output) {
        // std::set has operator<, so we can sort the vector of sets directly.
        std::sort(iteration.begin(), iteration.end());
    }
    return output;
}

void run_alignment_test(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const arma::vec& cohort_weights) {
    
    // Create differentially rotated cohort-specific factor matrices 
    // by right-multiplying the true factors by arbitrary full-rank matrices.
    const size_t C = observed_outcome_indices.size();
    const arma::uword r = true_factors.n_cols;
    std::vector<arma::mat> rotation_matrices = generate_rotation_matrices(C, r);
    
    std::vector<arma::mat> cohort_factor_matrices;
    for (size_t c = 0; c < C; ++c) {
        cohort_factor_matrices.push_back(
            true_factors.rows(observed_outcome_indices[c]) * rotation_matrices[c]);
    }

    // Run the alignment function.
    arma::mat aligned_factors = apm::align_factors_using_apm(cohort_factor_matrices, observed_outcome_indices, cohort_weights);

    // Check the result. The column space of the aligned factors should be the
    // same as the true factors. We test this by comparing their projection matrices.
    arma::mat proj_aligned = apm::internal::projection_matrix(aligned_factors);
    arma::mat proj_true = apm::internal::projection_matrix(true_factors);

    ASSERT_TRUE(arma::approx_equal(proj_aligned, proj_true, "absdiff", 1e-9))
        << "Projection matrix of aligned factors does not match projection matrix of true factors." << std::endl
        << "proj_aligned:" << std::endl << proj_aligned << std::endl
        << "proj_true:" << std::endl << proj_true;
}

void run_alignment_test(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices) {
    
    // Create differentially rotated cohort-specific factor matrices 
    // by right-multiplying the true factors by arbitrary full-rank matrices.
    const size_t C = observed_outcome_indices.size();
    const arma::uword r = true_factors.n_cols;
    std::vector<arma::mat> rotation_matrices = generate_rotation_matrices(C, r);
    
    std::vector<arma::mat> cohort_factor_matrices;
    for (size_t c = 0; c < C; ++c) {
        cohort_factor_matrices.push_back(
            true_factors.rows(observed_outcome_indices[c]) * rotation_matrices[c]);
    }

    // Run the alignment function.
    arma::mat aligned_factors = apm::align_factors_using_apm(cohort_factor_matrices, observed_outcome_indices);

    // Check the result. The column space of the aligned factors should be the
    // same as the true factors. We test this by comparing their projection matrices.
    arma::mat proj_aligned = apm::internal::projection_matrix(aligned_factors);
    arma::mat proj_true = apm::internal::projection_matrix(true_factors);

    ASSERT_TRUE(arma::approx_equal(proj_aligned, proj_true, "absdiff", 1e-9))
        << "Projection matrix of aligned factors does not match projection matrix of true factors." << std::endl
        << "proj_aligned:" << std::endl << proj_aligned << std::endl
        << "proj_true:" << std::endl << proj_true;
}

} // anonymous namespace

TEST(APMTest, GetVersionTest) {
    std::string version = apm::get_version();
    EXPECT_EQ(version, "0.1.0");
}

TEST(APMTest, AlignFactorsAPMStaircasePattern) {
    auto s = make_staircase_data();
    run_alignment_test(s.true_factors, s.observed_outcome_indices);
}

TEST(APMTest, AlignFactorsAPMStaircasePatternWeighted) {
    auto s = make_staircase_data();
    arma::vec cohort_weights = {1.0, 2.0, 1.0};
    run_alignment_test(s.true_factors, s.observed_outcome_indices, cohort_weights);
}

// Helper struct and function for estimation tests
struct EstimationTestData {
    arma::uword T = 4, r = 2, C = 2, q = 2;
    arma::mat G;
    arma::vec a;
    arma::vec g_0;
    std::vector<arma::mat> X_c_vec;
    std::vector<arma::uvec> observed_outcome_indices;
    std::vector<arma::vec> l_c;
};

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

TEST(APMTest, EstimateMeans_FactorsCovariatesFixedEffects) {
    auto data = setup_estimation_test_data();
    
    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0] + data.X_c_vec[0] * data.a + data.g_0).t();
    true_m.row(1) = (data.G * data.l_c[1] + data.X_c_vec[1] * data.a + data.g_0).t();

    std::vector<arma::vec> m_c_vec = {
        arma::vec(true_m.row(0).t()).elem(data.observed_outcome_indices[0]),
        arma::vec(true_m.row(1).t()).elem(data.observed_outcome_indices[1])
    };

    arma::mat estimated_m = apm::estimate_outcome_means_across_cohorts(
        data.G, data.g_0, data.a, data.observed_outcome_indices, m_c_vec, data.X_c_vec);

    ASSERT_TRUE(arma::approx_equal(estimated_m, true_m, "absdiff", 1e-9));
}

TEST(APMTest, EstimateMeans_FactorsAndFixedEffects) {
    auto data = setup_estimation_test_data();
    
    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0] + data.g_0).t();
    true_m.row(1) = (data.G * data.l_c[1] + data.g_0).t();

    std::vector<arma::vec> m_c_vec = {
        arma::vec(true_m.row(0).t()).elem(data.observed_outcome_indices[0]),
        arma::vec(true_m.row(1).t()).elem(data.observed_outcome_indices[1])
    };

    arma::mat estimated_m = apm::estimate_outcome_means_across_cohorts(
        data.G, data.g_0, data.observed_outcome_indices, m_c_vec);

    ASSERT_TRUE(arma::approx_equal(estimated_m, true_m, "absdiff", 1e-9));
}

TEST(APMTest, EstimateMeans_FactorsAndCovariates) {
    auto data = setup_estimation_test_data();
    
    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0] + data.X_c_vec[0] * data.a).t();
    true_m.row(1) = (data.G * data.l_c[1] + data.X_c_vec[1] * data.a).t();

    std::vector<arma::vec> m_c_vec = {
        arma::vec(true_m.row(0).t()).elem(data.observed_outcome_indices[0]),
        arma::vec(true_m.row(1).t()).elem(data.observed_outcome_indices[1])
    };

    arma::mat estimated_m = apm::estimate_outcome_means_across_cohorts(
        data.G, data.a, data.observed_outcome_indices, m_c_vec, data.X_c_vec);

    ASSERT_TRUE(arma::approx_equal(estimated_m, true_m, "absdiff", 1e-9));
}

TEST(APMTest, EstimateMeans_FactorsOnly) {
    auto data = setup_estimation_test_data();
    
    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0]).t();
    true_m.row(1) = (data.G * data.l_c[1]).t();

    std::vector<arma::vec> m_c_vec = {
        arma::vec(true_m.row(0).t()).elem(data.observed_outcome_indices[0]),
        arma::vec(true_m.row(1).t()).elem(data.observed_outcome_indices[1])
    };

    arma::mat estimated_m = apm::estimate_outcome_means_across_cohorts(
        data.G, data.observed_outcome_indices, m_c_vec);

    ASSERT_TRUE(arma::approx_equal(estimated_m, true_m, "absdiff", 1e-9));
}

TEST(APMTest, EstimateMeans_ParamsAndSuffStats_AllComponents) {
    auto data = setup_estimation_test_data();

    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0] + data.X_c_vec[0] * data.a + data.g_0).t();
    true_m.row(1) = (data.G * data.l_c[1] + data.X_c_vec[1] * data.a + data.g_0).t();

    apm::FactorModelParameters params(data.G, data.g_0, data.a);

    std::vector<apm::OutcomeMeanSufficientStatistics> suff_stats_vec;
    suff_stats_vec.reserve(data.C);
    for (arma::uword c = 0; c < data.C; ++c) {
        arma::vec m_c = arma::vec(true_m.row(c).t()).elem(data.observed_outcome_indices[c]);
        suff_stats_vec.emplace_back(m_c, data.X_c_vec[c]);
    }

    arma::mat estimated_m = apm::estimate_outcome_means_across_cohorts(
        params, data.observed_outcome_indices, suff_stats_vec);

    ASSERT_TRUE(arma::approx_equal(estimated_m, true_m, "absdiff", 1e-9));
}

TEST(APMTest, EstimateMeans_ParamsAndSuffStats_FixedEffectsOnly) {
    auto data = setup_estimation_test_data();

    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0] + data.g_0).t();
    true_m.row(1) = (data.G * data.l_c[1] + data.g_0).t();

    apm::FactorModelParameters params(data.G, data.g_0, std::nullopt);

    std::vector<apm::OutcomeMeanSufficientStatistics> suff_stats_vec;
    suff_stats_vec.reserve(data.C);
    for (arma::uword c = 0; c < data.C; ++c) {
        arma::vec m_c = arma::vec(true_m.row(c).t()).elem(data.observed_outcome_indices[c]);
        suff_stats_vec.emplace_back(m_c, std::nullopt);
    }

    arma::mat estimated_m = apm::estimate_outcome_means_across_cohorts(
        params, data.observed_outcome_indices, suff_stats_vec);

    ASSERT_TRUE(arma::approx_equal(estimated_m, true_m, "absdiff", 1e-9));
}

TEST(APMTest, EstimateMeans_ParamsAndSuffStats_CovariatesOnly) {
    auto data = setup_estimation_test_data();

    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0] + data.X_c_vec[0] * data.a).t();
    true_m.row(1) = (data.G * data.l_c[1] + data.X_c_vec[1] * data.a).t();

    apm::FactorModelParameters params(data.G, std::nullopt, data.a);

    std::vector<apm::OutcomeMeanSufficientStatistics> suff_stats_vec;
    suff_stats_vec.reserve(data.C);
    for (arma::uword c = 0; c < data.C; ++c) {
        arma::vec m_c = arma::vec(true_m.row(c).t()).elem(data.observed_outcome_indices[c]);
        suff_stats_vec.emplace_back(m_c, data.X_c_vec[c]);
    }

    arma::mat estimated_m = apm::estimate_outcome_means_across_cohorts(
        params, data.observed_outcome_indices, suff_stats_vec);

    ASSERT_TRUE(arma::approx_equal(estimated_m, true_m, "absdiff", 1e-9));
}

TEST(APMTest, EstimateMeans_ParamsAndSuffStats_FactorsOnly) {
    auto data = setup_estimation_test_data();

    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0]).t();
    true_m.row(1) = (data.G * data.l_c[1]).t();

    apm::FactorModelParameters params(data.G, std::nullopt, std::nullopt);

    std::vector<apm::OutcomeMeanSufficientStatistics> suff_stats_vec;
    suff_stats_vec.reserve(data.C);
    for (arma::uword c = 0; c < data.C; ++c) {
        arma::vec m_c = arma::vec(true_m.row(c).t()).elem(data.observed_outcome_indices[c]);
        suff_stats_vec.emplace_back(m_c, std::nullopt);
    }

    arma::mat estimated_m = apm::estimate_outcome_means_across_cohorts(
        params, data.observed_outcome_indices, suff_stats_vec);

    ASSERT_TRUE(arma::approx_equal(estimated_m, true_m, "absdiff", 1e-9));
}

TEST(APMTest, EstimateMeans_EstimatesWithBootstrap_AllComponents) {
    auto data = setup_estimation_test_data();

    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0] + data.X_c_vec[0] * data.a + data.g_0).t();
    true_m.row(1) = (data.G * data.l_c[1] + data.X_c_vec[1] * data.a + data.g_0).t();

    apm::FactorModelParameters params_point(data.G, data.g_0, data.a);

    const std::size_t B = 2;
    auto suff_stats_point = make_suff_stats_vec(true_m, data.observed_outcome_indices, data.X_c_vec);
    auto suff_est_vec = duplicate_bootstrap_suff(suff_stats_point, B);

    // Duplicate bootstrap replicates identical to the point parameters
    apm::FactorModelEstimates param_estimates = duplicate_bootstrap(params_point, B);

    apm::OutcomeMeansEstimates out = apm::estimate_outcome_means_across_cohorts(
        param_estimates, data.observed_outcome_indices, suff_est_vec);

    ASSERT_TRUE(arma::approx_equal(out.mean_outcomes, true_m, "absdiff", 1e-9));
    ASSERT_EQ(out.bootstrap_replicates.size(), B);
    for (std::size_t b = 0; b < B; ++b) {
        ASSERT_TRUE(arma::approx_equal(out.bootstrap_replicates[b], out.mean_outcomes, "absdiff", 1e-9));
    }
}

TEST(APMTest, EstimateMeans_EstimatesWithBootstrap_MismatchThrows) {
    auto data = setup_estimation_test_data();

    std::vector<apm::FactorModelParameters> param_boot;
    param_boot.emplace_back(data.G, data.g_0 + 0.1, data.a);
    param_boot.emplace_back(data.G, data.g_0 + 0.2, data.a);
    apm::FactorModelEstimates param_estimates(apm::FactorModelParameters(data.G, data.g_0, data.a),
                                              std::move(param_boot));

    std::vector<apm::OutcomeMeanSuffStatEstimates> suff_est_vec;
    suff_est_vec.reserve(data.C);
    for (arma::uword c = 0; c < data.C; ++c) {
        arma::mat X_c = data.X_c_vec[c];
        arma::vec m_c(data.observed_outcome_indices[c].n_elem, arma::fill::zeros);
        apm::OutcomeMeanSufficientStatistics stats_point(m_c, X_c);
        std::vector<apm::OutcomeMeanSufficientStatistics> boots;
        if (c == 0) {
            boots.push_back(stats_point);
        } else {
            boots.push_back(stats_point);
            boots.push_back(stats_point);
        }
        suff_est_vec.emplace_back(stats_point, std::move(boots));
    }

    EXPECT_THROW(
        (void)apm::estimate_outcome_means_across_cohorts(param_estimates,
                                                         data.observed_outcome_indices,
                                                         suff_est_vec),
        std::invalid_argument);
}

TEST(APMTest, ImputeOutcomes_Dispatcher_AllComponents) {
    auto data = setup_estimation_test_data();

    apm::FactorModelParameters params(data.G, data.g_0, data.a);
    const arma::uword c = 0;
    arma::uvec T_c = data.observed_outcome_indices[c];

    arma::vec m_full = (data.G * data.l_c[c] + data.X_c_vec[c] * data.a + data.g_0);
    arma::vec m_c = m_full.elem(T_c);
    apm::OutcomeMeanSufficientStatistics stats(m_c, data.X_c_vec[c]);

    arma::vec via_dispatch = apm::impute_outcomes(params, T_c, stats);
    arma::vec via_raw = apm::impute_outcomes(data.G, data.g_0, data.a, T_c, m_c, data.X_c_vec[c]);

    ASSERT_TRUE(arma::approx_equal(via_dispatch, via_raw, "absdiff", 1e-12));
}

TEST(APMTest, AlignFactorsAPMNonContiguousPattern) {
    // T=5 (max outcome index is 4), r=2.
    arma::mat true_factors = {
        {0.1, 0.6},
        {0.2, 0.7},
        {0.3, 0.8},
        {0.4, 0.9},
        {0.5, 1.0}
    };

    std::vector<arma::uvec> observed_outcome_indices = {
        {0, 1, 2},
        {1, 2, 3},
        {0, 3, 4} // Non-contiguous outcomes
    };

    run_alignment_test(true_factors, observed_outcome_indices);
}

TEST(APMTest, AggregateFactorModelParams_FactorsOnly) {
    auto s = make_staircase_data();
    const size_t C = s.observed_outcome_indices.size();
    const arma::uword r = s.true_factors.n_cols;
    std::vector<arma::mat> rotation_matrices = generate_rotation_matrices(C, r);

    // Build cohort-specific params with only G present
    std::vector<apm::FactorModelParameters> cohort_params;
    cohort_params.reserve(C);
    for (size_t c = 0; c < C; ++c) {
        arma::mat G_c = s.true_factors.rows(s.observed_outcome_indices[c]) * rotation_matrices[c];
        cohort_params.emplace_back(G_c, std::nullopt, std::nullopt);
    }

    apm::FactorModelParameters agg = apm::aggregate_cohort_specific_factor_model_params(
        cohort_params, s.observed_outcome_indices);

    // Check aligned G subspace
    arma::mat proj_aligned = apm::internal::projection_matrix(agg.G);
    arma::mat proj_true = apm::internal::projection_matrix(s.true_factors);
    ASSERT_TRUE(arma::approx_equal(proj_aligned, proj_true, "absdiff", 1e-9));

    // Optional fields should be absent
    EXPECT_FALSE(agg.g_0.has_value());
    EXPECT_FALSE(agg.a.has_value());
}

TEST(APMTest, AggregateFactorModelParams_WithAllParams_Weighted) {
    auto s = make_staircase_data();
    const size_t C = s.observed_outcome_indices.size();
    const arma::uword r = s.true_factors.n_cols;
    const arma::uword q = 2;
    std::vector<arma::mat> rotation_matrices = generate_rotation_matrices(C, r);

    arma::vec g0_true = s.g0_true;

    // Cohort weights (will be normalized inside the function)
    arma::vec cohort_weights = {1.0, 2.0, 1.0};
    arma::vec weights_norm = cohort_weights / arma::sum(cohort_weights);

    // Build cohort-specific parameter vectors
    std::vector<apm::FactorModelParameters> cohort_params;
    cohort_params.reserve(C);

    std::vector<arma::vec> a_c_vec;
    a_c_vec.reserve(C);

    for (size_t c = 0; c < C; ++c) {
        arma::mat G_c = s.true_factors.rows(s.observed_outcome_indices[c]) * rotation_matrices[c];
        arma::vec g0_c = g0_true.elem(s.observed_outcome_indices[c]);
        // Create distinct a per cohort to exercise weighting
        arma::vec a_c(q);
        a_c(0) = 0.5 + 0.1 * static_cast<double>(c);
        a_c(1) = 1.0 + 0.2 * static_cast<double>(c);
        a_c_vec.push_back(a_c);

        cohort_params.emplace_back(G_c, g0_c, a_c);
    }

    apm::FactorModelParameters agg = apm::aggregate_cohort_specific_factor_model_params(
        cohort_params, s.observed_outcome_indices, cohort_weights);

    // 1) Check aligned G subspace matches true_factors
    arma::mat proj_aligned = apm::internal::projection_matrix(agg.G);
    arma::mat proj_true = apm::internal::projection_matrix(s.true_factors);
    ASSERT_TRUE(arma::approx_equal(proj_aligned, proj_true, "absdiff", 1e-9));

    // 2) Check g_0 equals the true g0_true since each cohort uses the same per-t value
    ASSERT_TRUE(agg.g_0.has_value());
    ASSERT_EQ(agg.g_0->n_elem, g0_true.n_elem);
    ASSERT_TRUE(arma::approx_equal(*(agg.g_0), g0_true, "absdiff", 1e-12));

    // 3) Check weighted average for a
    ASSERT_TRUE(agg.a.has_value());
    arma::vec expected_a(q, arma::fill::zeros);
    for (size_t c = 0; c < C; ++c) {
        expected_a += a_c_vec[c] * weights_norm(c);
    }
    ASSERT_TRUE(arma::approx_equal(*(agg.a), expected_a, "absdiff", 1e-12));
}

TEST(APMTest, AggregateFactorModelParams_WithBootstrap_AllParams) {
    // True factors (T x r)
    arma::mat true_factors = {
        {0.1, 0.6},
        {0.2, 0.7},
        {0.3, 0.8},
        {0.4, 0.9},
        {0.5, 1.0}
    };

    std::vector<arma::uvec> observed_outcome_indices = {
        {0, 1, 2},
        {1, 2, 3},
        {2, 3, 4}
    };

    const size_t C = observed_outcome_indices.size();
    const arma::uword r = true_factors.n_cols;
    const arma::uword q = 2;
    std::vector<arma::mat> rotation_matrices = generate_rotation_matrices(C, r);

    arma::vec g0_true = {0.10, 0.20, 0.30, 0.40, 0.50};

    // Cohort weights for point and bootstrap (use same per-draw)
    arma::vec cohort_weights = arma::vec(C, arma::fill::ones);

    // Build cohort-specific parameter estimates with B identical bootstrap replicates
    const std::size_t B = 2;
    std::vector<apm::FactorModelEstimates> cohort_estimates;
    cohort_estimates.reserve(C);

    std::vector<arma::vec> a_c_vec;
    a_c_vec.reserve(C);

    for (size_t c = 0; c < C; ++c) {
        arma::mat G_c = true_factors.rows(observed_outcome_indices[c]) * rotation_matrices[c];
        arma::vec g0_c = g0_true.elem(observed_outcome_indices[c]);
        arma::vec a_c(q);
        a_c(0) = 0.5 + 0.1 * static_cast<double>(c);
        a_c(1) = 1.0 + 0.2 * static_cast<double>(c);
        a_c_vec.push_back(a_c);

        apm::FactorModelParameters point(G_c, g0_c, a_c);
        std::vector<apm::FactorModelParameters> boot(B, point); // identical replicates
        cohort_estimates.emplace_back(std::move(point), std::move(boot));
    }

    // Bootstrap cohort weights: B vectors of length C; use ones
    std::vector<arma::vec> bootstrap_cohort_weights(B, arma::vec(C, arma::fill::ones));

    apm::FactorModelEstimates agg = apm::aggregate_cohort_specific_factor_model_params(
        cohort_estimates, observed_outcome_indices, cohort_weights, bootstrap_cohort_weights);

    // Check aligned G subspace matches true_factors
    arma::mat proj_aligned = apm::internal::projection_matrix(agg.parameter_estimates.G);
    arma::mat proj_true = apm::internal::projection_matrix(true_factors);
    ASSERT_TRUE(arma::approx_equal(proj_aligned, proj_true, "absdiff", 1e-9));

    // Check g_0 equals the true g0_true (since each cohort uses same per-t value)
    ASSERT_TRUE(agg.parameter_estimates.g_0.has_value());
    ASSERT_EQ(agg.parameter_estimates.g_0->n_elem, g0_true.n_elem);
    ASSERT_TRUE(arma::approx_equal(*(agg.parameter_estimates.g_0), g0_true, "absdiff", 1e-12));

    // Weighted average for a with equal weights equals the simple mean of a_c across cohorts
    ASSERT_TRUE(agg.parameter_estimates.a.has_value());
    arma::vec expected_a(q, arma::fill::zeros);
    for (size_t c = 0; c < C; ++c) {
        expected_a += a_c_vec[c] * (1.0 / static_cast<double>(C));
    }
    ASSERT_TRUE(arma::approx_equal(*(agg.parameter_estimates.a), expected_a, "absdiff", 1e-12));

    // Bootstrap replicates should be identical to point since inputs were identical per draw
    ASSERT_TRUE(agg.has_bootstrap_replicates());
    ASSERT_EQ(agg.bootstrap_replicates.size(), B);
    for (std::size_t b = 0; b < B; ++b) {
        const auto& rep = agg.bootstrap_replicates[b];
        arma::mat proj_rep = apm::internal::projection_matrix(rep.G);
        ASSERT_TRUE(arma::approx_equal(proj_rep, proj_true, "absdiff", 1e-9));
        ASSERT_TRUE(rep.g_0.has_value());
        ASSERT_TRUE(arma::approx_equal(*(rep.g_0), g0_true, "absdiff", 1e-12));
        ASSERT_TRUE(rep.a.has_value());
        ASSERT_TRUE(arma::approx_equal(*(rep.a), expected_a, "absdiff", 1e-12));
    }
}

//==============================================================================
// O3 Algorithm Tests
//==============================================================================

TEST(APMTest, O3Algorithm_StaircasePattern) {
    // This test uses a "staircase" pattern of observed outcomes where each
    // cohort overlaps with the next, ensuring they all merge into a single
    // super cohort in one step. We set r=2.
    std::vector<arma::uvec> observed_outcome_indices = {
        {0, 1, 2},
        {1, 2, 3},
        {2, 3, 4}
    };
    unsigned int r = 2;

    auto super_cohort_iterations = apm::o3_algorithm(observed_outcome_indices, r);

    // Expected output: The algorithm should converge in one step after the
    // initial state, resulting in a single super cohort. The initial state
    // is not included in the output.
    std::vector<std::vector<std::set<arma::uword>>> expected_output = {
        {{0, 1, 2}}
    };

    // The order of super-cohorts within an iteration is not guaranteed, so we
    // canonicalize both the actual and expected results before comparison.
    ASSERT_EQ(canonicalize_o3_output(super_cohort_iterations),
              canonicalize_o3_output(expected_output));

    ASSERT_TRUE(apm::aligned_factors_identified(observed_outcome_indices, r));
}

TEST(APMTest, O3Algorithm_NonContiguous) {
    // This test uses a pattern designed to merge in two steps.
    // Step 1: cohorts 0 and 1 merge.
    // Step 2: cohort 2 merges with the new {0, 1} super cohort.
    std::vector<arma::uvec> observed_outcome_indices = {
        {0, 1, 2}, // Cohort 0
        {1, 2, 3}, // Cohort 1
        {0, 3, 4}  // Cohort 2
    };
    unsigned int r = 2;

    auto super_cohort_iterations = apm::o3_algorithm(observed_outcome_indices, r);

    // Expected output structure
    std::vector<std::vector<std::set<arma::uword>>> expected_output = {
        {{0, 1}, {2}}, // State after first merge
        {{0, 1, 2}}    // Final converged state
    };
    
    // For comparison, canonicalize the nested vectors.
    ASSERT_EQ(canonicalize_o3_output(super_cohort_iterations),
              canonicalize_o3_output(expected_output));

    ASSERT_TRUE(apm::aligned_factors_identified(observed_outcome_indices, r));
}

TEST(APMTest, FactorsNotIdentifiedOneIteration) {
    std::vector<arma::uvec> observed_outcome_indices = {
        {0, 1},    // Cohort 0
        {1, 2, 3}, // Cohort 1
        {2, 3, 4}  // Cohort 2
    };
    unsigned int r = 2;

    // With r=2, cohorts 1 and 2 merge because they share outcomes {2, 3}.
    // Cohort 0 does not merge with {1, 2} because they only share outcome {1}.
    auto super_cohort_iterations = apm::o3_algorithm(observed_outcome_indices, r);

    std::vector<std::vector<std::set<arma::uword>>> expected_output = {
        {{0}, {1, 2}}
    };

    // Canonicalize for stable comparison.
    ASSERT_EQ(canonicalize_o3_output(super_cohort_iterations),
              canonicalize_o3_output(expected_output));
    ASSERT_FALSE(apm::aligned_factors_identified(observed_outcome_indices, r));
}

TEST(APMTest, FactorsNotIdentifiedNoIterations) {
    std::vector<arma::uvec> observed_outcome_indices = {
        {0, 1},
        {1, 2},
        {2, 3, 4}
    };
    unsigned int r = 2;

    // With r=2, no two cohorts share at least 2 outcomes, so no merges occur.
    auto super_cohort_iterations = apm::o3_algorithm(observed_outcome_indices, r);

    // The algorithm should produce no iterations.
    ASSERT_TRUE(super_cohort_iterations.empty());
    ASSERT_FALSE(apm::aligned_factors_identified(observed_outcome_indices, r));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}