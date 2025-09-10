#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include "apm_core.h"
#include "linear_algebra_utils.h"
#include "test_helpers.h"

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