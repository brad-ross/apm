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

    run_alignment_test(true_factors, observed_outcome_indices);
}

TEST(APMTest, AlignFactorsAPMStaircasePatternWeighted) {
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

    arma::vec cohort_weights = {1.0, 2.0, 1.0};
    run_alignment_test(true_factors, observed_outcome_indices, cohort_weights);
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
    std::vector<arma::mat> rotation_matrices = generate_rotation_matrices(C, r);

    // Build cohort-specific params with only G present
    std::vector<apm::FactorModelParameters> cohort_params;
    cohort_params.reserve(C);
    for (size_t c = 0; c < C; ++c) {
        arma::mat G_c = true_factors.rows(observed_outcome_indices[c]) * rotation_matrices[c];
        cohort_params.emplace_back(G_c, std::nullopt, std::nullopt);
    }

    apm::FactorModelParameters agg = apm::aggregate_cohort_specific_factor_model_params(
        cohort_params, observed_outcome_indices);

    // Check aligned G subspace
    arma::mat proj_aligned = apm::internal::projection_matrix(agg.G);
    arma::mat proj_true = apm::internal::projection_matrix(true_factors);
    ASSERT_TRUE(arma::approx_equal(proj_aligned, proj_true, "absdiff", 1e-9));

    // Optional fields should be absent
    EXPECT_FALSE(agg.g_0.has_value());
    EXPECT_FALSE(agg.a.has_value());
}

TEST(APMTest, AggregateFactorModelParams_WithAllParams_Weighted) {
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

    // Cohort weights (will be normalized inside the function)
    arma::vec cohort_weights = {1.0, 2.0, 1.0};
    arma::vec weights_norm = cohort_weights / arma::sum(cohort_weights);

    // Build cohort-specific parameter vectors
    std::vector<apm::FactorModelParameters> cohort_params;
    cohort_params.reserve(C);

    std::vector<arma::vec> a_c_vec;
    a_c_vec.reserve(C);

    for (size_t c = 0; c < C; ++c) {
        arma::mat G_c = true_factors.rows(observed_outcome_indices[c]) * rotation_matrices[c];
        arma::vec g0_c = g0_true.elem(observed_outcome_indices[c]);
        // Create distinct a per cohort to exercise weighting
        arma::vec a_c(q);
        a_c(0) = 0.5 + 0.1 * static_cast<double>(c);
        a_c(1) = 1.0 + 0.2 * static_cast<double>(c);
        a_c_vec.push_back(a_c);

        cohort_params.emplace_back(G_c, g0_c, a_c);
    }

    apm::FactorModelParameters agg = apm::aggregate_cohort_specific_factor_model_params(
        cohort_params, observed_outcome_indices, cohort_weights);

    // 1) Check aligned G subspace matches true_factors
    arma::mat proj_aligned = apm::internal::projection_matrix(agg.G);
    arma::mat proj_true = apm::internal::projection_matrix(true_factors);
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