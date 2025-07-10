#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include "apm.h"
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
        {0, 2, 4} // Non-contiguous outcomes
    };

    run_alignment_test(true_factors, observed_outcome_indices);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(APMTest, EstimateOutcomeMeansAcrossCohortsTest) {
    const arma::uword T = 4, r = 2, C = 2, q = 2;
    std::vector<arma::uvec> observed_outcome_indices = {{0, 1, 2}, {1, 2, 3}};
    
    const arma::mat G = {{1, 1}, {2, 4}, {3, 9}, {4, 16}};
    const arma::vec a = {0.5, 1.0};

    const std::vector<arma::vec> l_c = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    arma::mat X1 = {
        {0.1, 0.5},
        {0.2, 0.6},
        {0.3, 0.7},
        {0.4, 0.8}
    };
    arma::mat X2 = {
        {1.1, 1.5},
        {1.2, 1.6},
        {1.3, 1.7},
        {1.4, 1.8}
    };
    std::vector<arma::mat> X_c_vec = {X1, X2};

    arma::mat m(C, T);
    m.row(0) = (G * l_c[0] + X1 * a).t();
    m.row(1) = (G * l_c[1] + X2 * a).t();
    
    const std::vector<arma::vec> m_c_vec = {
        arma::vec(m.row(0).t()).elem(observed_outcome_indices[0]),
        arma::vec(m.row(1).t()).elem(observed_outcome_indices[1])
    };

    arma::mat estimated_m = apm::estimate_outcome_means_across_cohorts(
        G, a, observed_outcome_indices, m_c_vec, X_c_vec
    );
    
    ASSERT_TRUE(arma::approx_equal(estimated_m, m, "absdiff", 1e-9));
}