#include <gtest/gtest.h>
#include <vector>
#include "apm.h"
#include "linear_algebra_utils.h"

namespace { // Anonymous namespace for test helpers

void run_alignment_test(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices) {
    
    // Create differentially rotated cohort-specific factor matrices 
    // by right-multiplying the true factors by arbitrary full-rank matrices.
    arma::mat R1 = {{1.0, 0.5}, {0.1, 1.2}};
    arma::mat R2 = {{0.8, -0.2}, {0.3, 1.1}};
    arma::mat R3 = {{1.5, 0.1}, {0.2, 0.9}};
    
    std::vector<arma::mat> cohort_factor_matrices;
    cohort_factor_matrices.push_back(true_factors.rows(observed_outcome_indices[0]) * R1);
    cohort_factor_matrices.push_back(true_factors.rows(observed_outcome_indices[1]) * R2);
    cohort_factor_matrices.push_back(true_factors.rows(observed_outcome_indices[2]) * R3);

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