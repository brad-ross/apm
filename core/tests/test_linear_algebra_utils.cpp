#include <gtest/gtest.h>
#include <armadillo>
#include "linear_algebra_utils.h"

// Function to check if a matrix is a projection matrix (idempotent and symmetric)
void check_is_projection(const arma::mat& P) {
    ASSERT_TRUE(P.is_symmetric());
    arma::mat P_squared = P * P;
    ASSERT_TRUE(arma::approx_equal(P, P_squared, "absdiff", 1e-9));
}

TEST(LinearAlgebraUtilsTest, ProjectionMatrixFullRank) {
    arma::mat X = {{1, 2}, {2, 4.1}, {3, 6}};
    arma::mat P = apm::internal::projection_matrix(X);
    check_is_projection(P);

    // Project a vector onto the column space of X
    arma::vec v = {1, 1, 1};
    arma::vec projected_v = P * v;
    
    // The residual should be orthogonal to the columns of X
    arma::vec residual = v - projected_v;
    ASSERT_NEAR(arma::dot(residual, X.col(0)), 0.0, 1e-9);
    ASSERT_NEAR(arma::dot(residual, X.col(1)), 0.0, 1e-9);
}

TEST(LinearAlgebraUtilsTest, ProjectionMatrixRankDeficient) {
    arma::mat X = {{1, 2}, {2, 4}, {3, 6}};
    arma::mat P = apm::internal::projection_matrix(X);
    check_is_projection(P);

    // The projection onto the col space of X should be the same as onto its first column
    arma::mat P_expected = apm::internal::projection_matrix(X.col(0));
    ASSERT_TRUE(arma::approx_equal(P, P_expected, "absdiff", 1e-9))
        << "Matrix P is not equal to P_expected:" << std::endl
        << "P:" << std::endl << P << std::endl
        << "P_expected:" << std::endl << P_expected;
}

TEST(LinearAlgebraUtilsTest, ProjectionMatrixZeroCols) {
    arma::mat X(3, 0);
    arma::mat P = apm::internal::projection_matrix(X);
    ASSERT_EQ(P.n_rows, 3);
    ASSERT_EQ(P.n_cols, 3);
    ASSERT_TRUE(arma::all(arma::vectorise(P) == 0.0));
} 