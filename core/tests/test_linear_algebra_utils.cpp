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

TEST(LinearAlgebraUtilsTest, MinNormSolveUnderdetermined) {
    // Underdetermined system: 2 equations, 3 unknowns.
    arma::mat A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    arma::mat B = {{7.0, 8.0}, {9.0, 10.0}};

    arma::mat X = apm::internal::multi_min_norm_solve(A, B);

    // 1. Check if the solution satisfies the equation.
    ASSERT_TRUE(arma::approx_equal(A * X, B, "absdiff", 1e-9));

    // 2. Check if the solution has the minimum norm.
    // The min-norm solution must be orthogonal to the null space of A.
    arma::mat null_space_basis = arma::null(A);
    ASSERT_GT(null_space_basis.n_cols, 0) << "Null space should not be empty for an underdetermined system.";

    ASSERT_TRUE(arma::approx_equal(X.t() * null_space_basis, 
        arma::zeros(X.n_cols, null_space_basis.n_cols), "absdiff", 1e-9));
} 