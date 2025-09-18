#ifndef LINEAR_ALGEBRA_UTILS_H
#define LINEAR_ALGEBRA_UTILS_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

namespace apm {
namespace internal {

/**
 * @brief Computes the projection matrix onto the column space of a matrix.
 *
 * This function computes the projection matrix using the Singular Value 
 * Decomposition (SVD). If X = USV', where the columns of U are orthonormal 
 * vectors spanning the column space of X, the projection matrix is UU'.
 *
 * @param X The input matrix.
 * @return The projection matrix.
 */
arma::mat projection_matrix(const arma::mat& X);

/**
 * @brief Computes the minimum-norm solution to a system of linear equations.
 *
 * This function solves the system AX = B. If the system is overdetermined,
 * it finds the solution that minimizes the Frobenius norm of the residual ||AX - B||_F.
 * If the system is underdetermined, it finds the solution with the minimum Frobenius
 * norm ||X||_F among all possible solutions.
 *
 * @param A The matrix of coefficients.
 * @param B The matrix of dependent values.
 * @return The minimum-frobenius norm solution matrix X.
 */
arma::mat multi_min_norm_solve(const arma::mat& A, const arma::mat& B);

/**
 * @brief Computes the minimum-norm solution to a system of linear equations.
 *
 * This function solves the system AX = b. If the system is overdetermined,
 * it finds the solution that minimizes the Frobenius norm of the residual ||AX - b||_F.
 * If the system is underdetermined, it finds the solution with the minimum Frobenius
 * norm ||X||_F among all possible solutions.
 *
 * @param A The matrix of coefficients.
 * @param b The vector of dependent values.
 * @return The minimum-frobenius norm solution vector X.
 */
arma::vec min_norm_solve(const arma::mat& A, const arma::vec& b);

} // namespace internal

} // namespace apm

#endif // LINEAR_ALGEBRA_UTILS_H 