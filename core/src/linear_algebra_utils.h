#ifndef LINEAR_ALGEBRA_UTILS_H
#define LINEAR_ALGEBRA_UTILS_H

#include <armadillo>

namespace apm {

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

} // namespace apm

#endif // LINEAR_ALGEBRA_UTILS_H 