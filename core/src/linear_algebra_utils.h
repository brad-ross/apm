#ifndef LINEAR_ALGEBRA_UTILS_H
#define LINEAR_ALGEBRA_UTILS_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif
#include <functional>
#include <optional>
#include <limits>

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

// ------------------------------------------------------------
// Matrix-free LSMR (Fong & Saunders) linear least-squares solver
// ------------------------------------------------------------

struct LinearOperator {
    arma::uword domain_dim;   // n
    arma::uword range_dim;    // m
    std::function<void(const arma::vec&, arma::vec&)> apply;           // y = A x
    std::function<void(const arma::vec&, arma::vec&)> apply_transpose; // z = A^T y
};

struct LSMROptions {
    double atol = 1e-6;        // relative tol on ||A^T r||
    double btol = 1e-6;        // relative tol on ||r||
    double conlim = 1e+8;      // condition limit
    std::size_t max_iters = std::numeric_limits<std::size_t>::max();
    double lambda = 0.0;       // Tikhonov damping; 0 disables
};

struct LSMRResult {
    arma::vec x;               // solution
    std::size_t iters;         // iterations used
    double rnorm;              // ||r||_2
    double arnorm;             // ||A^T r||_2
    double anorm;              // ||A|| estimate
    double acond;              // cond(A) estimate
    int flag;                  // 0=converged, 1=conlim, 2=maxit, 3=breakdown
};

LSMRResult lsmr(const LinearOperator& A,
                const arma::vec& b,
                const LSMROptions& opts,
                std::optional<arma::vec> x0 = std::nullopt);

} // namespace internal

} // namespace apm

#endif // LINEAR_ALGEBRA_UTILS_H 