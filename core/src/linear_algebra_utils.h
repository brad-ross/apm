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

/**
 * @brief Matrix-free linear operator A with forward and transpose actions.
 */
struct LinearOperator {
    arma::uword domain_dim;   ///< Domain dimension n.
    arma::uword range_dim;    ///< Range dimension m.
    std::function<void(const arma::vec&, arma::vec&)> apply;           ///< y = A x
    std::function<void(const arma::vec&, arma::vec&)> apply_transpose; ///< z = A^T y
};

/**
 * @brief Configuration for matrix-free LSMR solver.
 */
struct LSMROptions {
    double atol = 1e-12;        ///< Relative tol on ||A^T r||.
    double btol = 1e-12;        ///< Relative tol on ||r||.
    double conlim = 1e+8;       ///< Condition number limit.
    std::size_t max_iters = std::numeric_limits<std::size_t>::max(); ///< Max iterations.
    double lambda = 0.0;        ///< Tikhonov damping; 0 disables.
};

/**
 * @brief Results returned by LSMR.
 */
struct LSMRResult {
    arma::vec x;               ///< Solution vector.
    std::size_t iters;         ///< Iterations used.
    double rnorm;              ///< ||r||_2.
    double arnorm;             ///< ||A^T r||_2.
    double anorm;              ///< ||A|| estimate.
    double acond;              ///< cond(A) estimate.
    int flag;                  ///< 0=converged, 1=conlim, 2=maxit, 3=breakdown.
};

/**
 * @brief Exact column 2-norms for a matrix-free operator.
 *
 * @param A Linear operator.
 * @param eps Minimum norm floor to avoid zeros.
 * @return Column norms (length n).
 */
arma::vec exact_column_norms(const LinearOperator& A, double eps = 1e-12);
/**
 * @brief Hutchinson-estimated column 2-norms for a matrix-free operator.
 *
 * @param A Linear operator.
 * @param K Number of Hutchinson draws.
 * @param eps Minimum norm floor to avoid zeros.
 * @return Estimated column norms (length n).
 */
arma::vec hutchinson_column_norms(const LinearOperator& A, std::size_t K = 16, double eps = 1e-12);

// Build a column-scaled operator by computing d internally.
// - If num_diag_approx_draws == 0: use exact_column_norms(A, eps)
// - Else: use hutchinson_column_norms(A, num_diag_approx_draws, eps)
// Returns the scaled operator and the scaling vector d (needed to unscale x).
/**
 * @brief Build a column-scaled operator and return scaling vector.
 *
 * @param A Linear operator to scale.
 * @param num_diag_approx_draws 0 for exact norms; >0 uses Hutchinson with this many draws.
 * @param eps Minimum norm floor.
 * @return Pair {scaled operator, scaling vector d}.
 */
std::pair<LinearOperator, arma::vec> make_column_scaled_operator(
    const LinearOperator& A,
    std::size_t num_diag_approx_draws = 0,
    double eps = 1e-12);

/**
 * @brief Matrix-free LSMR solve with optional diagonal preconditioning and homotopy.
 *
 * @param A Linear operator.
 * @param b Right-hand side vector (length m).
 * @param opts LSMR options (tolerances, damping, iteration cap).
 * @param x0 Optional initial solution guess.
 * @param diagonal_precond If true, apply column scaling preconditioner.
 * @param num_diag_approx_draws Hutchinson draws for diagonal precond (0 = exact).
 * @param homotopy_iters Number of damped stages before final undamped solve.
 * @return LSMRResult containing solution and diagnostics.
 */
LSMRResult lsmr(const LinearOperator& A,
                const arma::vec& b,
                const LSMROptions& opts,
                std::optional<arma::vec> x0 = std::nullopt,
                bool diagonal_precond = false,
                std::size_t num_diag_approx_draws = 0,
                std::size_t homotopy_iters = 0);

} // namespace internal

} // namespace apm

#endif // LINEAR_ALGEBRA_UTILS_H 