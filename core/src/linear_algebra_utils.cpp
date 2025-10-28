#include "linear_algebra_utils.h"
#include <cmath>

namespace apm {
namespace internal {

arma::mat projection_matrix(const arma::mat& X) {
    if (X.n_cols == 0) {
        return arma::mat(X.n_rows, X.n_rows, arma::fill::zeros);
    }

    arma::mat U, V;
    arma::vec s;
    arma::svd_econ(U, s, V, X, "left");

    // Determine the rank by finding the number of singular values greater than a tolerance.
    double tol = std::max(X.n_rows, X.n_cols) * s.max() * arma::Datum<double>::eps;
    arma::uword r = arma::sum(s > tol);

    if (r == 0) {
        return arma::mat(X.n_rows, X.n_rows, arma::fill::zeros);
    }

    // Use only the first 'r' columns of U, which form a basis for the column space.
    arma::mat U_r = U.cols(0, r - 1);

    return U_r * U_r.t();
}

arma::mat multi_min_norm_solve(const arma::mat& A, const arma::mat& B) {
    if (A.n_rows != B.n_rows) {
        throw std::invalid_argument("The number of rows in A must match the number of rows in B.");
    }
    return arma::pinv(A) * B;
}

arma::vec min_norm_solve(const arma::mat& A, const arma::vec& b) {
    return arma::pinv(A) * b;
}

static inline double safe_div(double num, double den) {
    const double eps = std::numeric_limits<double>::epsilon();
    return num / std::max(den, eps);
}

LSMRResult lsmr(const LinearOperator& A,
                const arma::vec& b,
                const LSMROptions& o,
                std::optional<arma::vec> x0)
{
    const arma::uword n = A.domain_dim;
    const arma::uword m = A.range_dim;
    if (b.n_rows != m) {
        throw std::invalid_argument("lsmr: dimension mismatch: b length must equal A.range_dim");
    }

    arma::vec x = x0.value_or(arma::vec(n, arma::fill::zeros));

    // u = b - A x
    arma::vec u = b;
    if (x0) {
        arma::vec Ax(m, arma::fill::zeros);
        A.apply(*x0, Ax);
        u -= Ax;
    }
    double beta = arma::norm(u);
    if (beta == 0.0) {
        return {x, 0u, 0.0, 0.0, 0.0, 1.0, 0};
    }
    u /= beta;

    // v = A^T u
    arma::vec At_u(n, arma::fill::zeros);
    A.apply_transpose(u, At_u);
    double alpha = arma::norm(At_u);
    arma::vec v = (alpha > 0.0) ? (At_u / alpha) : arma::vec(n, arma::fill::zeros);
    arma::vec w = v;

    double rho_bar = alpha;
    double phi_bar = beta;
    double anorm = 0.0;
    double acond = 1.0;
    double rnorm = beta;
    double arnorm = alpha * beta;

    int flag = 2;
    std::size_t k = 0;

    arma::vec Av(m, arma::fill::zeros);
    for (; k < o.max_iters; ++k) {
        // Bidiagonalization
        A.apply(v, Av);
        Av -= alpha * u;
        double beta_new = arma::norm(Av);
        u = (beta_new > 0.0) ? (Av / beta_new) : Av;

        A.apply_transpose(u, At_u);
        At_u -= beta_new * v;
        double alpha_new = arma::norm(At_u);
        v = (alpha_new > 0.0) ? (At_u / alpha_new) : At_u;

        alpha = alpha_new;

        // LSMR orthogonal transformation (undamped): rho = hypot(rho_bar, beta)
        double rho = std::hypot(rho_bar, beta_new);
        double c = (rho > 0.0) ? (rho_bar / rho) : 1.0;
        double s = (rho > 0.0) ? (beta_new / rho) : 0.0;
        double theta = s * alpha;
        rho_bar = -c * alpha;
        double phi = c * phi_bar;
        phi_bar = s * phi_bar;

        // Updates
        x += safe_div(phi, rho) * w;
        w = v - safe_div(theta, rho) * w;

        // Norm and condition estimates (lightweight)
        anorm = std::hypot(anorm, alpha);
        anorm = std::hypot(anorm, beta_new);
        rnorm = std::abs(phi_bar);
        arnorm = alpha * std::abs(phi);

        // Stopping tests
        double xnorm = arma::norm(x);
        double bnorm = arma::norm(b);
        bool t1 = (arnorm <= o.atol * (anorm * rnorm + 1e-50));
        bool t2 = (rnorm <= o.btol * (anorm * xnorm + bnorm));
        bool t3 = (o.conlim > 0.0 && acond >= o.conlim);
        if (t1 || t2) { flag = 0; break; }
        if (t3) { flag = 1; break; }
        if (!(std::isfinite(alpha) && std::isfinite(beta_new))) { flag = 3; break; }

        beta = beta_new;
    }

    if (k == o.max_iters && flag == 2) {
        // max iters
    }

    return {x, k, rnorm, arnorm, anorm, acond, flag};
}

} // namespace internal

} // namespace apm 