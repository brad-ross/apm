#include "linear_algebra_utils.h"

namespace apm {

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

} // namespace apm 