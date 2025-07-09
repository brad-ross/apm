#include "linear_algebra_utils.h"

namespace apm {

arma::mat projection_matrix(const arma::mat& X) {
    if (X.n_cols == 0) {
        return arma::mat(X.n_rows, X.n_rows, arma::fill::zeros);
    }

    arma::mat U, V;
    arma::vec s;
    arma::svd_econ(U, s, V, X, "left");

    return U * U.t();
}

} // namespace apm 