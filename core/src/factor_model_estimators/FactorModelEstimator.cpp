#include "FactorModelEstimator.h"
#include <stdexcept>

namespace apm {

void FactorModelEstimator::add_data(const arma::uvec& unit_idxs,
                                    const arma::mat& Y,
                                    const arma::cube& X) {
    validate_data_dimensions(unit_idxs, Y, X);
    add_data_(unit_idxs, Y, X);
}

void FactorModelEstimator::add_datum(std::size_t unit_idx,
                                     const arma::vec& Y,
                                     const arma::mat& X) {
    arma::uvec unit_idxs(1);
    unit_idxs(0) = static_cast<arma::uword>(unit_idx);

    arma::mat Y_batch = Y.t(); // 1 x T_c_

    arma::cube X_batch;
    if (q_ == 0 || X.n_cols == 0) {
        X_batch = arma::cube(); // empty cube when no covariates
    } else {
        X_batch.set_size(1, T_c_, q_);
        for (std::size_t s = 0; s < q_; ++s) {
            X_batch.slice(static_cast<arma::uword>(s)).row(0) = X.col(static_cast<arma::uword>(s)).t();
        }
    }

    add_data(unit_idxs, Y_batch, X_batch);
}

void FactorModelEstimator::validate_data_dimensions(const arma::uvec& unit_idxs,
                                                    const arma::mat& Y,
                                                    const arma::cube& X) const {
    const std::size_t N = unit_idxs.n_elem;
    if (Y.n_rows != N || Y.n_cols != T_c_) {
        throw std::invalid_argument("add_data(): Y must be N x T_c.");
    }
    if (q_ == 0) {
        // Allow empty X when q==0
        if (!(X.is_empty() || X.n_slices == 0)) {
            throw std::invalid_argument("add_data(): q==0 so X must be empty (zero slices).");
        }
        return;
    }
    if (X.n_rows != N || X.n_cols != T_c_ || X.n_slices != q_) {
        throw std::invalid_argument("add_data(): X must be N x T_c x q.");
    }
}


} // namespace apm


