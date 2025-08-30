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
    validate_datum_dimensions(unit_idx, Y, X);
    add_datum_(unit_idx, Y, X);
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

void FactorModelEstimator::validate_datum_dimensions(std::size_t unit_idx,
                                                     const arma::vec& Y,
                                                     const arma::mat& X) const {
    if (Y.n_elem != T_c_) {
        throw std::invalid_argument("add_datum(): Y must have length T_c.");
    }
    if (q_ == 0) {
        if (!(X.is_empty() || X.n_cols == 0)) {
            throw std::invalid_argument("add_datum(): q==0 so X must have zero columns or be empty.");
        }
        return;
    }
    if (X.n_rows != T_c_ || X.n_cols != q_) {
        throw std::invalid_argument("add_datum(): X must be T_c x q.");
    }
}

} // namespace apm


