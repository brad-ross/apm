#include "nuisance_param_estimators.h"
#include <limits>
#include <stdexcept>
#include <cmath>

namespace apm {

CohortAuxiliaryDataMeanEstimator::CohortAuxiliaryDataMeanEstimator(std::size_t d,
                                                                   std::shared_ptr<const WeightedBootstrap> bootstrap)
    : d_(d), bootstrap_(std::move(bootstrap)),
      row_count_(0.0),
      aux_sum_(arma::zeros<arma::vec>(static_cast<arma::uword>(d_))),
      aux_count_(arma::zeros<arma::vec>(static_cast<arma::uword>(d_)))
{
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        boot_row_counts_ = arma::zeros<arma::vec>(static_cast<arma::uword>(B));
        boot_aux_sum_    = arma::zeros<arma::mat>(static_cast<arma::uword>(d_), static_cast<arma::uword>(B));
        boot_aux_weight_ = arma::zeros<arma::mat>(static_cast<arma::uword>(d_), static_cast<arma::uword>(B));
    }
}

void CohortAuxiliaryDataMeanEstimator::validate_data_dimensions(const arma::uvec& unit_idxs,
                                                                const arma::cube& A,
                                                                std::size_t d)
{
    const std::size_t N = unit_idxs.n_elem;
    if (A.n_rows != N || A.n_cols == 0 || A.n_slices != static_cast<arma::uword>(d)) {
        throw std::invalid_argument("CohortAuxiliaryDataMeanEstimator::add_data: A must be N x T x d.");
    }
}

void CohortAuxiliaryDataMeanEstimator::add_data(const arma::uvec& unit_idxs, const arma::cube& A)
{
    validate_data_dimensions(unit_idxs, A, d_);

    const std::size_t N = unit_idxs.n_elem;
    const std::size_t B = num_bootstraps();

    for (std::size_t i = 0; i < N; ++i) {
        const arma::uword iu = static_cast<arma::uword>(i);
        arma::vec w_b;
        if (B > 0) {
            arma::uvec one(1);
            one(0) = unit_idxs(iu);
            arma::mat rows = bootstrap_->obs(one); // 1 x B
            w_b = rows.row(0).t();                 // B x 1
        }

        const arma::uword T = A.n_cols;
        for (arma::uword t = 0; t < T; ++t) {
            bool any_finite = false;
            for (std::size_t j = 0; j < d_; ++j) {
                const arma::uword jj = static_cast<arma::uword>(j);
                const double v = A(iu, t, jj);
                if (std::isfinite(v)) {
                    any_finite = true;
                    aux_sum_(jj) += v;
                    aux_count_(jj) += 1.0;
                    if (B > 0) {
                        boot_aux_sum_.row(jj)    += v * w_b.t();
                        boot_aux_weight_.row(jj) += w_b.t();
                    }
                }
            }
            if (any_finite) {
                row_count_ += 1.0;
                if (B > 0) {
                    boot_row_counts_ += w_b;
                }
            }
        }
    }
}

void CohortAuxiliaryDataMeanEstimator::add_datum(std::size_t unit_idx, const arma::mat& A)
{
    if (A.n_cols != static_cast<arma::uword>(d_)) {
        throw std::invalid_argument("CohortAuxiliaryDataMeanEstimator::add_datum: A must be T x d.");
    }
    arma::uvec one(1);
    one(0) = static_cast<arma::uword>(unit_idx);
    // wrap: N x T x d, with N=1
    arma::cube C(1, A.n_rows, A.n_cols);
    for (arma::uword j = 0; j < A.n_cols; ++j) {
        C.slice(j).row(0) = A.col(j).t();
    }
    add_data(one, C);
}

CohortAuxiliaryDataMeanEstimates CohortAuxiliaryDataMeanEstimator::estimate(std::size_t total_rows) const
{
    arma::vec means(static_cast<arma::uword>(d_));
    for (std::size_t j = 0; j < d_; ++j) {
        const arma::uword jj = static_cast<arma::uword>(j);
        const double n = aux_count_(jj);
        means(jj) = (n > 0.0) ? (aux_sum_(jj) / n) : std::numeric_limits<double>::quiet_NaN();
    }
    const double denom = static_cast<double>(total_rows);
    const double share = (denom > 0.0) ? (row_count_ / denom)
                                            : std::numeric_limits<double>::quiet_NaN();
    CohortAuxiliaryDataMeans point(share, means);

    std::vector<CohortAuxiliaryDataMeans> boots;
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        boots.reserve(B);
        for (std::size_t b = 0; b < B; ++b) {
            const arma::uword bu = static_cast<arma::uword>(b);
            arma::vec means_b(static_cast<arma::uword>(d_));
            for (std::size_t j = 0; j < d_; ++j) {
                const arma::uword jj = static_cast<arma::uword>(j);
                const double wden = boot_aux_weight_(jj, bu);
                means_b(jj) = (wden > 0.0) ? (boot_aux_sum_(jj, bu) / wden)
                                           : std::numeric_limits<double>::quiet_NaN();
            }
            // Each bootstrap column sums to 1 across units
            const double share_b = boot_row_counts_(bu);
            boots.emplace_back(share_b, std::move(means_b));
        }
    }

    return CohortAuxiliaryDataMeanEstimates(std::move(point), std::move(boots));
}

} // namespace apm


