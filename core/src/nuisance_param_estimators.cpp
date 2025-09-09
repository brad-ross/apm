#include "nuisance_param_estimators.h"
#include <limits>
#include <stdexcept>
#include <cmath>

namespace apm {

static arma::mat combine_aux_means(
    const arma::cube& A,               // N x T x d
    const arma::vec& row_weights,      // N
    const arma::mat& current_means,    // T x d
    double current_total_weight)
{
    const double batch_weight = arma::accu(row_weights);
    if (batch_weight == 0.0) {
        return current_means;
    }
    const arma::uword T = A.n_cols;
    const arma::uword d = A.n_slices;
    arma::mat batch_means(T, d, arma::fill::zeros);
    for (arma::uword j = 0; j < d; ++j) {
        arma::rowvec weighted_sum = row_weights.t() * A.slice(j); // 1 x T
        batch_means.col(j) = (weighted_sum / batch_weight).t();   // T x 1
    }
    const double total_weight = current_total_weight + batch_weight;
    const double rel = batch_weight / total_weight;
    return rel * batch_means + (1.0 - rel) * current_means;
}

CohortAuxiliaryDataMeanEstimator::CohortAuxiliaryDataMeanEstimator(std::size_t T,
                                                                   std::size_t d,
                                                                   std::shared_ptr<const WeightedBootstrap> bootstrap)
    : T_(T), d_(d), bootstrap_(std::move(bootstrap)),
      total_weight_(0.0), aux_means_(arma::zeros<arma::mat>(static_cast<arma::uword>(T_), static_cast<arma::uword>(d_)))
{
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        total_boot_weights_ = arma::zeros<arma::vec>(static_cast<arma::uword>(B));
        boot_aux_means_     = arma::cube(static_cast<arma::uword>(T_), static_cast<arma::uword>(d_), static_cast<arma::uword>(B), arma::fill::zeros);
    }
}

void CohortAuxiliaryDataMeanEstimator::validate_data_dimensions(const arma::uvec& unit_idxs,
                                                                const arma::cube& A,
                                                                std::size_t T,
                                                                std::size_t d)
{
    const std::size_t N = unit_idxs.n_elem;
    if (A.n_rows != N || A.n_cols != static_cast<arma::uword>(T) || A.n_slices != static_cast<arma::uword>(d)) {
        throw std::invalid_argument("CohortAuxiliaryDataMeanEstimator::add_data: A must be N x T x d.");
    }
}

void CohortAuxiliaryDataMeanEstimator::add_data(const arma::uvec& unit_idxs, const arma::cube& A)
{
    validate_data_dimensions(unit_idxs, A, T_, d_);

    const std::size_t N = unit_idxs.n_elem;
    const std::size_t B = num_bootstraps();
    // Point estimate: equal weights for the batch
    arma::vec ones_w(static_cast<arma::uword>(N), arma::fill::ones);
    aux_means_ = combine_aux_means(A, ones_w, aux_means_, total_weight_);
    total_weight_ += arma::accu(ones_w);

    // Bootstrap estimates: weighted by replicate-specific unit weights
    if (B > 0) {
        arma::mat rows = bootstrap_->obs(unit_idxs); // N x B
        for (arma::uword b = 0; b < static_cast<arma::uword>(B); ++b) {
            arma::vec w_b = rows.col(b);
            boot_aux_means_.slice(b) = combine_aux_means(A, w_b, boot_aux_means_.slice(b), total_boot_weights_(b));
            total_boot_weights_(b) += arma::accu(w_b);
        }
    }

}

void CohortAuxiliaryDataMeanEstimator::add_datum(std::size_t unit_idx, const arma::mat& A)
{
    if (A.n_rows != static_cast<arma::uword>(T_) || A.n_cols != static_cast<arma::uword>(d_)) {
        throw std::invalid_argument("CohortAuxiliaryDataMeanEstimator::add_datum: A must be T x d.");
    }
    arma::uvec one(1);
    one(0) = static_cast<arma::uword>(unit_idx);
    // wrap: N x T x d, with N=1
    arma::cube C(1, static_cast<arma::uword>(T_), static_cast<arma::uword>(d_));
    for (arma::uword j = 0; j < static_cast<arma::uword>(d_); ++j) C.slice(j).row(0) = A.col(j).t();
    add_data(one, C);
}

CohortAuxiliaryDataMeanEstimates CohortAuxiliaryDataMeanEstimator::estimate(std::size_t total_units) const
{
    const double denom = static_cast<double>(total_units);
    const double share = (denom > 0.0) ? (total_weight_ / denom) : std::numeric_limits<double>::quiet_NaN();
    CohortAuxiliaryDataMeans point(share, aux_means_);

    std::vector<CohortAuxiliaryDataMeans> boots;
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        boots.reserve(B);
        for (std::size_t b = 0; b < B; ++b) {
            const arma::uword bu = static_cast<arma::uword>(b);
            arma::mat means_b = boot_aux_means_.slice(bu);
            const double share_b = total_boot_weights_(bu);
            boots.emplace_back(share_b, std::move(means_b));
        }
    }

    return CohortAuxiliaryDataMeanEstimates(std::move(point), std::move(boots));
}

} // namespace apm


