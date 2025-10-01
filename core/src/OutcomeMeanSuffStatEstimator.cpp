#include "OutcomeMeanSuffStatEstimator.h"
#include <limits>

namespace apm {

OutcomeMeanSuffStatEstimator::OutcomeMeanSuffStatEstimator(std::size_t T_c,
                                                   std::size_t T,
                                                   std::size_t q,
                                                   std::shared_ptr<const WeightedBootstrap> bootstrap)
    : T_c_(T_c), T_(T), q_(q), bootstrap_(std::move(bootstrap)),
      outcome_means_(arma::zeros<arma::vec>(T_c_)),
      total_weight_(0.0)
{
    if (q_ > 0) {
        if (T_ == 0) {
            throw std::invalid_argument("OutcomeMeanSuffStatEstimator: when q>0, T (total outcomes) must be provided.");
        }
        covar_means_ = arma::zeros<arma::mat>(T_, q_);
    }
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        total_boot_weights_ = arma::zeros<arma::vec>(B);
        boot_outcome_means_ = arma::zeros<arma::mat>(T_c_, B);
        if (q_ > 0) {
            boot_covar_means_ = arma::cube(T_, q_, B, arma::fill::zeros);
        }
    }
}

void OutcomeMeanSuffStatEstimator::add_data(const arma::uvec& unit_idxs,
                                        const arma::mat& Y,
                                        const arma::cube& X)
{
    validate_data_dimensions(unit_idxs, Y, X);

    arma::vec ones_w = arma::ones<arma::vec>(static_cast<arma::uword>(Y.n_rows));
    auto combined_main = combine_means(Y, ones_w, outcome_means_, total_weight_);
    if (q_ > 0) {
        covar_means_ = combine_covar_means(X, ones_w, covar_means_, total_weight_);
    }
    outcome_means_ = std::move(combined_main.first);
    total_weight_ = combined_main.second;

    const std::size_t B = num_bootstraps();
    if (B > 0) {
        for (std::size_t b = 0; b < B; ++b) {
            const arma::uword bu = static_cast<arma::uword>(b);
            arma::vec w_b = boot_weights_for_indices(unit_idxs, b);

            auto combined_b = combine_means(Y, w_b, boot_outcome_means_.col(bu), total_boot_weights_(bu));
            if (q_ > 0) {
                boot_covar_means_.slice(bu) =
                    combine_covar_means(X, w_b, boot_covar_means_.slice(bu), total_boot_weights_(bu));
            }
            boot_outcome_means_.col(bu) = std::move(combined_b.first);
            total_boot_weights_(bu) = combined_b.second;
        }
    }
}

void OutcomeMeanSuffStatEstimator::add_datum(std::size_t unit_idx,
                                         const arma::vec& Y,
                                         const arma::mat& X)
{
    arma::uvec unit_idxs(1);
    unit_idxs(0) = static_cast<arma::uword>(unit_idx);

    arma::mat Y_batch = Y.t();
    arma::cube X_batch;
    if (q_ == 0 || X.n_cols == 0) {
        X_batch = arma::cube();
    } else {
        if (X.n_rows != static_cast<arma::uword>(T_) || X.n_cols != static_cast<arma::uword>(q_)) {
            throw std::invalid_argument("OutcomeMeanSuffStatEstimator::add_datum: X must be T x q.");
        }
        X_batch.set_size(1, T_, q_);
        for (std::size_t s = 0; s < q_; ++s) {
            X_batch.slice(static_cast<arma::uword>(s)).row(0) = X.col(static_cast<arma::uword>(s)).t();
        }
    }
    add_data(unit_idxs, Y_batch, X_batch);
}

OutcomeMeanSuffStatEstimates OutcomeMeanSuffStatEstimator::estimate(std::size_t total_units) const
{
    std::optional<arma::mat> covars_opt = std::nullopt;
    if (q_ > 0) {
        covars_opt = covar_means_;
    }
    OutcomeMeanSufficientStatistics point(outcome_means_, covars_opt);
    const double denom = static_cast<double>(total_units);
    point.cohort_pop_share = (denom > 0.0) ? (total_weight_ / denom) : std::numeric_limits<double>::quiet_NaN();

    std::vector<OutcomeMeanSufficientStatistics> boot_reps;
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        boot_reps.reserve(B);
        for (std::size_t b = 0; b < B; ++b) {
            const arma::uword bu = static_cast<arma::uword>(b);
            std::optional<arma::mat> covars_b = std::nullopt;
            if (q_ > 0) {
                covars_b = boot_covar_means_.slice(bu);
            }
            OutcomeMeanSufficientStatistics rep(boot_outcome_means_.col(bu), covars_b);
            rep.cohort_pop_share = total_boot_weights_(bu);
            boot_reps.emplace_back(std::move(rep));
        }
    }
    return OutcomeMeanSuffStatEstimates(std::move(point), std::move(boot_reps));
}

void OutcomeMeanSuffStatEstimator::validate_data_dimensions(const arma::uvec& unit_idxs,
                                                        const arma::mat& Y,
                                                        const arma::cube& X) const
{
    const std::size_t N = unit_idxs.n_elem;
    if (Y.n_rows != N || Y.n_cols != T_c_) {
        throw std::invalid_argument("OutcomeMeanSuffStatEstimator::add_data: Y must be N x T_c.");
    }
    if (q_ == 0) {
        if (!(X.is_empty() || X.n_slices == 0)) {
            throw std::invalid_argument("OutcomeMeanSuffStatEstimator::add_data: q==0 so X must be empty.");
        }
        return;
    }
    if (X.n_rows != N || X.n_cols != T_ || X.n_slices != q_) {
        throw std::invalid_argument("OutcomeMeanSuffStatEstimator::add_data: X must be N x T x q.");
    }
}

arma::vec OutcomeMeanSuffStatEstimator::boot_weights_for_indices(const arma::uvec& unit_idxs, std::size_t b) const
{
    if (!bootstrap_) {
        throw std::runtime_error("OutcomeMeanSuffStatEstimator: no bootstrap present");
    }
    const std::size_t B = num_bootstraps();
    if (b >= B) {
        throw std::out_of_range("OutcomeMeanSuffStatEstimator: draw index out of range");
    }
    arma::mat rows = bootstrap_->obs(unit_idxs);
    return rows.col(static_cast<arma::uword>(b));
}

std::pair<arma::vec, double> OutcomeMeanSuffStatEstimator::combine_means(
    const arma::mat& Y,
    const arma::vec& row_weights,
    const arma::vec& current_mean,
    double current_total_weight)
{
    const double batch_weight = arma::accu(row_weights);
    if (batch_weight == 0.0) {
        return {current_mean, current_total_weight};
    }
    arma::rowvec weighted_sum = row_weights.t() * Y;
    arma::vec batch_mean = (weighted_sum / batch_weight).t();

    const double total_weight = current_total_weight + batch_weight;
    const double rel = batch_weight / total_weight;
    arma::vec combined = rel * batch_mean + (1.0 - rel) * current_mean;
    return {std::move(combined), total_weight};
}

arma::mat OutcomeMeanSuffStatEstimator::combine_covar_means(
    const arma::cube& X,
    const arma::vec& row_weights,
    const arma::mat& current_means,
    double current_total_weight)
{
    const double batch_weight = arma::accu(row_weights);
    if (batch_weight == 0.0) {
        return current_means;
    }
    const arma::uword T = X.n_cols;
    const arma::uword q = X.n_slices;

    arma::mat batch_means(T, q, arma::fill::zeros);
    for (arma::uword k = 0; k < q; ++k) {
        arma::rowvec weighted_sum = row_weights.t() * X.slice(k);
        batch_means.col(k) = (weighted_sum / batch_weight).t();
    }

    const double total_weight = current_total_weight + batch_weight;
    const double rel = batch_weight / total_weight;
    return rel * batch_means + (1.0 - rel) * current_means;
}

} // namespace apm


