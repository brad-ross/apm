// Implementation for PC estimators

#include "pc_estimators.h"

namespace apm {

PCBase::PCBase(std::size_t r,
               std::size_t T_c,
               std::shared_ptr<const WeightedBootstrap> bootstrap,
               std::size_t q)
    : FactorModelEstimator(r, T_c, std::move(bootstrap), q),
      N(0),
      outcome_second_moment_mat(arma::zeros<arma::mat>(T_c, T_c))
{
    // Initialize bootstrap-related storage if bootstrap is present
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        total_boot_weights = arma::zeros<arma::vec>(B);
        // Shape: T_c x T_c x B (slice per bootstrap draw)
        boot_outcome_second_moment_mats = arma::cube(T_c, T_c, B, arma::fill::zeros);
    }
}

void PCBase::add_data_(const arma::uvec& unit_idxs,
                       const arma::mat& Y,
                       const arma::cube& X)
{
    // Update unweighted second moment using equal row weights (ones)
    arma::vec ones_w = arma::ones<arma::vec>(static_cast<arma::uword>(Y.n_rows));
    auto combined_main = weighted_combine_second_moment_mats(
        Y, ones_w, outcome_second_moment_mat, static_cast<double>(N));
    outcome_second_moment_mat = std::move(combined_main.first);
    N = static_cast<std::size_t>(combined_main.second);

    // Update bootstrap-specific aggregates if present
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        const std::size_t Tc = T_c();
        for (std::size_t b = 0; b < B; ++b) {
            const arma::uword bu = static_cast<arma::uword>(b);
            arma::vec w_b = boot_weights_for_indices(unit_idxs, b); // length batch_N

            arma::mat curr_boot_second_moment_mat = boot_outcome_second_moment_mats.slice(bu);

            auto combined_b = weighted_combine_second_moment_mats(
                Y, w_b, curr_boot_second_moment_mat,
                static_cast<double>(total_boot_weights(bu)));

            // Write combined matrix back into slice b
            boot_outcome_second_moment_mats.slice(bu) = combined_b.first;

            total_boot_weights(bu) = combined_b.second;
        }
    }
}

std::pair<arma::mat, double> PCBase::weighted_combine_second_moment_mats(
    const arma::mat& Y,
    const arma::vec& row_weights,
    const arma::mat& current_second_moment_mat,
    double current_total_weight)
{
    // Compute weighted outer product mean for the batch: (Y' * diag(w) * Y) / sum(w)
    const double batch_weight = arma::accu(row_weights);
    if (batch_weight == 0.0) {
        return {current_second_moment_mat, current_total_weight}; // no contribution from empty/zero-weight batch
    }

    arma::mat WY = Y.each_col() % row_weights; // weight rows
    arma::mat batch_second_moment = (Y.t() * WY) / batch_weight; // T_c x T_c

    const double total_weight = current_total_weight + batch_weight;
    if (total_weight == 0.0) {
        return {current_second_moment_mat, 0.0}; // both zero → unchanged
    }

    const double rel_batch_weight  = batch_weight / total_weight;
    arma::mat combined = rel_batch_weight * batch_second_moment + (1 - rel_batch_weight) * current_second_moment_mat;
    return {std::move(combined), total_weight};
}

PCEstimatorWithFEs::PCEstimatorWithFEs(std::size_t r,
                                       std::size_t T_c,
                                       std::shared_ptr<const WeightedBootstrap> bootstrap,
                                       std::size_t q)
    : PCBase(r, T_c, std::move(bootstrap), q),
      outcome_means(arma::zeros<arma::vec>(T_c))
{
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        boot_outcome_means = arma::zeros<arma::mat>(T_c, B);
    }
}

void PCEstimatorWithFEs::add_data_(const arma::uvec& unit_idxs,
                                   const arma::mat& Y,
                                   const arma::cube& X)
{
    // First, update second moments and N via base logic
    PCBase::add_data_(unit_idxs, Y, X);

    // Unweighted means (equal weights)
    arma::vec ones_w = arma::ones<arma::vec>(static_cast<arma::uword>(Y.n_rows));
    outcome_means = weighted_combine_means(Y, ones_w, outcome_means, static_cast<double>(N - Y.n_rows));

    // Bootstrap-specific means
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        for (std::size_t b = 0; b < B; ++b) {
            const arma::uword bu = static_cast<arma::uword>(b);
            arma::vec w_b = boot_weights_for_indices(unit_idxs, b);
            arma::vec curr_mean_b = boot_outcome_means.col(bu);
            arma::vec combined_b = weighted_combine_means(
                Y, w_b, curr_mean_b,
                static_cast<double>(total_boot_weights(bu) - arma::accu(w_b)));
            boot_outcome_means.col(bu) = combined_b;
        }
    }
}

arma::vec PCEstimatorWithFEs::weighted_combine_means(
    const arma::mat& Y,
    const arma::vec& row_weights,
    const arma::vec& current_mean,
    double current_total_weight)
{
    const double batch_weight = arma::accu(row_weights);
    if (batch_weight == 0.0) {
        return current_mean;
    }

    // Weighted batch mean across rows: sum_i w_i * Y_i / sum(w)
    arma::rowvec weighted_sum = row_weights.t() * Y; // 1 x T_c
    arma::vec batch_mean = (weighted_sum / batch_weight).t(); // T_c

    const double total_weight = current_total_weight + batch_weight;
    if (total_weight == 0.0) {
        return current_mean;
    }

    const double rel_batch_weight = batch_weight / total_weight;
    arma::vec combined = rel_batch_weight * batch_mean + (1 - rel_batch_weight) * current_mean;
    return combined;
}

} // namespace apm