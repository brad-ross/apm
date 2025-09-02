#include "factor_model_parameter_structs.h"

namespace apm {

namespace {
arma::vec process_weights(arma::vec weights, arma::uword expected_length) {
    if (weights.n_elem == 0) {
        weights = arma::vec(expected_length, arma::fill::ones);
    }
    if (weights.n_elem != expected_length) {
        throw std::invalid_argument("OutcomeMeanSufficientStatistics: weights must have length N.");
    }
    if (arma::any(weights < 0)) {
        throw std::invalid_argument("OutcomeMeanSufficientStatistics: weights must be non-negative.");
    }
    const double sum_w = arma::sum(weights);
    if (sum_w == 0.0) {
        throw std::invalid_argument("OutcomeMeanSufficientStatistics: at least one weight must be positive.");
    }
    weights /= sum_w;
    return weights;
}

std::optional<arma::mat> covar_means_from(
    const std::optional<arma::cube>& covars,
    arma::uword N,
    const arma::vec& normalized_weights) {
    if (!covars) return std::nullopt;
    if (covars->n_rows != N) {
        throw std::invalid_argument("OutcomeMeanSufficientStatistics: covars and outcomes must both have N rows.");
    }

    const arma::uword q = covars->n_slices;
    arma::mat cm(covars->n_cols, q, arma::fill::zeros);
    for (arma::uword k = 0; k < q; ++k) {
        cm.col(k) = covars->slice(k).t() * normalized_weights;
    }
    return cm;
}
} // anonymous namespace

OutcomeMeanSufficientStatistics::OutcomeMeanSufficientStatistics(
    arma::vec observed_means_in,
    std::optional<arma::mat> covar_means_in)
    : observed_outcome_means(std::move(observed_means_in)),
      covar_means(std::move(covar_means_in)) {}

OutcomeMeanSufficientStatistics::OutcomeMeanSufficientStatistics(
    const arma::mat& outcomes,
    std::optional<arma::cube> covars,
    arma::vec weights)
    : OutcomeMeanSufficientStatistics(
          outcomes.t() * process_weights(arma::vec(weights), outcomes.n_rows),
          covar_means_from(covars, outcomes.n_rows,
                           process_weights(std::move(weights), outcomes.n_rows))) {}

OutcomeMeanSufficientStatEstimates::OutcomeMeanSufficientStatEstimates(
    OutcomeMeanSufficientStatistics stats,
    std::vector<OutcomeMeanSufficientStatistics> boot_reps)
    : suff_stat_estimates(std::move(stats)),
      bootstrap_replicates(std::move(boot_reps)) {}

OutcomeMeanSufficientStatEstimates::OutcomeMeanSufficientStatEstimates(
    const arma::mat& outcomes,
    std::shared_ptr<const apm::WeightedBootstrap> bootstrap,
    std::optional<arma::cube> covars,
    arma::uvec unit_idxs)
    : suff_stat_estimates(outcomes, covars) {
    const std::size_t N = static_cast<std::size_t>(outcomes.n_rows);

    if (!bootstrap) {
        bootstrap_replicates.clear();
        return;
    }

    const std::size_t B = bootstrap->n_bootstraps();
    if (bootstrap->n_obs() != N) {
        throw std::invalid_argument("OutcomeMeanSufficientStatEstimates: bootstrap n_obs must equal number of rows (N) in outcomes.");
    }

    // Validate/prepare unit indices
    if (unit_idxs.n_elem == 0) {
        unit_idxs = arma::regspace<arma::uvec>(0, static_cast<arma::uword>(N - 1));
    }
    if (unit_idxs.min() < 0 || unit_idxs.max() >= N) {
        throw std::invalid_argument("OutcomeMeanSufficientStatEstimates: unit_idxs out of range.");
    }

    bootstrap_replicates.resize(B);

    // Get weights for selected units across all draws and use column b directly
    arma::mat weights_rows = bootstrap->obs(unit_idxs); // (unit_idxs.size() x B)
    for (std::size_t b = 0; b < B; ++b) {
        arma::vec unit_idx_weights = weights_rows.col(static_cast<arma::uword>(b));
        bootstrap_replicates[b] = OutcomeMeanSufficientStatistics(outcomes, covars, std::move(unit_idx_weights));
    }
}

} // namespace apm


