// Implementation for PC estimators

#include "pc_estimators.h"
#include "../online_accumulators.h"

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
    return apm::stats::online_weighted_second_moment(
        Y, row_weights, current_second_moment_mat, current_total_weight);
}

arma::mat PCBase::top_r_eigenvectors_psd(const arma::mat& S, std::size_t r)
{
    const arma::uword n = S.n_rows;
    if (S.n_cols != n) {
        throw std::invalid_argument("top_r_eigenvectors_psd: input must be square.");
    }
    if (r > static_cast<std::size_t>(n)) {
        throw std::invalid_argument("top_r_eigenvectors_psd: r cannot exceed matrix dimension.");
    }
    if (r == 0) {
        return arma::mat(n, 0, arma::fill::zeros);
    }

    arma::vec eigval;
    arma::mat eigvec;
    if (!arma::eig_sym(eigval, eigvec, S)) {
        throw std::runtime_error("top_r_eigenvectors_psd: eig_sym failed.");
    }

    const arma::uword p = eigvec.n_cols;
    arma::uvec idxs = arma::regspace<arma::uvec>(p - static_cast<arma::uword>(r), p - 1);
    return eigvec.cols(idxs);
}

FactorModelEstimates PCEstimator::estimate()
{
    // Compute main estimate using accumulated second-moment matrix
    const arma::mat& S = outcome_second_moment();
    arma::mat G_hat = top_r_eigenvectors_psd(S, r());
    FactorModelParameters params(std::move(G_hat));

    // Bootstrap replicates, if any
    std::vector<FactorModelParameters> boot_reps;
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        boot_reps.reserve(B);
        for (std::size_t b = 0; b < B; ++b) {
            arma::mat S_b = boot_outcome_second_moment_mats.slice(static_cast<arma::uword>(b));
            arma::mat G_b = top_r_eigenvectors_psd(S_b, r());
            boot_reps.emplace_back(std::move(G_b));
        }
    }

    return FactorModelEstimates(std::move(params), std::move(boot_reps));
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
    auto res = apm::stats::online_weighted_mean(Y, row_weights, current_mean, current_total_weight);
    return std::move(res.first);
}

FactorModelEstimates PCEstimatorWithFEs::estimate()
{
    // Compute covariance matrix: E[YY'] - mu mu'
    arma::mat cov = outcome_second_moment() - outcome_means * outcome_means.t();

    arma::mat G_hat = top_r_eigenvectors_psd(cov, r());
    FactorModelParameters params(std::move(G_hat), outcome_means);

    std::vector<FactorModelParameters> boot_reps;
    const std::size_t B = num_bootstraps();
    if (B > 0) {
        boot_reps.reserve(B);
        for (std::size_t b = 0; b < B; ++b) {
            const arma::uword bu = static_cast<arma::uword>(b);
            arma::vec mu_b = boot_outcome_means.col(bu);
            arma::mat cov_b = boot_outcome_second_moment_mats.slice(bu) - mu_b * mu_b.t();

            arma::mat G_b = top_r_eigenvectors_psd(cov_b, r());
            boot_reps.emplace_back(std::move(G_b), std::move(mu_b));
        }
    }

    return FactorModelEstimates(std::move(params), std::move(boot_reps));
}

} // namespace apm