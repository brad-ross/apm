#include "twfe_estimator.h"
#include "../online_accumulators.h"

namespace apm {

TWFEEstimator::TWFEEstimator(std::size_t T_c,
                             std::shared_ptr<const WeightedBootstrap> bootstrap,
                             std::size_t q)
	: FactorModelEstimator(/*r=*/1, T_c, std::move(bootstrap), q),
	  N_(0),
	  total_weight_(0.0),
	  outcome_means_(arma::zeros<arma::vec>(T_c))
{
	const std::size_t B = num_bootstraps();
	if (B > 0) {
		total_boot_weights_ = arma::zeros<arma::vec>(B);
		boot_outcome_means_ = arma::zeros<arma::mat>(T_c, B);
	}
}

void TWFEEstimator::add_data_(const arma::uvec& unit_idxs,
                              const arma::mat& Y,
                              const arma::cube& /*X*/)
{
	// Unweighted means (equal weights)
	arma::vec ones_w = arma::ones<arma::vec>(static_cast<arma::uword>(Y.n_rows));
	auto mu_upd = apm::stats::online_weighted_mean(
		Y, ones_w, outcome_means_, total_weight_);
	outcome_means_ = std::move(mu_upd.first);
	total_weight_ = mu_upd.second;
	N_ += static_cast<std::size_t>(Y.n_rows);

	// Bootstrap-specific means (if present)
	const std::size_t B = num_bootstraps();
	if (B > 0) {
		for (std::size_t b = 0; b < B; ++b) {
			const arma::uword bu = static_cast<arma::uword>(b);
			arma::vec w_b = boot_weights_for_indices(unit_idxs, b);
			auto mu_b_upd = apm::stats::online_weighted_mean(
				Y, w_b, boot_outcome_means_.col(bu), static_cast<double>(total_boot_weights_(bu)));
			boot_outcome_means_.col(bu) = std::move(mu_b_upd.first);
			total_boot_weights_(bu) = mu_b_upd.second;
		}
	}
}

FactorModelEstimates TWFEEstimator::estimate()
{
	// Trivial factor: column of ones (enables unit-level demeaning downstream)
	arma::mat G_hat = arma::ones<arma::mat>(T_c(), /*r=*/1);

	// Dummy covariate coefficients (signal presence to downstream if q_ > 0)
	std::optional<arma::vec> a_opt = std::nullopt;
	if (q() > 0) {
		a_opt.emplace(arma::zeros<arma::vec>(static_cast<arma::uword>(q())));
	}

	FactorModelParameters params(std::move(G_hat), outcome_means_, a_opt);

	// Bootstrap replicates
	std::vector<FactorModelParameters> boot_reps;
	const std::size_t B = num_bootstraps();
	if (B > 0) {
		boot_reps.reserve(B);
		for (std::size_t b = 0; b < B; ++b) {
			arma::mat G_b = arma::ones<arma::mat>(T_c(), 1);
			arma::vec g0_b = boot_outcome_means_.col(static_cast<arma::uword>(b));
			boot_reps.emplace_back(std::move(G_b), std::move(g0_b), a_opt);
		}
	}

	return FactorModelEstimates(std::move(params), std::move(boot_reps));
}

} // namespace apm