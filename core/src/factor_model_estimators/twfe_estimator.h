#ifndef APM_TWFE_ESTIMATOR_H
#define APM_TWFE_ESTIMATOR_H

#include "FactorModelEstimator.h"

namespace apm {

/**
 * TWFEEstimator
 * - Emits a trivial factor structure with G = 1_T (T_c x 1 column of ones)
 * - Also provides outcome fixed effects g_0 (unweighted means across units)
 * - If q > 0, emits a dummy covariate coefficient vector a = 0_q
 * - Supports bootstrap: g_0 bootstrap means per draw; G is always ones
 */
class TWFEEstimator : public FactorModelEstimator {
public:
	explicit TWFEEstimator(std::size_t T_c,
	                      std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
	                      std::size_t q = 0);

	FactorModelEstimates estimate() override;

protected:
	void add_data_(const arma::uvec& unit_idxs,
	               const arma::mat& Y,
	               const arma::cube& X) override;

private:
	// Internal running state
	std::size_t N_;                 // number of accumulated rows
	double total_weight_;           // equal to N_ for equal weights
	arma::vec outcome_means_;       // length T_c_

	// Bootstrap aggregates (when present)
	arma::vec total_boot_weights_;  // length B
	arma::mat boot_outcome_means_;  // T_c_ x B
};

} // namespace apm

#endif // APM_TWFE_ESTIMATOR_H


