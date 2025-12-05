#ifndef APM_TWFE_ESTIMATOR_H
#define APM_TWFE_ESTIMATOR_H

//==============================================================================
// Two-way fixed effects (TWFE) baseline estimator.
//==============================================================================

#include "FactorModelEstimator.h"

namespace apm {

/**
 * @brief Two-way fixed effects baseline estimator (no latent factors).
 *
 * - Emits G = 1_T (T_c x 1 column of ones).
 * - Provides outcome fixed effects g_0 (unweighted means across units).
 * - If q > 0, emits a dummy covariate coefficient vector a = 0_q.
 * - Supports bootstrap: g_0 bootstrap means per draw; G is always ones.
 *
 * Useful as a baseline or for testing pipelines that expect a
 * FactorModelEstimator-compatible interface without learning latent factors.
 */
class TWFEEstimator : public FactorModelEstimator {
public:
    /**
     * @brief Construct a TWFE estimator.
     *
     * @param T_c Outcome dimension.
     * @param bootstrap Optional bootstrap weights.
     * @param q Covariate dimension (default 0).
     */
    explicit TWFEEstimator(std::size_t T_c,
                           std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
                           std::size_t q = 0);

	FactorModelEstimates estimate() override;

protected:
    /**
     * @brief Ingest a batch of data (pre-validated).
     *
     * @param unit_idxs Unit indices (length N).
     * @param Y N x T_c outcomes.
     * @param X N x T_c x q covariates (may be empty if q==0).
     */
    void add_data_(const arma::uvec& unit_idxs,
                   const arma::mat& Y,
                   const arma::cube& X) override;

private:
    // Internal running state
    std::size_t N_;                 ///< Number of accumulated rows.
    double total_weight_;           ///< Running total weight (equals N_ for equal weights).
    arma::vec outcome_means_;       ///< Outcome means (length T_c_).

    // Bootstrap aggregates (when present)
    arma::vec total_boot_weights_;  ///< Bootstrap total weights (length B).
    arma::mat boot_outcome_means_;  ///< Bootstrap outcome means (T_c_ x B).
};

} // namespace apm

#endif // APM_TWFE_ESTIMATOR_H


