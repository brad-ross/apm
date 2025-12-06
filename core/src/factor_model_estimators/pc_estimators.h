#ifndef APM_PC_ESTIMATORS_H
#define APM_PC_ESTIMATORS_H

//==============================================================================
// Principal components-based factor model estimators.
//==============================================================================

#include "FactorModelEstimator.h"

namespace apm {

/**
 * @brief Base class implementing accumulation for principal-components estimators.
 *
 * Maintains second-moment matrices (and bootstrap variants) and provides
 * utilities to extract top-r eigenvectors.
 */
class PCBase : public FactorModelEstimator {
public:
    /**
     * @brief Construct PCBase and initialize accumulators.
     *
     * @param r Factor rank.
     * @param T_c Outcome dimension.
     * @param bootstrap Optional bootstrap weights (shared across batches).
     * @param q Covariate dimension (default 0).
     */
    explicit PCBase(std::size_t r,
                    std::size_t T_c,
                    std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
                    std::size_t q = 0);

protected:
    void add_data_(const arma::uvec& unit_idxs,
                   const arma::mat& Y,
                   const arma::cube& X) override;

    /**
     * @brief Combine second-moment matrix with a weighted batch.
     *
     * @param Y N x T_c outcomes.
     * @param row_weights Length-N weights.
     * @param current_second_moment_mat Running second-moment (T_c x T_c).
     * @param current_total_weight Running total weight.
     * @return Pair {updated second-moment, updated total weight}.
     */
    static std::pair<arma::mat, double> weighted_combine_second_moment_mats(
        const arma::mat& Y,
        const arma::vec& row_weights,
        const arma::mat& current_second_moment_mat,
        double current_total_weight);

    /**
     * @brief Top-r eigenvectors of a symmetric PSD matrix (no symmetrization).
     *
     * @param S Symmetric PSD matrix.
     * @param r Number of eigenvectors to return.
     * @return n x r matrix of eigenvectors (columns).
     */
    static arma::mat top_r_eigenvectors_psd(const arma::mat& S, std::size_t r);

    std::size_t N; ///< Number of observations accumulated.

    // Accessor for subclasses to read the accumulated second moment matrix
    const arma::mat& outcome_second_moment() const { return outcome_second_moment_mat; }

    // Bootstrap aggregates accessible to subclasses
    arma::vec total_boot_weights;                ///< Bootstrap total weights (length B, if bootstrap present).
    arma::cube boot_outcome_second_moment_mats;  ///< Bootstrap second moments (T_c x T_c x B).

private:
    arma::mat outcome_second_moment_mat;         ///< Accumulated second-moment matrix (T_c x T_c).
};

/**
 * @brief Standard PC estimator returning factors only (no fixed effects).
 */
class PCEstimator : public PCBase {
public:
    using PCBase::PCBase;

    /**
     * @brief Compute factor estimates from accumulated data.
     * @return FactorModelEstimates containing factors and optional bootstrap replicates.
     */
    FactorModelEstimates estimate() override;
};

/**
 * @brief PC estimator with outcome fixed effects (g_0) in addition to factors.
 */
class PCEstimatorWithFEs : public PCBase {
public:
    /**
     * @brief Construct PC estimator with outcome fixed effects.
     *
     * @param r Factor rank.
     * @param T_c Outcome dimension.
     * @param bootstrap Optional bootstrap weights.
     * @param q Covariate dimension (default 0).
     */
    explicit PCEstimatorWithFEs(std::size_t r,
                                std::size_t T_c,
                                std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
                                std::size_t q = 0);

    /**
     * @brief Compute factor and fixed-effect estimates from accumulated data.
     * @return FactorModelEstimates containing factors, fixed effects, and optional bootstrap replicates.
     */
    FactorModelEstimates estimate() override;

protected:
    /**
     * @brief Ingest a batch of data (pre-validated) and update accumulators.
     *
     * @param unit_idxs Unit indices for the batch (length N).
     * @param Y N x T_c outcome matrix.
     * @param X N x T_c x q covariate cube (may be empty when q==0).
     */
    void add_data_(const arma::uvec& unit_idxs,
                   const arma::mat& Y,
                   const arma::cube& X) override;

    /**
     * @brief Combine mean vector with a weighted batch.
     *
     * @param Y N x T_c outcomes.
     * @param row_weights Length-N weights.
     * @param current_mean Running mean (length T_c).
     * @param current_total_weight Running total weight.
     * @return Updated mean vector.
     */
    static arma::vec weighted_combine_means(
        const arma::mat& Y,
        const arma::vec& row_weights,
        const arma::vec& current_mean,
        double current_total_weight);

private:
    arma::vec outcome_means;      ///< Running means (length T_c).
    arma::mat boot_outcome_means; ///< Bootstrap outcome means (T_c x B).
};

} // namespace apm

#endif // APM_PC_ESTIMATORS_H