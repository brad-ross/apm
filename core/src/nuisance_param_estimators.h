#pragma once
#ifndef APM_NUISANCE_PARAM_ESTIMATORS_H
#define APM_NUISANCE_PARAM_ESTIMATORS_H

//==============================================================================
// Estimators for auxiliary (nuisance) cohort-level quantities.
//==============================================================================

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "bootstrap.h"
#include "cohort_specific_param_structs.h"

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

namespace apm {

/**
 * @brief Online estimator of cohort-level auxiliary data means with optional bootstrap.
 *
 * Accumulates auxiliary matrices (T x d) across units, maintaining both point
 * estimates and bootstrap-weighted replicates when a WeightedBootstrap is
 * provided.
 */
class CohortAuxiliaryDataMeanEstimator {
public:
    /**
     * @brief Construct an estimator for cohort-level auxiliary means.
     *
     * @param T Number of outcomes (rows).
     * @param d Number of auxiliary columns.
     * @param bootstrap Optional weighted bootstrap controller.
     */
    explicit CohortAuxiliaryDataMeanEstimator(std::size_t T,
                                              std::size_t d,
                                              std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr);

    std::size_t T() const noexcept { return T_; }
    std::size_t d() const noexcept { return d_; }
    bool has_bootstrap() const noexcept { return static_cast<bool>(bootstrap_); }
    std::size_t num_bootstraps() const noexcept { return bootstrap_ ? bootstrap_->n_bootstraps() : 0; }

    // Batch add: eta is N_c x T x d (units x outcomes x aux-cols)
    /**
     * @brief Add a batch of auxiliary data (units x outcomes x aux-cols).
     *
     * @param unit_idxs Unit indices for the batch (length N_c).
     * @param eta Auxiliary data cube, shape N_c x T x d.
     */
    void add_data(const arma::uvec& unit_idxs, const arma::cube& eta);
    // Single unit add: eta is T x d for the unit
    /**
     * @brief Add a single unit's auxiliary data.
     *
     * @param unit_idx Unit index.
     * @param eta Auxiliary data matrix (T x d) for the unit.
     */
    void add_datum(std::size_t unit_idx, const arma::mat& eta);

    /**
     * @brief Finalize estimates and bootstrap replicates.
     * @return CohortAuxiliaryDataMeanEstimates with point and bootstrap means.
     */
    CohortAuxiliaryDataMeanEstimates estimate() const;

private:
    static void validate_data_dimensions(const arma::uvec& unit_idxs, const arma::cube& eta, std::size_t T, std::size_t d);

    std::size_t T_;                                       ///< Outcome dimension.
    std::size_t d_;                                       ///< Auxiliary column count.
    std::shared_ptr<const WeightedBootstrap> bootstrap_;  ///< Optional bootstrap controller.

    // Running point-estimate aggregates
    double total_weight_;      ///< Accumulated weight.
    arma::mat aux_means_;      ///< Point-estimate auxiliary means (T x d).

    // Running bootstrap aggregates
    arma::vec  total_boot_weights_; ///< Bootstrap weights per draw (length B).
    arma::cube boot_aux_means_;     ///< Bootstrap auxiliary means (T x d x B).
};

} // namespace apm

#endif // APM_NUISANCE_PARAM_ESTIMATORS_H


