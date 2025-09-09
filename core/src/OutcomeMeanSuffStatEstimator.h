#pragma once
#ifndef APM_OUTCOME_MEAN_SUFF_STAT_ESTIMATOR_H
#define APM_OUTCOME_MEAN_SUFF_STAT_ESTIMATOR_H

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
 * @brief Incrementally computes sufficient statistics for outcome means (and optional covariate means),
 *        with optional bootstrap replicates.
 *
 * This estimator aggregates, across units, the observed outcome means for the cohort and,
 * when covariates are provided, the mean of covariates by outcome. Data can be supplied in
 * batches via `add_data` or one unit at a time via `add_datum`. If a `WeightedBootstrap`
 * is attached, bootstrap-weighted sufficient statistics are accumulated in parallel.
 *
 * Dimensions:
 *  - T_c: number of observed outcomes in the cohort (rows of observed Y used for identification)
 *  - T:   total number of outcomes (rows of X when covariates are provided)
 *  - q:   number of covariates (columns of X when provided)
 */
class OutcomeMeanSuffStatEstimator {
public:
    /**
     * @brief Construct an estimator.
     * @param T_c Number of observed outcomes in this cohort.
     * @param T   Total number of outcomes (required only if covariates are used). Default 0.
     * @param q   Number of covariates (required only if covariates are used). Default 0.
     * @param bootstrap Optional weighted bootstrap controller. If provided, bootstrap replicates
     *                  are accumulated alongside point estimates.
     */
    explicit OutcomeMeanSuffStatEstimator(std::size_t T_c,
                                          std::size_t T = 0,
                                          std::size_t q = 0,
                                          std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr);

    /** @return T_c: number of observed outcomes in the cohort. */
    std::size_t T_c() const noexcept { return T_c_; }
    /** @return T: total number of outcomes (0 if covariates are not tracked). */
    std::size_t T()   const noexcept { return T_; }
    /** @return q: number of covariates (0 if covariates are not tracked). */
    std::size_t q()   const noexcept { return q_; }

    /** @return true if a bootstrap object is attached. */
    bool has_bootstrap() const noexcept { return static_cast<bool>(bootstrap_); }
    /** @return number of bootstrap replicates managed by the attached bootstrap (0 if none). */
    std::size_t num_bootstraps() const noexcept { return bootstrap_ ? bootstrap_->n_bootstraps() : 0; }

    /**
     * @brief Add a batch of units' data.
     * @param unit_idxs Indexes of units included in this batch (used for bootstrap weighting).
     * @param Y Matrix of outcomes with shape T_c x N_batch, where each column corresponds to a unit.
     * @param X Optional cube of covariates with shape T x q x N_batch, aligned to Y's unit order.
     *
     * Aggregates point-estimate means and, when a bootstrap is attached, aggregates bootstrap-weighted
     * means for each replicate using weights derived from `unit_idxs`.
     */
    void add_data(const arma::uvec& unit_idxs,
                  const arma::mat& Y,
                  const arma::cube& X = arma::cube());

    /**
     * @brief Add a single unit's data.
     * @param unit_idx Index of the unit (used for bootstrap weighting).
     * @param Y Vector of outcomes of length T_c for this unit.
     * @param X Optional matrix of covariates with shape T x q for this unit.
     */
    void add_datum(std::size_t unit_idx,
                   const arma::vec& Y,
                   const arma::mat& X = arma::mat());

    /**
     * @brief Finalize and return sufficient statistics and optional bootstrap replicates.
     * @return Aggregated outcome mean sufficient statistics and, when available, bootstrap replicates.
     */
    OutcomeMeanSuffStatEstimates estimate() const;

private:
    /**
     * @brief Validate dimensions of inputs for a batch addition.
     * @throws std::invalid_argument on any dimension mismatch.
     */
    void validate_data_dimensions(const arma::uvec& unit_idxs,
                                  const arma::mat& Y,
                                  const arma::cube& X) const;

    /**
     * @brief Retrieve bootstrap weights corresponding to `unit_idxs` for replicate b.
     * @param unit_idxs Batch unit indices.
     * @param b Bootstrap replicate index.
     * @return Vector of nonnegative weights of length equal to the batch size.
     */
    arma::vec boot_weights_for_indices(const arma::uvec& unit_idxs, std::size_t b) const;

    /**
     * @brief Combine a batch's outcomes into running outcome means.
     * @param Y            Batch outcomes (T_c x N_batch).
     * @param row_weights  Weights per unit/column (length N_batch).
     * @param current_mean Current running mean vector (length T_c).
     * @param current_total_weight Current total weight accumulated so far.
     * @return Pair of updated mean vector and updated total weight.
     */
    static std::pair<arma::vec, double> combine_means(
        const arma::mat& Y,
        const arma::vec& row_weights,
        const arma::vec& current_mean,
        double current_total_weight);

    /**
     * @brief Combine a batch's covariates into running covariate means.
     * @param X            Batch covariates (T x q x N_batch).
     * @param row_weights  Weights per unit (length N_batch).
     * @param current_means Current running means matrix (T x q).
     * @param current_total_weight Current total weight accumulated so far.
     * @return Updated covariate means matrix (T x q).
     */
    static arma::mat combine_covar_means(
        const arma::cube& X,
        const arma::vec& row_weights,
        const arma::mat& current_means,
        double current_total_weight);

    // Dimensions
    std::size_t T_c_;
    std::size_t T_;
    std::size_t q_;

    // Optional bootstrap state
    std::shared_ptr<const WeightedBootstrap> bootstrap_;

    // Running point-estimate aggregates
    arma::vec outcome_means_;
    double total_weight_;
    arma::mat covar_means_;

    // Running bootstrap aggregates
    arma::vec  total_boot_weights_;
    arma::mat  boot_outcome_means_;
    arma::cube boot_covar_means_;
};

} // namespace apm

#endif // APM_OUTCOME_MEAN_SUFF_STAT_ESTIMATOR_H


