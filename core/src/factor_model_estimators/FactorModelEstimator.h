#ifndef APM_FACTOR_MODEL_ESTIMATOR_H
#define APM_FACTOR_MODEL_ESTIMATOR_H

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../bootstrap.h"
#include "../cohort_specific_param_structs.h"

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

namespace apm {

/**
 * @brief Abstract base class for linear factor model estimators.
 *
 * This class defines a common interface for estimators of linear factor models
 * with rank \( r \), outcome dimension \( T_c \) and optional covariates of
 * dimension \( q \). Implementations may optionally use bootstrap weights
 * provided via a `WeightedBootstrap` object.
 */
class FactorModelEstimator {
public:
    virtual ~FactorModelEstimator() = default;

    //==============================================================================
    // Dimensions and configuration
    //==============================================================================

    /**
     * @brief Returns the model rank (number of factors).
     * @return Rank r.
     */
    std::size_t r() const noexcept { return r_; }

    /**
     * @brief Returns the outcome dimension.
     * @return Outcome dimension T_c.
     */
    std::size_t T_c() const noexcept { return T_c_; }

    /**
     * @brief Indicates whether bootstrap weights were provided.
     * @return `true` if a bootstrap object is present, `false` otherwise.
     */
    bool has_bootstrap() const noexcept { return static_cast<bool>(bootstrap_); }

    /**
     * @brief Returns the bootstrap weights object, if any.
     * @return Shared pointer to immutable `WeightedBootstrap` (may be null).
     */
    std::shared_ptr<const WeightedBootstrap> bootstrap() const noexcept { return bootstrap_; }

    /**
     * @brief Returns the number of bootstrap replicates.
     * @return B when bootstrap present; otherwise 0.
     */
    std::size_t num_bootstraps() const noexcept {
        return bootstrap_ ? bootstrap_->n_bootstraps() : 0;
    }

    /**
     * @brief Returns bootstrap weights for the specified unit indices and draw b.
     * @param unit_idxs Vector of unit indices (rows in the bootstrap weight matrix).
     * @param b Bootstrap draw index in [0, B).
     * @return Vector of length unit_idxs.n_elem containing weights for draw b.
     * @throws std::runtime_error if no bootstrap is present or b is out of range.
     */
    arma::vec boot_weights_for_indices(const arma::uvec& unit_idxs, std::size_t b) const;

    /**
     * @brief Returns the covariate dimension.
     * @return Covariate dimension q.
     */
    std::size_t q() const noexcept { return q_; }

    //==============================================================================
    // Data ingestion interface (Template Method)
    //==============================================================================

    /**
     * @brief Adds a batch of observations.
     *
     * @param unit_idxs A vector of length N containing unit indices for the batch.
     * @param Y An N x T_c matrix of observed outcomes for the batch.
     * @param X An N x T_c x q cube of covariates for the batch. If q==0, X may be empty
     *          (zero slices; rows/cols may be zero or match N and T_c).
     */
    void add_data(const arma::uvec& unit_idxs,
                  const arma::mat& Y,
                  const arma::cube& X = arma::cube());

    /**
     * @brief Adds a single observation by wrapping into a batch and delegating to add_data.
     * 
     * @param unit_idx The index of the unit to add.
     * @param Y A T_c vector of outcomes for the unit.
     * @param X A T_c x q matrix of covariates for the unit.
     */
    void add_datum(std::size_t unit_idx,
                   const arma::vec& Y,
                   const arma::mat& X = arma::mat());

    /**
     * @brief Produces parameter estimates (and optional bootstrap replicates) from the accumulated data.
     * @return FactorModelEstimates containing point estimates and (if applicable) bootstrap replicates.
     */
    virtual FactorModelEstimates estimate() = 0;

protected:
    /**
     * @brief Constructs the estimator interface with dimensions and optional bootstrap.
     *
     * @param r The model rank (number of factors).
     * @param T_c The outcome dimension.
     * @param bootstrap Optional shared pointer to immutable bootstrap weights.
     * @param q The covariate dimension (default 0).
     */
    explicit FactorModelEstimator(std::size_t r,
                                  std::size_t T_c,
                                  std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
                                  std::size_t q = 0)
        : r_(r), T_c_(T_c), bootstrap_(std::move(bootstrap)), q_(q) {
        if (r_ > T_c_) {
            throw std::invalid_argument("FactorModelEstimator: r must be <= T_c.");
        }
    }

    //==============================================================================
    // Validation helpers
    //==============================================================================

    /**
     * @brief Validates dimensions for a batch add.
     *
     * Requirements:
     *  - Y must be N x T_c where N = unit_idxs.n_elem
     *  - X must be N x T_c x q
     *  - If a bootstrap is present, unit indices are not range-checked here (left to subclasses).
     *
     * @param unit_idxs Unit indices for the batch (length N).
     * @param Y N x T_c outcome matrix.
     * @param X N x T_c x q covariate cube.
     */
    void validate_data_dimensions(const arma::uvec& unit_idxs,
                                  const arma::mat& Y,
                                  const arma::cube& X) const;

    // No single-datum validation needed; add_data performs validation for wrapped batches

    //==============================================================================
    // Hooks to be implemented by subclasses
    //==============================================================================

    /**
     * @brief Subclass hook for adding a batch (pre-validated inputs).
     *
     * @param unit_idxs Unit indices for the batch (length N).
     * @param Y N x T_c outcome matrix.
     * @param X N x T_c x q covariate cube (may be empty when q==0).
     */
    virtual void add_data_(const arma::uvec& unit_idxs,
                           const arma::mat& Y,
                           const arma::cube& X) = 0;

    std::size_t r_;                                        ///< Model rank.
    std::size_t T_c_;                                      ///< Outcome dimension.
    std::shared_ptr<const WeightedBootstrap> bootstrap_;   ///< Optional bootstrap weights.
    std::size_t q_;                                        ///< Covariate dimension.
};

} // namespace apm

#endif // APM_FACTOR_MODEL_ESTIMATOR_H