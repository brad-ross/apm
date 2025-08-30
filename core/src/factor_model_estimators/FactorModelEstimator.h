#ifndef APM_FACTOR_MODEL_ESTIMATOR_H
#define APM_FACTOR_MODEL_ESTIMATOR_H

#include <cstddef>
#include <memory>
#include <utility>

#include "../bootstrap.h"
#include "../FactorModelParameters.h"

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
     * @brief Adds a single observation.
     *
     * @param unit_idx The unit index for the observation.
     * @param Y A vector of length T_c containing observed outcomes for the unit.
     * @param X A T_c x q matrix of covariates for the unit. If q==0, X may have
     *          zero columns (rows may be zero or T_c).
     */
    void add_datum(std::size_t unit_idx,
                   const arma::vec& Y,
                   const arma::mat& X = arma::mat());

    /**
     * @brief Produces parameter estimates from the accumulated data.
     * @return FactorModelParameters containing estimate of the T_c x r factor matrix G, 
     * optional T_c-dimensional outcome fixed effects g_0 and optional q-dimensional 
     * covariate coefficients a.
     */
    virtual FactorModelParameters estimate() = 0;

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
        : r_(r), T_c_(T_c), bootstrap_(std::move(bootstrap)), q_(q) {}

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
     */
    void validate_data_dimensions(const arma::uvec& unit_idxs,
                                  const arma::mat& Y,
                                  const arma::cube& X) const;

    /**
     * @brief Validates dimensions for a single observation add.
     *
     * Requirements:
     *  - Y must have length T_c
     *  - X must be T_c x q (may have zero columns if q == 0)
     */
    void validate_datum_dimensions(std::size_t unit_idx,
                                   const arma::vec& Y,
                                   const arma::mat& X) const;

    //==============================================================================
    // Hooks to be implemented by subclasses
    //==============================================================================

    /**
     * @brief Subclass hook for adding a batch (pre-validated inputs).
     */
    virtual void add_data_(const arma::uvec& unit_idxs,
                           const arma::mat& Y,
                           const arma::cube& X) = 0;

    /**
     * @brief Subclass hook for adding a single observation (pre-validated inputs).
     */
    virtual void add_datum_(std::size_t unit_idx,
                            const arma::vec& Y,
                            const arma::mat& X) = 0;

    std::size_t r_;                                        // model rank
    std::size_t T_c_;                                      // outcome dimension
    std::shared_ptr<const WeightedBootstrap> bootstrap_;   // optional bootstrap weights
    std::size_t q_;                                        // covariate dimension
};

} // namespace apm

#endif // APM_FACTOR_MODEL_ESTIMATOR_H


