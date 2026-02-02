#ifndef BOOTSTRAP_H
#define BOOTSTRAP_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

namespace apm {

/**
 * @brief Immutable container for bootstrap weights (N observations x B draws).
 *
 * Convention: columns correspond to bootstrap draws; rows correspond to observations.
 * Each column sums to 1.
 */
class WeightedBootstrap {
public:
    virtual ~WeightedBootstrap() = default;

    // Dimensions
    /** @return Number of observations N. */
    std::size_t n_obs() const noexcept { return weights_.n_rows; }
    /** @return Number of bootstrap draws B. */
    std::size_t n_bootstraps() const noexcept { return weights_.n_cols; }

    // Accessors
    /**
     * @brief Get weights for a single bootstrap draw.
     * @param b Bootstrap draw index in [0, B).
     * @return Length-N vector of weights for draw b.
     */
    arma::vec draw(std::size_t b) const;
    /**
     * @brief Get weights across all draws for a single observation.
     * @param i Observation index in [0, N).
     * @return Length-B vector of weights for observation i.
     */
    arma::vec obs(std::size_t i) const;
    /**
     * @brief Get weights for a set of observation indices (arma::uvec).
     * @param idx Observation indices.
     * @return Submatrix of weights with selected rows.
     */
    arma::mat obs(const arma::uvec& idx) const;
    /**
     * @brief Get weights for a set of observation indices (std::vector).
     * @param idx Observation indices.
     * @return Submatrix of weights with selected rows.
     */
    arma::mat obs(const std::vector<std::size_t>& idx) const;
    /**
     * @brief Access the full N x B weight matrix.
     */
    const arma::mat& weights() const noexcept { return weights_; }

    // Convenience
    /**
     * @brief Human-readable description of dimensions (N, B).
     * @return Description string.
     */
    std::string desc() const;

protected:
    explicit WeightedBootstrap(arma::mat weights);

private:
    arma::mat weights_;
    static void normalize_columns(arma::mat& W);
    static void validate_nonempty(const arma::mat& W);
    static void validate_nonnegative(const arma::mat& W);
    static void validate_index(std::size_t idx, std::size_t lim, const char* what);
    static void validate_indices(const arma::uvec& idx, std::size_t lim, const char* what);
};

/**
 * @brief Classical multinomial bootstrap (Efron). Each draw samples N indices with replacement uniformly.
 * Weights are counts / N, normalized to sum to 1 per draw.
 */
class MultinomialBootstrap final : public WeightedBootstrap {
public:
    /**
     * @param N Number of observations.
     * @param B Number of bootstrap draws.
     * @param seed RNG seed.
     */
    explicit MultinomialBootstrap(std::size_t N, std::size_t B, std::uint64_t seed = 0);
};

/**
 * @brief Bayesian bootstrap (Rubin). Each draw uses i.i.d. Gamma(1,1) (Exp(1)) raw weights and normalizes to sum to 1.
 */
class BayesianBootstrap final : public WeightedBootstrap {
public:
    /**
     * @param N Number of observations.
     * @param B Number of bootstrap draws.
     * @param seed RNG seed.
     */
    explicit BayesianBootstrap(std::size_t N, std::size_t B, std::uint64_t seed = 0);
};

/**
 * @brief Results from bootstrap-based simultaneous inference.
 */
struct SimultaneousInferenceResults {
    arma::vec point_ests;           ///< Point estimates (length p)
    arma::vec pointwise_t_stats;    ///< t-stats using bootstrap SEs
    arma::vec pointwise_p_vals;     ///< Unadjusted p-values
    arma::vec std_errs;             ///< Robust IQR-based SEs: row_sd / sqrt(N)
    double sig_level;               ///< Significance level used for intervals/bands
    arma::vec ci_lb;                ///< Pointwise CI lower bounds
    arma::vec ci_ub;                ///< Pointwise CI upper bounds
    arma::vec fwer_control_p_vals;  ///< Romano–Wolf stepdown adjusted p-values
    arma::vec cb_lb;                ///< Simultaneous confidence band lower bounds
    arma::vec cb_ub;                ///< Simultaneous confidence band upper bounds
};

/**
 * @brief Computes per-parameter CIs and a simultaneous confidence band from bootstrap replicates.
 * 
 * @param point_ests p-vector of point estimates.
 * @param bootstrap_replicates p x B matrix of bootstrap estimates.
 * @param N Sample size used for the estimates.
 * @param sig_level Significance level in (0,1).
 * @return SimultaneousInferenceResults containing point estimates, t-stats, p-values, CIs, and simultaneous bands.
 */
SimultaneousInferenceResults get_bootstrap_inference(
    const arma::vec& point_ests,
    const arma::mat& bootstrap_replicates,
    std::size_t N,
    double sig_level
);

} // namespace apm

#endif // BOOTSTRAP_H


