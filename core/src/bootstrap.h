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
    std::size_t n_obs() const noexcept { return weights_.n_rows; }
    std::size_t n_bootstraps() const noexcept { return weights_.n_cols; }

    // Accessors
    arma::vec draw(std::size_t b) const;            // weights across all observations for draw b
    arma::vec obs(std::size_t i) const;             // weights across all draws for observation i
    arma::mat obs(const arma::uvec& idx) const;     // rows for a set of observation indices
    arma::mat obs(const std::vector<std::size_t>& idx) const; // rows for a set of observation indices
    const arma::mat& weights() const noexcept { return weights_; }

    // Convenience
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
    explicit MultinomialBootstrap(std::size_t N, std::size_t B, std::uint64_t seed = 0);
};

/**
 * @brief Bayesian bootstrap (Rubin). Each draw uses i.i.d. Gamma(1,1) (Exp(1)) raw weights and normalizes to sum to 1.
 */
class BayesianBootstrap final : public WeightedBootstrap {
public:
    explicit BayesianBootstrap(std::size_t N, std::size_t B, std::uint64_t seed = 0);
};

} // namespace apm

#endif // BOOTSTRAP_H


