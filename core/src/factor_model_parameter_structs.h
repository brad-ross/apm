#ifndef APM_FACTOR_MODEL_PARAMETERS_H
#define APM_FACTOR_MODEL_PARAMETERS_H

#include <optional>
#include <stdexcept>
#include <vector>
#include <utility>

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

namespace apm {

/**
 * @brief Container for estimated factor model parameters.
 *
 * Fields:
 *  - G: T_c x r matrix of factor estimates.
 *  - g_0: optional T_c vector of outcome fixed effects.
 *  - a: optional q vector of covariate coefficient estimates.
 */
struct FactorModelParameters {
    arma::mat G;                           // T_c x r
    std::optional<arma::vec> g_0;          // length T_c (if present)
    std::optional<arma::vec> a;            // length q (if present)

    // Constructors
    FactorModelParameters() = default;

    FactorModelParameters(arma::mat G_in,
                          std::optional<arma::vec> g0_in = std::nullopt,
                          std::optional<arma::vec> a_in = std::nullopt)
        : G(std::move(G_in)), g_0(std::move(g0_in)), a(std::move(a_in)) {
        if (g_0 && g_0->n_elem != G.n_rows) {
            throw std::invalid_argument("FactorModelParameters: g_0 length must equal number of ows in G (T_c).");
        }
        // 'a' length validation requires external knowledge of q; callers may validate separately.
    }

    // Presence checks
    bool has_fixed_effects() const noexcept { return static_cast<bool>(g_0); }
    bool has_covariate_coefs() const noexcept { return static_cast<bool>(a); }

    // Convenience dimensions
    std::size_t T_c() const noexcept { return static_cast<std::size_t>(G.n_rows); }
    std::size_t r() const noexcept { return static_cast<std::size_t>(G.n_cols); }
    std::size_t q() const noexcept { return a ? static_cast<std::size_t>(a->n_elem) : 0; }
};

/**
 * @brief Container for point estimates and (optional) bootstrap replicates of factor model parameters.
 *
 * Fields:
 *  - parameter_estimates: the primary parameter estimates (G, optional g_0, optional a).
 *  - bootstrap_replicates: a vector of parameter estimates, one per bootstrap draw when present,
 *    or empty if no bootstrap is attached.
 */
struct FactorModelEstimates {
    FactorModelParameters parameter_estimates;                 // point estimates
    std::vector<FactorModelParameters> bootstrap_replicates;  // length B if bootstrap is present; otherwise 0

    /**
     * @brief Move-construct from a precomputed vector of bootstrap replicates.
     * @param params Point estimates.
     * @param boot_reps Vector of bootstrap parameter estimates (moved into place).
     */
    FactorModelEstimates(FactorModelParameters params,
                         std::vector<FactorModelParameters> boot_reps)
        : parameter_estimates(std::move(params)),
          bootstrap_replicates(std::move(boot_reps)) {}

    /**
     * @brief Indicates whether bootstrap replicates are present (non-empty).
     * @return `true` if replicates exist, `false` otherwise.
     */
    bool has_bootstrap_replicates() const noexcept { return !bootstrap_replicates.empty(); }
};

} // namespace apm

#endif // APM_FACTOR_MODEL_PARAMETERS_H