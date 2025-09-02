#ifndef APM_FACTOR_MODEL_PARAMETERS_H
#define APM_FACTOR_MODEL_PARAMETERS_H

#include <optional>
#include <stdexcept>
#include <vector>
#include <utility>
#include <memory>

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif
#include "bootstrap.h"

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
    /**
     * @brief Indicates whether outcome fixed effects (g_0) are present.
     * @return true if g_0 has a value; false otherwise.
     */
    bool has_fixed_effects() const noexcept { return static_cast<bool>(g_0); }
    /**
     * @brief Indicates whether covariate coefficients (a) are present.
     * @return true if a has a value; false otherwise.
     */
    bool has_covariate_coefs() const noexcept { return static_cast<bool>(a); }

    // Convenience dimensions
    /**
     * @brief Returns the number of observed outcomes in this cohort (T_c).
     * @return Number of rows in G.
     */
    std::size_t T_c() const noexcept { return static_cast<std::size_t>(G.n_rows); }
    /**
     * @brief Returns the factor rank (r).
     * @return Number of columns in G.
     */
    std::size_t r() const noexcept { return static_cast<std::size_t>(G.n_cols); }
    /**
     * @brief Returns the number of covariates (q) if covariate coefficients are present, else 0.
     * @return Length of a if present; otherwise 0.
     */
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
                         std::vector<FactorModelParameters> boot_reps = {})
        : parameter_estimates(std::move(params)),
          bootstrap_replicates(std::move(boot_reps)) {}

    /**
     * @brief Indicates whether bootstrap replicates are present (non-empty).
     * @return `true` if replicates exist, `false` otherwise.
     */
    bool has_bootstrap_replicates() const noexcept { return !bootstrap_replicates.empty(); }
};

/**
 * @brief Container for sufficient statistics for identifying and estimating outcome means.
 *
 * Fields:
 *  - observed_outcome_means: length T_c vector of observed outcome means.
 *  - covar_means: optional T x q matrix of covariate means (per outcome).
 */
struct OutcomeMeanSufficientStatistics {
    arma::vec observed_outcome_means;                   // length T_c
    std::optional<arma::mat> covar_means;               // T x q (if present)

    // Constructors
    OutcomeMeanSufficientStatistics() = default;

    OutcomeMeanSufficientStatistics(arma::vec observed_means_in,
                                    std::optional<arma::mat> covar_means_in = std::nullopt);

    /**
     * @brief Construct from raw outcomes and optional covariates, aggregating across units.
     *
     * @param outcomes N x T_c matrix of observed outcomes across N units.
     * @param covars Optional N x T x q cube of covariates across N units.
     *               When provided, covariate means are computed by averaging over the N dimension
     *               (weighted if weights provided), yielding a T x q matrix.
     * @param weights Optional length-N vector of non-negative weights. If empty, uses equal weights.
     *                Weights are normalized to sum to one before aggregation.
     */
    OutcomeMeanSufficientStatistics(const arma::mat& outcomes,
                                    std::optional<arma::cube> covars = std::nullopt,
                                    arma::vec weights = arma::vec());

    // Presence checks
    /**
     * @brief Indicates whether covariate means are present.
     * @return true if covar_means has a value; false otherwise.
     */
    bool has_covar_means() const noexcept { return static_cast<bool>(covar_means); }

    // Convenience dimensions
    /**
     * @brief Returns the number of observed outcomes (T_c).
     * @return Length of observed_outcome_means.
     */
    std::size_t T_c() const noexcept { return static_cast<std::size_t>(observed_outcome_means.n_elem); }
    /**
     * @brief Returns the number of outcomes (T) if covariate means are present, else 0.
     * @return Number of rows in covar_means, or 0 if absent.
     */
    std::size_t T() const noexcept { return covar_means ? static_cast<std::size_t>(covar_means->n_rows) : 0; }
    /**
     * @brief Returns the number of covariates (q) if covariate means are present, else 0.
     * @return Number of columns in covar_means, or 0 if absent.
     */
    std::size_t q() const noexcept { return covar_means ? static_cast<std::size_t>(covar_means->n_cols) : 0; }
};

/**
 * @brief Aggregated sufficient statistics estimates with optional bootstrap replicates.
 */
struct OutcomeMeanSufficientStatEstimates {
    OutcomeMeanSufficientStatistics suff_stat_estimates;                  // point sufficient stats
    std::vector<OutcomeMeanSufficientStatistics> bootstrap_replicates;   // length B if present; otherwise 0

    OutcomeMeanSufficientStatEstimates(OutcomeMeanSufficientStatistics stats,
                                       std::vector<OutcomeMeanSufficientStatistics> boot_reps = {});

    /**
     * @brief Indicates whether bootstrap replicates are present (non-empty).
     * @return true if replicates exist; false otherwise.
     */
    bool has_bootstrap_replicates() const noexcept { return !bootstrap_replicates.empty(); }

    /**
     * @brief Construct sufficient statistics and bootstrap replicates from raw data and a bootstrap object.
     *
     * @param outcomes N x T_c matrix of observed outcomes.
     * @param bootstrap shared_ptr to WeightedBootstrap providing B bootstrap draws. If null, no replicates are created.
     * @param covars Optional N x T x q cube of covariates.
     * @param unit_idxs Optional indices of units to include (0-based). If empty, all units are used.
     */
    OutcomeMeanSufficientStatEstimates(const arma::mat& outcomes,
                                       std::shared_ptr<const apm::WeightedBootstrap> bootstrap,
                                       std::optional<arma::cube> covars = std::nullopt,
                                       arma::uvec unit_idxs = arma::uvec());
};

} // namespace apm

#endif // APM_FACTOR_MODEL_PARAMETERS_H