#ifndef APM_FACTOR_MODEL_PARAMETERS_H
#define APM_FACTOR_MODEL_PARAMETERS_H

//==============================================================================
// Cohort-specific parameter containers for the APM factor model.
//
// These lightweight structs transport estimated factors, fixed effects,
// covariate coefficients, and sufficient statistics across the estimation
// pipeline. They are intentionally header-only to keep the core API obvious to
// callers (e.g., language bindings) while the implementation lives in .cpp
// files. Unless noted otherwise, dimensions follow the notation:
//   - T: total outcomes, 
//   - T_c: outcomes observed in a cohort
//   - r: factor rank, 
//   - q: number of covariates, 
//   - d: auxiliary columns
//==============================================================================

#include <optional>
#include <limits>
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
    arma::mat G;                           ///< T_c x r factor matrix.
    std::optional<arma::vec> g_0;          ///< Optional length-T_c outcome fixed effects.
    std::optional<arma::vec> a;            ///< Optional length-q covariate coefficients.
    std::optional<arma::mat> L;            ///< Optional N x r matrix of unit factor scores.

    // Constructors
    FactorModelParameters() = default;

    /**
     * @brief Construct parameters with optional fixed effects and covariates.
     *
     * @param G_in T_c x r factor matrix.
     * @param g0_in Optional outcome fixed effects (length T_c).
     * @param a_in Optional covariate coefficients (length q).
     *
     * @throws std::invalid_argument if g_0 length does not equal rows of G.
     */
    FactorModelParameters(arma::mat G_in,
                          std::optional<arma::vec> g0_in = std::nullopt,
                          std::optional<arma::vec> a_in = std::nullopt)
        : G(std::move(G_in)), g_0(std::move(g0_in)), a(std::move(a_in)), L(std::nullopt) {
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
    FactorModelParameters parameter_estimates;                 ///< Point estimates.
    std::vector<FactorModelParameters> bootstrap_replicates;   ///< Bootstrap replicates (length B or empty).

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
    /**
     * @brief Returns the number of bootstrap replicates (0 if none).
     */
    std::size_t n_bootstrap_replicates() const noexcept { return bootstrap_replicates.size(); }
};

/**
 * @brief Container for sufficient statistics for identifying and estimating outcome means.
 *
 * Fields:
 *  - observed_outcome_means: length T_c vector of observed outcome means.
 *  - covar_means: optional T x q matrix of covariate means (per outcome).
 */
struct OutcomeMeanSufficientStatistics {
    arma::vec observed_outcome_means;                   ///< Length T_c observed outcome means.
    std::optional<arma::mat> covar_means;               ///< Optional T x q covariate means.
    double cohort_pop_share;                            ///< Share of all rows belonging to this cohort.

    // Constructors
    OutcomeMeanSufficientStatistics() = default;

    OutcomeMeanSufficientStatistics(arma::vec observed_means_in,
                                    std::optional<arma::mat> covar_means_in = std::nullopt)
        : observed_outcome_means(std::move(observed_means_in)),
          covar_means(std::move(covar_means_in)),
          cohort_pop_share(std::numeric_limits<double>::quiet_NaN()) {}

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
struct OutcomeMeanSuffStatEstimates {
    OutcomeMeanSufficientStatistics suff_stat_estimates;                  ///< Point sufficient stats.
    std::vector<OutcomeMeanSufficientStatistics> bootstrap_replicates;   ///< Bootstrap sufficient stats (length B or empty).

    OutcomeMeanSuffStatEstimates(OutcomeMeanSufficientStatistics stats,
                                 std::vector<OutcomeMeanSufficientStatistics> boot_reps = {})
        : suff_stat_estimates(std::move(stats)),
          bootstrap_replicates(std::move(boot_reps)) {}

    /**
     * @brief Indicates whether bootstrap replicates are present (non-empty).
     * @return true if replicates exist; false otherwise.
     */
    bool has_bootstrap_replicates() const noexcept { return !bootstrap_replicates.empty(); }
    /**
     * @brief Returns the number of bootstrap replicates (0 if none).
     */
    std::size_t n_bootstrap_replicates() const noexcept { return bootstrap_replicates.size(); }
};

// No heavy constructors or factories; use OutcomeSuffStatEstimator to build these.

/**
 * @brief Container for cohort weighting estimates (point and optional bootstrap replicates).
 *
 * Fields:
 *  - cohort_weights: length C vector of cohort weights (point estimate).
 *  - bootstrap_cohort_weights: optional vector of length-B replicate weight vectors (each length C).
 */
struct CohortWeightEstimates {
    arma::vec cohort_weights;                    ///< Length-C cohort weights (point estimate).
    std::vector<arma::vec> bootstrap_cohort_weights; ///< Optional bootstrap weights (length B, each length C).

    bool has_bootstrap_replicates() const noexcept { return !bootstrap_cohort_weights.empty(); }
    std::size_t n_bootstrap_replicates() const noexcept { return bootstrap_cohort_weights.size(); }
};

/**
 * @brief Cohort-level auxiliary data means.
 *
 * Fields:
 *  - auxiliary_means:  T x d matrix of means over auxiliary columns
 */
struct CohortAuxiliaryDataMeans {
    /**
     * @brief Row-outcome by column-auxiliary matrix of cohort means (T x d).
     *
     * Rows correspond to outcomes in the panel-wide outcome index; columns
     * correspond to auxiliary data series supplied alongside the panel.
     */
    arma::mat auxiliary_means; ///< T x d matrix of auxiliary means.

    CohortAuxiliaryDataMeans() = default;

    explicit CohortAuxiliaryDataMeans(arma::mat aux_means_in)
        : auxiliary_means(std::move(aux_means_in)) {}

    std::size_t T() const noexcept { return static_cast<std::size_t>(auxiliary_means.n_rows); }
    std::size_t d() const noexcept { return static_cast<std::size_t>(auxiliary_means.n_cols); }
};

/**
 * @brief Aggregated cohort auxiliary data mean estimates with optional bootstrap replicates.
 */
struct CohortAuxiliaryDataMeanEstimates {
    CohortAuxiliaryDataMeans estimates;                          ///< Point estimate.
    std::vector<CohortAuxiliaryDataMeans> bootstrap_replicates;  ///< Bootstrap replicates (length B or empty).

    CohortAuxiliaryDataMeanEstimates() = default;

    CohortAuxiliaryDataMeanEstimates(CohortAuxiliaryDataMeans est,
                                     std::vector<CohortAuxiliaryDataMeans> boot_reps = {})
        : estimates(std::move(est)), bootstrap_replicates(std::move(boot_reps)) {}

    bool has_bootstrap_replicates() const noexcept { return !bootstrap_replicates.empty(); }
    std::size_t n_bootstrap_replicates() const noexcept { return bootstrap_replicates.size(); }
};

} // namespace apm

#endif // APM_FACTOR_MODEL_PARAMETERS_H