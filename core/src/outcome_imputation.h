#pragma once
#ifndef APM_OUTCOME_IMPUTATION_H
#define APM_OUTCOME_IMPUTATION_H

//==============================================================================
// High-level orchestration of outcome imputation and parameter completion.
//==============================================================================

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <optional>
#include <vector>

#include "panels/AbstractUnbalancedPanel.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "cohort_specific_param_structs.h"
#include "utils.h"
#include "bootstrap.h"

namespace apm {

inline constexpr double DEFAULT_TOL = 1e-12;
inline constexpr std::size_t DEFAULT_MAX_ITERS = std::numeric_limits<std::size_t>::max();

enum class AccelMethod {
    None,      ///< No acceleration beyond fixed-point updates
    IronsTuck  ///< Irons-Tuck averaging acceleration
};

enum class ImputationSolver {
    FixedPoint, ///< Fixed-point iteration in factor space
    LSMR        ///< Matrix-free LSMR linear solver
};

/**
 * @brief Options controlling fixed-point/LSMR solvers and accelerations.
 */
struct ImputationOptions {
    double tol = DEFAULT_TOL;                  ///< Convergence tolerance.
    std::size_t max_iters = DEFAULT_MAX_ITERS; ///< Maximum iterations (0 = none).
    AccelMethod method = AccelMethod::IronsTuck; ///< Acceleration method.
    std::size_t grand_period = 0;   ///< Grand acceleration period (0 disables).
    std::size_t grand_k = 4;        ///< h(X) = f^grand_k(X) exponent for grand acceleration.
    std::size_t stabilize_after = 0; ///< Stabilization start iteration (0 disables).
    std::size_t extra_proj = 0;     ///< Extra projections before acceleration.
    ImputationSolver solver = ImputationSolver::FixedPoint; ///< Solver choice (FixedPoint or LSMR).
    // LSMR-specific options
    bool lsmr_diagonal_precond = false; ///< Enable diagonal preconditioning.
	std::size_t lsmr_num_diag_approx_draws = 0; ///< Hutchinson draws for diag precond (0 = exact).
	std::size_t lsmr_homotopy_iters = 0; ///< Homotopy stages (0 disables).
    // LSMR solver tolerance/limits (fed into internal LSMROptions)
    double lsmr_atol = 1e-6;      ///< Relative tol on ||A^T r||.
    double lsmr_btol = 1e-6;      ///< Relative tol on ||r||.
    double lsmr_conlim = 1e+8;    ///< Condition limit.
    std::size_t lsmr_max_iters = 0; ///< Max iterations (0 = auto, 2*(T+dim)).
    double lsmr_lambda = 0.0;     ///< Tikhonov damping; 0 disables.
};

// High-level orchestration that returns G, optional a, g_0, and L
/**
 * @brief Compute imputation components from point estimates and an observed panel.
 *
 * Runs unit/outcome-specific parameter estimation (lambda, optional g_0) and
 * returns completed FactorModelParameters with optional covariates and loadings.
 *
 * @param panel Unbalanced panel of individual-level data.
 * @param factor_model_params Point factor model parameters (G, optional g_0, a).
 * @param cohort_outcome_mean_suff_stats Optional cohort outcome sufficient statistics (per cohort).
 * @param unit_weights_opt Optional unit weights for estimation.
 * @param effective_ooi_opt Optional observed outcome indices override.
 * @param fp ImputationOptions controlling solver/acceleration.
 * @return Completed factor model parameters including inferred L (and updated g_0/a if applicable).
 */
FactorModelParameters comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    const std::vector<OutcomeMeanSufficientStatistics>& cohort_outcome_mean_suff_stats = std::vector<OutcomeMeanSufficientStatistics>(),
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    const ImputationOptions& fp = ImputationOptions());

// Overload that accepts FactorModelEstimates and OutcomeMeanSuffStatEstimates, with optional parallelization
/**
 * @brief Compute imputation components with bootstrap-aware inputs.
 *
 * @param panel Unbalanced panel of individual-level data.
 * @param factor_model_ests Point estimates plus bootstrap replicates.
 * @param cohort_outcome_mean_suff_stat_ests Cohort sufficient stats (point + bootstrap).
 * @param wb Optional weighted bootstrap controller.
 * @param effective_ooi_opt Optional observed outcome indices override.
 * @param fp ImputationOptions controlling solver/acceleration.
 * @param num_threads Optional thread cap.
 * @return FactorModelEstimates with completed parameters (point + bootstrap).
 */
FactorModelEstimates comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelEstimates& factor_model_ests,
    const std::vector<OutcomeMeanSuffStatEstimates>& cohort_outcome_mean_suff_stat_ests = std::vector<OutcomeMeanSuffStatEstimates>(),
    std::shared_ptr<const WeightedBootstrap> wb = nullptr,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    const ImputationOptions& fp = ImputationOptions(),
    std::optional<std::size_t> num_threads = std::nullopt);

// Map-of-estimators overload (dispatches per key)
/**
 * @brief Compute imputation components for multiple estimator specifications.
 *
 * @param panel Unbalanced panel of individual-level data.
 * @param factor_model_estimates_map Map spec -> FactorModelEstimates (point + bootstrap).
 * @param cohort_outcome_mean_suff_stat_ests Cohort sufficient stats (point + bootstrap).
 * @param wb Optional weighted bootstrap controller.
 * @param effective_ooi_opt Optional observed outcome indices override.
 * @param fp ImputationOptions controlling solver/acceleration.
 * @param num_threads Optional thread cap.
 * @return Map spec -> completed FactorModelEstimates.
 */
std::unordered_map<std::string, FactorModelEstimates> comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, FactorModelEstimates>& factor_model_estimates_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& cohort_outcome_mean_suff_stat_ests = std::vector<OutcomeMeanSuffStatEstimates>(),
    std::shared_ptr<const WeightedBootstrap> wb = nullptr,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    const ImputationOptions& fp = ImputationOptions(),
    std::optional<std::size_t> num_threads = std::nullopt);

} // namespace apm

#endif // APM_OUTCOME_IMPUTATION_H
