#pragma once
#ifndef APM_OUTCOME_IMPUTATION_H
#define APM_OUTCOME_IMPUTATION_H

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
    None,
    IronsTuck
};

enum class ImputationSolver {
    FixedPoint,
    LSMR
};

struct ImputationOptions {
    double tol = DEFAULT_TOL;
    std::size_t max_iters = DEFAULT_MAX_ITERS;
    AccelMethod method = AccelMethod::IronsTuck;
    std::size_t grand_period = 0;   // 0 disables grand acceleration
    std::size_t grand_k = 4;        // h(X) = f^grand_k(X)
    std::size_t stabilize_after = 0; // 0 disables stabilization
    std::size_t extra_proj = 0;     // extra projections before acceleration
    ImputationSolver solver = ImputationSolver::FixedPoint; // LSMR or FixedPoint
    // LSMR-specific options
    bool lsmr_diagonal_precond = false; // enable diagonal preconditioning
	std::size_t lsmr_num_diag_approx_draws = 0; // 0 = exact; >0 uses Hutchinson with this many draws
	std::size_t lsmr_homotopy_iters = 0; // number of homotopy iterations (0 disables)
    // LSMR solver tolerance/limits (fed into internal LSMROptions)
    double lsmr_atol = 1e-6;      // relative tol on ||A^T r|| (default preserves prior behavior)
    double lsmr_btol = 1e-6;      // relative tol on ||r||
    double lsmr_conlim = 1e+8;    // condition limit
    std::size_t lsmr_max_iters = 0; // 0 = auto (use 2*(T + dim)), >0 overrides
    double lsmr_lambda = 0.0;     // Tikhonov damping; 0 disables
};

// High-level orchestration that returns G, optional a, g_0, and L
FactorModelParameters comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    const std::vector<OutcomeMeanSufficientStatistics>& cohort_outcome_mean_suff_stats = std::vector<OutcomeMeanSufficientStatistics>(),
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    const ImputationOptions& fp = ImputationOptions());

// Overload that accepts FactorModelEstimates and OutcomeMeanSuffStatEstimates, with optional parallelization
FactorModelEstimates comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelEstimates& factor_model_ests,
    const std::vector<OutcomeMeanSuffStatEstimates>& cohort_outcome_mean_suff_stat_ests = std::vector<OutcomeMeanSuffStatEstimates>(),
    std::shared_ptr<const WeightedBootstrap> wb = nullptr,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    const ImputationOptions& fp = ImputationOptions(),
    std::optional<std::size_t> num_threads = std::nullopt);

// Map-of-estimators overload (dispatches per key)
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
