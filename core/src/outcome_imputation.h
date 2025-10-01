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
inline constexpr const char* DEFAULT_FP_METHOD = "irons-tuck";

// High-level orchestration that returns G, optional a, g_0, and L
FactorModelParameters comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    const std::vector<OutcomeMeanSufficientStatistics>& cohort_outcome_mean_suff_stats = std::vector<OutcomeMeanSufficientStatistics>(),
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = DEFAULT_TOL,
    std::size_t max_iters = DEFAULT_MAX_ITERS,
    const std::string& fixed_point_method = DEFAULT_FP_METHOD);

// Overload that accepts FactorModelEstimates and OutcomeMeanSuffStatEstimates, with optional parallelization
FactorModelEstimates comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelEstimates& factor_model_ests,
    const std::vector<OutcomeMeanSuffStatEstimates>& cohort_outcome_mean_suff_stat_ests = std::vector<OutcomeMeanSuffStatEstimates>(),
    std::shared_ptr<const WeightedBootstrap> wb = nullptr,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = DEFAULT_TOL,
    std::size_t max_iters = DEFAULT_MAX_ITERS,
    const std::string& fixed_point_method = DEFAULT_FP_METHOD,
    std::optional<std::size_t> num_threads = std::nullopt);

// Map-of-estimators overload (dispatches per key)
std::unordered_map<std::string, FactorModelEstimates> comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, FactorModelEstimates>& factor_model_estimates_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& cohort_outcome_mean_suff_stat_ests = std::vector<OutcomeMeanSuffStatEstimates>(),
    std::shared_ptr<const WeightedBootstrap> wb = nullptr,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = DEFAULT_TOL,
    std::size_t max_iters = DEFAULT_MAX_ITERS,
    const std::string& fixed_point_method = DEFAULT_FP_METHOD,
    std::optional<std::size_t> num_threads = std::nullopt);

} // namespace apm

#endif // APM_OUTCOME_IMPUTATION_H
