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
#include "cohort_specific_param_structs.h"
#include "utils.h"

namespace apm {

struct VariableSpec {
    enum class Kind { Outcome, Covariate };
    Kind kind;
    std::size_t covariate_index = 0;

    static VariableSpec outcome() { return VariableSpec{Kind::Outcome, 0}; }
    static VariableSpec covariate(std::size_t j) { return VariableSpec{Kind::Covariate, j}; }
};

// High-level orchestration that returns G, optional a, g_0, and L
FactorModelParameters comp_imputation_components(
    const AbstractUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    const std::vector<OutcomeMeanSufficientStatistics>& cohort_outcome_mean_suff_stats = std::vector<OutcomeMeanSufficientStatistics>(),
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = 1e-10,
    std::size_t max_iters = 1000,
    const std::string& fixed_point_method = "irons-tuck");

// Overload that accepts FactorModelEstimates and OutcomeMeanSuffStatEstimates, with optional parallelization
FactorModelEstimates comp_imputation_components(
    const AbstractUnbalancedPanel& panel,
    const FactorModelEstimates& factor_model_ests,
    const std::vector<OutcomeMeanSuffStatEstimates>& cohort_outcome_mean_suff_stat_ests = std::vector<OutcomeMeanSuffStatEstimates>(),
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = 1e-10,
    std::size_t max_iters = 1000,
    const std::string& fixed_point_method = "irons-tuck",
    std::optional<std::size_t> num_threads = std::nullopt);

// Map-of-estimators overload (dispatches per key)
std::unordered_map<std::string, FactorModelEstimates> comp_imputation_components(
    const AbstractUnbalancedPanel& panel,
    const std::unordered_map<std::string, FactorModelEstimates>& factor_model_estimates_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& cohort_outcome_mean_suff_stat_ests = std::vector<OutcomeMeanSuffStatEstimates>(),
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = 1e-10,
    std::size_t max_iters = 1000,
    const std::string& fixed_point_method = "irons-tuck",
    std::optional<std::size_t> num_threads = std::nullopt);

} // namespace apm

#endif // APM_OUTCOME_IMPUTATION_H
