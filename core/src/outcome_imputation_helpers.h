#pragma once
#ifndef APM_OUTCOME_IMPUTATION_HELPERS_H
#define APM_OUTCOME_IMPUTATION_HELPERS_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <optional>
#include <utility>
#include <string>

#include "panels/InMemoryUnbalancedPanel.h"
#include "cohort_specific_param_structs.h"
#include "outcome_imputation.h" // for VariableSpec

namespace apm {
namespace internal {

std::pair<arma::vec, std::optional<arma::mat>> comp_unit_and_outcome_specific_params(
    const arma::vec& g_0_prev,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    bool store_unit_params = false);

arma::vec comp_outcome_specific_params(
    const arma::vec& g_0_prev,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt);

arma::vec comp_outcome_specific_params_fixed_point(
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = 1e-10,
    std::size_t max_iters = 1000,
    const std::string& fixed_point_method = "");

} // namespace internal
} // namespace apm

#endif // APM_OUTCOME_IMPUTATION_HELPERS_H


