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

#include "panels/AbstractUnbalancedPanel.h"
#include "cohort_specific_param_structs.h"
#include "outcome_imputation.h" // for VariableSpec

namespace apm {
namespace internal {

// Centralized lambda computation
arma::vec comp_lambda_i(const arma::mat& G_c, const arma::vec& w_i);

// Pair of optional outputs: optional g_0 and optional lambda
std::pair<std::optional<arma::vec>, std::optional<arma::mat>> comp_unit_and_outcome_specific_params(
    std::optional<arma::vec> g_0_prev,
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt,
    bool store_unit_params = false);

// Convenience wrapper for lambda-only computation
std::optional<arma::mat> comp_unit_specific_params(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt);

arma::vec comp_outcome_specific_params(
    const arma::vec& g_0_prev,
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt);

arma::vec comp_outcome_specific_params_fixed_point(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = 1e-10,
    std::size_t max_iters = 1000,
    const std::string& fixed_point_method = "vanilla");

std::pair<std::optional<arma::vec>, std::optional<arma::mat>> comp_unit_and_outcome_specific_params_fixed_point(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = 1e-10,
    std::size_t max_iters = 1000,
    const std::string& fixed_point_method = "vanilla",
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt,
    bool store_unit_params = false);

// Covariate coefficient estimation
arma::vec comp_covar_coefs(
    const AbstractUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> g_0_init = std::nullopt,
    std::vector<arma::vec> g_0_init_covars = {},
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt);

} // namespace internal
} // namespace apm

#endif // APM_OUTCOME_IMPUTATION_HELPERS_H


