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
#include "outcome_imputation.h" // for DEFAULT_* constants
// VariableSpec moved here from outcome_imputation.h

namespace apm {

struct VariableSpec {
    enum class Kind { Outcome, Covariate };
    Kind kind;
    std::size_t covariate_index = 0;

    static VariableSpec outcome() { return VariableSpec{Kind::Outcome, 0}; }
    static VariableSpec covariate(std::size_t j) { return VariableSpec{Kind::Covariate, j}; }
};
namespace internal {

// Centralized lambda computation
arma::vec comp_lambda_i(const arma::mat& G_c, const arma::vec& w_i);

// Pair of optional outputs: optional g_0 and optional lambda
std::pair<std::optional<arma::vec>, std::optional<arma::mat>> comp_unit_and_outcome_specific_params(
    std::optional<arma::vec> g_0_prev,
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt,
    bool store_unit_params = false);

// Convenience wrapper for lambda-only computation
std::optional<arma::mat> comp_unit_specific_params(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt);

arma::vec comp_outcome_specific_params(
    const arma::vec& g_0_prev,
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt);

arma::vec comp_outcome_specific_params_fixed_point(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = DEFAULT_TOL,
    std::size_t max_iters = DEFAULT_MAX_ITERS,
    const std::string& fixed_point_method = DEFAULT_FP_METHOD);

std::pair<std::optional<arma::vec>, std::optional<arma::mat>> comp_unit_and_outcome_specific_params_fixed_point(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = DEFAULT_TOL,
    std::size_t max_iters = DEFAULT_MAX_ITERS,
    const std::string& fixed_point_method = DEFAULT_FP_METHOD,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt,
    bool store_unit_params = false);

// Covariate coefficient estimation
arma::vec comp_covar_coefs(
    const AbstractUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> g_0_init = std::nullopt,
    std::vector<arma::vec> g_0_init_covars = {},
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt);

} // namespace internal
} // namespace apm

#endif // APM_OUTCOME_IMPUTATION_HELPERS_H


