#pragma once
#ifndef APM_OUTCOME_IMPUTATION_HELPERS_H
#define APM_OUTCOME_IMPUTATION_HELPERS_H

//==============================================================================
// Internal helpers for outcome imputation (lambda and g_0 estimation).
//==============================================================================

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
#include "outcome_imputation.h"
// VariableSpec moved here from outcome_imputation.h

namespace apm {

/**
 * @brief Specifies whether to work with outcomes or a specific covariate column.
 */
struct VariableSpec {
    enum class Kind { Outcome, Covariate };
    Kind kind;                      ///< Outcome or Covariate.
    std::size_t covariate_index = 0; ///< Covariate index (used only when kind == Covariate).

    static VariableSpec outcome() { return VariableSpec{Kind::Outcome, 0}; }
    static VariableSpec covariate(std::size_t j) { return VariableSpec{Kind::Covariate, j}; }
};
namespace internal {

/**
 * @brief Compute minimum-norm loadings lambda_i for a cohort given factors G_c and outcomes w_i.
 *
 * @param G_c T_c x r factor submatrix for a cohort.
 * @param w_i Length-T_c outcome (or covariate) vector for a unit.
 * @return Length-r loading vector lambda_i.
 */
arma::vec comp_lambda_i(const arma::mat& G_c, const arma::vec& w_i);

/**
 * @brief Estimate outcome-specific parameters g_0 (optional) and unit-specific lambdas.
 *
 * If g_0_prev is provided, performs one pass to update g_0 (projection onto null(G^T))
 * and, optionally, per-unit lambdas; otherwise only lambdas are produced when
 * store_unit_params is true.
 *
 * @param g_0_prev Previous g_0 (length T) or nullopt to compute lambdas only.
 * @param panel Unbalanced panel providing data assembly.
 * @param var VariableSpec choosing outcome or a specific covariate column.
 * @param factor_model_params Factor model parameters (G, optional g_0/a).
 * @param unit_weights_opt Optional unit weights (length = num_units()).
 * @param effective_ooi_opt Optional override for observed outcome indices.
 * @param covar_coefs_for_residualization Optional covariate coefs to residualize outcomes.
 * @param store_unit_params Whether to return per-unit lambda matrix.
 * @return Pair {optional g_0, optional lambda matrix (units x r)}.
 */
std::pair<std::optional<arma::vec>, std::optional<arma::mat>> get_unit_and_outcome_specific_params(
    std::optional<arma::vec> g_0_prev,
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt,
    bool store_unit_params = false);

/**
 * @brief Convenience wrapper to compute only unit-specific lambdas.
 *
 * @param panel Unbalanced panel providing data assembly.
 * @param var VariableSpec choosing outcome or covariate.
 * @param factor_model_params Factor model parameters (G, optional g_0/a).
 * @param unit_weights_opt Optional unit weights.
 * @param effective_ooi_opt Optional override for observed outcome indices.
 * @param covar_coefs_for_residualization Optional covariate coefs to residualize outcomes.
 * @return Optional lambda matrix (units x r); empty if store_unit_params is false in underlying call.
 */
std::optional<arma::mat> get_unit_specific_params(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt);

/**
 * @brief Compute outcome-specific parameters g_0 given a previous guess.
 *
 * @param g_0_prev Previous g_0 (length T).
 * @param panel Unbalanced panel providing data assembly.
 * @param var VariableSpec choosing outcome or covariate.
 * @param factor_model_params Factor model parameters (G, optional a).
 * @param unit_weights_opt Optional unit weights.
 * @param effective_ooi_opt Optional override for observed outcome indices.
 * @param covar_coefs_for_residualization Optional covariate coefs to residualize outcomes.
 * @return Updated g_0 (length T).
 */
arma::vec get_outcome_specific_params(
    const arma::vec& g_0_prev,
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt);

/**
 * @brief Fixed-point solver for g_0 (optional) and lambdas.
 *
 * @param panel Unbalanced panel providing data assembly.
 * @param var VariableSpec choosing outcome or covariate.
 * @param factor_model_params Factor model parameters (G, optional g_0/a).
 * @param unit_weights_opt Optional unit weights.
 * @param effective_ooi_opt Optional override for observed outcome indices.
 * @param fp Imputation options (tolerances, acceleration).
 * @param covar_coefs_for_residualization Optional covariate coefs to residualize outcomes.
 * @param store_unit_params Whether to return per-unit lambda matrix.
 * @return Pair {optional g_0, optional lambda matrix}.
 */
std::pair<std::optional<arma::vec>, std::optional<arma::mat>> comp_unit_and_outcome_specific_params_fixed_point(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp,
    std::optional<arma::vec> covar_coefs_for_residualization,
    bool store_unit_params);

/**
 * @brief Dispatch between fixed-point and LSMR solvers for g_0 and lambdas.
 *
 * @param panel Unbalanced panel providing data assembly.
 * @param var VariableSpec choosing outcome or covariate.
 * @param factor_model_params Factor model parameters (G, optional g_0/a).
 * @param unit_weights_opt Optional unit weights.
 * @param effective_ooi_opt Optional override for observed outcome indices.
 * @param fp Imputation options (solver choice, tolerances).
 * @param covar_coefs_for_residualization Optional covariate coefs to residualize outcomes.
 * @param store_unit_params Whether to return per-unit lambda matrix.
 * @return Pair {optional g_0, optional lambda matrix}.
 */
std::pair<std::optional<arma::vec>, std::optional<arma::mat>> comp_unit_and_outcome_specific_params(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    const apm::ImputationOptions& fp = apm::ImputationOptions(),
    std::optional<arma::vec> covar_coefs_for_residualization = std::nullopt,
    bool store_unit_params = false);

/**
 * @brief Estimate covariate coefficients (alpha) via residualized regressions.
 *
 * @param panel Unbalanced panel providing data assembly.
 * @param factor_model_params Factor model parameters (G, optional g_0/a).
 * @param g_0_init Optional initial g_0.
 * @param g_0_init_covars Optional per-covariate initial g_0 adjustments.
 * @param unit_weights_opt Optional unit weights.
 * @param effective_ooi_opt Optional override for observed outcome indices.
 * @return Covariate coefficient vector (length q) or empty if q==0.
 */
arma::vec comp_covar_coefs(
    const AbstractUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> g_0_init = std::nullopt,
    std::vector<arma::vec> g_0_init_covars = {},
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt);

/**
 * @brief Compute g_0 only, dispatching to the chosen solver.
 *
 * @param panel Unbalanced panel.
 * @param var VariableSpec (typically outcome()).
 * @param factor_model_params Factor model parameters (G, optional a).
 * @param unit_weights_opt Optional unit weights.
 * @param effective_ooi_opt Optional override for observed outcome indices.
 * @param fp Imputation options (solver choice, tolerances).
 * @return g_0 vector (length T).
 */
arma::vec comp_outcome_specific_params(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt = std::nullopt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    const apm::ImputationOptions& fp = apm::ImputationOptions());

/**
 * @brief Fixed-point computation of g_0 (no LSMR).
 *
 * @param panel Unbalanced panel.
 * @param var VariableSpec (typically outcome()).
 * @param factor_model_params Factor model parameters (G, optional a).
 * @param unit_weights_opt Optional unit weights.
 * @param effective_ooi_opt Optional override for observed outcome indices.
 * @param fp Imputation options (tolerances, acceleration).
 * @return g_0 vector (length T).
 */
arma::vec comp_outcome_specific_params_fixed_point(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp);

/**
 * @brief LSMR computation of g_0.
 *
 * @param panel Unbalanced panel.
 * @param var VariableSpec (typically outcome()).
 * @param factor_model_params Factor model parameters (G, optional a).
 * @param unit_weights_opt Optional unit weights.
 * @param effective_ooi_opt Optional override for observed outcome indices.
 * @param fp Imputation options (tolerances, solver limits).
 * @return g_0 vector (length T).
 */
arma::vec comp_outcome_specific_params_lsmr(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp);

} // namespace internal

} // namespace apm

#endif // APM_OUTCOME_IMPUTATION_HELPERS_H


