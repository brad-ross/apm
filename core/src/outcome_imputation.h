#pragma once
#ifndef APM_OUTCOME_IMPUTATION_H
#define APM_OUTCOME_IMPUTATION_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <optional>
#include <utility>

#include "panels/InMemoryUnbalancedPanel.h"
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

arma::mat comp_unit_specific_params(
    const arma::vec& g_0,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt
);

// Compute outcome-specific fixed effects by averaging residuals given unit-specific lambda
arma::vec comp_outcome_specific_params(
    const arma::mat& lambda,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt
);

// Overload that nests lambda computation inside using previous g_0
arma::vec comp_outcome_specific_params(
    const arma::vec& g_0_prev,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt
);

// Iterate outcome-specific parameter computation (nested lambda) to a fixed point
arma::vec comp_unit_and_outcome_specific_params_vanilla_fixed_point(
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt = std::nullopt,
    double tol = 1e-10,
    std::size_t max_iters = 1000
);

} // namespace apm

#endif // APM_OUTCOME_IMPUTATION_H
