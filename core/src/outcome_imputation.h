#pragma once
#ifndef APM_OUTCOME_IMPUTATION_H
#define APM_OUTCOME_IMPUTATION_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <optional>

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

// Note: more outcome-imputation routines are available in outcome_imputation_helpers.h for tests/internal use only.

} // namespace apm

#endif // APM_OUTCOME_IMPUTATION_H
