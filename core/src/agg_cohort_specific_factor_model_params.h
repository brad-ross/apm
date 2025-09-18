#ifndef APM_AGG_COHORT_SPECIFIC_FACTOR_MODEL_PARAMS_H
#define APM_AGG_COHORT_SPECIFIC_FACTOR_MODEL_PARAMS_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <vector>
#include <unordered_map>

#include "cohort_specific_param_structs.h"
#include "utils.h"

namespace apm {

/**
 * @brief Aggregate cohort-specific parameter estimates including bootstrap replicates.
 *
 * Aggregates point estimates and each bootstrap replicate independently using
 * optional weights (and optional per-bootstrap weights), returning a
 * FactorModelEstimates containing aggregated point estimates and aggregated
 * bootstrap replicates.
 */
FactorModelEstimates aggregate_cohort_specific_factor_model_params(
    const std::vector<FactorModelEstimates>& cohort_specific_factor_model_param_ests,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const CohortWeightEstimates& cohort_weight_estimates = CohortWeightEstimates());

/**
 * @brief Aggregate factor model estimates per estimator specification.
 *
 * Validates keys across maps (symmetric presence) and aggregates per spec by
 * dispatching to the vector-based overload of
 * `aggregate_cohort_specific_factor_model_params`.
 */
std::unordered_map<std::string, FactorModelEstimates> aggregate_cohort_specific_factor_model_params(
    const std::unordered_map<std::string, std::vector<FactorModelEstimates>>& cohort_specific_factor_ests,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::unordered_map<std::string, CohortWeightEstimates>& cohort_weights);

} // namespace apm

#endif // APM_AGG_COHORT_SPECIFIC_FACTOR_MODEL_PARAMS_H
