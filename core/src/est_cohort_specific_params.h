#pragma once
#ifndef APM_EST_COHORT_SPECIFIC_PARAMS_H
#define APM_EST_COHORT_SPECIFIC_PARAMS_H

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.h"                    // ObservedOutcomeIndices, CohortOutcomeMask
#include "cohort_specific_param_structs.h"

namespace apm {

class WeightedBootstrap; // forward declaration

struct EstimatorSpecification {
    std::string factor_model_estimator; // e.g., "principal_components"
    bool include_outcome_fes;
    std::size_t r;
    std::string cohort_weighting = "by_size"; // "by_size" (default) or "equal"
};

struct CohortSpecificEstimates {
    std::unordered_map<std::string, std::vector<FactorModelEstimates>> cohort_specific_factor_ests;
    std::vector<OutcomeMeanSuffStatEstimates> cohort_outcome_mean_ests;
    std::unordered_map<std::string, CohortWeightEstimates> cohort_weights;
    // Optional: only non-empty when auxiliary data to be averaged are defined
    std::vector<CohortAuxiliaryDataMeanEstimates> cohort_auxiliary_means;
    // Optional: present only when masking is applied
    std::optional<ObservedOutcomeIndices> masked_observed_outcome_indices;
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_cohort_outcome_means;
    std::optional<CohortOutcomeMask> cohort_outcome_mask;
};

// New panel-based entry point. Indices inside panel may be 0- or 1-based; panel handles it.
class InMemoryUnbalancedPanel; // fwd
CohortSpecificEstimates estimate_cohort_specific_params_from_internal_panel_rep(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
    std::optional<std::size_t> num_threads = std::nullopt,
    const CohortOutcomeMask& cohort_outcomes_to_mask = CohortOutcomeMask()
);

} // namespace apm

#endif // APM_EST_COHORT_SPECIFIC_PARAMS_H


