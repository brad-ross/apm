#pragma once
#ifndef APM_EST_COHORT_SPECIFIC_PARAMS_H
#define APM_EST_COHORT_SPECIFIC_PARAMS_H

//==============================================================================
// Entry points for estimating cohort-specific factor model parameters.
//==============================================================================

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

/**
 * @brief Estimator configuration for a cohort-specific factor model.
 */
struct EstimatorSpecification {
    std::string factor_model_estimator; ///< Estimator key, e.g., "principal_components".
    bool include_outcome_fes;           ///< Whether to include outcome fixed effects.
    std::size_t r;                      ///< Factor rank.
    std::string cohort_weighting = "equal"; ///< Cohort weighting scheme: "equal" (default) or "by_size".
};

/**
 * @brief Outputs from cohort-specific estimation.
 */
struct CohortSpecificEstimates {
    std::unordered_map<std::string, std::vector<FactorModelEstimates>> cohort_specific_factor_ests; ///< Per-spec factor estimates by cohort.
    std::vector<OutcomeMeanSuffStatEstimates> cohort_outcome_mean_ests; ///< Outcome mean sufficient stats per cohort.
    std::unordered_map<std::string, CohortWeightEstimates> cohort_weights; ///< Cohort weights (point + bootstrap) per spec.
    // Optional: only non-empty when auxiliary data to be averaged are defined
    std::vector<CohortAuxiliaryDataMeanEstimates> cohort_auxiliary_means; ///< Optional auxiliary means per cohort.
    // Optional: present only when masking is applied
    std::optional<ObservedOutcomeIndices> masked_observed_outcome_indices; ///< Observed outcomes after masking (if applied).
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_cohort_outcome_means; ///< Masked cohort outcome means (if applied).
    std::optional<CohortOutcomeMask> cohort_outcome_mask; ///< Mask specification applied (if any).
};

// New panel-based entry point. Indices inside panel may be 0- or 1-based; panel handles it.
class InMemoryUnbalancedPanel; // fwd
/**
 * @brief Estimate cohort-specific factor models and related statistics from a panel.
 *
 * Dispatches per-estimator-spec factor model estimators, computes outcome mean
 * sufficient statistics, and (optionally) auxiliary means and masking variants.
 *
 * @param panel In-memory unbalanced panel (handles 0/1-based indices internally).
 * @param est_specs Map from estimator name -> specification (rank, estimator kind, etc.).
 * @param bootstrap Optional weighted bootstrap controller shared across estimators.
 * @param num_threads Optional thread cap for parallel sections.
 * @param cohort_outcomes_to_mask Outcomes to drop per cohort (e.g., for holdout tasks).
 * @return CohortSpecificEstimates containing per-spec factor estimates, outcome sufficient
 *         statistics, cohort weights, and optional auxiliary means / masks.
 */
CohortSpecificEstimates estimate_cohort_specific_params_from_internal_panel_rep(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
    std::optional<std::size_t> num_threads = std::nullopt,
    const CohortOutcomeMask& cohort_outcomes_to_mask = CohortOutcomeMask()
);

} // namespace apm

#endif // APM_EST_COHORT_SPECIFIC_PARAMS_H


