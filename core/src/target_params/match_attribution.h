#pragma once
#ifndef APM_MATCH_ATTRIBUTION_H
#define APM_MATCH_ATTRIBUTION_H

#include "est_target_params.h"
#include "../utils.h"

namespace apm {

TargetFn get_fgw_bipartite_match_outcome_diff_params_fn(
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices);

TargetParameterEstimates est_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<CohortAuxiliaryDataMeanEstimates>& eta_by_cohort,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

std::unordered_map<std::string, TargetParameterEstimates> est_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::unordered_map<std::string, std::vector<OutcomeMeanSuffStatEstimates>>& stats_map,
    const std::unordered_map<std::string, std::vector<CohortAuxiliaryDataMeanEstimates>>& eta_map,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

} // namespace apm

#endif // APM_MATCH_ATTRIBUTION_H