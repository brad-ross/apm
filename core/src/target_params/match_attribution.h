#pragma once
#ifndef APM_MATCH_ATTRIBUTION_H
#define APM_MATCH_ATTRIBUTION_H

#include "est_target_params.h"
#include "../utils.h"

namespace apm {

TargetFn get_fgw_bipartite_match_outcome_diff_params_fn(
    const arma::uvec& outcome_indices_1,
    const arma::uvec& outcome_indices_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt);

TargetFn get_fgw_bipartite_match_outcome_diff_params_fn(
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices);

TargetParameterEstimates est_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices_1,
    const arma::uvec& outcome_indices_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

TargetParameterEstimates est_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

std::unordered_map<std::string, TargetParameterEstimates> est_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices_1,
    const arma::uvec& outcome_indices_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

std::unordered_map<std::string, TargetParameterEstimates> est_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

TargetFn get_avg_fgw_bipartite_match_outcome_diff_params_fn(
    const std::vector<arma::uvec>& outcome_groupings,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt);

TargetFn get_avg_fgw_bipartite_match_outcome_diff_params_fn(
    const arma::uvec& outcome_indices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt);

TargetParameterEstimates est_avg_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<arma::uvec>& outcome_groupings,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

TargetParameterEstimates est_avg_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

std::unordered_map<std::string, TargetParameterEstimates> est_avg_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<arma::uvec>& outcome_groupings,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

std::unordered_map<std::string, TargetParameterEstimates> est_avg_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

} // namespace apm

#endif // APM_MATCH_ATTRIBUTION_H