#pragma once
#ifndef APM_MATCH_ATTRIBUTION_H
#define APM_MATCH_ATTRIBUTION_H

//==============================================================================
// Target functions and estimators for bipartite matching attribution (FGW).
//==============================================================================

#include "est_target_params.h"
#include "../utils.h"

namespace apm {

/**
 * @brief Build a target function for FGW bipartite match outcome differences (group version).
 *
 * @param outcome_indices_1 Outcome indices for group 1.
 * @param outcome_indices_2 Outcome indices for group 2.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param outcome_weights Optional weights per outcome (length T); defaults to 1.
 * @return TargetFn mapping cohort outcome means/stats to a two-parameter difference vector.
 */
TargetFn get_fgw_bipartite_match_outcome_diff_params_fn(
    const arma::uvec& outcome_indices_1,
    const arma::uvec& outcome_indices_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt);

/**
 * @brief Build a target function for FGW bipartite match outcome difference (scalar version).
 *
 * @param outcome_idx_1 Single outcome index in group 1.
 * @param outcome_idx_2 Single outcome index in group 2.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @return TargetFn mapping cohort outcome means/stats to a two-parameter difference vector.
 */
TargetFn get_fgw_bipartite_match_outcome_diff_params_fn(
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices);

/**
 * @brief Estimate FGW bipartite match outcome differences for given outcome groups.
 *
 * @param ome Outcome means (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient stats (point + optional bootstrap).
 * @param outcome_indices_1 Outcome indices for group 1.
 * @param outcome_indices_2 Outcome indices for group 2.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param outcome_weights Optional weights per outcome (length T); defaults to 1.
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return TargetParameterEstimates (point + bootstrap).
 */
TargetParameterEstimates est_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices_1,
    const arma::uvec& outcome_indices_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Estimate FGW bipartite match outcome differences for two single outcomes.
 *
 * @param ome Outcome means (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient stats (point + optional bootstrap).
 * @param outcome_idx_1 Single outcome index for group 1.
 * @param outcome_idx_2 Single outcome index for group 2.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return TargetParameterEstimates (point + bootstrap).
 */
TargetParameterEstimates est_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Estimate FGW bipartite match outcome differences for multiple estimator specs.
 *
 * @param ome_map Map spec -> outcome means (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient stats (point + optional bootstrap).
 * @param outcome_indices_1 Outcome indices for group 1.
 * @param outcome_indices_2 Outcome indices for group 2.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param outcome_weights Optional weights per outcome (length T); defaults to 1.
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return Map spec -> TargetParameterEstimates.
 */
std::unordered_map<std::string, TargetParameterEstimates> est_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices_1,
    const arma::uvec& outcome_indices_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Estimate FGW bipartite match outcome differences for two single outcomes across specs.
 *
 * @param ome_map Map spec -> outcome means (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient stats (point + optional bootstrap).
 * @param outcome_idx_1 Single outcome index for group 1.
 * @param outcome_idx_2 Single outcome index for group 2.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return Map spec -> TargetParameterEstimates.
 */
std::unordered_map<std::string, TargetParameterEstimates> est_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Build a target function for average FGW bipartite match differences over pairs of outcome groups.
 *
 * @param outcome_groupings Vector of outcome index groups.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param outcome_weights Optional weights per outcome (length T); defaults to 1.
 * @return TargetFn mapping cohort outcome means/stats to averaged two-parameter differences across pairs of groups.
 */
TargetFn get_avg_fgw_bipartite_match_outcome_diff_params_fn(
    const std::vector<arma::uvec>& outcome_groupings,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt);

/**
 * @brief Build a target function for average FGW bipartite match differences over individual outcomes.
 *
 * @param outcome_indices Outcome indices to consider (length >= 2).
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param outcome_weights Optional weights per outcome (length T); defaults to 1.
 * @return TargetFn mapping cohort outcome means/stats to averaged two-parameter differences over all pairs.
 */
TargetFn get_avg_fgw_bipartite_match_outcome_diff_params_fn(
    const arma::uvec& outcome_indices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt);

/**
 * @brief Estimate average FGW bipartite match differences over pairs of outcome groups.
 *
 * @param ome Outcome means (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient stats (point + optional bootstrap).
 * @param outcome_groupings Vector of outcome index groups.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param outcome_weights Optional weights per outcome (length T); defaults to 1.
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return TargetParameterEstimates (point + bootstrap).
 */
TargetParameterEstimates est_avg_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<arma::uvec>& outcome_groupings,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Estimate average FGW bipartite match differences over all pairs from a single outcome set.
 *
 * @param ome Outcome means (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient stats (point + optional bootstrap).
 * @param outcome_indices Outcome indices to consider (pairs formed across all elements).
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return TargetParameterEstimates (point + bootstrap).
 */
TargetParameterEstimates est_avg_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Estimate average FGW bipartite match differences for multiple estimator specs.
 *
 * @param ome_map Map spec -> outcome means (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient stats (point + optional bootstrap).
 * @param outcome_groupings Vector of outcome index groups.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param outcome_weights Optional weights per outcome (length T); defaults to 1.
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return Map spec -> TargetParameterEstimates.
 */
std::unordered_map<std::string, TargetParameterEstimates> est_avg_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<arma::uvec>& outcome_groupings,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Estimate average FGW bipartite match differences over all pairs from a single outcome set across specs.
 *
 * @param ome_map Map spec -> outcome means (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient stats (point + optional bootstrap).
 * @param outcome_indices Outcome indices to consider (pairs formed across all elements).
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return Map spec -> TargetParameterEstimates.
 */
std::unordered_map<std::string, TargetParameterEstimates> est_avg_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads = std::nullopt);

} // namespace apm

#endif // APM_MATCH_ATTRIBUTION_H