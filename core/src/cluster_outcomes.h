#pragma once
#ifndef APM_CLUSTER_OUTCOMES_H
#define APM_CLUSTER_OUTCOMES_H

#include <cstddef>
#include <utility>
#include <unordered_map>

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include "utils.h" // ObservedOutcomeIndices

namespace apm {

class InMemoryUnbalancedPanel; // forward declaration

/**
 * Summary metrics for a given k-means outcome clustering evaluated via O^3.
 */
struct OutcomeClusteringSummary {
    std::size_t largest_super_cohort_size;            // total units in largest final super cohort
    double       largest_super_cohort_share;          // share of panel units
    std::size_t  min_cohort_size_in_largest_super;    // min combined cohort size within that super cohort
    std::size_t  num_o3_iterations;                   // number of O^3 iterations (length of super_cohort_iterates)
};

// For each k in [min_k, max_k], cluster outcome CDF rows, combine cohorts,
// run O^3 at rank max_model_rank, and report mapping + summary.
/**
 * Cluster outcome CDF rows for k in [min_k, max_k], combine cohorts accordingly,
 * run O^3 with max_model_rank, and return per-k outcome->cluster mappings (0-based)
 * alongside O^3-based summary metrics.
 */
std::pair<std::vector<arma::uvec>, std::vector<OutcomeClusteringSummary>>
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t max_model_rank,
    std::size_t min_k,
    std::size_t max_k);

// Overload: single k (min_k == max_k == k)
/**
 * Overload for a single k: equivalent to the range version with min_k = max_k = k.
 */
std::pair<arma::uvec, OutcomeClusteringSummary>
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t max_model_rank,
    std::size_t k);

} // namespace apm

#endif // APM_CLUSTER_OUTCOMES_H