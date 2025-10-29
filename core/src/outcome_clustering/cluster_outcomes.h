#pragma once
#ifndef APM_CLUSTER_OUTCOMES_H
#define APM_CLUSTER_OUTCOMES_H

#include <cstddef>
#include <utility>
#include <optional>

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include "utils.h" // ObservedOutcomeIndices

namespace apm {

class InMemoryUnbalancedPanel; // forward declaration

// Mapping-only APIs
// Compute outcome cluster assignments for k in [min_k, max_k].
std::vector<arma::uvec>
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t min_k,
    std::size_t max_k,
    std::optional<std::size_t> n_inits = std::nullopt,
    std::optional<uint64_t> seed = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

// Single-k mapping
arma::uvec
comp_outcome_clustering(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t k,
    std::optional<std::size_t> n_inits = std::nullopt,
    std::optional<uint64_t> seed = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

// Public API: compute new cohort groupings after combining outcome indices.
// Returns pair { new ObservedOutcomeIndices, new cohort sizes }.
std::pair<ObservedOutcomeIndices, arma::uvec>
get_new_cohorts_from_combining_outcomes(
    const ObservedOutcomeIndices& ooi,
    const arma::uvec& cohort_sizes,
    const arma::uvec& old_to_new_outcome);

#ifdef APM_TESTS
namespace test {
// Test-only access to outcome CDFs and counts used by clustering.
// Returns pair { T x G matrix of per-outcome CDF values at interior quantile grid,
//                length-T vector of per-outcome finite observation counts }.
std::pair<arma::mat, arma::uvec>
comp_outcome_dists_test(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size);
}
#endif

} // namespace apm

#endif // APM_CLUSTER_OUTCOMES_H