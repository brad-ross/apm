#pragma once
#ifndef APM_CLUSTER_OUTCOMES_H
#define APM_CLUSTER_OUTCOMES_H

//==============================================================================
// Outcome clustering utilities for combining outcomes by similarity.
//==============================================================================

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

/**
 * @brief Compute outcome clusterings across a k grid of CDF values using weighted k-means.
 *
 * Uses CDF-based outcome embeddings (G grid points) and outcome counts as
 * weights. Runs multiple random initializations per k and returns the best
 * assignment per k based on weighted SSE.
 *
 * @param panel Panel of outcomes (used to compute outcome CDF embeddings).
 * @param grid_size Number of interior quantile points G for the CDF embedding (G>0).
 * @param min_k Minimum clusters k (>=1).
 * @param max_k Maximum clusters k (>=min_k, <= T).
 * @param n_inits Optional initializations per k (default 10).
 * @param seed Optional RNG seed (offset per init).
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return Vector of cluster assignments (one arma::uvec per k in [min_k,max_k]), length T each.
 */
std::vector<arma::uvec>
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t min_k,
    std::size_t max_k,
    std::optional<std::size_t> n_inits = std::nullopt,
    std::optional<uint64_t> seed = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Compute a single outcome clustering with k clusters.
 *
 * Convenience wrapper around comp_outcome_clusterings for a single k.
 *
 * @param panel Panel of outcomes.
 * @param grid_size Number of interior quantile points G for the CDF embedding.
 * @param k Number of clusters.
 * @param n_inits Optional initializations per k (default 10).
 * @param seed Optional RNG seed.
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return Cluster assignments (length T) for the given k.
 */
arma::uvec
comp_outcome_clustering(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t k,
    std::optional<std::size_t> n_inits = std::nullopt,
    std::optional<uint64_t> seed = std::nullopt,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Remap observed outcome indices after clustering merges outcomes.
 *
 * @param ooi Original observed outcome indices per cohort.
 * @param cohort_sizes Cohort sizes (length C).
 * @param old_to_new_outcome Mapping from old outcome index to new cluster index (length T).
 * @return Pair {new ObservedOutcomeIndices, new cohort sizes} reflecting merged outcomes.
 */
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
/**
 * @brief Test helper exposing outcome CDFs and counts used in clustering.
 *
 * @param panel Panel of outcomes.
 * @param grid_size Number of interior quantile points G.
 * @return Pair {T x G CDF matrix, length-T counts}.
 */
std::pair<arma::mat, arma::uvec>
comp_outcome_dists_test(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size);
}
#endif

} // namespace apm

#endif // APM_CLUSTER_OUTCOMES_H