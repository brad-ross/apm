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

// Mapping-only APIs
// Compute outcome cluster assignments for k in [min_k, max_k].
std::vector<arma::uvec>
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t min_k,
    std::size_t max_k);

// Single-k mapping
arma::uvec
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t k);

} // namespace apm

#endif // APM_CLUSTER_OUTCOMES_H