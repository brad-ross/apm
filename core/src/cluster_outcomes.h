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

// Compute per-outcome empirical CDFs at an interior G-point probability grid.
// Returns:
//  - outcome_val_cdfs: T x G matrix (rows = outcomes 0..T-1, cols = probs g/(G+1))
//  - outcome_counts: T-length vector of integer counts per outcome
std::pair<arma::mat, arma::uvec>
comp_outcome_dists(const InMemoryUnbalancedPanel& panel, std::size_t G);

// Combine cohorts whose mapped outcome sets are identical; sum sizes.
// Inputs:
//  - ooi: ObservedOutcomeIndices (per-cohort old outcome indices, 0-based)
//  - cohort_sizes: length C vector of atomic cohort sizes
//  - oldToNewOutcome: map old outcome -> new merged outcome index
// Returns:
//  - combined_ooi: unique, sorted new outcome index sets (one per merged cohort)
//  - combined_cohort_sizes: summed sizes matching combined_ooi
std::pair<ObservedOutcomeIndices, arma::uvec>
get_new_cohorts_from_combining_outcomes(
    const ObservedOutcomeIndices& ooi,
    const arma::uvec& cohort_sizes,
    const std::unordered_map<int,int>& oldToNewOutcome);

} // namespace apm

#endif // APM_CLUSTER_OUTCOMES_H