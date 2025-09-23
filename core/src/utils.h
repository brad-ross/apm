#pragma once
#ifndef APM_UTILS_H
#define APM_UTILS_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <vector>
#include <unordered_map>
#include <optional>
#include <memory>
#ifdef APM_HAS_TBB
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>
#endif

namespace apm {

// Central alias for observed outcome indices per cohort (0-based)
using ObservedOutcomeIndices = std::vector<arma::uvec>;

// Map from 0-based cohort id -> vector of 0-based outcome indices to mask
using CohortOutcomeMask = std::unordered_map<int, arma::uvec>;

// Apply cohort_outcomes_to_mask to observed_outcome_indices, removing masked outcomes
// from each cohort's arma::uvec while preserving original order. If the mask is empty,
// returns a copy of observed_outcome_indices.
ObservedOutcomeIndices get_masked_observed_outcome_indices(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const CohortOutcomeMask& cohort_outcomes_to_mask);

// Count total outcomes T across all cohorts (1 + max index if any, else 0)
arma::uword num_outcomes(const ObservedOutcomeIndices& observed_outcome_indices);

// Threading utilities
struct ParallelismScope {
	std::size_t nt;

	explicit ParallelismScope(std::optional<std::size_t> num_threads);
	~ParallelismScope() = default;

private:
#ifdef APM_HAS_TBB
	std::unique_ptr<oneapi::tbb::global_control> gc_;
#endif
};

// Return default concurrency (uses oneTBB when available, else 1)
std::size_t get_cpp_default_concurrency();

} // namespace apm

#endif // APM_UTILS_H


