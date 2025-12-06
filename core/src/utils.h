#pragma once
#ifndef APM_UTILS_H
#define APM_UTILS_H

//==============================================================================
// Shared utilities and type aliases for APM factor-model estimation.
//==============================================================================

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

/**
 * @brief Observed outcome indices per cohort (0-based).
 *
 * Each entry is the length-T_c vector of outcome indices observed in cohort c,
 * ordered as they appear in the input panel. The largest index across all
 * cohorts defines the global outcome count T = 1 + max index (or 0 if empty).
 */
using ObservedOutcomeIndices = std::vector<arma::uvec>;

/**
 * @brief Map from cohort id (0-based) to outcome indices to drop for that cohort.
 *
 * Used to temporarily mask outcomes (e.g., for placebo or cross-validation) while
 * preserving the original ordering of observed outcomes for each cohort.
 */
using CohortOutcomeMask = std::unordered_map<int, arma::uvec>;

/**
 * @brief Apply an outcome mask per cohort to observed outcome indices.
 *
 * Removes masked outcome indices from each cohort's observed list while
 * preserving order. If the mask is empty, the input is returned unchanged.
 */
ObservedOutcomeIndices get_masked_observed_outcome_indices(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const CohortOutcomeMask& cohort_outcomes_to_mask);

/**
 * @brief Count total outcomes T across all cohorts.
 *
 * @return 1 + max observed index (or 0 when no outcomes).
 */
arma::uword num_outcomes(const ObservedOutcomeIndices& observed_outcome_indices);

// Threading utilities
/**
 * @brief Scoped thread-parallelism limiter.
 */
struct ParallelismScope {
	std::size_t nt; ///< Threads requested (or defaulted).

    /**
     * @brief Set a scoped upper bound on threads for algorithms.
     *
     * When oneTBB is available, constructs a `global_control` to cap the maximum
     * allowed parallelism to `num_threads` (or leaves default if std::nullopt).
     * The scope ends when the object is destroyed.
     *
     * @param num_threads Optional thread cap; std::nullopt keeps default concurrency.
     */
	explicit ParallelismScope(std::optional<std::size_t> num_threads);
	~ParallelismScope() = default;

private:
#ifdef APM_HAS_TBB
	std::unique_ptr<oneapi::tbb::global_control> gc_;
#endif
};

/**
 * @brief Return the default concurrency available to the library.
 *
 * Uses oneTBB's reported thread count when present; otherwise returns 1.
 *
 * @return Concurrency level (threads).
 */
std::size_t get_cpp_default_concurrency();

} // namespace apm

#endif // APM_UTILS_H


