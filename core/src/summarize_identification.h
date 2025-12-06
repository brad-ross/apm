#pragma once
#ifndef APM_SUMMARIZE_IDENTIFICATION_H
#define APM_SUMMARIZE_IDENTIFICATION_H

//==============================================================================
// Identification summaries and overlap-based checks using the O^3 algorithm.
//==============================================================================

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include "utils.h"            // ObservedOutcomeIndices

namespace apm {

/**
 * @brief Summary metrics for identification of aligned factors.
 */
struct IdentificationSummary {
  std::size_t largest_super_cohort_size;             ///< Total units in largest final super cohort.
  double       largest_super_cohort_share;           ///< Share of panel units.
  std::size_t  min_cohort_size_in_largest_super;     ///< Min combined cohort size within that super cohort.
  std::size_t  num_outcomes_in_largest_super_cohort; ///< Unique outcomes observed by at least one cohort in the largest super cohort.
  double       total_outcome_weight_in_largest_super_cohort;  ///< Sum of weights across union outcomes in largest super cohort.
  double       share_outcomes_in_largest_super_cohort;        ///< Union outcome count divided by total outcomes.
  double       share_outcome_weight_in_largest_super_cohort;  ///< Union outcome weight divided by total outcome weight.
  std::size_t  num_o3_iterations;                    ///< Number of O^3 iterations (length of super_cohort_iterates).
};

/**
 * @brief Get sorted cohort indices of the largest super cohort at a given iteration.
 *
 * @param observed_outcome_indices Observed outcomes per cohort.
 * @param cohort_sizes Units per cohort (length C).
 * @param max_model_rank Maximum model rank r used in O^3.
 * @param iter Iteration selector; iter >= 0 clamps to last if too large, iter < 0 counts from end (e.g., -1 is final).
 * @return arma::uvec of cohort indices in the largest super cohort.
 */
arma::uvec
get_largest_super_cohort(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    int iter);

/**
 * @brief Wrapper: largest super cohort at final iteration (iter = -1).
 *
 * @param observed_outcome_indices Observed outcomes per cohort.
 * @param cohort_sizes Units per cohort (length C).
 * @param max_model_rank Maximum model rank r used in O^3.
 * @return arma::uvec of cohort indices in the largest final super cohort.
 */
arma::uvec
get_largest_super_cohort(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank);

/**
 * @brief Summarize identification for multiple panels at a specific iteration.
 *
 * @param observed_outcome_indices_vec Vector of ObservedOutcomeIndices, one per panel.
 * @param cohort_sizes_vec Vector of cohort size vectors, aligned with observed_outcome_indices_vec.
 * @param max_model_rank Maximum model rank r to test in O^3.
 * @param iter Iteration selector; iter >= 0 clamps to last if too large, iter < 0 counts from end (e.g., -1 is final).
 * @return Vector of IdentificationSummary, one per panel.
 */
std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank,
    int iter);

/**
 * @brief Wrapper: summarize identification for multiple panels at final iteration.
 *
 * @param observed_outcome_indices_vec Vector of ObservedOutcomeIndices, one per panel.
 * @param cohort_sizes_vec Vector of cohort size vectors, aligned with observed_outcome_indices_vec.
 * @param max_model_rank Maximum model rank r to test in O^3.
 * @return Vector of IdentificationSummary, one per panel.
 */
std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank);

/**
 * @brief Summarize identification for a single panel at a specific iteration.
 *
 * @param observed_outcome_indices Observed outcomes per cohort (0-based).
 * @param cohort_sizes Units per cohort (length C).
 * @param max_model_rank Maximum model rank r to test in O^3.
 * @param iter Iteration selector; iter >= 0 clamps to last if too large, iter < 0 counts from end (e.g., -1 is final).
 * @return IdentificationSummary at the specified iteration.
 */
IdentificationSummary
summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    int iter);

/**
 * @brief Summarize identification for a single panel at a specific iteration with explicit outcome weights.
 *
 * @param observed_outcome_indices Observed outcomes per cohort (0-based).
 * @param cohort_sizes Units per cohort (length C).
 * @param max_model_rank Maximum model rank r to test in O^3.
 * @param iter Iteration selector; iter >= 0 clamps to last if too large, iter < 0 counts from end (e.g., -1 is final).
 * @param outcome_weights Outcome weights (length T); defaults to ones when absent.
 * @return IdentificationSummary at the specified iteration.
 */
IdentificationSummary
summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    int iter,
    const arma::vec& outcome_weights);

/**
 * @brief Wrapper: summarize identification for a single panel at final iteration.
 *
 * @param observed_outcome_indices Observed outcomes per cohort (0-based).
 * @param cohort_sizes Units per cohort (length C).
 * @param max_model_rank Maximum model rank r to test in O^3.
 * @return IdentificationSummary at the final iteration.
 */
IdentificationSummary
summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank);

/**
 * @brief Summarize identification for a single panel with explicit outcome weights.
 *
 * @param observed_outcome_indices Observed outcomes per cohort (0-based).
 * @param cohort_sizes Units per cohort (length C).
 * @param max_model_rank Maximum model rank r to test in O^3.
 * @param outcome_weights Outcome weights (length T); defaults to ones when absent.
 * @return IdentificationSummary at the final iteration.
 */
IdentificationSummary
summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    const arma::vec& outcome_weights);

/**
 * @brief Summarize identification for multiple panels with explicit weights and iteration.
 *
 * @param observed_outcome_indices_vec Vector of ObservedOutcomeIndices, one per panel.
 * @param cohort_sizes_vec Vector of cohort size vectors, aligned with observed_outcome_indices_vec.
 * @param max_model_rank Maximum model rank r to test in O^3.
 * @param iter Iteration selector; iter >= 0 clamps to last if too large, iter < 0 counts from end (e.g., -1 is final).
 * @param outcome_weights_vec Outcome weights per panel (length T_i each).
 * @return Vector of IdentificationSummary, one per panel.
 */
std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank,
    int iter,
    const std::vector<arma::vec>& outcome_weights_vec);

/**
 * @brief Wrapper: summarize identification for multiple panels at final iteration with weights.
 *
 * @param observed_outcome_indices_vec Vector of ObservedOutcomeIndices, one per panel.
 * @param cohort_sizes_vec Vector of cohort size vectors, aligned with observed_outcome_indices_vec.
 * @param max_model_rank Maximum model rank r to test in O^3.
 * @param outcome_weights_vec Outcome weights per panel (length T_i each).
 * @return Vector of IdentificationSummary, one per panel.
 */
std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank,
    const std::vector<arma::vec>& outcome_weights_vec);

/**
 * @brief Check if aligned factors are identified using the O^3 overlap test.
 *
 * @param observed_outcome_indices Observed outcomes per cohort (0-based).
 * @param r Model rank.
 * @return true if a single super cohort covers all cohorts; false otherwise.
 */
bool aligned_factors_identified(
    const ObservedOutcomeIndices& observed_outcome_indices,
    unsigned int r);

// For each focal cohort, accumulate the total outcome weight of the unique
// outcomes that appear in any cohort whose overlap with the focal cohort
// contains at least `rank` outcomes. The focal cohort's own outcomes always
// contribute to its total. Cohort indices are 0-based. When no weights are
// supplied, each outcome weighs 1.
/**
 * @brief Count outcome weights per cohort when overlap contains at least rank outcomes.
 *
 * @param observed_outcome_indices Observed outcomes per cohort (0-based).
 * @param rank Overlap threshold.
 * @return Length-C vector of total outcome counts (weights = 1).
 */
arma::vec
count_outcomes_with_rank_overlap_per_cohort(
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::size_t rank);

/**
 * @brief Weighted version of count_outcomes_with_rank_overlap_per_cohort.
 *
 * @param observed_outcome_indices Observed outcomes per cohort (0-based).
 * @param rank Overlap threshold.
 * @param outcome_weights Outcome weights (length T).
 * @return Length-C vector of weighted totals.
 */
arma::vec
count_outcomes_with_rank_overlap_per_cohort(
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::size_t rank,
    const arma::vec& outcome_weights);

} // namespace apm

#endif // APM_SUMMARIZE_IDENTIFICATION_H