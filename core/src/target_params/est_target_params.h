#pragma once
#ifndef APM_TARGET_PARAM_ESTIMATES_H
#define APM_TARGET_PARAM_ESTIMATES_H

//==============================================================================
// Target parameter estimation and inference built on outcome means and aux data.
//==============================================================================

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "est_outcome_means.h"
#include "cohort_specific_param_structs.h"
#include "est_cohort_specific_params.h"
#include "../outcome_imputation.h"
#include "../bootstrap.h"

namespace apm {

/**
 * @brief Target parameter estimates with optional bootstrap replicates.
 */
struct TargetParameterEstimates {
    arma::vec point;                ///< Point estimates (length p).
    arma::mat bootstrap_replicates; ///< p x B bootstrap replicates (cols), or p x 0 if none.

    TargetParameterEstimates() = default;

    TargetParameterEstimates(arma::vec point_in, arma::mat boots = arma::mat())
        : point(std::move(point_in)), bootstrap_replicates(std::move(boots)) {
        if (bootstrap_replicates.n_rows == 0 && bootstrap_replicates.n_cols == 0) {
            bootstrap_replicates.set_size(static_cast<arma::uword>(point.n_elem), arma::uword(0));
        }
    }

    // Convenience: accept a vector of bootstrap vectors and pack into a matrix
    TargetParameterEstimates(arma::vec point_in, const std::vector<arma::vec>& boots_vec);

    bool has_bootstrap_replicates() const noexcept { return bootstrap_replicates.n_cols > 0; }
    std::size_t n_bootstrap_replicates() const noexcept { return bootstrap_replicates.n_cols; }
    std::size_t p() const noexcept { return static_cast<std::size_t>(point.n_elem); }
};

using TargetFn = std::function<arma::vec(
    const arma::mat& Y,
    const std::vector<OutcomeMeanSufficientStatistics>& stats_all,
    const std::vector<CohortAuxiliaryDataMeans>& eta_all)>;

/**
 * @brief Estimate target parameters for a single estimator specification.
 *
 * @param ome Outcome mean estimates (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient statistics (point + optional bootstrap).
 * @param eta_by_cohort Cohort auxiliary means (point + optional bootstrap).
 * @param fn Target function mapping (C x T outcome mean matrix, cohort sufficient statistics, cohort auxiliary means) -> target parameter vector.
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return TargetParameterEstimates (point + bootstrap).
 */
TargetParameterEstimates est_target_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<CohortAuxiliaryDataMeanEstimates>& eta_by_cohort,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Estimate target parameters for multiple estimator specifications.
 *
 * @param ome_map Map spec -> outcome means (point + optional bootstrap).
 * @param stats_by_cohort Cohort sufficient statistics (point + optional bootstrap).
 * @param eta_by_cohort Cohort auxiliary means (point + optional bootstrap).
 * @param fn Target function mapping (C x T outcome mean matrix, cohort sufficient statistics, cohort auxiliary means) -> target parameter vector.
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return Map spec -> TargetParameterEstimates.
 */
std::unordered_map<std::string, TargetParameterEstimates> est_target_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<CohortAuxiliaryDataMeanEstimates>& eta_by_cohort,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Difference of two target parameter estimate sets (point and bootstrap).
 *
 * @param target_params_1 First target estimates.
 * @param target_params_2 Second target estimates.
 * @return Difference (target_params_1 - target_params_2) with aligned bootstrap.
 */
TargetParameterEstimates get_target_param_diff_ests(
    const TargetParameterEstimates& target_params_1,
    const TargetParameterEstimates& target_params_2);

TargetParameterEstimates combine_target_param_ests(
    const TargetParameterEstimates& target_params_1,
    const TargetParameterEstimates& target_params_2);

TargetParameterEstimates combine_target_param_ests(
    const std::vector<TargetParameterEstimates>& targets);

TargetParameterEstimates subset_target_param_ests(
    const TargetParameterEstimates& target_params,
    const arma::uvec& indices);

TargetParameterEstimates subset_target_param_ests(
    const TargetParameterEstimates& target_params,
    std::size_t index);

//------------------------------------------------------------------------------
// End-to-end components from panel (used by R bindings)
//------------------------------------------------------------------------------

struct TargetParamComponents {
    std::unordered_map<std::string, OutcomeMeansEstimates> outcome_means_by_spec; ///< Outcome means per estimator spec.
    std::vector<OutcomeMeanSuffStatEstimates> cohort_outcome_mean_ests; ///< Cohort outcome means (point + bootstrap).
    std::vector<CohortAuxiliaryDataMeanEstimates> cohort_auxiliary_means; ///< Cohort auxiliary means (point + bootstrap).
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_cohort_outcome_means; ///< Masked cohort means (if masking applied).
    std::optional<ObservedOutcomeIndices> masked_observed_outcome_indices; ///< Masked observed outcomes (if applied).
    std::optional<CohortOutcomeMask> cohort_outcome_mask; ///< Mask specification (if applied).
};

class InMemoryUnbalancedPanel; // fwd
/**
 * @brief End-to-end pipeline to estimate outcome means, auxiliary means, and masks from a panel.
 *
 * @param panel Unbalanced panel input.
 * @param est_specs Map spec -> estimator specification.
 * @param bootstrap Optional shared bootstrap controller.
 * @param num_threads Optional thread cap; defaults to library concurrency (see get_cpp_default_concurrency()).
 * @param cohort_outcomes_to_mask Outcomes to mask per cohort (optional).
 * @param est_outcome_means_via_imputation If true, use imputation-based outcome means; else direct.
 * @param imputation_opts Options for imputation if used.
 * @return TargetParamComponents containing means, auxiliary data, and masks.
 */
TargetParamComponents est_target_param_components_from_panel(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
    std::optional<std::size_t> num_threads = std::nullopt,
    const CohortOutcomeMask& cohort_outcomes_to_mask = CohortOutcomeMask(),
    bool est_outcome_means_via_imputation = true,
    const ImputationOptions& imputation_opts = ImputationOptions());

// Bootstrap-based inference for target parameters
/**
 * @brief Bootstrap-based inference for target parameters.
 *
 * @param ests Target parameter estimates (point + bootstrap).
 * @param panel Underlying panel (for N).
 * @param sig_level Significance level (default 0.05).
 * @return SimultaneousInferenceResults with t-stats, CIs, and bands.
 */
SimultaneousInferenceResults target_param_inference(
    const TargetParameterEstimates& ests,
    const InMemoryUnbalancedPanel& panel,
    double sig_level = 0.05);

// Overload: inference for multiple specs
/**
 * @brief Bootstrap-based inference for multiple target-parameter specs.
 *
 * @param ests_by_spec Map spec -> target parameter estimates.
 * @param panel Underlying panel (for N).
 * @param sig_level Significance level (default 0.05).
 * @return Map spec -> SimultaneousInferenceResults.
 */
std::unordered_map<std::string, SimultaneousInferenceResults> target_param_inference(
    const std::unordered_map<std::string, TargetParameterEstimates>& ests_by_spec,
    const InMemoryUnbalancedPanel& panel,
    double sig_level = 0.05);

} // namespace apm

#endif // APM_TARGET_PARAM_ESTIMATES_H


