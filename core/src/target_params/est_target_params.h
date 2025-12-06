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
 *
 * This struct holds the output of target parameter estimation, where target
 * parameters are user-defined functions of cohort outcome means. Common examples
 * include treatment effects, average differences across time periods, or
 * policy-relevant aggregations.
 *
 * The struct stores both point estimates (a p-dimensional vector) and optional
 * bootstrap replicates (a p x B matrix where each column is a bootstrap draw).
 * Bootstrap replicates enable inference via target_param_inference().
 */
struct TargetParameterEstimates {
    arma::vec point;                ///< Point estimates (length p).
    arma::mat bootstrap_replicates; ///< p x B bootstrap replicates (cols), or p x 0 if none.

    /**
     * @brief Default constructor.
     */
    TargetParameterEstimates() = default;

    /**
     * @brief Construct from point estimates and optional bootstrap matrix.
     *
     * @param point_in Point estimate vector of length p.
     * @param boots Bootstrap replicates matrix (p x B). If empty, an empty
     *              matrix with p rows and 0 columns is created.
     */
    TargetParameterEstimates(arma::vec point_in, arma::mat boots = arma::mat())
        : point(std::move(point_in)), bootstrap_replicates(std::move(boots)) {
        if (bootstrap_replicates.n_rows == 0 && bootstrap_replicates.n_cols == 0) {
            bootstrap_replicates.set_size(static_cast<arma::uword>(point.n_elem), arma::uword(0));
        }
    }

    /**
     * @brief Construct from point estimates and a vector of bootstrap vectors.
     *
     * Packs the bootstrap vectors into a p x B matrix where each column corresponds
     * to one bootstrap replicate.
     *
     * @param point_in Point estimate vector of length p.
     * @param boots_vec Vector of B bootstrap replicate vectors, each of length p.
     * @throws std::runtime_error If any bootstrap vector has inconsistent length.
     */
    TargetParameterEstimates(arma::vec point_in, const std::vector<arma::vec>& boots_vec);

    /**
     * @brief Check if bootstrap replicates are available.
     * @return True if bootstrap_replicates has B > 0 columns.
     */
    bool has_bootstrap_replicates() const noexcept { return bootstrap_replicates.n_cols > 0; }

    /**
     * @brief Get the number of bootstrap replicates.
     * @return Number of columns in bootstrap_replicates (0 if none).
     */
    std::size_t n_bootstrap_replicates() const noexcept { return bootstrap_replicates.n_cols; }

    /**
     * @brief Get the dimension of the target parameter vector.
     * @return Length of the point estimate vector.
     */
    std::size_t p() const noexcept { return static_cast<std::size_t>(point.n_elem); }
};

/**
 * @brief Type alias for target parameter functions.
 *
 * A TargetFn maps estimated outcome means and cohort-level statistics to a
 * vector of target parameters. The function signature is:
 *
 * @code
 * arma::vec fn(
 *     const arma::mat& Y,                                      // C x T outcome means matrix
 *     const std::vector<OutcomeMeanSufficientStatistics>& stats_all,  // Per-cohort statistics
 *     const std::vector<CohortAuxiliaryDataMeans>& eta_all     // Per-cohort auxiliary data
 * );
 * @endcode
 *
 * @param Y C x T matrix of (imputed) cohort-by-outcome means.
 * @param stats_all Vector of per-cohort sufficient statistics including cohort
 *                  population shares, observed outcome means, and covariate means.
 * @param eta_all Vector of per-cohort auxiliary data means.
 * @return Numeric vector of length p (the target parameter dimension).
 */
using TargetFn = std::function<arma::vec(
    const arma::mat& Y,
    const std::vector<OutcomeMeanSufficientStatistics>& stats_all,
    const std::vector<CohortAuxiliaryDataMeans>& eta_all)>;

/**
 * @brief Estimate target parameters for a single estimator specification.
 *
 * Applies a user-defined target function to estimated outcome means to compute
 * target parameters of interest. When bootstrap replicates are present in the
 * inputs, the function is applied to each bootstrap replicate to enable inference.
 *
 * All inputs must have the same number of bootstrap replicates (or all have none).
 * Bootstrap replicates are processed in parallel when TBB is available.
 *
 * @param ome Outcome mean estimates (point + optional bootstrap).
 * @param stats_by_cohort Per-cohort sufficient statistics (point + optional bootstrap).
 *                        Provides cohort population shares, observed outcome means,
 *                        and covariate means.
 * @param eta_by_cohort Per-cohort auxiliary data means (point + optional bootstrap).
 * @param fn Target function mapping (C x T outcome mean matrix, cohort statistics,
 *           cohort auxiliary means) to a target parameter vector of length p.
 * @param num_threads Optional thread cap for parallel bootstrap processing.
 *                    Defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return TargetParameterEstimates containing point estimates and bootstrap replicates.
 * @throws std::invalid_argument If inputs have mismatched bootstrap replicate counts.
 * @throws std::runtime_error If target function returns inconsistent vector lengths.
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
 * Applies a user-defined target function to outcome means from multiple estimator
 * specifications, producing target parameter estimates for each specification.
 * Cohort-level statistics and auxiliary data are shared across all specifications.
 *
 * @param ome_map Map from specification name to OutcomeMeansEstimates.
 * @param stats_by_cohort Per-cohort sufficient statistics, shared across specs.
 * @param eta_by_cohort Per-cohort auxiliary data means, shared across specs.
 * @param fn Target function mapping (C x T outcome means, cohort statistics,
 *           cohort auxiliary means) to a target parameter vector of length p.
 * @param num_threads Optional thread cap for parallel bootstrap processing.
 *                    Defaults to library concurrency (see get_cpp_default_concurrency()).
 * @return Map from specification name to TargetParameterEstimates.
 * @throws std::invalid_argument If inputs have mismatched bootstrap replicate counts.
 */
std::unordered_map<std::string, TargetParameterEstimates> est_target_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<CohortAuxiliaryDataMeanEstimates>& eta_by_cohort,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads = std::nullopt);

/**
 * @brief Compute elementwise difference of two target parameter estimates.
 *
 * Returns target_params_1 - target_params_2 for both point estimates and
 * bootstrap replicates (when present). Useful for computing treatment effects
 * or contrasts between two estimators or specifications.
 *
 * @param target_params_1 First target parameter estimates.
 * @param target_params_2 Second target parameter estimates. Must have the same
 *                        dimension p() and bootstrap structure as target_params_1.
 * @return TargetParameterEstimates containing the elementwise difference with
 *         the same dimension p and bootstrap count B as the inputs.
 * @throws std::invalid_argument If point vectors have different lengths.
 * @throws std::invalid_argument If one input has bootstrap replicates and the other does not.
 * @throws std::invalid_argument If bootstrap replicate counts differ.
 */
TargetParameterEstimates get_target_param_diff_ests(
    const TargetParameterEstimates& target_params_1,
    const TargetParameterEstimates& target_params_2);

/**
 * @brief Concatenate two target parameter estimates.
 *
 * Combines two TargetParameterEstimates by vertically stacking their point
 * estimate vectors and bootstrap replicate matrices. The resulting object
 * has dimension p1 + p2.
 *
 * Both inputs must have the same bootstrap structure: either both have
 * bootstrap replicates with the same count B, or neither has bootstrap replicates.
 *
 * @param target_params_1 First target parameter estimates (dimension p1).
 * @param target_params_2 Second target parameter estimates (dimension p2).
 * @return TargetParameterEstimates with dimension p1 + p2.
 * @throws std::invalid_argument If one input has bootstrap replicates and the other does not.
 * @throws std::invalid_argument If bootstrap replicate counts differ.
 */
TargetParameterEstimates combine_target_param_ests(
    const TargetParameterEstimates& target_params_1,
    const TargetParameterEstimates& target_params_2);

/**
 * @brief Concatenate multiple target parameter estimates.
 *
 * Combines a vector of TargetParameterEstimates by vertically stacking their
 * point estimate vectors and bootstrap replicate matrices. The resulting object
 * has dimension equal to the sum of all input dimensions.
 *
 * All inputs must have the same bootstrap structure: either all have bootstrap
 * replicates with the same count B, or none have bootstrap replicates.
 *
 * @param targets Non-empty vector of TargetParameterEstimates to combine.
 * @return TargetParameterEstimates with dimension equal to sum of input dimensions.
 * @throws std::invalid_argument If input vector is empty.
 * @throws std::invalid_argument If inputs have inconsistent bootstrap structures.
 */
TargetParameterEstimates combine_target_param_ests(
    const std::vector<TargetParameterEstimates>& targets);

/**
 * @brief Extract a subset of target parameters by indices.
 *
 * Creates a new TargetParameterEstimates containing only the elements at
 * the specified indices. Both point estimates and bootstrap replicates
 * (if present) are subsetted.
 *
 * @param target_params Source target parameter estimates.
 * @param indices Vector of 0-based indices to extract. All indices must be
 *                in the range [0, p()).
 * @return TargetParameterEstimates with dimension equal to indices.n_elem.
 * @throws std::invalid_argument If any index is out of range.
 */
TargetParameterEstimates subset_target_param_ests(
    const TargetParameterEstimates& target_params,
    const arma::uvec& indices);

/**
 * @brief Extract a single target parameter by index.
 *
 * Convenience overload that extracts a single element. Equivalent to
 * subset_target_param_ests(target_params, arma::uvec{index}).
 *
 * @param target_params Source target parameter estimates.
 * @param index 0-based index to extract. Must be in the range [0, p()).
 * @return TargetParameterEstimates with dimension 1.
 * @throws std::invalid_argument If index is out of range.
 */
TargetParameterEstimates subset_target_param_ests(
    const TargetParameterEstimates& target_params,
    std::size_t index);

//------------------------------------------------------------------------------
// End-to-end components from panel (used by R bindings)
//------------------------------------------------------------------------------

/**
 * @brief Container for all intermediate results from end-to-end target parameter estimation.
 *
 * This struct holds the output of est_target_param_components_from_panel(), which
 * performs the complete estimation pipeline from panel data to outcome means.
 * The components can then be passed to est_target_params() along with a user-defined
 * target function to compute final target parameters.
 */
struct TargetParamComponents {
    /**
     * @brief Outcome means per estimator specification.
     *
     * Map from specification name to OutcomeMeansEstimates containing
     * C x T cohort-by-outcome mean matrices (point + optional bootstrap).
     */
    std::unordered_map<std::string, OutcomeMeansEstimates> outcome_means_by_spec;

    /**
     * @brief Per-cohort outcome mean sufficient statistics.
     *
     * Vector of length C containing cohort-level statistics including
     * population shares, observed outcome means, and covariate means.
     * Point estimates and bootstrap replicates are included.
     */
    std::vector<OutcomeMeanSuffStatEstimates> cohort_outcome_mean_ests;

    /**
     * @brief Per-cohort auxiliary data means.
     *
     * Vector of length C containing cohort-level auxiliary data means.
     * Empty if no auxiliary data was provided in the panel.
     */
    std::vector<CohortAuxiliaryDataMeanEstimates> cohort_auxiliary_means;

    /**
     * @brief True outcome means for masked outcomes (if masking was applied).
     *
     * Map from 0-based cohort index to OutcomeMeanSufficientStatistics
     * containing the true means for outcomes that were masked during estimation.
     * Used for cross-validation or out-of-sample evaluation.
     */
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_cohort_outcome_means;

    /**
     * @brief Observed outcome indices after masking (if applied).
     *
     * Contains the outcome indices that remain observed after masking.
     * std::nullopt if no masking was applied.
     */
    std::optional<ObservedOutcomeIndices> masked_observed_outcome_indices;

    /**
     * @brief The mask specification that was applied (if any).
     *
     * Map from cohort name to vector of outcome indices that were masked.
     * std::nullopt if no masking was applied.
     */
    std::optional<CohortOutcomeMask> cohort_outcome_mask;
};

class InMemoryUnbalancedPanel; // fwd

/**
 * @brief End-to-end pipeline to estimate outcome means from panel data.
 *
 * Performs the complete estimation pipeline from raw panel data to imputed
 * outcome means, combining:
 * 1. Cohort-specific factor model estimation
 * 2. Factor aggregation across cohorts
 * 3. Outcome mean estimation via imputation (optional)
 *
 * The output contains all intermediate results needed for target parameter
 * estimation and inference via est_target_params() and target_param_inference().
 *
 * @param panel Unbalanced panel containing unit-level outcome data.
 * @param est_specs Map from specification name to EstimatorSpecification defining
 *                  the factor model estimator, number of factors, etc.
 * @param bootstrap Optional shared bootstrap controller for inference.
 *                  If nullptr, only point estimates are computed.
 * @param num_threads Optional thread cap for parallel processing.
 *                    Defaults to library concurrency (see get_cpp_default_concurrency()).
 * @param cohort_outcomes_to_mask Map from cohort name to outcome indices to mask.
 *                                Masked outcomes are excluded from estimation but
 *                                their true values are preserved for evaluation.
 * @param est_outcome_means_via_imputation If true (default), compute outcome means
 *                                         by imputing unobserved outcomes using the
 *                                         estimated factor model. If false, use only
 *                                         observed outcome means.
 * @param imputation_opts Options controlling the imputation algorithm including
 *                        solver choice, convergence tolerances, and LSMR settings.
 * @return TargetParamComponents containing outcome means, cohort statistics,
 *         auxiliary data, and masking information.
 *
 * @see est_target_params() For computing target parameters from the returned components.
 * @see target_param_inference() For computing inference on target parameters.
 */
TargetParamComponents est_target_param_components_from_panel(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
    std::optional<std::size_t> num_threads = std::nullopt,
    const CohortOutcomeMask& cohort_outcomes_to_mask = CohortOutcomeMask(),
    bool est_outcome_means_via_imputation = true,
    const ImputationOptions& imputation_opts = ImputationOptions());

//------------------------------------------------------------------------------
// Bootstrap-based inference for target parameters
//------------------------------------------------------------------------------

/**
 * @brief Compute bootstrap-based inference for target parameters.
 *
 * Uses bootstrap replicates to compute:
 * - Robust standard errors (IQR-based)
 * - Pointwise t-statistics and p-values
 * - Pointwise confidence intervals
 * - Romano-Wolf stepdown adjusted p-values (FWER-controlling)
 * - Simultaneous confidence bands (FWER-controlling)
 *
 * @param ests Target parameter estimates with bootstrap replicates.
 *             Must have has_bootstrap_replicates() == true.
 * @param panel Underlying panel data (used to determine sample size N).
 * @param sig_level Significance level for confidence intervals and bands.
 *                  Must be in (0, 1). Default is 0.05 for 95% intervals.
 * @return SimultaneousInferenceResults containing standard errors, t-statistics,
 *         confidence intervals, adjusted p-values, and confidence bands.
 * @throws std::invalid_argument If panel.num_units() == 0.
 * @throws std::invalid_argument If ests has no bootstrap replicates.
 *
 * @see SimultaneousInferenceResults For accessing the inference output.
 */
SimultaneousInferenceResults target_param_inference(
    const TargetParameterEstimates& ests,
    const InMemoryUnbalancedPanel& panel,
    double sig_level = 0.05);

/**
 * @brief Compute bootstrap-based inference for multiple target parameter specs.
 *
 * Applies target_param_inference() to each specification in the input map,
 * returning a map of inference results with matching keys.
 *
 * @param ests_by_spec Map from specification name to TargetParameterEstimates.
 *                     Each must have bootstrap replicates.
 * @param panel Underlying panel data (used to determine sample size N).
 * @param sig_level Significance level for confidence intervals and bands.
 *                  Must be in (0, 1). Default is 0.05 for 95% intervals.
 * @return Map from specification name to SimultaneousInferenceResults.
 * @throws std::invalid_argument If panel.num_units() == 0.
 * @throws std::invalid_argument If any estimate set has no bootstrap replicates.
 *
 * @see target_param_inference() For single-spec inference.
 */
std::unordered_map<std::string, SimultaneousInferenceResults> target_param_inference(
    const std::unordered_map<std::string, TargetParameterEstimates>& ests_by_spec,
    const InMemoryUnbalancedPanel& panel,
    double sig_level = 0.05);

} // namespace apm

#endif // APM_TARGET_PARAM_ESTIMATES_H