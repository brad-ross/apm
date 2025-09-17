#ifndef APM_EST_OUTCOME_MEANS_H
#define APM_EST_OUTCOME_MEANS_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <vector>
#include <unordered_map>

#include "cohort_specific_param_structs.h"
#include "utils.h"

namespace apm {

/**
 * @brief Aggregate cohort-specific parameter estimates including bootstrap replicates.
 *
 * Aggregates point estimates and each bootstrap replicate independently using
 * optional weights (and optional per-bootstrap weights), returning a
 * FactorModelEstimates containing aggregated point estimates and aggregated
 * bootstrap replicates.
 */
FactorModelEstimates aggregate_cohort_specific_factor_model_params(
    const std::vector<FactorModelEstimates>& cohort_specific_factor_model_param_ests,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const CohortWeightEstimates& cohort_weight_estimates = CohortWeightEstimates());

/**
 * @brief Aggregate factor model estimates per estimator specification.
 *
 * Validates keys across maps (symmetric presence) and aggregates per spec by
 * dispatching to the vector-based overload of
 * `aggregate_cohort_specific_factor_model_params`.
 */
std::unordered_map<std::string, FactorModelEstimates> aggregate_cohort_specific_factor_model_params(
    const std::unordered_map<std::string, std::vector<FactorModelEstimates>>& cohort_specific_factor_ests,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::unordered_map<std::string, CohortWeightEstimates>& cohort_weights);

/**
 * @brief Container for cohort mean outcome estimates with optional bootstrap replicates.
 */
struct OutcomeMeansEstimates {
    arma::mat mean_outcomes;                    // C x T matrix
    std::vector<arma::mat> bootstrap_replicates; // optional vector of C x T matrices

    OutcomeMeansEstimates(arma::mat point, std::vector<arma::mat> boot = {})
        : mean_outcomes(std::move(point)), bootstrap_replicates(std::move(boot)) {}

    bool has_bootstrap_replicates() const noexcept { return !bootstrap_replicates.empty(); }
    std::size_t n_bootstrap_replicates() const noexcept { return bootstrap_replicates.size(); }
};

/**
 * @brief Estimates cohort mean outcomes using parameter estimates (with optional bootstrap) and
 *        sufficient statistics estimates (with optional bootstrap).
 *
 * Computes point estimates using `apm::FactorModelEstimates::parameter_estimates` and
 * `apm::OutcomeMeanSuffStatEstimates::suff_stat_estimates` for each cohort. If bootstrap
 * replicates are present in both inputs, computes mean outcomes for each bootstrap draw using the
 * corresponding replicate of parameters and sufficient statistics.
 *
 * @param factor_model_estimates `apm::FactorModelEstimates` containing point parameters and optional replicates.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param suff_stat_estimates_vec Vector of `apm::OutcomeMeanSuffStatEstimates`, one per cohort.
 * @return `apm::OutcomeMeansEstimates` containing a C x T point matrix and optional bootstrap matrices.
 */
OutcomeMeansEstimates estimate_outcome_means_across_cohorts(
    const FactorModelEstimates& factor_model_estimates,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<OutcomeMeanSuffStatEstimates>& suff_stat_estimates_vec);

/**
 * @brief Estimates cohort mean outcomes per estimator specification.
 *
 * Iterates over input map keys and dispatches to the single-spec overload,
 * returning an unordered_map keyed by estimator specification.
 */
std::unordered_map<std::string, OutcomeMeansEstimates> estimate_outcome_means_across_cohorts(
    const std::unordered_map<std::string, FactorModelEstimates>& factor_model_estimates_map,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<OutcomeMeanSuffStatEstimates>& suff_stat_estimates_vec);

/**
 * @brief Return the default number of threads available to C++ parallel runtime.
 *
 * Uses oneTBB when available; otherwise returns 1.
 */
std::size_t get_cpp_default_concurrency();

} // namespace apm

#endif // APM_EST_OUTCOME_MEANS_H