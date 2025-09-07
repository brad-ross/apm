#pragma once
#ifndef APM_EST_COHORT_SPECIFIC_PARAMS_H
#define APM_EST_COHORT_SPECIFIC_PARAMS_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif
#ifdef APM_HAS_TBB
#include <oneapi/tbb/info.h>
#endif

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "apm_core.h"                 // ObservedOutcomeIndices
#include "bootstrap.h"
#include "factor_model_estimators/FactorModelEstimator.h"
#include "factor_model_estimators/pc_estimators.h"
#include "cohort_specific_param_structs.h"
#include "OutcomeMeanSuffStatEstimator.h"

namespace apm {

struct EstimatorSpecification {
    std::string factor_model_estimator; // e.g., "principal_components"
    bool include_outcome_fes;
    std::size_t r;
    std::string cohort_weighting = "equal"; // "equal" or "by_size"
};

struct CohortSpecificEstimates {
    std::unordered_map<std::string, std::vector<FactorModelEstimates>> cohort_specific_factor_ests;
    std::vector<OutcomeMeanSuffStatEstimates> cohort_outcome_mean_ests;
    std::unordered_map<std::string, CohortWeightEstimates> cohort_weights;
    std::vector<CohortAuxiliaryDataMeanEstimates> cohort_auxiliary_means;
};

/**
 * @brief Estimate cohort-specific factor parameters and outcome mean sufficient statistics
 *        directly from raw column pointers of a processed panel.
 *
 * All indices (unit, cohort, outcome) are assumed to be 0-based.
 * The observed outcome order per cohort is provided via ObservedOutcomeIndices (0-based).
 * T for covariates is computed as 1 + max(observed outcomes for the cohort).
 */
CohortSpecificEstimates estimate_cohort_specific_params_from_raw(
    const int* unit_idx,         // length n_rows, 0-based
    const int* cohort_id,        // length n_rows, 0-based
    const int* outcome_idx,      // length n_rows, 0-based
    const double* y,             // length n_rows
    const std::vector<const double*>& covar_cols, // size q, each length n_rows
    const std::vector<const double*>& auxiliary_cols,  // size d, each length n_rows
    std::size_t n_rows,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    const ObservedOutcomeIndices& observed_outcome_indices, // 0-based per cohort, index with cohort_id
    std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
#ifdef APM_HAS_TBB
    std::size_t num_threads = oneapi::tbb::info::default_concurrency()
#else
    std::size_t num_threads = 1
#endif
);

} // namespace apm

#endif // APM_EST_COHORT_SPECIFIC_PARAMS_H


