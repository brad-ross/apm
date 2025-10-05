#pragma once
#ifndef APM_TARGET_PARAM_ESTIMATES_H
#define APM_TARGET_PARAM_ESTIMATES_H

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

struct TargetParameterEstimates {
    arma::vec point;
    arma::mat bootstrap_replicates;

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

TargetParameterEstimates est_target_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<CohortAuxiliaryDataMeanEstimates>& eta_by_cohort,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads = std::nullopt);

std::unordered_map<std::string, TargetParameterEstimates> est_target_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::unordered_map<std::string, std::vector<OutcomeMeanSuffStatEstimates>>& stats_map,
    const std::unordered_map<std::string, std::vector<CohortAuxiliaryDataMeanEstimates>>& eta_map,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads = std::nullopt);

//------------------------------------------------------------------------------
// End-to-end components from panel (used by R bindings)
//------------------------------------------------------------------------------

struct TargetParamComponents {
    std::unordered_map<std::string, OutcomeMeansEstimates> outcome_means_by_spec;
    std::vector<OutcomeMeanSuffStatEstimates> cohort_outcome_mean_ests;
    std::vector<CohortAuxiliaryDataMeanEstimates> cohort_auxiliary_means;
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_cohort_outcome_means;
    std::optional<ObservedOutcomeIndices> masked_observed_outcome_indices;
};

class InMemoryUnbalancedPanel; // fwd
TargetParamComponents est_target_param_components_from_panel(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr,
    std::optional<std::size_t> num_threads = std::nullopt,
    const CohortOutcomeMask& cohort_outcomes_to_mask = CohortOutcomeMask(),
    bool est_outcome_means_via_imputation = true,
    double imputation_tol = DEFAULT_TOL,
    std::size_t imputation_max_iters = DEFAULT_MAX_ITERS,
    const std::string& imputation_fixed_point_method = DEFAULT_FP_METHOD);

// Bootstrap-based inference for target parameters
SimultaneousInferenceResults target_param_inference(
    const TargetParameterEstimates& ests,
    const InMemoryUnbalancedPanel& panel,
    double sig_level = 0.05);

} // namespace apm

#endif // APM_TARGET_PARAM_ESTIMATES_H


