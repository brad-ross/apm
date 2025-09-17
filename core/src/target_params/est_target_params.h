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

namespace apm {

struct TargetParameterEstimates {
    arma::vec point;
    std::vector<arma::vec> bootstrap_replicates;

    TargetParameterEstimates() = default;

    TargetParameterEstimates(arma::vec point_in, std::vector<arma::vec> boots = {})
        : point(std::move(point_in)), bootstrap_replicates(std::move(boots)) {}

    bool has_bootstrap_replicates() const noexcept { return !bootstrap_replicates.empty(); }
    std::size_t n_bootstrap_replicates() const noexcept { return bootstrap_replicates.size(); }
    std::size_t p() const noexcept { return static_cast<std::size_t>(point.n_elem); }
};

using TargetFn = std::function<arma::vec(const arma::mat& Y,
                                         const std::vector<CohortAuxiliaryDataMeans>& eta_all)>;

TargetParameterEstimates est_target_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<CohortAuxiliaryDataMeanEstimates>& eta_by_cohort,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads = std::nullopt);

std::unordered_map<std::string, TargetParameterEstimates> est_target_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::unordered_map<std::string, std::vector<CohortAuxiliaryDataMeanEstimates>>& eta_map,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads = std::nullopt);

} // namespace apm

#endif // APM_TARGET_PARAM_ESTIMATES_H


