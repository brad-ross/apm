#pragma once

#ifndef APM_EST_MASKED_OUTCOME_MEAN_ERR_METRICS_H
#define APM_EST_MASKED_OUTCOME_MEAN_ERR_METRICS_H

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "est_target_params.h"

namespace apm {

struct OutcomeMeanErrMetrics {
    std::unordered_map<std::string, double> bias_by_spec;
    std::unordered_map<std::string, double> se_by_spec;
    std::unordered_map<std::string, double> rmse_by_spec;
    double cohort_pop_share = std::numeric_limits<double>::quiet_NaN();
};

using CohortOutcomeIndex = std::pair<std::size_t, std::size_t>;

struct CohortOutcomeIndexHash {
    std::size_t operator()(const CohortOutcomeIndex& p) const noexcept {
        // FNV-1a style hash combine on the two coordinates
        std::size_t h = 1469598103934665603ull;
        h ^= p.first + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= p.second + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

std::unordered_map<CohortOutcomeIndex, OutcomeMeanErrMetrics, CohortOutcomeIndexHash>
est_masked_outcome_mean_err_metrics(const TargetParamComponents& components);

} // namespace apm

#endif // APM_EST_MASKED_OUTCOME_MEAN_ERR_METRICS_H