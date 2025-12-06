#pragma once

#ifndef APM_EST_MASKED_OUTCOME_MEAN_ERR_METRICS_H
#define APM_EST_MASKED_OUTCOME_MEAN_ERR_METRICS_H

//==============================================================================
// Error metrics for masked outcome means (bias, SE, RMSE) by cohort/outcome.
//==============================================================================

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "est_target_params.h"

namespace apm {

/**
 * @brief Bias, standard error, and RMSE for outcome mean estimates by spec.
 */
struct OutcomeMeanErrMetrics {
    std::unordered_map<std::string, double> bias_by_spec; ///< Bias per estimator spec (masked predicted mean minus true masked mean).
    std::unordered_map<std::string, double> se_by_spec;   ///< Standard error per estimator spec (bootstrap or analytic).
    std::unordered_map<std::string, double> rmse_by_spec; ///< Root mean squared error per estimator spec.
    double cohort_pop_share = std::numeric_limits<double>::quiet_NaN(); ///< Cohort population share.
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

/**
 * @brief Estimate masked outcome mean error metrics (bias, SE, RMSE) by estimator specification for all cohort/outcome pairs.
 *
 * @param components Target parameter components including masks and means.
 * @return Map from (cohort, outcome) -> OutcomeMeanErrMetrics (bias, SE, RMSE per spec, plus cohort share).
 */
std::unordered_map<CohortOutcomeIndex, OutcomeMeanErrMetrics, CohortOutcomeIndexHash>
est_masked_outcome_mean_err_metrics(const TargetParamComponents& components);

} // namespace apm

#endif // APM_EST_MASKED_OUTCOME_MEAN_ERR_METRICS_H