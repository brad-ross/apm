#include "est_outcome_means.h"
#include "agg_cohort_specific_factor_model_params.h"
#include "apm_core.h"

#include <stdexcept>
#include <algorithm>
#include <string>
#include <set>
#ifdef APM_HAS_TBB
#include <oneapi/tbb/info.h>
#endif

namespace apm {

OutcomeMeansEstimates estimate_outcome_means_across_cohorts(
    const FactorModelEstimates& factor_model_estimates,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<OutcomeMeanSuffStatEstimates>& suff_stat_estimates_vec) {
    const std::size_t C = static_cast<std::size_t>(observed_outcome_indices.size());
    if (suff_stat_estimates_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }

    // Point estimates
    std::vector<OutcomeMeanSufficientStatistics> suff_stats_point;
    suff_stats_point.reserve(C);
    for (std::size_t c = 0; c < C; ++c) {
        suff_stats_point.push_back(suff_stat_estimates_vec[c].suff_stat_estimates);
    }
    arma::mat point_means = estimate_outcome_means_across_cohorts(
        factor_model_estimates.parameter_estimates,
        observed_outcome_indices,
        suff_stats_point);

    // Bootstrap replicates
    const bool has_param_boot = factor_model_estimates.has_bootstrap_replicates();
    std::size_t B = has_param_boot ? factor_model_estimates.n_bootstrap_replicates() : 0;
    if (has_param_boot) {
        // Validate all cohorts have same number of bootstraps
        for (std::size_t c = 0; c < C; ++c) {
            if (suff_stat_estimates_vec[c].n_bootstrap_replicates() != B) {
                throw std::invalid_argument("All cohorts must have the same number of bootstrap replicates as the number of bootstrap replicates in the factor model estimates.");
            }
        }
    }

    std::vector<arma::mat> bootstrap_means;
    bootstrap_means.reserve(B);
    for (std::size_t b = 0; b < B; ++b) {
        // Build suff stats slice for bootstrap b
        std::vector<OutcomeMeanSufficientStatistics> suff_stats_b;
        suff_stats_b.reserve(C);
        for (std::size_t c = 0; c < C; ++c) {
            suff_stats_b.push_back(suff_stat_estimates_vec[c].bootstrap_replicates[b]);
        }

        const FactorModelParameters& params_b = factor_model_estimates.bootstrap_replicates[b];
        bootstrap_means.push_back(
            estimate_outcome_means_across_cohorts(params_b, observed_outcome_indices, suff_stats_b));
    }

    return OutcomeMeansEstimates(std::move(point_means), std::move(bootstrap_means));
}

std::unordered_map<std::string, OutcomeMeansEstimates> estimate_outcome_means_across_cohorts(
    const std::unordered_map<std::string, FactorModelEstimates>& factor_model_estimates_map,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<OutcomeMeanSuffStatEstimates>& suff_stat_estimates_vec) {

    std::unordered_map<std::string, OutcomeMeansEstimates> out;
    out.reserve(factor_model_estimates_map.size());

    for (const auto& kv : factor_model_estimates_map) {
        const std::string& key = kv.first;
        const FactorModelEstimates& ests = kv.second;
        OutcomeMeansEstimates ome = estimate_outcome_means_across_cohorts(
            ests, observed_outcome_indices, suff_stat_estimates_vec);
        out.emplace(key, std::move(ome));
    }

    return out;
}

} // namespace apm