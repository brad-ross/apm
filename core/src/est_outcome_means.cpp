#include "est_outcome_means.h"
#include "apm_core.h"

#include <stdexcept>
#include <algorithm>
#include <string>
#include <set>

namespace apm {

FactorModelEstimates aggregate_cohort_specific_factor_model_params(
    const std::vector<FactorModelEstimates>& cohort_specific_factor_model_param_ests,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const CohortWeightEstimates& cohort_weight_estimates) {
    const std::size_t C = cohort_specific_factor_model_param_ests.size();

    // Aggregate point estimates
    std::vector<FactorModelParameters> point_params;
    point_params.reserve(C);
    for (const auto& est : cohort_specific_factor_model_param_ests) {
        point_params.push_back(est.parameter_estimates);
    }

    FactorModelParameters agg_point = aggregate_cohort_specific_factor_model_params(
        point_params, observed_outcome_indices, cohort_weight_estimates.cohort_weights);

    // Determine number of bootstrap replicates from first cohort (0 if none)
    std::size_t B = cohort_specific_factor_model_param_ests.empty()
        ? 0
        : cohort_specific_factor_model_param_ests.front().n_bootstrap_replicates();

    if (cohort_weight_estimates.bootstrap_cohort_weights.size() != B) {
        throw std::invalid_argument("bootstrap_cohort_weights must have length equal to number of bootstrap replicates.");
    }

    std::vector<FactorModelParameters> agg_bootstrap;
    agg_bootstrap.reserve(B);

    for (std::size_t b = 0; b < B; ++b) {
        std::vector<FactorModelParameters> params_b;
        params_b.reserve(C);
        for (const auto& est : cohort_specific_factor_model_param_ests) {
            if (b == 0) {
                if (est.n_bootstrap_replicates() != B) {
                    throw std::invalid_argument("All cohorts must have the same number of bootstrap replicates.");
                }
            }
            params_b.push_back(est.bootstrap_replicates[b]);
        }

        agg_bootstrap.push_back(
            aggregate_cohort_specific_factor_model_params(
                params_b, observed_outcome_indices, cohort_weight_estimates.bootstrap_cohort_weights[b]));
    }

    return FactorModelEstimates(std::move(agg_point), std::move(agg_bootstrap));
}

std::unordered_map<std::string, FactorModelEstimates> aggregate_cohort_specific_factor_model_params(
    const std::unordered_map<std::string, std::vector<FactorModelEstimates>>& cohort_specific_factor_ests,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::unordered_map<std::string, CohortWeightEstimates>& cohort_weights) {

    std::unordered_map<std::string, FactorModelEstimates> out;
    out.reserve(std::max(cohort_specific_factor_ests.size(), cohort_weights.size()));

    // Build union of keys, mirroring build_return_list pattern
    std::set<std::string> union_keys;
    for (const auto& kv : cohort_specific_factor_ests) union_keys.insert(kv.first);
    for (const auto& kw : cohort_weights) union_keys.insert(kw.first);

    for (const auto& key : union_keys) {
        auto fit = cohort_specific_factor_ests.find(key);
        auto wit = cohort_weights.find(key);
        if (fit == cohort_specific_factor_ests.end()) {
            throw std::invalid_argument("Spec key present in cohort weights but missing in factor estimates: " + key);
        }
        if (wit == cohort_weights.end()) {
            throw std::invalid_argument("Spec key present in factor estimates but missing in cohort weights: " + key);
        }

        const auto& ests = fit->second;
        const auto& weights = wit->second;
        FactorModelEstimates agg = aggregate_cohort_specific_factor_model_params(
            ests, observed_outcome_indices, weights);
        out.emplace(key, std::move(agg));
    }

    return out;
}

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

} // namespace apm


