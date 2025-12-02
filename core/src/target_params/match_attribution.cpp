#include "match_attribution.h"

#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace apm {

namespace {

std::vector<std::unordered_set<std::size_t>> build_observed_sets(const ObservedOutcomeIndices& observed_outcome_indices) {
    std::vector<std::unordered_set<std::size_t>> sets;
    sets.reserve(observed_outcome_indices.size());
    for (const auto& cohort_indices : observed_outcome_indices) {
        std::unordered_set<std::size_t> cohort_set;
        cohort_set.reserve(cohort_indices.n_elem);
        for (arma::uword j = 0; j < cohort_indices.n_elem; ++j) {
            cohort_set.insert(static_cast<std::size_t>(cohort_indices[j]));
        }
        sets.push_back(std::move(cohort_set));
    }
    return sets;
}

bool any_cohort_observes(const std::vector<std::unordered_set<std::size_t>>& obs_sets, std::size_t outcome_idx) {
    for (const auto& cohort_set : obs_sets) {
        if (cohort_set.find(outcome_idx) != cohort_set.end()) return true;
    }
    return false;
}

} // namespace

TargetFn get_fgw_bipartite_match_outcome_diff_params_fn(
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices)
{
    const std::vector<std::unordered_set<std::size_t>> obs_sets = build_observed_sets(observed_outcome_indices);
    if (!any_cohort_observes(obs_sets, outcome_idx_1)) {
        throw std::invalid_argument("outcome_idx_1 is not observed by any cohort.");
    }
    if (!any_cohort_observes(obs_sets, outcome_idx_2)) {
        throw std::invalid_argument("outcome_idx_2 is not observed by any cohort.");
    }

    return [outcome_idx_1, outcome_idx_2, obs_sets](
               const arma::mat& Y,
               const std::vector<OutcomeMeanSufficientStatistics>& stats_all,
               const std::vector<CohortAuxiliaryDataMeans>& /*eta_all*/) -> arma::vec {
        if (outcome_idx_1 >= static_cast<std::size_t>(Y.n_cols) || outcome_idx_2 >= static_cast<std::size_t>(Y.n_cols)) {
            throw std::invalid_argument("Outcome index exceeds Y column dimension.");
        }
        if (stats_all.size() != obs_sets.size()) {
            throw std::invalid_argument("stats_all size must match observed_outcome_indices size.");
        }
        if (Y.n_rows != static_cast<arma::uword>(stats_all.size())) {
            throw std::invalid_argument("Y row dimension must match number of cohorts.");
        }

        const std::size_t C = stats_all.size();

        auto compute_obs_weighted_avg = [&](std::size_t outcome_idx) {
            double weight_sum = 0.0;
            double weighted_total = 0.0;
            for (std::size_t c = 0; c < C; ++c) {
                if (obs_sets[c].find(outcome_idx) != obs_sets[c].end()) {
                    const double w = stats_all[c].cohort_pop_share;
                    weight_sum += w;
                    weighted_total += w * Y(static_cast<arma::uword>(c), static_cast<arma::uword>(outcome_idx));
                }
            }
            return (weight_sum > 0.0) ? (weighted_total / weight_sum) : 0.0;
        };

        auto compute_pop_weighted_avg = [&](std::size_t outcome_idx) {
            double weight_sum = 0.0;
            double weighted_total = 0.0;
            for (std::size_t c = 0; c < C; ++c) {
                const double w = stats_all[c].cohort_pop_share;
                weight_sum += w;
                weighted_total += w * Y(static_cast<arma::uword>(c), static_cast<arma::uword>(outcome_idx));
            }
            return (weight_sum > 0.0) ? (weighted_total / weight_sum) : 0.0;
        };

        const double obs_diff = compute_obs_weighted_avg(outcome_idx_1) - compute_obs_weighted_avg(outcome_idx_2);
        const double pop_diff = compute_pop_weighted_avg(outcome_idx_1) - compute_pop_weighted_avg(outcome_idx_2);
        const double ratio = (std::abs(obs_diff) > 1e-15) ? (pop_diff / obs_diff) : 0.0;
        return arma::vec({ratio, 1.0 - ratio});
    };
}

TargetParameterEstimates est_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<CohortAuxiliaryDataMeanEstimates>& eta_by_cohort,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads)
{
    TargetFn fn = get_fgw_bipartite_match_outcome_diff_params_fn(outcome_idx_1, outcome_idx_2, observed_outcome_indices);
    return est_target_params(ome, stats_by_cohort, eta_by_cohort, fn, num_threads);
}

std::unordered_map<std::string, TargetParameterEstimates> est_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::unordered_map<std::string, std::vector<OutcomeMeanSuffStatEstimates>>& stats_map,
    const std::unordered_map<std::string, std::vector<CohortAuxiliaryDataMeanEstimates>>& eta_map,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<std::size_t> num_threads)
{
    TargetFn fn = get_fgw_bipartite_match_outcome_diff_params_fn(outcome_idx_1, outcome_idx_2, observed_outcome_indices);
    std::unordered_map<std::string, TargetParameterEstimates> out;
    out.reserve(ome_map.size());
    for (const auto& kv : ome_map) {
        const std::string& key = kv.first;
        auto stats_it = stats_map.find(key);
        auto eta_it = eta_map.find(key);
        const std::vector<OutcomeMeanSuffStatEstimates> empty_stats;
        const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
        const auto& stats_vec = (stats_it == stats_map.end()) ? empty_stats : stats_it->second;
        const auto& eta_vec = (eta_it == eta_map.end()) ? empty_eta : eta_it->second;
        out.emplace(key, est_target_params(kv.second, stats_vec, eta_vec, fn, num_threads));
    }
    return out;
}

} // namespace apm

