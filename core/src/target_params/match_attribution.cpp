#include "match_attribution.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

std::vector<std::unordered_map<std::size_t, arma::uword>> build_observed_position_maps(
    const ObservedOutcomeIndices& observed_outcome_indices)
{
    std::vector<std::unordered_map<std::size_t, arma::uword>> pos_maps;
    pos_maps.reserve(observed_outcome_indices.size());
    for (const auto& cohort_indices : observed_outcome_indices) {
        std::unordered_map<std::size_t, arma::uword> pos_map;
        pos_map.reserve(cohort_indices.n_elem);
        for (arma::uword j = 0; j < cohort_indices.n_elem; ++j) {
            pos_map.emplace(static_cast<std::size_t>(cohort_indices[j]), j);
        }
        pos_maps.push_back(std::move(pos_map));
    }
    return pos_maps;
}

bool any_cohort_observes(const std::vector<std::unordered_set<std::size_t>>& obs_sets, std::size_t outcome_idx) {
    for (const auto& cohort_set : obs_sets) {
        if (cohort_set.find(outcome_idx) != cohort_set.end()) return true;
    }
    return false;
}

arma::uvec project_observed_subset(const std::unordered_set<std::size_t>& cohort_set,
                                   const arma::uvec& outcome_indices)
{
    arma::uvec observed_subset(outcome_indices.n_elem);
    arma::uword subset_size = 0;
    for (arma::uword i = 0; i < outcome_indices.n_elem; ++i) {
        const arma::uword idx = outcome_indices[i];
        if (cohort_set.find(static_cast<std::size_t>(idx)) == cohort_set.end()) continue;
        observed_subset[subset_size++] = idx;
    }
    observed_subset.set_size(subset_size);
    return observed_subset;
}

arma::vec build_outcome_weights(const ObservedOutcomeIndices& observed_outcome_indices,
                                std::optional<arma::vec> outcome_weights_opt)
{
    const arma::uword total_outcomes = num_outcomes(observed_outcome_indices);
    if (total_outcomes == 0) {
        throw std::invalid_argument("observed_outcome_indices must describe at least one outcome.");
    }

    arma::vec weights;
    if (outcome_weights_opt.has_value()) {
        weights = outcome_weights_opt.value();
    } else {
        weights = arma::vec(total_outcomes, arma::fill::ones);
    }

    if (weights.n_elem != total_outcomes) {
        throw std::invalid_argument("outcome_weights length must equal the total number of outcomes.");
    }

    double positive_sum = 0.0;
    for (arma::uword i = 0; i < weights.n_elem; ++i) {
        const double w = weights[i];
        if (w < 0.0) {
            throw std::invalid_argument("outcome_weights entries must be nonnegative.");
        }
        positive_sum += w;
    }
    if (positive_sum <= 0.0) {
        throw std::invalid_argument("outcome_weights entries must sum to a positive value.");
    }

    return weights;
}

double subset_weight_sum(const arma::vec& weights, const arma::uvec& outcome_indices) {
    double sum = 0.0;
    for (arma::uword i = 0; i < outcome_indices.n_elem; ++i) {
        sum += weights[static_cast<arma::uword>(outcome_indices[i])];
    }
    return sum;
}

void validate_outcome_indices(const arma::uvec& outcome_indices,
                              const std::vector<std::unordered_set<std::size_t>>& obs_sets,
                              std::size_t total_outcomes,
                              const arma::vec& outcome_weights,
                              const char* arg_name)
{
    if (outcome_indices.n_elem == 0) {
        throw std::invalid_argument(std::string(arg_name) + " must contain at least one outcome index.");
    }
    for (arma::uword i = 0; i < outcome_indices.n_elem; ++i) {
        const std::size_t idx = static_cast<std::size_t>(outcome_indices[i]);
        if (idx >= total_outcomes) {
            throw std::invalid_argument(std::string(arg_name) + " contains an index that is not observed (exceeds available outcomes).");
        }
        if (!any_cohort_observes(obs_sets, idx)) {
            throw std::invalid_argument(std::string(arg_name) + " contains an index that is not observed by any cohort.");
        }
    }
    if (subset_weight_sum(outcome_weights, outcome_indices) <= 0.0) {
        throw std::invalid_argument(std::string(arg_name) + " corresponds to zero total outcome weight.");
    }
}

} // namespace

TargetFn get_fgw_bipartite_match_outcome_diff_params_fn(
    const arma::uvec& outcome_indices_1,
    const arma::uvec& outcome_indices_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights,
    bool use_observed_outcome_means)
{
    const std::vector<std::unordered_set<std::size_t>> obs_sets = build_observed_sets(observed_outcome_indices);
    const std::vector<std::unordered_map<std::size_t, arma::uword>> obs_pos_maps =
        build_observed_position_maps(observed_outcome_indices);
    const arma::vec weights = build_outcome_weights(observed_outcome_indices, std::move(outcome_weights));
    const std::size_t total_outcomes = static_cast<std::size_t>(weights.n_elem);
    validate_outcome_indices(outcome_indices_1, obs_sets, total_outcomes, weights, "outcome_indices_1");
    validate_outcome_indices(outcome_indices_2, obs_sets, total_outcomes, weights, "outcome_indices_2");
    const arma::uword max_required_idx = std::max(
        outcome_indices_1.max(),
        outcome_indices_2.max());

    return [outcome_indices_1,
            outcome_indices_2,
            obs_sets,
            weights,
            max_required_idx,
            obs_pos_maps,
            use_observed_outcome_means](
               const arma::mat& Y,
               const std::vector<OutcomeMeanSufficientStatistics>& stats_all,
               const std::vector<CohortAuxiliaryDataMeans>& /*eta_all*/) -> arma::vec {
        if (max_required_idx >= Y.n_cols) {
            throw std::invalid_argument("Outcome index exceeds Y column dimension.");
        }
        if (stats_all.size() != obs_sets.size()) {
            throw std::invalid_argument("stats_all size must match observed_outcome_indices size.");
        }
        if (Y.n_rows != static_cast<arma::uword>(stats_all.size())) {
            throw std::invalid_argument("Y row dimension must match number of cohorts.");
        }

        const std::size_t C = stats_all.size();

        auto read_observed_mean = [&](std::size_t cohort_idx, arma::uword outcome_idx) -> double {
            const arma::vec& observed_means = stats_all[cohort_idx].observed_outcome_means;
            if (observed_means.n_elem == 0) {
                throw std::invalid_argument("use_observed_outcome_means requires observed outcome means for each cohort.");
            }
            const auto& pos_map = obs_pos_maps[cohort_idx];
            const auto pos_it = pos_map.find(static_cast<std::size_t>(outcome_idx));
            if (pos_it == pos_map.end()) {
                throw std::runtime_error("Outcome index missing from observed_outcome_indices for cohort.");
            }
            const arma::uword pos = pos_it->second;
            if (pos >= observed_means.n_elem) {
                throw std::runtime_error("Observed outcome mean index exceeds bounds.");
            }
            return observed_means[pos];
        };

        auto compute_weighted_avg = [&](const arma::uvec& outcome_indices,
                                        bool observed_only,
                                        bool use_observed_means_for_row) {
            double weight_sum = 0.0;
            double weighted_total = 0.0;
            for (std::size_t c = 0; c < C; ++c) {
                arma::uvec indices_for_row;
                if (observed_only) {
                    indices_for_row = project_observed_subset(obs_sets[c], outcome_indices);
                    if (indices_for_row.n_elem == 0) continue;
                } else {
                    indices_for_row = outcome_indices;
                }

                const bool use_observed_means_here = observed_only && use_observed_means_for_row;
                double row_outcome_weight_sum = 0.0;
                double row_outcome_weighted_total = 0.0;
                for (arma::uword i = 0; i < indices_for_row.n_elem; ++i) {
                    const arma::uword idx = indices_for_row[i];
                    const double outcome_w = weights[idx];
                    row_outcome_weight_sum += outcome_w;
                    const double measurement = use_observed_means_here
                        ? read_observed_mean(c, idx)
                        : Y(static_cast<arma::uword>(c), idx);
                    row_outcome_weighted_total += outcome_w * measurement;
                }
                if (row_outcome_weight_sum <= 0.0) continue;

                const double row_weight = stats_all[c].cohort_pop_share;
                weight_sum += row_weight * row_outcome_weight_sum;
                weighted_total += row_weight * row_outcome_weighted_total;
            }
            return (weight_sum > 0.0) ? (weighted_total / weight_sum) : 0.0;
        };

        const double obs_diff = compute_weighted_avg(
                                    outcome_indices_1,
                                    /*observed_only=*/true,
                                    use_observed_outcome_means) -
                                compute_weighted_avg(
                                    outcome_indices_2,
                                    /*observed_only=*/true,
                                    use_observed_outcome_means);
        const double pop_diff = compute_weighted_avg(
                                    outcome_indices_1,
                                    /*observed_only=*/false,
                                    /*use_observed_means_for_row=*/false) -
                                compute_weighted_avg(
                                    outcome_indices_2,
                                    /*observed_only=*/false,
                                    /*use_observed_means_for_row=*/false);
        const double ratio = (std::abs(obs_diff) > 1e-15) ? (pop_diff / obs_diff) : 0.0;
        return arma::vec({ratio, 1.0 - ratio});
    };
}

TargetFn get_fgw_bipartite_match_outcome_diff_params_fn(
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    bool use_observed_outcome_means)
{
    arma::uvec outcome_indices_1({static_cast<arma::uword>(outcome_idx_1)});
    arma::uvec outcome_indices_2({static_cast<arma::uword>(outcome_idx_2)});
    return get_fgw_bipartite_match_outcome_diff_params_fn(
        outcome_indices_1,
        outcome_indices_2,
        observed_outcome_indices,
        std::nullopt,
        use_observed_outcome_means);
}

TargetParameterEstimates est_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices_1,
    const arma::uvec& outcome_indices_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights,
    bool use_observed_outcome_means,
    std::optional<std::size_t> num_threads)
{
    TargetFn fn = get_fgw_bipartite_match_outcome_diff_params_fn(
        outcome_indices_1,
        outcome_indices_2,
        observed_outcome_indices,
        std::move(outcome_weights),
        use_observed_outcome_means);
    const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
    return est_target_params(ome, stats_by_cohort, empty_eta, fn, num_threads);
}

TargetParameterEstimates est_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    bool use_observed_outcome_means,
    std::optional<std::size_t> num_threads)
{
    arma::uvec outcome_indices_1({static_cast<arma::uword>(outcome_idx_1)});
    arma::uvec outcome_indices_2({static_cast<arma::uword>(outcome_idx_2)});
    return est_fgw_bipartite_match_outcome_diff_params(
        ome,
        stats_by_cohort,
        outcome_indices_1,
        outcome_indices_2,
        observed_outcome_indices,
        std::nullopt,
        use_observed_outcome_means,
        num_threads);
}

std::unordered_map<std::string, TargetParameterEstimates> est_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices_1,
    const arma::uvec& outcome_indices_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights,
    bool use_observed_outcome_means,
    std::optional<std::size_t> num_threads)
{
    TargetFn fn = get_fgw_bipartite_match_outcome_diff_params_fn(
        outcome_indices_1,
        outcome_indices_2,
        observed_outcome_indices,
        std::move(outcome_weights),
        use_observed_outcome_means);
    std::unordered_map<std::string, TargetParameterEstimates> out;
    out.reserve(ome_map.size());
    const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
    for (const auto& kv : ome_map) {
        out.emplace(
            kv.first,
            est_target_params(kv.second, stats_by_cohort, empty_eta, fn, num_threads));
    }
    return out;
}

std::unordered_map<std::string, TargetParameterEstimates> est_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    std::size_t outcome_idx_1,
    std::size_t outcome_idx_2,
    const ObservedOutcomeIndices& observed_outcome_indices,
    bool use_observed_outcome_means,
    std::optional<std::size_t> num_threads)
{
    arma::uvec outcome_indices_1({static_cast<arma::uword>(outcome_idx_1)});
    arma::uvec outcome_indices_2({static_cast<arma::uword>(outcome_idx_2)});
    return est_fgw_bipartite_match_outcome_diff_params(
        ome_map,
        stats_by_cohort,
        outcome_indices_1,
        outcome_indices_2,
        observed_outcome_indices,
        std::nullopt,
        use_observed_outcome_means,
        num_threads);
}

TargetFn get_avg_fgw_bipartite_match_outcome_diff_params_fn(
    const std::vector<arma::uvec>& outcome_groupings,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights,
    bool use_observed_outcome_means)
{
    if (outcome_groupings.size() < 2) {
        throw std::invalid_argument("outcome_groupings must contain at least two groups.");
    }

    const std::vector<std::unordered_set<std::size_t>> obs_sets = build_observed_sets(observed_outcome_indices);
    const arma::vec weights = build_outcome_weights(observed_outcome_indices, std::move(outcome_weights));
    const std::size_t total_outcomes = static_cast<std::size_t>(weights.n_elem);

    std::vector<double> group_weight_sums;
    group_weight_sums.reserve(outcome_groupings.size());
    for (std::size_t g = 0; g < outcome_groupings.size(); ++g) {
        const std::string arg_name = "outcome_groupings[" + std::to_string(g) + "]";
        validate_outcome_indices(outcome_groupings[g], obs_sets, total_outcomes, weights, arg_name.c_str());
        group_weight_sums.push_back(subset_weight_sum(weights, outcome_groupings[g]));
    }

    struct PairFnData {
        TargetFn fn;
        double weight;
    };

    std::vector<PairFnData> pair_fns;
    pair_fns.reserve(outcome_groupings.size() * (outcome_groupings.size() - 1) / 2);
    double total_pair_weight = 0.0;
    for (std::size_t i = 0; i < outcome_groupings.size(); ++i) {
        for (std::size_t j = i + 1; j < outcome_groupings.size(); ++j) {
            const double pair_weight = group_weight_sums[i] + group_weight_sums[j];
            if (pair_weight <= 0.0) continue;
            pair_fns.push_back(PairFnData{
                get_fgw_bipartite_match_outcome_diff_params_fn(
                    outcome_groupings[i],
                    outcome_groupings[j],
                    observed_outcome_indices,
                    weights,
                    use_observed_outcome_means),
                pair_weight});
            total_pair_weight += pair_weight;
        }
    }

    if (pair_fns.empty() || total_pair_weight <= 0.0) {
        throw std::invalid_argument("outcome_groupings must yield at least one pair with positive total weight.");
    }

    return [pair_fns = std::move(pair_fns), total_pair_weight](
               const arma::mat& Y,
               const std::vector<OutcomeMeanSufficientStatistics>& stats_all,
               const std::vector<CohortAuxiliaryDataMeans>& eta_all) -> arma::vec {
        arma::vec weighted_sum(2, arma::fill::zeros);
        for (const auto& pair_data : pair_fns) {
            const arma::vec pair_value = pair_data.fn(Y, stats_all, eta_all);
            if (pair_value.n_elem != 2) {
                throw std::runtime_error("Pair function returned unexpected parameter length.");
            }
            weighted_sum += pair_data.weight * pair_value;
        }
        return weighted_sum / total_pair_weight;
    };
}

TargetFn get_avg_fgw_bipartite_match_outcome_diff_params_fn(
    const arma::uvec& outcome_indices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights,
    bool use_observed_outcome_means)
{
    if (outcome_indices.n_elem < 2) {
        throw std::invalid_argument("outcome_indices must contain at least two elements.");
    }
    std::vector<arma::uvec> groupings;
    groupings.reserve(outcome_indices.n_elem);
    for (arma::uword i = 0; i < outcome_indices.n_elem; ++i) {
        arma::uvec group(1);
        group[0] = outcome_indices[i];
        groupings.push_back(std::move(group));
    }
    return get_avg_fgw_bipartite_match_outcome_diff_params_fn(
        groupings,
        observed_outcome_indices,
        std::move(outcome_weights),
        use_observed_outcome_means);
}

TargetParameterEstimates est_avg_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<arma::uvec>& outcome_groupings,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights,
    bool use_observed_outcome_means,
    std::optional<std::size_t> num_threads)
{
    TargetFn fn = get_avg_fgw_bipartite_match_outcome_diff_params_fn(
        outcome_groupings,
        observed_outcome_indices,
        std::move(outcome_weights),
        use_observed_outcome_means);
    const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
    return est_target_params(ome, stats_by_cohort, empty_eta, fn, num_threads);
}

TargetParameterEstimates est_avg_fgw_bipartite_match_outcome_diff_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    bool use_observed_outcome_means,
    std::optional<std::size_t> num_threads)
{
    TargetFn fn = get_avg_fgw_bipartite_match_outcome_diff_params_fn(
        outcome_indices,
        observed_outcome_indices,
        std::nullopt,
        use_observed_outcome_means);
    const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
    return est_target_params(ome, stats_by_cohort, empty_eta, fn, num_threads);
}

std::unordered_map<std::string, TargetParameterEstimates> est_avg_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<arma::uvec>& outcome_groupings,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::optional<arma::vec> outcome_weights,
    bool use_observed_outcome_means,
    std::optional<std::size_t> num_threads)
{
    TargetFn fn = get_avg_fgw_bipartite_match_outcome_diff_params_fn(
        outcome_groupings,
        observed_outcome_indices,
        std::move(outcome_weights),
        use_observed_outcome_means);
    std::unordered_map<std::string, TargetParameterEstimates> out;
    out.reserve(ome_map.size());
    const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
    for (const auto& kv : ome_map) {
        out.emplace(
            kv.first,
            est_target_params(kv.second, stats_by_cohort, empty_eta, fn, num_threads));
    }
    return out;
}

std::unordered_map<std::string, TargetParameterEstimates> est_avg_fgw_bipartite_match_outcome_diff_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const arma::uvec& outcome_indices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    bool use_observed_outcome_means,
    std::optional<std::size_t> num_threads)
{
    TargetFn fn = get_avg_fgw_bipartite_match_outcome_diff_params_fn(
        outcome_indices,
        observed_outcome_indices,
        std::nullopt,
        use_observed_outcome_means);
    std::unordered_map<std::string, TargetParameterEstimates> out;
    out.reserve(ome_map.size());
    const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
    for (const auto& kv : ome_map) {
        out.emplace(
            kv.first,
            est_target_params(kv.second, stats_by_cohort, empty_eta, fn, num_threads));
    }
    return out;
}

} // namespace apm