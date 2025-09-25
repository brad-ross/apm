#include "est_target_params.h"

#include <algorithm>
#include <optional>
#include <memory>
#ifdef APM_HAS_TBB
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/global_control.h>
#endif

#include "../agg_cohort_specific_factor_model_params.h"
#include "../est_outcome_means.h"
#include "../panels/InMemoryUnbalancedPanel.h"

namespace apm {

static std::vector<CohortAuxiliaryDataMeans>
collect_eta_across_cohorts(const std::vector<CohortAuxiliaryDataMeanEstimates>& v,
                           std::optional<std::size_t> b_opt) {
    std::vector<CohortAuxiliaryDataMeans> out;
    out.reserve(v.size());
    for (const auto& e : v) {
        if (b_opt.has_value() && e.n_bootstrap_replicates() > *b_opt) out.push_back(e.bootstrap_replicates[*b_opt]);
        else out.push_back(e.estimates);
    }
    return out;
}

static std::vector<OutcomeMeanSufficientStatistics>
collect_stats_across_cohorts(const std::vector<OutcomeMeanSuffStatEstimates>& v,
                             std::optional<std::size_t> b_opt) {
    std::vector<OutcomeMeanSufficientStatistics> out;
    out.reserve(v.size());
    for (const auto& e : v) {
        if (b_opt.has_value() && e.n_bootstrap_replicates() > *b_opt) out.push_back(e.bootstrap_replicates[*b_opt]);
        else out.push_back(e.suff_stat_estimates);
    }
    return out;
}

TargetParameterEstimates est_target_params(
    const OutcomeMeansEstimates& ome,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<CohortAuxiliaryDataMeanEstimates>& eta_by_cohort,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads)
{
    const std::size_t B = ome.n_bootstrap_replicates();
    // Enforce equal number of bootstrap replicates across all inputs
    for (const auto& e : eta_by_cohort) {
        if (e.n_bootstrap_replicates() != B) {
            throw std::invalid_argument("All parameter estimates must have the same number of bootstrap replicates.");
        }
    }
    for (const auto& s : stats_by_cohort) {
        if (s.n_bootstrap_replicates() != B) {
            throw std::invalid_argument("All parameter estimates must have the same number of bootstrap replicates.");
        }
    }

    arma::vec point = fn(ome.mean_outcomes,
                         collect_stats_across_cohorts(stats_by_cohort, std::nullopt),
                         collect_eta_across_cohorts(eta_by_cohort, std::nullopt));

    std::vector<arma::vec> boots;
    if (B > 0) {
        boots.resize(B);
        auto process_boot = [&](std::size_t b) {
            boots[b] = fn(ome.bootstrap_replicates[b],
                          collect_stats_across_cohorts(stats_by_cohort, b),
                          collect_eta_across_cohorts(eta_by_cohort, b));
        };

#ifdef APM_HAS_TBB
        std::size_t nt = num_threads.has_value() ? *num_threads : oneapi::tbb::info::default_concurrency();
        std::unique_ptr<oneapi::tbb::global_control> tbb_gc;
        if (nt > 1) {
            tbb_gc = std::make_unique<oneapi::tbb::global_control>(
                oneapi::tbb::global_control::max_allowed_parallelism, nt);
            oneapi::tbb::parallel_for(std::size_t(0), B, [&](std::size_t b) { process_boot(b); });
        } else
#endif
        {
            for (std::size_t b = 0; b < B; ++b) process_boot(b);
        }
    }

    return TargetParameterEstimates(std::move(point), std::move(boots));
}

std::unordered_map<std::string, TargetParameterEstimates> est_target_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::unordered_map<std::string, std::vector<OutcomeMeanSuffStatEstimates>>& stats_map,
    const std::unordered_map<std::string, std::vector<CohortAuxiliaryDataMeanEstimates>>& eta_map,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads)
{
    std::unordered_map<std::string, TargetParameterEstimates> out;
    out.reserve(ome_map.size());
    for (const auto& kv : ome_map) {
        const std::string& key = kv.first;
        const OutcomeMeansEstimates& ome = kv.second;
        auto it_eta = eta_map.find(key);
        auto it_stats = stats_map.find(key);
        const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
        const std::vector<OutcomeMeanSuffStatEstimates> empty_stats;
        const auto& eta_vec = (it_eta == eta_map.end() ? empty_eta : it_eta->second);
        const auto& stats_vec = (it_stats == stats_map.end() ? empty_stats : it_stats->second);
        out.emplace(key, est_target_params(ome, stats_vec, eta_vec, fn, num_threads));
    }
    return out;
}

TargetParamComponents est_target_param_components_from_panel(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    std::shared_ptr<const WeightedBootstrap> bootstrap,
    std::optional<std::size_t> num_threads,
    const CohortOutcomeMask& cohort_outcomes_to_mask)
{
    // 1) Cohort-specific estimates (raw C++) via panel-based core
    CohortSpecificEstimates ests = estimate_cohort_specific_params_from_internal_panel_rep(
        panel,
        est_specs,
        bootstrap,
        num_threads,
        cohort_outcomes_to_mask);

    // 2) Observed outcome indices (prefer masked if present)
    const ObservedOutcomeIndices obs_idx_panel = panel.observed_outcome_indices();
    const ObservedOutcomeIndices& obs_idx_eff = ests.masked_observed_outcome_indices.has_value()
        ? *ests.masked_observed_outcome_indices
        : obs_idx_panel;

    // 3) Aggregate factor model params by spec (use obs_idx_eff)
    std::unordered_map<std::string, FactorModelEstimates> agg_by_spec =
        aggregate_cohort_specific_factor_model_params(
            ests.cohort_specific_factor_ests,
            obs_idx_eff,
            ests.cohort_weights);

    // 4) Estimate outcome means by spec (use obs_idx_eff)
    std::unordered_map<std::string, OutcomeMeansEstimates> ome_by_spec =
        estimate_outcome_means_across_cohorts(
            agg_by_spec,
            obs_idx_eff,
            ests.cohort_outcome_mean_ests);

    TargetParamComponents res;
    res.outcome_means_by_spec = std::move(ome_by_spec);
    res.cohort_outcome_mean_ests = std::move(ests.cohort_outcome_mean_ests);
    res.cohort_auxiliary_means = std::move(ests.cohort_auxiliary_means);
    res.masked_cohort_outcome_means = std::move(ests.masked_cohort_outcome_means);
    res.masked_observed_outcome_indices = std::move(ests.masked_observed_outcome_indices);
    return res;
}

} // namespace apm


