#include "est_target_params.h"

#include <algorithm>
#include <optional>
#include <memory>
#ifdef APM_HAS_TBB
#include <oneapi/tbb/parallel_for.h>
#endif

#include "../agg_cohort_specific_factor_model_params.h"
#include "../est_outcome_means.h"
#include "../panels/InMemoryUnbalancedPanel.h"
#include "../utils.h"
#include "../bootstrap.h"

namespace apm {

namespace {
arma::mat pack_bootstrap_columns(const std::vector<arma::vec>& columns, std::size_t expected_p) {
    const std::size_t B = columns.size();
    arma::mat out;
    out.set_size(static_cast<arma::uword>(expected_p), static_cast<arma::uword>(B));
    for (std::size_t b = 0; b < B; ++b) {
        if (static_cast<std::size_t>(columns[b].n_elem) != expected_p) {
            throw std::runtime_error("Target function returned a vector of inconsistent length across bootstrap draws.");
        }
        out.col(static_cast<arma::uword>(b)) = columns[b];
    }
    return out;
}
} // anonymous namespace

TargetParameterEstimates::TargetParameterEstimates(arma::vec point_in, const std::vector<arma::vec>& boots_vec)
    : point(std::move(point_in))
{
    const std::size_t p = static_cast<std::size_t>(point.n_elem);
    if (boots_vec.empty()) {
        bootstrap_replicates.set_size(static_cast<arma::uword>(p), arma::uword(0));
    } else {
        bootstrap_replicates = pack_bootstrap_columns(boots_vec, p);
    }
}

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

    arma::mat boots_mat; // p x B (each column a bootstrap draw)
    const std::size_t p = static_cast<std::size_t>(point.n_elem);
    if (B == 0) {
        boots_mat.set_size(static_cast<arma::uword>(p), arma::uword(0));
    } else {
        // Compute each bootstrap replicate vector in parallel into a temporary container
        std::vector<arma::vec> tmp(B);
        auto process_boot = [&](std::size_t b) {
            tmp[b] = fn(ome.bootstrap_replicates[b],
                        collect_stats_across_cohorts(stats_by_cohort, b),
                        collect_eta_across_cohorts(eta_by_cohort, b));
        };

        {
            ParallelismScope pscope(num_threads);
#ifdef APM_HAS_TBB
            if (pscope.nt > 1) {
                oneapi::tbb::parallel_for(std::size_t(0), B, [&](std::size_t b) { process_boot(b); });
            } else
#endif
            {
                for (std::size_t b = 0; b < B; ++b) process_boot(b);
            }
        }

        boots_mat = pack_bootstrap_columns(tmp, p);
    }

    return TargetParameterEstimates(std::move(point), std::move(boots_mat));
}

std::unordered_map<std::string, TargetParameterEstimates> est_target_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& stats_by_cohort,
    const std::vector<CohortAuxiliaryDataMeanEstimates>& eta_by_cohort,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads)
{
    std::unordered_map<std::string, TargetParameterEstimates> out;
    out.reserve(ome_map.size());
    for (const auto& kv : ome_map) {
        out.emplace(
            kv.first,
            est_target_params(kv.second, stats_by_cohort, eta_by_cohort, fn, num_threads));
    }
    return out;
}

TargetParamComponents est_target_param_components_from_panel(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    std::shared_ptr<const WeightedBootstrap> bootstrap,
    std::optional<std::size_t> num_threads,
    const CohortOutcomeMask& cohort_outcomes_to_mask,
    bool est_outcome_means_via_imputation,
    const ImputationOptions& imputation_opts)
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

    // 4) Optionally compute imputation components by spec (use obs_idx_eff)
    std::unordered_map<std::string, FactorModelEstimates> params_for_means = agg_by_spec;
    if (est_outcome_means_via_imputation) {
        params_for_means = comp_imputation_components(
            panel,
            agg_by_spec,
            ests.cohort_outcome_mean_ests,
            bootstrap,
            std::optional<ObservedOutcomeIndices>{obs_idx_eff},
            imputation_opts,
            num_threads);
    }

    // 5) Estimate outcome means by spec (use obs_idx_eff)
    std::unordered_map<std::string, OutcomeMeansEstimates> ome_by_spec =
        estimate_outcome_means_across_cohorts(
            params_for_means,
            obs_idx_eff,
            ests.cohort_outcome_mean_ests);

    TargetParamComponents res;
    res.outcome_means_by_spec = std::move(ome_by_spec);
    res.cohort_outcome_mean_ests = std::move(ests.cohort_outcome_mean_ests);
    res.cohort_auxiliary_means = std::move(ests.cohort_auxiliary_means);
    res.masked_cohort_outcome_means = std::move(ests.masked_cohort_outcome_means);
    res.masked_observed_outcome_indices = std::move(ests.masked_observed_outcome_indices);
    res.cohort_outcome_mask = std::move(ests.cohort_outcome_mask);
    return res;
}

SimultaneousInferenceResults target_param_inference(
    const TargetParameterEstimates& ests,
    const InMemoryUnbalancedPanel& panel,
    double sig_level)
{
    const std::size_t N = panel.num_units();
    if (N == 0) {
        throw std::invalid_argument("target_param_inference: panel.num_units() must be > 0.");
    }
    if (!ests.has_bootstrap_replicates()) {
        throw std::invalid_argument("target_param_inference: bootstrap_replicates must have B>0 columns.");
    }
    return get_bootstrap_inference(ests.point, ests.bootstrap_replicates, N, sig_level);
}

std::unordered_map<std::string, SimultaneousInferenceResults> target_param_inference(
    const std::unordered_map<std::string, TargetParameterEstimates>& ests_by_spec,
    const InMemoryUnbalancedPanel& panel,
    double sig_level)
{
    const std::size_t N = panel.num_units();
    if (N == 0) {
        throw std::invalid_argument("target_param_inference: panel.num_units() must be > 0.");
    }

    std::unordered_map<std::string, SimultaneousInferenceResults> out;
    out.reserve(ests_by_spec.size());
    for (const auto& kv : ests_by_spec) {
        out.emplace(kv.first, target_param_inference(kv.second, panel, sig_level));
    }
    return out;
}

} // namespace apm


