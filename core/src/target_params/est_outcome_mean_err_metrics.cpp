#include "est_outcome_mean_err_metrics.h"

#include <armadillo>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace apm {

namespace {

void ensure_bootstrap_available(const TargetParamComponents& components) {
    std::vector<std::string> specs_without_bootstrap;
    for (const auto& kv : components.outcome_means_by_spec) {
        if (!kv.second.has_bootstrap_replicates()) {
            specs_without_bootstrap.push_back(kv.first);
        }
    }

    if (!specs_without_bootstrap.empty()) {
        std::string msg = "est_masked_outcome_mean_err_metrics: missing bootstrap replicates for specs: ";
        for (std::size_t i = 0; i < specs_without_bootstrap.size(); ++i) {
            msg += specs_without_bootstrap[i];
            if (i + 1 < specs_without_bootstrap.size()) msg += ", ";
        }
        throw std::invalid_argument(msg);
    }
}

void validate_component_dimensions(const TargetParamComponents& components, std::size_t cohort_index) {
    const auto cohort_rows = components.outcome_means_by_spec.begin()->second.mean_outcomes.n_rows;
    if (cohort_index >= cohort_rows) {
        throw std::invalid_argument("est_masked_outcome_mean_err_metrics: cohort index out of range for outcome means");
    }

    if (cohort_index >= components.cohort_outcome_mean_ests.size()) {
        throw std::invalid_argument("est_masked_outcome_mean_err_metrics: cohort index out of range for cohort_outcome_mean_ests");
    }
}

} // namespace

std::unordered_map<CohortOutcomeIndex, OutcomeMeanErrMetrics, CohortOutcomeIndexHash>
est_masked_outcome_mean_err_metrics(const TargetParamComponents& components) {
    if (!components.cohort_outcome_mask.has_value() || components.cohort_outcome_mask->empty()) {
        throw std::invalid_argument("est_masked_outcome_mean_err_metrics: cohort_outcome_mask must be provided and non-empty");
    }

    if (components.outcome_means_by_spec.empty()) {
        throw std::invalid_argument("est_masked_outcome_mean_err_metrics: outcome_means_by_spec is empty");
    }

    ensure_bootstrap_available(components);

    const CohortOutcomeMask& mask = *components.cohort_outcome_mask;

    std::size_t total_pairs = 0;
    for (const auto& kv : mask) {
        total_pairs += static_cast<std::size_t>(kv.second.n_elem);
    }

    std::unordered_map<CohortOutcomeIndex, OutcomeMeanErrMetrics, CohortOutcomeIndexHash> metrics_map;
    metrics_map.reserve(total_pairs);

    for (const auto& mask_entry : mask) {
        const int cohort_id = mask_entry.first;
        if (cohort_id < 0) {
            throw std::invalid_argument("est_masked_outcome_mean_err_metrics: cohort ids in cohort_outcome_mask must be non-negative");
        }
        const std::size_t cohort_idx = static_cast<std::size_t>(cohort_id);

        validate_component_dimensions(components, cohort_idx);

        const arma::uvec& masked_outcome_indices = mask_entry.second;
        if (masked_outcome_indices.n_elem == 0) {
            continue; // nothing to compute for this cohort
        }

        auto masked_mean_it = components.masked_cohort_outcome_means.find(cohort_id);
        if (masked_mean_it == components.masked_cohort_outcome_means.end()) {
            throw std::invalid_argument("est_masked_outcome_mean_err_metrics: masked outcome means missing for cohort id " + std::to_string(cohort_id));
        }
        const OutcomeMeanSufficientStatistics& masked_stats = masked_mean_it->second;
        if (masked_stats.observed_outcome_means.n_elem != masked_outcome_indices.n_elem) {
            throw std::invalid_argument("est_masked_outcome_mean_err_metrics: mismatch between mask length and masked outcome means length for cohort id " + std::to_string(cohort_id));
        }

        const double cohort_share = components.cohort_outcome_mean_ests[cohort_idx].suff_stat_estimates.cohort_pop_share;

        for (arma::uword local_idx = 0; local_idx < masked_outcome_indices.n_elem; ++local_idx) {
            const arma::uword outcome_absolute_idx = masked_outcome_indices[local_idx];
            const std::size_t outcome_idx = static_cast<std::size_t>(outcome_absolute_idx);
            const double truth = masked_stats.observed_outcome_means[local_idx];

            OutcomeMeanErrMetrics metrics;
            metrics.cohort_pop_share = cohort_share;
            metrics.bias_by_spec.reserve(components.outcome_means_by_spec.size());
            metrics.se_by_spec.reserve(components.outcome_means_by_spec.size());
            metrics.rmse_by_spec.reserve(components.outcome_means_by_spec.size());

            for (const auto& spec_entry : components.outcome_means_by_spec) {
                const std::string& spec_name = spec_entry.first;
                const OutcomeMeansEstimates& ome = spec_entry.second;

                if (cohort_idx >= static_cast<std::size_t>(ome.mean_outcomes.n_rows)) {
                    throw std::invalid_argument("est_masked_outcome_mean_err_metrics: cohort index out of range for spec '" + spec_name + "'");
                }
                if (outcome_idx >= static_cast<std::size_t>(ome.mean_outcomes.n_cols)) {
                    throw std::invalid_argument("est_masked_outcome_mean_err_metrics: outcome index out of range for spec '" + spec_name + "'");
                }

                const std::size_t B = ome.n_bootstrap_replicates();
                if (B == 0) {
                    throw std::logic_error("est_masked_outcome_mean_err_metrics: internal error, missing bootstrap replicates for spec '" + spec_name + "'");
                }

                arma::vec draws(static_cast<arma::uword>(B));
                for (std::size_t b = 0; b < B; ++b) {
                    const arma::mat& boot_mat = ome.bootstrap_replicates[b];
                    if (cohort_idx >= static_cast<std::size_t>(boot_mat.n_rows) || outcome_idx >= static_cast<std::size_t>(boot_mat.n_cols)) {
                        throw std::invalid_argument("est_masked_outcome_mean_err_metrics: bootstrap replicate dimensions do not match for spec '" + spec_name + "'");
                    }
                    draws(static_cast<arma::uword>(b)) = boot_mat(static_cast<arma::uword>(cohort_idx), static_cast<arma::uword>(outcome_idx));
                }

                const double mean_est = arma::mean(draws);
                const double bias = mean_est - truth;
                const double variance = (B > 1) ? arma::var(draws) : std::numeric_limits<double>::quiet_NaN();
                const double se = (B > 1) ? std::sqrt(variance) : std::numeric_limits<double>::quiet_NaN();
                const arma::vec diff = draws - truth;
                const double rmse = std::sqrt(arma::dot(diff, diff) / static_cast<double>(B));

                metrics.bias_by_spec.emplace(spec_name, bias);
                metrics.se_by_spec.emplace(spec_name, se);
                metrics.rmse_by_spec.emplace(spec_name, rmse);
            }

            metrics_map.emplace(CohortOutcomeIndex{cohort_idx, outcome_idx}, std::move(metrics));
        }
    }

    return metrics_map;
}

} // namespace apm