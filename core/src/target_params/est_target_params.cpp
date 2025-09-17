#include "est_target_params.h"

#include <algorithm>
#include <optional>
#include <memory>
#ifdef APM_HAS_TBB
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/global_control.h>
#endif

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

TargetParameterEstimates est_target_params(
    const OutcomeMeansEstimates& ome,
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

    arma::vec point = fn(ome.mean_outcomes, collect_eta_across_cohorts(eta_by_cohort, std::nullopt));

    std::vector<arma::vec> boots;
    if (B > 0) {
        boots.resize(B);
        auto process_boot = [&](std::size_t b) {
            boots[b] = fn(ome.bootstrap_replicates[b], collect_eta_across_cohorts(eta_by_cohort, b));
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
    const std::unordered_map<std::string, std::vector<CohortAuxiliaryDataMeanEstimates>>& eta_map,
    const TargetFn& fn,
    std::optional<std::size_t> num_threads)
{
    std::unordered_map<std::string, TargetParameterEstimates> out;
    out.reserve(ome_map.size());
    for (const auto& kv : ome_map) {
        const std::string& key = kv.first;
        const OutcomeMeansEstimates& ome = kv.second;
        auto it = eta_map.find(key);
        const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
        const auto& eta_vec = (it == eta_map.end() ? empty_eta : it->second);
        out.emplace(key, est_target_params(ome, eta_vec, fn, num_threads));
    }
    return out;
}

} // namespace apm


