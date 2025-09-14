#include "est_target_params.h"

#include <algorithm>

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
    const TargetFn& fn)
{
    const std::size_t B_ome = ome.n_bootstrap_replicates();
    std::size_t B_eta = 0;
    for (const auto& e : eta_by_cohort) B_eta = std::max(B_eta, e.n_bootstrap_replicates());
    const std::size_t B = std::max(B_ome, B_eta);

    arma::vec point = fn(ome.mean_outcomes, collect_eta_across_cohorts(eta_by_cohort, std::nullopt));

    std::vector<arma::vec> boots;
    if (B > 0) {
        boots.reserve(B);
        for (std::size_t b = 0; b < B; ++b) {
            const arma::mat& Y_b = (B_ome > 0 ? ome.bootstrap_replicates[b] : ome.mean_outcomes);
            boots.emplace_back(fn(Y_b, collect_eta_across_cohorts(eta_by_cohort, b)));
        }
    }

    return TargetParameterEstimates(std::move(point), std::move(boots));
}

std::unordered_map<std::string, TargetParameterEstimates> est_target_params(
    const std::unordered_map<std::string, OutcomeMeansEstimates>& ome_map,
    const std::unordered_map<std::string, std::vector<CohortAuxiliaryDataMeanEstimates>>& eta_map,
    const TargetFn& fn)
{
    std::unordered_map<std::string, TargetParameterEstimates> out;
    out.reserve(ome_map.size());
    for (const auto& kv : ome_map) {
        const std::string& key = kv.first;
        const OutcomeMeansEstimates& ome = kv.second;
        auto it = eta_map.find(key);
        const std::vector<CohortAuxiliaryDataMeanEstimates> empty_eta;
        const auto& eta_vec = (it == eta_map.end() ? empty_eta : it->second);
        out.emplace(key, est_target_params(ome, eta_vec, fn));
    }
    return out;
}

} // namespace apm


