#include "utils.h"
#include <unordered_set>

namespace apm {

arma::uword num_outcomes(const ObservedOutcomeIndices& observed_outcome_indices) {
    arma::uword max_idx = 0;
    bool has_observations = false;
    for (const auto& T_c : observed_outcome_indices) {
        if (!T_c.empty()) {
            has_observations = true;
            max_idx = std::max(max_idx, T_c.max());
        }
    }

    return has_observations ? static_cast<arma::uword>(max_idx + 1u) : 0u;
}

ObservedOutcomeIndices get_masked_observed_outcome_indices(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const CohortOutcomeMask& cohort_outcomes_to_mask)
{
    if (cohort_outcomes_to_mask.empty()) return observed_outcome_indices;

    ObservedOutcomeIndices ooi_effective = observed_outcome_indices;
    for (const auto& kv : cohort_outcomes_to_mask) {
        const int c = kv.first;
        if (c < 0 || static_cast<std::size_t>(c) >= ooi_effective.size()) continue;
        const arma::uvec& src = observed_outcome_indices[static_cast<std::size_t>(c)];
        const arma::uvec& to_drop = kv.second;
        std::unordered_set<arma::uword> drop;
        drop.reserve(to_drop.n_elem * 2);
        for (arma::uword j = 0; j < to_drop.n_elem; ++j) drop.insert(to_drop[j]);
        std::vector<arma::uword> kept;
        kept.reserve(src.n_elem);
        for (arma::uword j = 0; j < src.n_elem; ++j) {
            if (drop.find(src[j]) == drop.end()) kept.push_back(src[j]);
        }
        ooi_effective[static_cast<std::size_t>(c)] = arma::uvec(kept);
    }
    return ooi_effective;
}

} // namespace apm


