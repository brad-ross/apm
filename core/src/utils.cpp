#include "utils.h"

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

} // namespace apm


