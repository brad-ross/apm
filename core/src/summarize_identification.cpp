#include "summarize_identification.h"
#include "apm_core.h" // o3_algorithm

#include <set>
#include <limits>
#include <unordered_map>
#include <stdexcept>

namespace apm {

namespace {

struct LargestSuperCohortStats {
    std::size_t total_size;
    std::size_t min_cohort_size;
};

LargestSuperCohortStats largest_super_cohort_stats(
    const std::vector<std::set<arma::uword>>& o3_level_super_cohorts,
    const arma::uvec& combined_sizes)
{
    std::size_t best_total = 0;
    std::size_t best_min = 0;
    for (const auto& super_set : o3_level_super_cohorts) {
        std::size_t total = 0;
        std::size_t min_sz = std::numeric_limits<std::size_t>::max();
        for (arma::uword idx : super_set) {
            const std::size_t v = static_cast<std::size_t>(combined_sizes(idx));
            total += v;
            if (v < min_sz) min_sz = v;
        }
        if (total > best_total) {
            best_total = total;
            best_min = (min_sz == std::numeric_limits<std::size_t>::max()) ? 0 : min_sz;
        }
    }
    return {best_total, best_min};
}

} // anonymous namespace

IdentificationSummary summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank)
{
    auto super_cohort_iterates = o3_algorithm(observed_outcome_indices, static_cast<unsigned int>(max_model_rank));
    const auto& o3_level_super_cohorts = super_cohort_iterates.back();
    auto stats = largest_super_cohort_stats(o3_level_super_cohorts, cohort_sizes);
    const arma::uword total_units_u = arma::accu(cohort_sizes);
    double share = (total_units_u == 0) ? 0.0 : static_cast<double>(stats.total_size) / static_cast<double>(total_units_u);
    return IdentificationSummary{stats.total_size, share, stats.min_cohort_size, super_cohort_iterates.size()};
}

std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank)
{
    if (observed_outcome_indices_vec.size() != cohort_sizes_vec.size()) {
        throw std::invalid_argument("summarize_identification: inputs must have the same length");
    }
    std::vector<IdentificationSummary> summaries;
    const std::size_t n = observed_outcome_indices_vec.size();
    summaries.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        summaries.push_back(summarize_identification(
            observed_outcome_indices_vec[i],
            cohort_sizes_vec[i],
            max_model_rank));
    }
    return summaries;
}

} // namespace apm