#include "summarize_identification.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "apm_core.h" // o3_algorithm

#include <set>
#include <limits>
#include <unordered_map>

namespace apm {

namespace {

struct LargestSuperCohortStats {
    std::size_t total_size;
    std::size_t min_cohort_size;
};

static LargestSuperCohortStats largest_final_super_cohort_stats(
    const std::vector<std::set<arma::uword>>& final_level,
    const arma::uvec& combined_sizes)
{
    std::size_t best_total = 0;
    std::size_t best_min = 0;
    for (const auto& super_set : final_level) {
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

static std::pair<ObservedOutcomeIndices, arma::uvec>
get_new_cohorts_from_combining_outcomes(
    const ObservedOutcomeIndices& ooi,
    const arma::uvec& cohort_sizes,
    const std::unordered_map<int,int>& old_to_new_outcome);

static OutcomeClusteringSummary summarize_single_mapping(
    const InMemoryUnbalancedPanel& panel,
    const arma::uvec& mapping,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank)
{
    const std::size_t T = panel.T();
    std::unordered_map<int,int> old_to_new_outcome;
    old_to_new_outcome.reserve(T);
    for (std::size_t t = 0; t < T; ++t) {
        old_to_new_outcome.emplace(static_cast<int>(t), static_cast<int>(mapping(static_cast<arma::uword>(t))));
    }
    auto combined = get_new_cohorts_from_combining_outcomes(panel.observed_outcome_indices(), cohort_sizes, old_to_new_outcome);
    const ObservedOutcomeIndices& combined_ooi = combined.first;
    const arma::uvec& combined_sizes = combined.second;
    auto super_cohort_iterates = o3_algorithm(combined_ooi, static_cast<unsigned int>(max_model_rank));
    const auto& final_level = super_cohort_iterates.back();
    auto stats = largest_final_super_cohort_stats(final_level, combined_sizes);
    double share = (panel.num_units() == 0) ? 0.0 : static_cast<double>(stats.total_size) / static_cast<double>(panel.num_units());
    return OutcomeClusteringSummary{stats.total_size, share, stats.min_cohort_size, super_cohort_iterates.size()};
}

} // anonymous namespace

std::vector<OutcomeClusteringSummary>
summarize_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    const std::vector<arma::uvec>& mappings,
    std::size_t max_model_rank)
{
    arma::uvec cohort_sizes = panel.get_cohort_sizes();
    std::vector<OutcomeClusteringSummary> summaries;
    summaries.reserve(mappings.size());
    for (const auto& mapping : mappings) {
        summaries.push_back(summarize_single_mapping(panel, mapping, cohort_sizes, max_model_rank));
    }
    return summaries;
}

OutcomeClusteringSummary
summarize_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    const arma::uvec& mapping,
    std::size_t max_model_rank)
{
    return summarize_single_mapping(panel, mapping, panel.get_cohort_sizes(), max_model_rank);
}

} // namespace apm


