#include "summarize_identification.h"
#include "apm_core.h" // o3_algorithm

#include <set>
#include <limits>
#include <unordered_map>
#include <stdexcept>

namespace apm {

namespace {

struct SuperCohortStats {
    std::size_t total_size;
    std::size_t min_cohort_size;
};

// Compute stats for a single super cohort represented by cohort indices.
SuperCohortStats super_cohort_stats(
    const arma::uvec& super_cohort_indices,
    const arma::uvec& combined_sizes)
{
    std::size_t total = 0;
    std::size_t min_sz = std::numeric_limits<std::size_t>::max();
    for (arma::uword idx : super_cohort_indices) {
        const std::size_t v = static_cast<std::size_t>(combined_sizes(idx));
        total += v;
        if (v < min_sz) min_sz = v;
    }
    return { total, (min_sz == std::numeric_limits<std::size_t>::max()) ? 0 : min_sz };
}

} // anonymous namespace

arma::uvec
get_largest_super_cohort(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    int iter)
{
    auto super_cohort_iterates = o3_algorithm(observed_outcome_indices, static_cast<unsigned int>(max_model_rank));
    if (super_cohort_iterates.empty()) {
        return arma::uvec();
    }

    const std::size_t n_iters = super_cohort_iterates.size();
    std::size_t level_idx = 0;
    if (iter >= 0) {
        const std::size_t pos = static_cast<std::size_t>(iter);
        level_idx = (pos >= n_iters) ? (n_iters - 1) : pos;
    } else {
        const std::size_t back = static_cast<std::size_t>(-iter);
        if (back >= n_iters) {
            level_idx = 0;
        } else {
            level_idx = n_iters - back;
        }
    }

    const auto& level = super_cohort_iterates[level_idx];
    if (level.empty()) {
        return arma::uvec();
    }

    std::size_t best_total = 0;
    const std::set<arma::uword>* best_set = nullptr;
    for (const auto& super_set : level) {
        std::size_t total = 0;
        for (arma::uword idx : super_set) {
            total += static_cast<std::size_t>(cohort_sizes(idx));
        }
        if (best_set == nullptr || total > best_total) {
            best_total = total;
            best_set = &super_set;
        }
    }

    if (best_set == nullptr) {
        return arma::uvec();
    }

    arma::uvec result(best_set->size());
    std::size_t i = 0;
    for (arma::uword idx : *best_set) {
        result(i++) = idx;
    }
    return result;
}

arma::uvec
get_largest_super_cohort(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank)
{
    return get_largest_super_cohort(observed_outcome_indices, cohort_sizes, max_model_rank, -1);
}

IdentificationSummary summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    int iter)
{
    arma::uvec largest_super = get_largest_super_cohort(
        observed_outcome_indices, cohort_sizes, max_model_rank, iter);

    auto stats = super_cohort_stats(largest_super, cohort_sizes);

    auto super_cohort_iterates = o3_algorithm(observed_outcome_indices, static_cast<unsigned int>(max_model_rank));
    if (super_cohort_iterates.empty()) {
        return IdentificationSummary{0, 0.0, 0, 0};
    }
    const arma::uword total_units_u = arma::accu(cohort_sizes);
    double share = (total_units_u == 0) ? 0.0 : static_cast<double>(stats.total_size) / static_cast<double>(total_units_u);
    return IdentificationSummary{stats.total_size, share, stats.min_cohort_size, super_cohort_iterates.size()};
}

IdentificationSummary summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank)
{
    return summarize_identification(observed_outcome_indices, cohort_sizes, max_model_rank, -1);
}

std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank,
    int iter)
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
            max_model_rank,
            iter));
    }
    return summaries;
}

std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank)
{
    return summarize_identification(observed_outcome_indices_vec, cohort_sizes_vec, max_model_rank, -1);
}

} // namespace apm