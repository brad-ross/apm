#include "summarize_identification.h"
#include "apm_core.h" // o3_algorithm

#include <set>
#include <limits>
#include <unordered_set>
#include <stdexcept>
#include <vector>

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

struct OutcomeUnionStats {
    std::size_t count;
    double weight_sum;
};

arma::vec make_default_outcome_weights(const ObservedOutcomeIndices& observed_outcome_indices)
{
    const arma::uword total_outcomes_u = num_outcomes(observed_outcome_indices);
    arma::vec weights(total_outcomes_u, arma::fill::ones);
    return weights;
}

OutcomeUnionStats outcome_union_stats_for_super_cohort(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& super_cohort_indices,
    const arma::vec& outcome_weights,
    arma::uword total_outcomes_u)
{
    if (outcome_weights.n_elem != total_outcomes_u) {
        throw std::invalid_argument("outcome_weights length must equal number of outcomes");
    }

    const std::size_t total_outcomes = static_cast<std::size_t>(total_outcomes_u);
    if (total_outcomes == 0 || super_cohort_indices.n_elem == 0) {
        return {0u, 0.0};
    }

    std::vector<char> seen(total_outcomes, 0);
    std::size_t union_count = 0;
    double union_weight_sum = 0.0;
    for (arma::uword cohort_idx : super_cohort_indices) {
        const arma::uvec& outcomes = observed_outcome_indices[static_cast<std::size_t>(cohort_idx)];
        for (arma::uword outcome : outcomes) {
            if (outcome >= total_outcomes_u) {
                continue;
            }
            if (!seen[outcome]) {
                seen[outcome] = 1;
                ++union_count;
                union_weight_sum += outcome_weights(static_cast<arma::uword>(outcome));
            }
        }
    }

    return {union_count, union_weight_sum};
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
    arma::vec default_weights = make_default_outcome_weights(observed_outcome_indices);
    return summarize_identification(
        observed_outcome_indices,
        cohort_sizes,
        max_model_rank,
        iter,
        default_weights);
}

IdentificationSummary summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    int iter,
    const arma::vec& outcome_weights)
{
    const arma::uword total_outcomes_u = num_outcomes(observed_outcome_indices);
    if (outcome_weights.n_elem != total_outcomes_u) {
        throw std::invalid_argument("summarize_identification: outcome_weights length must equal number of outcomes");
    }

    arma::uvec largest_super = get_largest_super_cohort(
        observed_outcome_indices, cohort_sizes, max_model_rank, iter);

    auto stats = super_cohort_stats(largest_super, cohort_sizes);

    OutcomeUnionStats union_stats = outcome_union_stats_for_super_cohort(
        observed_outcome_indices,
        largest_super,
        outcome_weights,
        total_outcomes_u);

    auto super_cohort_iterates = o3_algorithm(observed_outcome_indices, static_cast<unsigned int>(max_model_rank));
    if (super_cohort_iterates.empty()) {
        return IdentificationSummary{0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0};
    }

    const arma::uword total_units_u = arma::accu(cohort_sizes);
    const std::size_t total_outcomes = static_cast<std::size_t>(total_outcomes_u);
    const double total_outcome_weight = arma::accu(outcome_weights);

    const double share_units = (total_units_u == 0)
        ? 0.0
        : static_cast<double>(stats.total_size) / static_cast<double>(total_units_u);
    const double share_outcomes = (total_outcomes == 0)
        ? 0.0
        : static_cast<double>(union_stats.count) / static_cast<double>(total_outcomes);
    const double share_outcome_weight = (total_outcome_weight == 0.0)
        ? 0.0
        : union_stats.weight_sum / total_outcome_weight;

    return IdentificationSummary{
        stats.total_size,
        share_units,
        stats.min_cohort_size,
        union_stats.count,
        union_stats.weight_sum,
        share_outcomes,
        share_outcome_weight,
        super_cohort_iterates.size()};
}

IdentificationSummary summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank)
{
    return summarize_identification(observed_outcome_indices, cohort_sizes, max_model_rank, -1);
}

IdentificationSummary summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    const arma::vec& outcome_weights)
{
    return summarize_identification(
        observed_outcome_indices,
        cohort_sizes,
        max_model_rank,
        -1,
        outcome_weights);
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
    const std::size_t n = observed_outcome_indices_vec.size();
    std::vector<arma::vec> default_weights;
    default_weights.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        default_weights.push_back(make_default_outcome_weights(observed_outcome_indices_vec[i]));
    }
    return summarize_identification(
        observed_outcome_indices_vec,
        cohort_sizes_vec,
        max_model_rank,
        iter,
        default_weights);
}

std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank,
    int iter,
    const std::vector<arma::vec>& outcome_weights_vec)
{
    if (observed_outcome_indices_vec.size() != cohort_sizes_vec.size()) {
        throw std::invalid_argument("summarize_identification: inputs must have the same length");
    }
    if (observed_outcome_indices_vec.size() != outcome_weights_vec.size()) {
        throw std::invalid_argument("summarize_identification: outcome_weights_vec must have the same length as observed_outcome_indices_vec");
    }

    std::vector<IdentificationSummary> summaries;
    const std::size_t n = observed_outcome_indices_vec.size();
    summaries.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        summaries.push_back(summarize_identification(
            observed_outcome_indices_vec[i],
            cohort_sizes_vec[i],
            max_model_rank,
            iter,
            outcome_weights_vec[i]));
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

std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank,
    const std::vector<arma::vec>& outcome_weights_vec)
{
    return summarize_identification(
        observed_outcome_indices_vec,
        cohort_sizes_vec,
        max_model_rank,
        -1,
        outcome_weights_vec);
}

bool aligned_factors_identified(
    const ObservedOutcomeIndices& observed_outcome_indices,
    unsigned int r)
{
    const arma::uword C = observed_outcome_indices.size();
    if (C <= 1) return true;

    auto super_cohort_iterations = o3_algorithm(observed_outcome_indices, r);
    const auto& final_super_cohorts = super_cohort_iterations.back();
    return final_super_cohorts.size() == 1 && final_super_cohorts[0].size() == C;
}

arma::vec
count_outcomes_with_rank_overlap_per_cohort(
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::size_t rank)
{
    arma::vec default_weights = make_default_outcome_weights(observed_outcome_indices);
    return count_outcomes_with_rank_overlap_per_cohort(
        observed_outcome_indices,
        rank,
        default_weights);
}

arma::vec
count_outcomes_with_rank_overlap_per_cohort(
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::size_t rank,
    const arma::vec& outcome_weights)
{
    const std::size_t C = observed_outcome_indices.size();

    arma::vec counts(static_cast<arma::uword>(C), arma::fill::zeros);
    if (C == 0) {
        return counts;
    }

    const arma::uword total_outcomes_u = num_outcomes(observed_outcome_indices);
    if (outcome_weights.n_elem != total_outcomes_u) {
        throw std::invalid_argument("count_outcomes_with_rank_overlap_per_cohort: outcome_weights length must equal number of outcomes");
    }
    const std::size_t total_outcomes = static_cast<std::size_t>(total_outcomes_u);

    std::vector<std::unordered_set<arma::uword>> membership_sets;
    membership_sets.reserve(C);
    for (const auto& outcomes : observed_outcome_indices) {
        std::unordered_set<arma::uword> s;
        s.reserve(outcomes.n_elem);
        for (arma::uword v : outcomes) {
            s.insert(v);
        }
        membership_sets.push_back(std::move(s));
    }

    std::vector<char> union_mask(total_outcomes, 0);
    std::vector<arma::uword> touched;
    touched.reserve(total_outcomes);

    for (std::size_t i = 0; i < C; ++i) {
        double total_weight = 0.0;
        const arma::uvec& focal = observed_outcome_indices[i];
        for (std::size_t j = 0; j < C; ++j) {
            const arma::uvec& comparison = observed_outcome_indices[j];
            if (comparison.n_elem == 0) {
                if (j != i) {
                    continue;
                }
                // Focal cohort contributes nothing but still considered.
            }

            bool include_cohort = (i == j) || (rank == 0);
            if (!include_cohort) {
                const arma::uvec* smaller_vec = &focal;
                const std::unordered_set<arma::uword>* other_set = &membership_sets[j];
                bool focal_is_smaller = (focal.n_elem <= comparison.n_elem);
                if (!focal_is_smaller) {
                    smaller_vec = &comparison;
                    other_set = &membership_sets[i];
                }

                std::size_t overlap = 0;
                for (arma::uword val : *smaller_vec) {
                    if (other_set->find(val) != other_set->end()) {
                        ++overlap;
                        if (overlap >= rank) {
                            break;
                        }
                    }
                }
                include_cohort = (overlap >= rank);
            }

            if (!include_cohort) {
                continue;
            }

            for (arma::uword outcome : comparison) {
                if (outcome >= total_outcomes_u) {
                    continue;
                }
                if (!union_mask[outcome]) {
                    union_mask[outcome] = 1;
                    touched.push_back(outcome);
                    total_weight += outcome_weights(static_cast<arma::uword>(outcome));
                }
            }
        }

        counts(static_cast<arma::uword>(i)) = total_weight;
        for (arma::uword idx : touched) {
            union_mask[idx] = 0;
        }
        touched.clear();
    }

    return counts;
}

} // namespace apm