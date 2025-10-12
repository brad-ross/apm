#include "cluster_outcomes.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "online_accumulators.h"
#include "apm_core.h"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include <set>

namespace apm {

namespace {
// Pass 1: collect observed outcome values and compute global interior-quantile grid
arma::vec compute_quantile_grid_from_panel(const InMemoryUnbalancedPanel& panel, std::size_t G) {
    std::vector<double> obs_y;
    obs_y.reserve(panel.num_units());
    arma::vec Y;
    for (const auto& blk : panel.cohort_blocks()) {
        const arma::uvec& T_idxs = panel.T_idx_for_cohort(blk.cohort);
        if (T_idxs.n_elem == 0) continue;
        const auto& pos_map = panel.pos_T_idx_for_cohort(blk.cohort);
        for (const auto& ur : blk.unit_runs) {
            panel.assemble_Y_for_unit(ur, T_idxs, pos_map, Y);
            for (arma::uword k = 0; k < Y.n_elem; ++k) {
                const double v = Y(k);
                if (std::isfinite(v)) obs_y.push_back(v);
            }
        }
    }
    if (obs_y.empty()) {
        throw std::invalid_argument("comp_outcome_dists: no observed outcomes to compute quantiles");
    }
    arma::vec probs(static_cast<arma::uword>(G));
    for (std::size_t g = 0; g < G; ++g) {
        probs(static_cast<arma::uword>(g)) = static_cast<double>(g + 1) / static_cast<double>(G + 1);
    }
    return arma::quantile(arma::vec(obs_y), probs);
}

// Pass 2: stream units and update per-outcome CDF rows via online means
void stream_update_outcome_cdfs(
    const InMemoryUnbalancedPanel& panel,
    const arma::vec& grid,
    arma::mat& outcome_val_cdfs,
    arma::uvec& outcome_counts)
{
    const std::size_t G = static_cast<std::size_t>(grid.n_elem);
    arma::vec Y;
    for (const auto& blk : panel.cohort_blocks()) {
        const arma::uvec& T_idxs = panel.T_idx_for_cohort(blk.cohort);
        if (T_idxs.n_elem == 0) continue;
        const auto& pos_map = panel.pos_T_idx_for_cohort(blk.cohort);
        for (const auto& ur : blk.unit_runs) {
            panel.assemble_Y_for_unit(ur, T_idxs, pos_map, Y);
            for (arma::uword k = 0; k < Y.n_elem; ++k) {
                const double v = Y(k);
                if (!std::isfinite(v)) continue;
                const arma::uword t_global = T_idxs(k);

                arma::mat Z(1, static_cast<arma::uword>(G));
                for (arma::uword j = 0; j < static_cast<arma::uword>(G); ++j) {
                    Z(0, j) = (v <= grid(j)) ? 1.0 : 0.0;
                }
                arma::vec w(1); w(0) = 1.0;
                arma::vec prev = outcome_val_cdfs.row(t_global).t();
                auto upd = apm::stats::online_weighted_mean(Z, w, prev, static_cast<double>(outcome_counts(t_global)));
                outcome_val_cdfs.row(t_global) = upd.first.t();
                outcome_counts(t_global) = static_cast<arma::uword>(upd.second);
            }
        }
    }
}

static std::pair<arma::mat, arma::uvec>
comp_outcome_dists(const InMemoryUnbalancedPanel& panel, std::size_t grid_size)
{
    if (grid_size == 0) {
        throw std::invalid_argument("comp_outcome_dists: G must be > 0");
    }

    const std::size_t T = panel.T();
    arma::mat outcome_val_cdfs(static_cast<arma::uword>(T), static_cast<arma::uword>(grid_size), arma::fill::zeros);
    arma::uvec outcome_counts(static_cast<arma::uword>(T), arma::fill::zeros);

    // Pass 1: quantile grid
    arma::vec grid = compute_quantile_grid_from_panel(panel, grid_size);
    // Pass 2: stream updates
    stream_update_outcome_cdfs(panel, grid, outcome_val_cdfs, outcome_counts);

    return {std::move(outcome_val_cdfs), std::move(outcome_counts)};
}

struct VectorHasher {
    std::size_t operator()(const std::vector<arma::uword>& v) const noexcept {
        std::size_t seed = v.size();
        for (arma::uword x : v) {
            seed ^= std::hash<arma::uword>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
struct VectorEq {
    bool operator()(const std::vector<arma::uword>& a, const std::vector<arma::uword>& b) const noexcept {
        return a == b;
    }
};

// Given data (G x T) and centers (G x k), return length-T 0-based cluster IDs.
static arma::uvec assign_outcomes_to_centers(const arma::mat& data, const arma::mat& centers) {
    const arma::uword T = data.n_cols;
    arma::uvec assignment(T);
    for (arma::uword i = 0; i < T; ++i) {
        arma::vec dists = arma::sum(arma::square(centers.each_col() - data.col(i)), 0).t();
        assignment(i) = dists.index_min();
    }
    return assignment;
}

struct LargestSuperCohortStats {
    std::size_t total_size;
    std::size_t min_cohort_size;
};

// Forward declaration: internal combiner used below
static std::pair<ObservedOutcomeIndices, arma::uvec>
get_new_cohorts_from_combining_outcomes(
    const ObservedOutcomeIndices& ooi,
    const arma::uvec& cohort_sizes,
    const std::unordered_map<int,int>& old_to_new_outcome);

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

static std::pair<arma::uvec, OutcomeClusteringSummary> cluster_outcomes_single_k(
    const InMemoryUnbalancedPanel& panel,
    const arma::mat& outcome_val_cdfs,   // T x G
    const arma::uvec& cohort_sizes,      // length C
    std::size_t max_model_rank,
    std::size_t k)
{
    const std::size_t T = panel.T();
    if (k == 0 || k > T) {
        throw std::invalid_argument("cluster_outcomes_single_k: invalid k");
    }

    // Prepare data (columns are outcomes) and run k-means
    arma::mat data = outcome_val_cdfs.t(); // G x T
    arma::mat centers;
    if (!arma::kmeans(
            centers,
            data,
            static_cast<arma::uword>(k),
            arma::static_subset,
            static_cast<arma::uword>(25),
            false))
    {
        throw std::runtime_error("cluster_outcomes_single_k: kmeans failed");
    }

    // Assign outcomes to nearest center
    arma::uvec outcome_to_cluster = assign_outcomes_to_centers(data, centers);

    // Build mapping outcome -> cluster id for combining
    std::unordered_map<int,int> old_to_new_outcome;
    old_to_new_outcome.reserve(T);
    for (std::size_t t = 0; t < T; ++t) {
        old_to_new_outcome.emplace(
            static_cast<int>(t),
            static_cast<int>(outcome_to_cluster(static_cast<arma::uword>(t))));
    }

    auto combined = get_new_cohorts_from_combining_outcomes(
        panel.observed_outcome_indices(),
        cohort_sizes,
        old_to_new_outcome);
    const ObservedOutcomeIndices& combined_ooi = combined.first;
    const arma::uvec& combined_sizes = combined.second;

    auto super_cohort_iterates = o3_algorithm(combined_ooi, static_cast<unsigned int>(max_model_rank));
    const auto& final_level = super_cohort_iterates.back();

    auto stats = largest_final_super_cohort_stats(final_level, combined_sizes);
    const double share = (panel.num_units() == 0)
        ? 0.0
        : static_cast<double>(stats.total_size) / static_cast<double>(panel.num_units());
    const std::size_t num_o3_iterations = super_cohort_iterates.size();

    OutcomeClusteringSummary summary{stats.total_size, share, stats.min_cohort_size, num_o3_iterations};
    return {std::move(outcome_to_cluster), summary};
}

static std::pair<ObservedOutcomeIndices, arma::uvec>
get_new_cohorts_from_combining_outcomes(
    const ObservedOutcomeIndices& ooi,
    const arma::uvec& cohort_sizes,
    const std::unordered_map<int,int>& old_to_new_outcome)
{
    if (ooi.size() != cohort_sizes.n_elem) {
        throw std::invalid_argument("get_new_cohorts_from_combining_outcomes: size mismatch between ooi and cohort_sizes");
    }

    ObservedOutcomeIndices combined_ooi;
    combined_ooi.reserve(ooi.size());
    std::vector<arma::uword> combined_sizes_vec;
    combined_sizes_vec.reserve(ooi.size());

    std::unordered_map<std::vector<arma::uword>, std::size_t, VectorHasher, VectorEq> key_to_group;
    key_to_group.reserve(ooi.size() * 2);

    for (std::size_t c = 0; c < ooi.size(); ++c) {
        // Map and canonicalize
        std::vector<arma::uword> mapped;
        const arma::uvec& occ = ooi[c];
        mapped.reserve(occ.n_elem);
        for (arma::uword t_old : occ) {
            auto it = old_to_new_outcome.find(static_cast<int>(t_old));
            if (it == old_to_new_outcome.end()) continue; // skip unmapped
            mapped.push_back(static_cast<arma::uword>(it->second));
        }
        std::sort(mapped.begin(), mapped.end());
        mapped.erase(std::unique(mapped.begin(), mapped.end()), mapped.end());

        auto itg = key_to_group.find(mapped);
        const arma::uword size_c = cohort_sizes(static_cast<arma::uword>(c));
        if (itg == key_to_group.end()) {
            std::size_t g = combined_ooi.size();
            key_to_group.emplace(mapped, g);
            combined_ooi.emplace_back(arma::uvec(mapped));
            combined_sizes_vec.push_back(size_c);
        } else {
            std::size_t g = itg->second;
            combined_sizes_vec[g] += size_c;
        }
    }

    arma::uvec combined_sizes(static_cast<arma::uword>(combined_sizes_vec.size()));
    for (std::size_t i = 0; i < combined_sizes_vec.size(); ++i) {
        combined_sizes(static_cast<arma::uword>(i)) = combined_sizes_vec[i];
    }

    return {std::move(combined_ooi), std::move(combined_sizes)};
}

} // anonymous namespace

std::pair<std::vector<arma::uvec>, std::vector<OutcomeClusteringSummary>>
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t max_model_rank,
    std::size_t min_k,
    std::size_t max_k)
{
    if (grid_size == 0) {
        throw std::invalid_argument("comp_outcome_clusterings: G must be > 0");
    }
    if (min_k == 0) {
        throw std::invalid_argument("comp_outcome_clusterings: min_k must be > 0");
    }
    if (min_k > max_k) {
        throw std::invalid_argument("comp_outcome_clusterings: min_k > max_k");
    }
    const std::size_t T = panel.T();
    if (max_k > T) {
        throw std::invalid_argument("comp_outcome_clusterings: max_k cannot exceed T");
    }

    // Get empirical CDFs of outcomes
    auto dists = comp_outcome_dists(panel, grid_size);
    arma::mat outcome_val_cdfs = std::move(dists.first);  // T x G
    // arma::uvec outcome_counts = std::move(dists.second); // currently unused

    // Precompute cohort sizes
    arma::uvec cohort_sizes = panel.get_cohort_sizes();

    std::vector<arma::uvec> mappings;
    std::vector<OutcomeClusteringSummary> summaries;
    mappings.reserve(max_k - min_k + 1);
    summaries.reserve(max_k - min_k + 1);

    for (std::size_t k = min_k; k <= max_k; ++k) {
        auto result = cluster_outcomes_single_k(panel, outcome_val_cdfs, cohort_sizes, max_model_rank, k);
        mappings.push_back(std::move(result.first));
        summaries.push_back(std::move(result.second));
    }

    return {std::move(mappings), std::move(summaries)};
}

std::pair<arma::uvec, OutcomeClusteringSummary>
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t max_model_rank,
    std::size_t k)
{
    if (grid_size == 0) {
        throw std::invalid_argument("comp_outcome_clusterings: grid_size must be > 0");
    }

    // Compute empirical CDFs
    auto dists = comp_outcome_dists(panel, grid_size);
    const arma::mat& outcome_val_cdfs = dists.first;  // T x grid_size

    // Cohort sizes once
    arma::uvec cohort_sizes = panel.get_cohort_sizes();

    // Single-k driver returns mapping and summary
    return cluster_outcomes_single_k(panel, outcome_val_cdfs, cohort_sizes, max_model_rank, k);
}

} // namespace apm