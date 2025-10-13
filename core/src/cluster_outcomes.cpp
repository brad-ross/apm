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

// Shared validation helpers
static void validate_grid(std::size_t grid_size) {
    if (grid_size == 0) {
        throw std::invalid_argument("grid_size must be > 0");
    }
}

static void validate_k_range(std::size_t T, std::size_t min_k, std::size_t max_k) {
    if (min_k == 0) {
        throw std::invalid_argument("min_k must be > 0");
    }
    if (min_k > max_k) {
        throw std::invalid_argument("min_k > max_k");
    }
    if (max_k > T) {
        throw std::invalid_argument("max_k cannot exceed T");
    }
}

// Shared single-k mapping helper
static arma::uvec compute_single_k_mapping(const arma::mat& outcome_val_cdfs, std::size_t k) {
    const arma::uword T = outcome_val_cdfs.n_rows;
    if (k == 0 || k > T) {
        throw std::invalid_argument("compute_single_k_mapping: invalid k");
    }
    arma::mat data = outcome_val_cdfs.t(); // G x T
    arma::mat centers;
    if (!arma::kmeans(centers, data, static_cast<arma::uword>(k), arma::static_subset, static_cast<arma::uword>(25), false)) {
        throw std::runtime_error("compute_single_k_mapping: kmeans failed");
    }
    return assign_outcomes_to_centers(data, centers);
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

std::vector<arma::uvec>
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t min_k,
    std::size_t max_k)
{
    validate_grid(grid_size);
    validate_k_range(panel.T(), min_k, max_k);

    auto dists = comp_outcome_dists(panel, grid_size);
    const arma::mat& outcome_val_cdfs = dists.first;  // T x grid_size

    std::vector<arma::uvec> mappings;
    mappings.reserve(max_k - min_k + 1);
    for (std::size_t k = min_k; k <= max_k; ++k) {
        mappings.push_back(compute_single_k_mapping(outcome_val_cdfs, k));
    }
    return mappings;
}

arma::uvec
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t k)
{
    validate_grid(grid_size);
    auto dists = comp_outcome_dists(panel, grid_size);
    const arma::mat& outcome_val_cdfs = dists.first;
    return compute_single_k_mapping(outcome_val_cdfs, k);
}

} // namespace apm