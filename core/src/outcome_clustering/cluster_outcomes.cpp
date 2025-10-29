#include "outcome_clustering/cluster_outcomes.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "online_accumulators.h"
#include "utils.h"
#include "outcome_clustering/weighted_kmeans.h"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include <set>
#include <mutex>


#ifdef APM_HAS_TBB
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#endif

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

std::pair<arma::mat, arma::uvec>
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

// Given data (G x T) and centers (G x k), return length-T 0-based cluster IDs.
arma::uvec assign_outcomes_to_centers(const arma::mat& data, const arma::mat& centers) {
    const arma::uword T = data.n_cols;
    arma::uvec assignment(T);
    for (arma::uword i = 0; i < T; ++i) {
        arma::vec dists = arma::sum(arma::square(centers.each_col() - data.col(i)), 0).t();
        assignment(i) = dists.index_min();
    }
    return assignment;
}

// Shared validation helpers
void validate_grid(std::size_t grid_size) {
    if (grid_size == 0) {
        throw std::invalid_argument("grid_size must be > 0");
    }
}

void validate_k_range(std::size_t T, std::size_t min_k, std::size_t max_k) {
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

// Shared inputs container for clustering
struct ClusteringInputs {
    arma::mat data;    // G x T
    arma::vec weights; // length T
};

ClusteringInputs build_clustering_inputs(const InMemoryUnbalancedPanel& panel, std::size_t grid_size) {
    validate_grid(grid_size);
    auto dists = comp_outcome_dists(panel, grid_size);
    const arma::mat& outcome_val_cdfs = dists.first;  // T x G
    const arma::uvec& outcome_counts = dists.second;  // T

    ClusteringInputs out;
    out.data = outcome_val_cdfs.t();                  // G x T
    out.weights = arma::conv_to<arma::vec>::from(outcome_counts);
    return out;
}

inline std::optional<uint64_t> per_init_seed(std::optional<uint64_t> base, std::size_t init) {
    return base ? std::optional<uint64_t>(*base + static_cast<uint64_t>(init)) : std::nullopt;
}

inline arma::uvec to_uvec(const arma::Row<size_t>& r) {
    arma::uvec out(r.n_elem);
    for (arma::uword i = 0; i < r.n_elem; ++i) out(i) = static_cast<arma::uword>(r(i));
    return out;
}

struct BestForK {
    std::mutex m;
    double best_sse = std::numeric_limits<double>::infinity();
    arma::Row<size_t> best_assignments;
    void consider(double sse, arma::Row<size_t>&& asg) {
        if (sse < best_sse) {
            best_sse = sse;
            best_assignments = std::move(asg);
        }
    }
};

inline std::pair<std::size_t, std::size_t> unflatten(std::size_t idx,
                                                     std::size_t num_k,
                                                     std::size_t n_inits)
{
    (void)num_k; // not needed for computation
    const std::size_t k_idx = idx / n_inits;
    const std::size_t init  = idx % n_inits;
    return {k_idx, init};
}

// Canonicalize cluster labels by order of first occurrence to ensure stable labeling.
void canonicalize_assignments(arma::Row<size_t>& assignments) {
    const arma::uword T = assignments.n_elem;
    if (T == 0) return;
    const size_t k_guess = static_cast<size_t>(assignments.max() + 1);
    std::vector<arma::uword> first_index(k_guess, std::numeric_limits<arma::uword>::max());
    for (arma::uword i = 0; i < T; ++i) {
        const size_t lbl = assignments(i);
        if (lbl >= k_guess) continue;
        if (i < first_index[lbl]) first_index[lbl] = i;
    }
    std::vector<std::pair<arma::uword, size_t>> ordering;
    ordering.reserve(k_guess);
    for (size_t lbl = 0; lbl < k_guess; ++lbl) {
        if (first_index[lbl] != std::numeric_limits<arma::uword>::max()) {
            ordering.emplace_back(first_index[lbl], lbl);
        }
    }
    std::sort(ordering.begin(), ordering.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
    std::vector<size_t> remap(k_guess, static_cast<size_t>(0));
    size_t next = 0;
    for (const auto& p : ordering) {
        remap[p.second] = next++;
    }
    for (arma::uword i = 0; i < T; ++i) {
        const size_t old = assignments(i);
        if (old < remap.size()) assignments(i) = remap[old];
    }
}

} // anonymous namespace

std::vector<arma::uvec>
comp_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t min_k,
    std::size_t max_k,
    std::optional<std::size_t> n_inits,
    std::optional<uint64_t> seed,
    std::optional<std::size_t> num_threads)
{
    validate_k_range(panel.T(), min_k, max_k);

    const ClusteringInputs inputs = build_clustering_inputs(panel, grid_size);

    const std::size_t num_k = max_k - min_k + 1;
    const std::size_t num_inits_val = n_inits.value_or(static_cast<std::size_t>(10));
    if (num_inits_val == 0) {
        throw std::invalid_argument("n_inits must be > 0");
    }

    std::vector<BestForK> trackers(num_k);

#ifdef APM_HAS_TBB
    apm::ParallelismScope par_scope(num_threads);
    std::size_t nt = par_scope.nt;
    if (nt > 1) {
        const std::size_t total_tasks = num_k * num_inits_val;
        oneapi::tbb::parallel_for(std::size_t(0), total_tasks, [&](std::size_t idx){
            auto [k_idx, init] = unflatten(idx, num_k, num_inits_val);
            const std::size_t k_val = min_k + k_idx;
            auto [sse, asg] = kmeans_weighted(inputs.data, inputs.weights, k_val, per_init_seed(seed, init));
            std::lock_guard<std::mutex> lock(trackers[k_idx].m);
            trackers[k_idx].consider(sse, std::move(asg));
        });
    } else
#endif
    {
        for (std::size_t k_idx = 0; k_idx < num_k; ++k_idx) {
            const std::size_t k_val = min_k + k_idx;
            for (std::size_t init = 0; init < num_inits_val; ++init) {
                auto [sse, asg] = kmeans_weighted(inputs.data, inputs.weights, k_val, per_init_seed(seed, init));
                trackers[k_idx].consider(sse, std::move(asg));
            }
        }
    }

    std::vector<arma::uvec> mappings;
    mappings.reserve(num_k);
    for (std::size_t k_idx = 0; k_idx < num_k; ++k_idx) {
        canonicalize_assignments(trackers[k_idx].best_assignments);
        mappings.push_back(to_uvec(trackers[k_idx].best_assignments));
    }
    return mappings;
}

arma::uvec
comp_outcome_clustering(
    const InMemoryUnbalancedPanel& panel,
    std::size_t grid_size,
    std::size_t k,
    std::optional<std::size_t> n_inits,
    std::optional<uint64_t> seed,
    std::optional<std::size_t> num_threads)
{
    auto res = comp_outcome_clusterings(panel, grid_size, k, k, n_inits, seed, num_threads);
    if (res.empty()) {
        throw std::runtime_error("Internal error: empty result for single-k clustering");
    }
    return res.front();
}

// Exposed API: compute new cohort groupings after combining outcomes
std::pair<ObservedOutcomeIndices, arma::uvec>
get_new_cohorts_from_combining_outcomes(
    const ObservedOutcomeIndices& ooi,
    const arma::uvec& cohort_sizes,
    const arma::uvec& old_to_new_outcome)
{
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
            if (t_old >= old_to_new_outcome.n_elem) {
                throw std::out_of_range("get_new_cohorts_from_combining_outcomes: t_old index exceeds mapping length");
            }
            mapped.push_back(static_cast<arma::uword>(old_to_new_outcome(t_old)));
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

} // namespace apm

#ifdef APM_TESTS
namespace apm { namespace test {
std::pair<arma::mat, arma::uvec>
comp_outcome_dists_test(const InMemoryUnbalancedPanel& panel, std::size_t grid_size) {
    return comp_outcome_dists(panel, grid_size);
}
}} // namespace apm::test
#endif