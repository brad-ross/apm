#include "est_cohort_specific_params.h"
#include <algorithm>
#include <limits>
#ifdef APM_HAS_TBB
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/global_control.h>
#endif

namespace apm {

//==============================
// Grouping data structures
//==============================
struct UnitRun {
    std::size_t start;  // [start, end)
    std::size_t end;
    int unit;           // 0-based
};

struct CohortBlock {
    std::size_t start;  // [start, end)
    std::size_t end;
    int cohort;         // 0-based
    std::vector<UnitRun> unit_runs;
};

//==============================
// Helpers: construction
//==============================
static std::vector<CohortBlock> build_cohort_blocks_with_unit_runs(
    const int* unit_idx,
    const int* cohort_id,
    std::size_t n_rows)
{
    std::vector<CohortBlock> blocks;
    if (n_rows == 0) return blocks;

    std::size_t cohort_start = 0;
    int curr_c = cohort_id[0];

    std::vector<UnitRun> unit_runs;
    unit_runs.reserve(64);

    std::size_t unit_start = 0;
    int curr_u = unit_idx[0];

    auto finalize_unit = [&](std::size_t end_exclusive) {
        unit_runs.push_back(UnitRun{unit_start, end_exclusive, curr_u});
    };
    auto finalize_cohort = [&](std::size_t end_exclusive) {
        blocks.push_back(CohortBlock{cohort_start, end_exclusive, curr_c, unit_runs});
    };

    for (std::size_t i = 1; i < n_rows; ++i) {
        const int c = cohort_id[i];
        const int u = unit_idx[i];

        if (c != curr_c) {
            finalize_unit(i);
            finalize_cohort(i);
            cohort_start = i;
            curr_c = c;
            unit_runs.clear();
            curr_u = u;
            unit_start = i;
            continue;
        }

        if (u != curr_u) {
            finalize_unit(i);
            curr_u = u;
            unit_start = i;
        }
    }

    finalize_unit(n_rows);
    finalize_cohort(n_rows);

    return blocks;
}

//==============================
// Helpers: cohort dictionaries
//==============================
struct CohortDictionaries {
    arma::uvec T_idx_0b;                                 // observed outcomes for this cohort (0-based)
    std::unordered_map<int, std::size_t> pos_T_idx;     // outcome -> position in Y/X_obs
    std::size_t T;                                       // 1 + max(T_idx_0b)
};

static std::unordered_map<int, std::size_t> make_pos_map(const arma::uvec& idx0) {
    std::unordered_map<int, std::size_t> mp;
    mp.reserve(static_cast<std::size_t>(idx0.n_elem) * 2);
    for (arma::uword k = 0; k < idx0.n_elem; ++k)
        mp.emplace(static_cast<int>(idx0[k]), static_cast<std::size_t>(k));
    return mp;
}

static std::size_t compute_T_from_T_idx(const arma::uvec& T_idx_0b) {
    if (T_idx_0b.is_empty()) return 0;
    arma::uword mx = T_idx_0b.max();
    return static_cast<std::size_t>(mx + 1u);
}

static CohortDictionaries make_cohort_dicts(
    int cohort_0b,
    const ObservedOutcomeIndices& observed_outcome_indices)
{
    CohortDictionaries d;
    d.T_idx_0b = observed_outcome_indices.at(static_cast<std::size_t>(cohort_0b));
    d.pos_T_idx = make_pos_map(d.T_idx_0b);
    d.T = compute_T_from_T_idx(d.T_idx_0b);
    return d;
}

//==============================
// Helpers: estimator factories
//==============================
static std::unordered_map<std::string, std::unique_ptr<FactorModelEstimator>>
make_factor_estimators_for_cohort(
    const std::unordered_map<std::string, EstimatorSpecification>& specs,
    std::size_t T_c,
    std::size_t q,
    const std::shared_ptr<const WeightedBootstrap>& bootstrap)
{
    std::unordered_map<std::string, std::unique_ptr<FactorModelEstimator>> out;
    out.reserve(specs.size());
    for (const auto& kv : specs) {
        const auto& sp = kv.second;
        if (sp.factor_model_estimator != "principal_components") {
            throw std::invalid_argument("Only 'principal_components' is supported");
        }
        if (sp.include_outcome_fes) {
            out.emplace(kv.first, std::make_unique<PCEstimatorWithFEs>(sp.r, T_c, bootstrap, q));
        } else {
            out.emplace(kv.first, std::make_unique<PCEstimator>(sp.r, T_c, bootstrap, q));
        }
    }
    return out;
}

//==============================
// Helpers: per-unit assembly
//==============================
static void assemble_Y_X_for_unit_run(
    const UnitRun& ur,
    const int* outcome_idx,
    const double* y_col,
    const std::vector<const double*>& covar_cols, // size q, may be 0
    const CohortDictionaries& dicts,
    arma::vec& Y,          // resized to T_c
    arma::mat& X_full,     // resized to T x q if q>0, filled with NaN then rows set
    arma::mat& X_obs       // resized to T_c x q if q>0
) {
    const std::size_t T_c = static_cast<std::size_t>(dicts.T_idx_0b.n_elem);
    const std::size_t T   = dicts.T;
    const std::size_t q   = covar_cols.size();

    // Y over observed outcomes (T_c)
    Y.set_size(T_c);
    for (std::size_t k = 0; k < T_c; ++k) {
        Y(static_cast<arma::uword>(k)) = std::numeric_limits<double>::quiet_NaN();
    }
    for (std::size_t r = ur.start; r < ur.end; ++r) {
        const int o = outcome_idx[r];
        auto it = dicts.pos_T_idx.find(o);
        if (it != dicts.pos_T_idx.end()) {
            Y(static_cast<arma::uword>(it->second)) = y_col[r];
        }
    }

    if (q == 0) {
        X_full.reset();
        X_obs.reset();
        return;
    }

    // X_full: T x q (initialize to NaN)
    X_full.set_size(T, q);
    X_full.fill(std::numeric_limits<double>::quiet_NaN());
    for (std::size_t r = ur.start; r < ur.end; ++r) {
        const int o = outcome_idx[r];
        if (static_cast<std::size_t>(o) >= T) continue; // guard if data has stray larger outcome ids
        const arma::uword row = static_cast<arma::uword>(o);
        for (std::size_t j = 0; j < q; ++j) {
            X_full(row, static_cast<arma::uword>(j)) = covar_cols[j][r];
        }
    }

    // X_obs: T_c x q by selecting rows of X_full in the T_idx order
    X_obs.set_size(T_c, q);
    for (std::size_t k = 0; k < T_c; ++k) {
        const arma::uword row_full = static_cast<arma::uword>(dicts.T_idx_0b[k]); // 0-based
        X_obs.row(static_cast<arma::uword>(k)) = X_full.row(row_full);
    }
}

//==============================
// Helpers: finalize outputs
//==============================
static CohortSpecificEstimates build_cohort_specific_estimates(
    std::unordered_map<std::string, std::vector<std::optional<FactorModelEstimates>>>& tmp_factor,
    std::vector<std::optional<OutcomeMeanSuffStatEstimates>>& tmp_outcome)
{
    const std::size_t C = tmp_outcome.size();

    CohortSpecificEstimates out;
    out.cohort_outcome_mean_ests.reserve(C);
    for (std::size_t i = 0; i < C; ++i) {
        out.cohort_outcome_mean_ests.push_back(std::move(*tmp_outcome[i]));
    }

    for (auto& kv : tmp_factor) {
        auto& name = kv.first;
        auto& vec_opt = kv.second;
        std::vector<FactorModelEstimates> vec;
        vec.reserve(C);
        for (std::size_t i = 0; i < C; ++i) vec.push_back(std::move(*vec_opt[i]));
        out.cohort_specific_factor_ests.emplace(name, std::move(vec));
    }

    return out;
}

//==============================
// Main entry
//==============================
CohortSpecificEstimates estimate_cohort_specific_params_from_raw(
    const int* unit_idx,
    const int* cohort_id,
    const int* outcome_idx,
    const double* y,
    const std::vector<const double*>& covar_cols,
    std::size_t n_rows,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    const ObservedOutcomeIndices& observed_outcome_indices,
    std::shared_ptr<const WeightedBootstrap> bootstrap,
    std::size_t num_threads)
{
    const std::size_t q = covar_cols.size();

#ifdef APM_HAS_TBB
    std::unique_ptr<oneapi::tbb::global_control> tbb_gc;
    if (num_threads > 1) {
        tbb_gc = std::make_unique<oneapi::tbb::global_control>(
            oneapi::tbb::global_control::max_allowed_parallelism,
            static_cast<std::size_t>(num_threads)
        );
    }
#endif

    // Single pass: build cohort blocks and their unit runs
    std::vector<CohortBlock> blocks = build_cohort_blocks_with_unit_runs(
        unit_idx, cohort_id, n_rows);
    const std::size_t C = blocks.size();

    // Prepare thread-safe output buffers
    std::unordered_map<std::string, std::vector<std::optional<FactorModelEstimates>>> tmp_factor;
    tmp_factor.reserve(est_specs.size());
    for (const auto& kv : est_specs) {
        tmp_factor.emplace(kv.first, std::vector<std::optional<FactorModelEstimates>>(C));
    }
    std::vector<std::optional<OutcomeMeanSuffStatEstimates>> tmp_outcome(C);

    // Parallelize across cohorts
    auto process_cohort = [&](std::size_t cidx) {
        const CohortBlock& blk = blocks[cidx];

        // Dictionaries for this cohort (T_c order and T from max+1)
        CohortDictionaries dicts = make_cohort_dicts(blk.cohort, observed_outcome_indices);
        const std::size_t T_c = static_cast<std::size_t>(dicts.T_idx_0b.n_elem);
        const std::size_t T   = dicts.T;

        // Estimators
        OutcomeMeanSuffStatEstimator omsse(
            T_c,
            /*T=*/(q > 0 ? T : 0),
            /*q=*/q,
            bootstrap
        );
        auto ests = make_factor_estimators_for_cohort(est_specs, T_c, q, bootstrap);

        // Reusable buffers
        arma::vec Y;
        arma::mat X_full, X_obs;

        // Stream units
        for (const auto& ur : blk.unit_runs) {
            assemble_Y_X_for_unit_run(ur, outcome_idx, y, covar_cols, dicts, Y, X_full, X_obs);

            // Add datum to factor estimators
            for (auto& kv : ests) {
                if (q > 0) {
                    kv.second->add_datum(static_cast<std::size_t>(ur.unit), Y, X_obs);
                } else {
                    kv.second->add_datum(static_cast<std::size_t>(ur.unit), Y);
                }
            }

            // Add datum to outcome mean estimator
            if (q > 0) {
                omsse.add_datum(static_cast<std::size_t>(ur.unit), Y, X_full);
            } else {
                omsse.add_datum(static_cast<std::size_t>(ur.unit), Y);
            }
        }

        // Estimate and store
        tmp_outcome[cidx].emplace(omsse.estimate());
        for (auto& kv : ests) {
            tmp_factor[kv.first][cidx].emplace(kv.second->estimate());
        }
    };

    // Parallelize across cohorts if requested; otherwise use serial loop
#ifdef APM_HAS_TBB
    if (num_threads <= 1) {
        for (std::size_t cidx = 0; cidx < C; ++cidx) {
            process_cohort(cidx);
        }
    } else {
        oneapi::tbb::parallel_for(std::size_t(0), C, [&](std::size_t cidx) {
            process_cohort(cidx);
        });
    }
#else
    for (std::size_t cidx = 0; cidx < C; ++cidx) {
        process_cohort(cidx);
    }
#endif

    // Finalize outputs
    return build_cohort_specific_estimates(tmp_factor, tmp_outcome);
}

} // namespace apm