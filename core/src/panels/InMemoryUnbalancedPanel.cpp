#include "InMemoryUnbalancedPanel.h"
#include <limits>

namespace apm {

static std::unordered_map<int, std::size_t> make_pos_map(const arma::uvec& idx0) {
    std::unordered_map<int, std::size_t> mp;
    mp.reserve(static_cast<std::size_t>(idx0.n_elem) * 2);
    for (arma::uword k = 0; k < idx0.n_elem; ++k)
        mp.emplace(static_cast<int>(idx0[k]), static_cast<std::size_t>(k));
    return mp;
}

std::vector<CohortBlock> InMemoryUnbalancedPanel::build_cohort_blocks_with_unit_runs(
    const int* unit_idx,
    const int* cohort_id,
    std::size_t n_rows)
{
    std::vector<CohortBlock> blocks;
    if (n_rows == 0) return blocks;

    std::size_t cohort_start = 0;
    std::size_t curr_c = static_cast<std::size_t>(cohort_id[0]);

    std::vector<UnitRun> unit_runs;
    unit_runs.reserve(64);

    std::size_t unit_start = 0;
    std::size_t curr_u = static_cast<std::size_t>(unit_idx[0]);

    auto finalize_unit = [&](std::size_t end_exclusive) {
        unit_runs.push_back(UnitRun{unit_start, end_exclusive, curr_u});
    };
    auto finalize_cohort = [&](std::size_t end_exclusive) {
        blocks.push_back(CohortBlock{cohort_start, end_exclusive, curr_c, unit_runs});
    };

    for (std::size_t i = 1; i < n_rows; ++i) {
        const std::size_t c = static_cast<std::size_t>(cohort_id[i]);
        const std::size_t u = static_cast<std::size_t>(unit_idx[i]);

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

InMemoryUnbalancedPanel::InMemoryUnbalancedPanel(
    const int* unit_idx,
    const int* cohort_id,
    const int* outcome_idx,
    const double* y,
    const std::vector<const double*>& covar_cols,
    const std::vector<const double*>& auxiliary_cols,
    std::size_t n_rows,
    ObservedOutcomeIndices observed_outcome_indices)
    : unit_idx_(unit_idx)
    , cohort_id_(cohort_id)
    , outcome_idx_(outcome_idx)
    , y_(y)
    , covar_cols_(covar_cols)
    , auxiliary_cols_(auxiliary_cols)
    , n_rows_(n_rows)
    , observed_outcome_indices_(std::move(observed_outcome_indices))
    , cohort_blocks_(build_cohort_blocks_with_unit_runs(unit_idx, cohort_id, n_rows))
{
    pos_T_idx_by_cohort_.resize(observed_outcome_indices_.size());
    for (std::size_t c = 0; c < observed_outcome_indices_.size(); ++c) {
        pos_T_idx_by_cohort_[c] = make_pos_map(observed_outcome_indices_[c]);
    }
}

void InMemoryUnbalancedPanel::assemble_Y_for_unit(
    const UnitRun& ur,
    const arma::uvec& T_idxs_for_cohort,
    const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
    arma::vec& Y) const
{
    const std::size_t T_c = static_cast<std::size_t>(T_idxs_for_cohort.n_elem);
    Y.set_size(static_cast<arma::uword>(T_c));
    Y.fill(std::numeric_limits<double>::quiet_NaN());

    for (std::size_t r = ur.start; r < ur.end; ++r) {
        const int o = outcome_idx_[r];
        auto it = pos_T_idx_for_cohort.find(o);
        if (it != pos_T_idx_for_cohort.end()) {
            Y(static_cast<arma::uword>(it->second)) = y_[r];
        }
    }
}

void InMemoryUnbalancedPanel::assemble_X_for_unit(
    const UnitRun& ur,
    std::size_t T,
    const arma::uvec& T_idxs_for_cohort,
    arma::mat& X_full,
    arma::mat& X_obs) const
{
    const std::size_t q = covar_cols_.size();
    if (q == 0) { X_full.reset(); X_obs.reset(); return; }

    X_full.set_size(static_cast<arma::uword>(T), static_cast<arma::uword>(q));
    X_full.fill(std::numeric_limits<double>::quiet_NaN());
    for (std::size_t r = ur.start; r < ur.end; ++r) {
        const int o = outcome_idx_[r];
        if (static_cast<std::size_t>(o) >= T) continue;
        const arma::uword row = static_cast<arma::uword>(o);
        for (std::size_t j = 0; j < q; ++j) {
            X_full(row, static_cast<arma::uword>(j)) = covar_cols_[j][r];
        }
    }

    const std::size_t T_c = static_cast<std::size_t>(T_idxs_for_cohort.n_elem);
    X_obs.set_size(static_cast<arma::uword>(T_c), static_cast<arma::uword>(q));
    for (std::size_t k = 0; k < T_c; ++k) {
        const arma::uword row_full = static_cast<arma::uword>(T_idxs_for_cohort[k]);
        X_obs.row(static_cast<arma::uword>(k)) = X_full.row(row_full);
    }
}

void InMemoryUnbalancedPanel::assemble_YX_for_unit(
    const UnitRun& ur,
    std::size_t T,
    const arma::uvec& T_idxs_for_cohort,
    const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
    arma::vec& Y,
    arma::mat& X_full,
    arma::mat& X_obs) const
{
    assemble_Y_for_unit(ur, T_idxs_for_cohort, pos_T_idx_for_cohort, Y);
    if (!covar_cols_.empty()) {
        assemble_X_for_unit(ur, T, T_idxs_for_cohort, X_full, X_obs);
    } else {
        X_full.reset();
        X_obs.reset();
    }
}

void InMemoryUnbalancedPanel::assemble_aux_for_unit(
    const UnitRun& ur,
    std::size_t T,
    arma::mat& A_out) const
{
    const std::size_t d = auxiliary_cols_.size();
    if (d == 0) { A_out.reset(); return; }
    A_out.set_size(static_cast<arma::uword>(T), static_cast<arma::uword>(d));
    A_out.fill(std::numeric_limits<double>::quiet_NaN());
    for (std::size_t r = ur.start; r < ur.end; ++r) {
        const int o = outcome_idx_[r];
        if (static_cast<std::size_t>(o) >= T) continue;
        const arma::uword row = static_cast<arma::uword>(o);
        for (arma::uword j = 0; j < A_out.n_cols; ++j) {
            A_out(row, j) = auxiliary_cols_[static_cast<std::size_t>(j)][r];
        }
    }
}

} // namespace apm


