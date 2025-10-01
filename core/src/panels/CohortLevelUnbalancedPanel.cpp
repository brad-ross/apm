#include "CohortLevelUnbalancedPanel.h"
#include <stdexcept>

namespace apm {

namespace {
static std::size_t infer_q_global(const std::vector<OutcomeMeanSufficientStatistics>& stats) {
    std::optional<std::size_t> q_opt;
    for (const auto& s : stats) {
        if (s.covar_means) {
            std::size_t q_here = static_cast<std::size_t>(s.covar_means->n_cols);
            if (q_opt && q_opt.value() != q_here) {
                throw std::invalid_argument("CohortLevelUnbalancedPanel: Inconsistent q across cohorts.");
            }
            q_opt = q_here;
        }
    }
    return q_opt.value_or(0);
}
} // anonymous namespace

std::vector<std::unordered_map<int, std::size_t>>
CohortLevelUnbalancedPanel::build_pos_maps(const ObservedOutcomeIndices& obs) {
    std::vector<std::unordered_map<int, std::size_t>> out;
    out.reserve(obs.size());
    for (const auto& idxs : obs) {
        std::unordered_map<int, std::size_t> pos;
        pos.reserve(static_cast<std::size_t>(idxs.n_elem) * 2);
        for (std::size_t j = 0; j < static_cast<std::size_t>(idxs.n_elem); ++j) {
            pos[static_cast<int>(idxs[j])] = j;
        }
        out.emplace_back(std::move(pos));
    }
    return out;
}

std::vector<CohortBlock>
CohortLevelUnbalancedPanel::build_single_unit_cohort_blocks(std::size_t n_cohorts) {
    std::vector<CohortBlock> blocks;
    blocks.reserve(n_cohorts);
    for (std::size_t c = 0; c < n_cohorts; ++c) {
        CohortBlock cb;
        cb.start = 0;
        cb.end = 1;
        cb.cohort = c;
        cb.unit_runs = { UnitRun{0, 1, c} };
        blocks.emplace_back(std::move(cb));
    }
    return blocks;
}

CohortLevelUnbalancedPanel::CohortLevelUnbalancedPanel(
    std::vector<OutcomeMeanSufficientStatistics> suff_stats_by_cohort,
    ObservedOutcomeIndices observed_outcome_indices)
    : T_(static_cast<std::size_t>(num_outcomes(observed_outcome_indices)))
    , q_(infer_q_global(suff_stats_by_cohort))
    , num_units_(suff_stats_by_cohort.size())
    , observed_outcome_indices_(std::move(observed_outcome_indices))
    , pos_T_idx_by_cohort_(build_pos_maps(observed_outcome_indices_))
    , suff_stats_by_cohort_(std::move(suff_stats_by_cohort))
    , cohort_blocks_(build_single_unit_cohort_blocks(num_units_))
{
    if (suff_stats_by_cohort_.size() != observed_outcome_indices_.size()) {
        throw std::invalid_argument("CohortLevelUnbalancedPanel: stats vector size must equal number of cohorts in observed_outcome_indices.");
    }
    for (std::size_t c = 0; c < suff_stats_by_cohort_.size(); ++c) {
        const auto& stats = suff_stats_by_cohort_[c];
        const auto& T_idx = observed_outcome_indices_[c];
        if (static_cast<std::size_t>(stats.observed_outcome_means.n_elem) != static_cast<std::size_t>(T_idx.n_elem)) {
            throw std::invalid_argument("CohortLevelUnbalancedPanel: observed_outcome_means length must equal T_c for each cohort.");
        }
        if (stats.covar_means) {
            if (static_cast<std::size_t>(stats.covar_means->n_rows) != T_) {
                throw std::invalid_argument("CohortLevelUnbalancedPanel: covar_means must have T rows.");
            }
            if (q_ != static_cast<std::size_t>(stats.covar_means->n_cols)) {
                throw std::invalid_argument("CohortLevelUnbalancedPanel: covar_means inconsistent columns with global q.");
            }
        }
    }
}

void CohortLevelUnbalancedPanel::assemble_Y_for_unit(
    const UnitRun& ur,
    const arma::uvec& T_idxs_for_cohort,
    const std::unordered_map<int, std::size_t>& /*pos_T_idx_for_cohort*/,
    arma::vec& Y) const
{
    const std::size_t c = ur.unit;
    const auto& stats = suff_stats_by_cohort_.at(c);
    const std::size_t Tc = static_cast<std::size_t>(T_idxs_for_cohort.n_elem);
    Y.set_size(static_cast<arma::uword>(Tc));
    Y = stats.observed_outcome_means;
}

void CohortLevelUnbalancedPanel::assemble_X_for_unit(
    const UnitRun& ur,
    std::size_t T,
    const arma::uvec& T_idxs_for_cohort,
    arma::mat* X_full,
    arma::mat* X_obs,
    std::optional<std::size_t> covariate_index) const
{
    if (T != T_) {
        throw std::invalid_argument("CohortLevelUnbalancedPanel::assemble_X_for_unit: Provided T does not match panel T.");
    }

    const std::size_t Tc = static_cast<std::size_t>(T_idxs_for_cohort.n_elem);
    const std::size_t q = q_;

    if (X_full) { X_full->set_size(static_cast<arma::uword>(T), static_cast<arma::uword>(q)); X_full->zeros(); }
    if (X_obs)  { X_obs->set_size(static_cast<arma::uword>(Tc), static_cast<arma::uword>(q)); X_obs->zeros(); }
    if (q == 0) return;

    const std::size_t c = ur.unit;
    const auto& stats = suff_stats_by_cohort_.at(c);
    if (!stats.covar_means) return; // leave zeros

    if (X_full) {
        if (covariate_index) {
            X_full->col(static_cast<arma::uword>(0)) = stats.covar_means->col(static_cast<arma::uword>(*covariate_index));
        } else {
            *X_full = *stats.covar_means;
        }
    }

    if (X_obs) {
        if (covariate_index) {
            for (std::size_t j = 0; j < Tc; ++j) {
                const arma::uword t = static_cast<arma::uword>(T_idxs_for_cohort[j]);
                (*X_obs)(static_cast<arma::uword>(j), 0) = stats.covar_means->at(t, static_cast<arma::uword>(*covariate_index));
            }
        } else {
            for (std::size_t j = 0; j < Tc; ++j) {
                const arma::uword t = static_cast<arma::uword>(T_idxs_for_cohort[j]);
                X_obs->row(static_cast<arma::uword>(j)) = stats.covar_means->row(t);
            }
        }
    }
}

void CohortLevelUnbalancedPanel::assemble_YX_for_unit(
    const UnitRun& ur,
    std::size_t T,
    const arma::uvec& T_idxs_for_cohort,
    const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
    arma::vec& Y,
    arma::mat& X_full,
    arma::mat& X_obs) const
{
    assemble_Y_for_unit(ur, T_idxs_for_cohort, pos_T_idx_for_cohort, Y);
    assemble_X_for_unit(ur, T, T_idxs_for_cohort, &X_full, &X_obs, std::nullopt);
}

void CohortLevelUnbalancedPanel::assemble_aux_for_unit(
    const UnitRun& /*ur*/,
    std::size_t T,
    arma::mat& A_out) const
{
    A_out.set_size(static_cast<arma::uword>(T), 0);
}

} // namespace apm


