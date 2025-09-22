#pragma once
#ifndef APM_COHORT_LEVEL_UNBALANCED_PANEL_H
#define APM_COHORT_LEVEL_UNBALANCED_PANEL_H

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <optional>

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include "AbstractUnbalancedPanel.h"
#include "../cohort_specific_param_structs.h" // OutcomeMeanSufficientStatistics

namespace apm {

class CohortLevelUnbalancedPanel : public AbstractUnbalancedPanel {
public:
    CohortLevelUnbalancedPanel(
        std::vector<OutcomeMeanSufficientStatistics> suff_stats_by_cohort,
        ObservedOutcomeIndices observed_outcome_indices);

    // Dimensions
    std::size_t T() const override { return T_; }
    std::size_t q() const override { return q_; }
    std::size_t d() const override { return 0; }
    std::size_t num_units() const override { return num_units_; }

    // Grouping
    const std::vector<CohortBlock>& cohort_blocks() const override { return cohort_blocks_; }

    // Per-cohort indices
    const arma::uvec& T_idx_for_cohort(std::size_t cohort_0b) const override {
        return observed_outcome_indices_.at(cohort_0b);
    }
    const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort(std::size_t cohort_0b) const override {
        return pos_T_idx_by_cohort_.at(cohort_0b);
    }

    // Assembly
    void assemble_Y_for_unit(
        const UnitRun& ur,
        const arma::uvec& T_idxs_for_cohort,
        const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
        arma::vec& Y) const override;

    void assemble_X_for_unit(
        const UnitRun& ur,
        std::size_t T,
        const arma::uvec& T_idxs_for_cohort,
        arma::mat* X_full,
        arma::mat* X_obs,
        std::optional<std::size_t> covariate_index = std::nullopt) const override;

    void assemble_YX_for_unit(
        const UnitRun& ur,
        std::size_t T,
        const arma::uvec& T_idxs_for_cohort,
        const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
        arma::vec& Y,
        arma::mat& X_full,
        arma::mat& X_obs) const override;

    void assemble_aux_for_unit(
        const UnitRun& ur,
        std::size_t T,
        arma::mat& A_out) const override;

private:
    std::size_t T_;
    std::size_t q_;
    std::size_t num_units_;

    ObservedOutcomeIndices observed_outcome_indices_;
    std::vector<std::unordered_map<int, std::size_t>> pos_T_idx_by_cohort_;
    std::vector<OutcomeMeanSufficientStatistics> suff_stats_by_cohort_;
    std::vector<CohortBlock> cohort_blocks_;

    static std::vector<std::unordered_map<int, std::size_t>>
    build_pos_maps(const ObservedOutcomeIndices& obs);
    static std::vector<CohortBlock>
    build_single_unit_cohort_blocks(std::size_t n_cohorts);
};

} // namespace apm

#endif // APM_COHORT_LEVEL_UNBALANCED_PANEL_H


