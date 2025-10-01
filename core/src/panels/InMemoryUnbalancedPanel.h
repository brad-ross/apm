#pragma once
#ifndef APM_INMEMORY_UNBALANCED_PANEL_H
#define APM_INMEMORY_UNBALANCED_PANEL_H

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <optional>
#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include "../utils.h" // ObservedOutcomeIndices, num_outcomes
#include "AbstractUnbalancedPanel.h"

namespace apm {

class InMemoryUnbalancedPanel : public AbstractUnbalancedPanel {
public:
    InMemoryUnbalancedPanel(
        const int* unit_idx,
        const int* cohort_id,
        const int* outcome_idx,
        const double* y,
        const std::vector<const double*>& covar_cols,
        const std::vector<const double*>& auxiliary_cols,
        std::size_t n_rows,
        ObservedOutcomeIndices observed_outcome_indices,
        bool one_indexed,
        std::optional<std::size_t> num_units = std::nullopt);

    // Accessors to original inputs
    const int* outcome_idx() const { return outcome_idx_; }
    const double* y() const { return y_; }
    const std::vector<const double*>& covar_cols() const { return covar_cols_; }
    const std::vector<const double*>& auxiliary_cols() const { return auxiliary_cols_; }
    const ObservedOutcomeIndices& observed_outcome_indices() const override { return observed_outcome_indices_; }
    std::size_t num_units() const override { return num_units_; }

    // Dimensions
    std::size_t T() const override { return static_cast<std::size_t>(num_outcomes(observed_outcome_indices_)); }
    std::size_t q() const override { return covar_cols_.size(); }
    std::size_t d() const override { return auxiliary_cols_.size(); }

    // Grouping
    const std::vector<CohortBlock>& cohort_blocks() const override { return cohort_blocks_; }

    // Access precomputed per-cohort indices and position maps
    const arma::uvec& T_idx_for_cohort(std::size_t cohort_0b) const override {
        return observed_outcome_indices_.at(cohort_0b);
    }
    const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort(std::size_t cohort_0b) const override {
        return pos_T_idx_by_cohort_.at(cohort_0b);
    }

    // Indexed assembly helpers
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
    // inputs
    const int* unit_idx_;
    const int* cohort_id_;
    const int* outcome_idx_;
    const double* y_;
    const std::vector<const double*>& covar_cols_;
    const std::vector<const double*>& auxiliary_cols_;
    std::size_t n_rows_;
    ObservedOutcomeIndices observed_outcome_indices_;
    bool one_indexed_;
    std::size_t num_units_;

    // precomputed grouping
    std::vector<CohortBlock> cohort_blocks_;

    // per-cohort map: outcome -> position in T_idxs_for_cohort
    std::vector<std::unordered_map<int, std::size_t>> pos_T_idx_by_cohort_;

    // helper (moved from est_cohort_specific_params.cpp)
    static std::vector<CohortBlock> build_cohort_blocks_with_unit_runs(
        const int* unit_idx,
        const int* cohort_id,
        std::size_t n_rows,
        bool one_indexed);
};

} // namespace apm

#endif // APM_INMEMORY_UNBALANCED_PANEL_H


