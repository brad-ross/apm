#pragma once
#ifndef APM_INMEMORY_UNBALANCED_PANEL_H
#define APM_INMEMORY_UNBALANCED_PANEL_H

//==============================================================================
// Concrete in-memory panel backed by raw arrays (e.g., R/Python bindings).
//==============================================================================

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

/**
 * @brief In-memory implementation of an unbalanced panel backed by column pointers.
 *
 * Accepts raw arrays of unit ids, cohort ids, outcome indices, outcomes, and
 * optional covariate/auxiliary columns. Provides fast assembly helpers used by
 * estimation and imputation routines.
 */
class InMemoryUnbalancedPanel : public AbstractUnbalancedPanel {
public:
    /**
     * @param unit_idx Pointer to length-n_rows unit ids.
     * @param cohort_id Pointer to length-n_rows cohort ids.
     * @param outcome_idx Pointer to length-n_rows outcome indices (0- or 1-based).
     * @param y Pointer to length-n_rows outcome values.
     * @param covar_cols Vector of q column pointers to covariate data (length n_rows each).
     * @param auxiliary_cols Vector of d column pointers to auxiliary data (length n_rows each).
     * @param n_rows Number of rows in the stacked long-format input.
     * @param observed_outcome_indices Per-cohort observed outcome indices (0-based).
     * @param one_indexed If true, indices in input arrays are 1-based and will be shifted to 0-based.
     * @param num_units Optional explicit unit count; inferred from unit_idx max+1 when omitted.
     */
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
    /** @return Pointer to raw outcome indices array. */
    const int* outcome_idx() const { return outcome_idx_; }
    /** @return Pointer to raw outcomes array. */
    const double* y() const { return y_; }
    /** @return Covariate column pointers (size q, may be empty). */
    const std::vector<const double*>& covar_cols() const { return covar_cols_; }
    /** @return Auxiliary column pointers (size d, may be empty). */
    const std::vector<const double*>& auxiliary_cols() const { return auxiliary_cols_; }
    /** @return Observed outcome indices per cohort. */
    const ObservedOutcomeIndices& observed_outcome_indices() const override { return observed_outcome_indices_; }
    /** @return Number of unique units. */
    std::size_t num_units() const override { return num_units_; }

    // Returns a length-C vector with unit counts per atomic cohort (0-based cohort ids).
    /**
     * @brief Return unit counts per cohort (length C, 0-based cohort ids).
     */
    arma::uvec get_cohort_sizes() const;

    // Dimensions
    /** @return Total outcomes T. */
    std::size_t T() const override { return static_cast<std::size_t>(num_outcomes(observed_outcome_indices_)); }
    /** @return Covariate dimension q. */
    std::size_t q() const override { return covar_cols_.size(); }
    /** @return Auxiliary dimension d. */
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
    /**
     * @brief Assemble outcomes for a unit into Y (length T_c).
     *
     * @param ur UnitRun describing the row range and unit id.
     * @param T_idxs_for_cohort Observed outcome indices for the cohort.
     * @param pos_T_idx_for_cohort Map from outcome index to position within T_idxs_for_cohort.
     * @param Y Output vector (length T_c).
     */
    void assemble_Y_for_unit(
        const UnitRun& ur,
        const arma::uvec& T_idxs_for_cohort,
        const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
        arma::vec& Y) const override;

    /**
     * @brief Assemble covariates for a unit (full T and observed-only views).
     *
     * @param ur UnitRun describing the row range and unit id.
     * @param T Total outcomes T.
     * @param T_idxs_for_cohort Observed outcome indices for the cohort.
     * @param X_full Output pointer for full covariates (T x q); may be null.
     * @param X_obs Output pointer for observed covariates (T_c x q); may be null.
     * @param covariate_index Optional single covariate column; if unset, all covariates are filled.
     */
    void assemble_X_for_unit(
        const UnitRun& ur,
        std::size_t T,
        const arma::uvec& T_idxs_for_cohort,
        arma::mat* X_full,
        arma::mat* X_obs,
        std::optional<std::size_t> covariate_index = std::nullopt) const override;

    /**
     * @brief Assemble outcomes and covariates together for a unit.
     *
     * @param ur UnitRun describing the row range and unit id.
     * @param T Total outcomes T.
     * @param T_idxs_for_cohort Observed outcome indices for the cohort.
     * @param pos_T_idx_for_cohort Map from outcome index to position within T_idxs_for_cohort.
     * @param Y Output vector (length T_c).
     * @param X_full Output full covariate matrix (T x q).
     * @param X_obs Output observed covariate matrix (T_c x q).
     */
    void assemble_YX_for_unit(
        const UnitRun& ur,
        std::size_t T,
        const arma::uvec& T_idxs_for_cohort,
        const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
        arma::vec& Y,
        arma::mat& X_full,
        arma::mat& X_obs) const override;

    /**
     * @brief Assemble auxiliary data matrix (T x d) for a unit.
     *
     * @param ur UnitRun describing the row range and unit id.
     * @param T Total outcomes T.
     * @param A_out Output auxiliary matrix (T x d).
     */
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


