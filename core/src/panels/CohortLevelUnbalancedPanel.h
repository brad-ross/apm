#pragma once
#ifndef APM_COHORT_LEVEL_UNBALANCED_PANEL_H
#define APM_COHORT_LEVEL_UNBALANCED_PANEL_H

//==============================================================================
// Synthetic panel backed by cohort-level sufficient statistics (no raw units).
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

#include "AbstractUnbalancedPanel.h"
#include "../cohort_specific_param_structs.h" // OutcomeMeanSufficientStatistics

namespace apm {

/**
 * @brief Panel wrapper for cohort-level sufficient statistics (no raw unit rows).
 *
 * Treats each cohort as a single "unit" whose data are cohort-level averages
 * (outcome means and optional covariate means). Assembly methods therefore
 * return cohort-average outcome vectors, covariate matrices, and auxiliary
 * matrices rather than raw unit-level data. Useful for target-parameter
 * estimation and imputation components that operate on sufficient statistics
 * instead of microdata.
 */
class CohortLevelUnbalancedPanel : public AbstractUnbalancedPanel {
public:
    /**
     * @brief Construct from sufficient statistics and observed outcome indices.
     *
     * @param suff_stats_by_cohort Vector of cohort sufficient statistics (length C).
     * @param observed_outcome_indices Per-cohort observed outcome indices (0-based).
     */
    CohortLevelUnbalancedPanel(
        std::vector<OutcomeMeanSufficientStatistics> suff_stats_by_cohort,
        ObservedOutcomeIndices observed_outcome_indices);

    // Dimensions
    /** @return Total outcomes T. */
    std::size_t T() const override { return T_; }
    /** @return Covariate dimension q (0 if covariates absent). */
    std::size_t q() const override { return q_; }
    /** @return Auxiliary dimension d (always 0 for cohort-level panels). */
    std::size_t d() const override { return 0; }
    /** @return Number of synthetic units (equals number of cohorts). */
    std::size_t num_units() const override { return num_units_; }

    // Grouping
    const std::vector<CohortBlock>& cohort_blocks() const override { return cohort_blocks_; }

    // Observed outcome indices
    const ObservedOutcomeIndices& observed_outcome_indices() const override { return observed_outcome_indices_; }

    // Per-cohort indices
    const arma::uvec& T_idx_for_cohort(std::size_t cohort_0b) const override {
        return observed_outcome_indices_.at(cohort_0b);
    }
    const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort(std::size_t cohort_0b) const override {
        return pos_T_idx_by_cohort_.at(cohort_0b);
    }

    // Assembly
    /**
     * @brief Assemble outcomes for a synthetic unit (cohort means only).
     *
     * @param ur UnitRun describing the cohort (single synthetic unit).
     * @param T_idxs_for_cohort Observed outcome indices for the cohort.
     * @param pos_T_idx_for_cohort Map from outcome index to position within T_idxs_for_cohort.
     * @param Y Output vector of cohort-average outcomes (length T_c).
     */
    void assemble_Y_for_unit(
        const UnitRun& ur,
        const arma::uvec& T_idxs_for_cohort,
        const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
        arma::vec& Y) const override;

    /**
     * @brief Assemble covariate means for a synthetic unit (if present).
     *
     * @param ur UnitRun describing the cohort (single synthetic unit).
     * @param T Total outcomes T.
     * @param T_idxs_for_cohort Observed outcome indices for the cohort.
     * @param X_full Output pointer for full T x q covariate means (may be null).
     * @param X_obs Output pointer for observed T_c x q covariate means (may be null).
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
     * @brief Assemble cohort-average outcomes and covariates together.
     *
     * @param ur UnitRun describing the cohort (single synthetic unit).
     * @param T Total outcomes T.
     * @param T_idxs_for_cohort Observed outcome indices for the cohort.
     * @param pos_T_idx_for_cohort Map from outcome index to position within T_idxs_for_cohort.
     * @param Y Output vector of cohort-average outcomes (length T_c).
     * @param X_full Output full covariate means (T x q).
     * @param X_obs Output observed covariate means (T_c x q).
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
     * @brief Assemble auxiliary data matrix (T x d) for a cohort (none present; fills zeros).
     *
     * @param ur UnitRun describing the cohort (single synthetic unit).
     * @param T Total outcomes T.
     * @param A_out Output auxiliary matrix (T x d), zero-filled (d=0 here).
     */
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


