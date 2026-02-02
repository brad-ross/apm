#pragma once
#ifndef APM_ABSTRACT_UNBALANCED_PANEL_H
#define APM_ABSTRACT_UNBALANCED_PANEL_H

//==============================================================================
// Panel abstraction for unbalanced cohort-outcome data used by APM estimators.
//
// Implementations expose consistent access to cohort grouping, observed outcome
// indices, and routines to assemble per-unit outcomes, covariates, and
// auxiliary data in the shapes expected by estimators and imputers.
// Panels are expected to store rows sorted by unit, and units sorted by cohort,
// so that cohort and unit blocks are contiguous for fast access.
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

namespace apm {

/**
 * @brief Half-open index range describing a contiguous run of rows for a unit.
 */
struct UnitRun {
    std::size_t start;  ///< Inclusive start row (0-based, half-open interval).
    std::size_t end;    ///< Exclusive end row.
    std::size_t unit;   ///< Unit id (0-based; may equal cohort id in synthetic panels).
};

/**
 * @brief Aggregated block of rows belonging to a cohort, with unit runs inside.
 */
struct CohortBlock {
    std::size_t start;               ///< Inclusive start row (0-based, half-open interval).
    std::size_t end;                 ///< Exclusive end row.
    std::size_t cohort;              ///< Cohort id (0-based).
    std::vector<UnitRun> unit_runs;  ///< Unit runs contained in this block.
};

/**
 * @brief Abstract interface for unbalanced panels supplied to estimation and imputation.
 *
 * Concrete panels provide dimension queries, cohort groupings, observed outcome
 * indices, and helper methods to assemble per-unit slices of outcomes,
 * covariates, and auxiliary data into contiguous Armadillo objects.
 */
class AbstractUnbalancedPanel {
public:
    virtual ~AbstractUnbalancedPanel() = default;

    // Dimensions
    /** @return Total number of outcomes T across the panel. */
    virtual std::size_t T() const = 0;
    /** @return Number of covariates q (0 if none). */
    virtual std::size_t q() const = 0;
    /** @return Number of auxiliary data columns d (0 if none). */
    virtual std::size_t d() const = 0;
    /** @return Number of units (unique unit ids) in the panel. */
    virtual std::size_t num_units() const = 0;

    // Grouping
    /** @return Cohort blocks delimiting contiguous row ranges. */
    virtual const std::vector<CohortBlock>& cohort_blocks() const = 0;

    // Observed outcome indices per cohort
    /** @return Observed outcome indices T_c for each cohort (0-based). */
    virtual const ObservedOutcomeIndices& observed_outcome_indices() const = 0;

    // Per-cohort observed outcome indices and position map
    /** @return Outcome indices observed in cohort_0b (0-based). */
    virtual const arma::uvec& T_idx_for_cohort(std::size_t cohort_0b) const = 0;
    /**
     * @return Map from outcome index -> position in T_idx_for_cohort for cohort_0b.
     */
    virtual const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort(std::size_t cohort_0b) const = 0;

    // Indexed assembly helpers
    /**
     * @brief Assemble outcomes Y for a unit into a length-T_c vector.
     *
     * @param ur UnitRun describing the row range and unit id.
     * @param T_idxs_for_cohort Observed outcome indices for the cohort.
     * @param pos_T_idx_for_cohort Map from outcome index to position within T_idxs_for_cohort.
     * @param Y Output vector (length T_c) to fill.
     */
    virtual void assemble_Y_for_unit(
        const UnitRun& ur,
        const arma::uvec& T_idxs_for_cohort,
        const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
        arma::vec& Y) const = 0;

    /**
     * @brief Assemble covariates X for a unit in both full-T and observed-only forms.
     *
     * @param ur UnitRun describing the row range and unit id.
     * @param T Total outcomes T.
     * @param T_idxs_for_cohort Observed outcome indices for the cohort.
     * @param X_full Output pointer for full T x q covariates (may be null).
     * @param X_obs Output pointer for observed T_c x q covariates (may be null).
     * @param covariate_index Optional single covariate column to extract; if unset,
     *        all covariates are populated.
     */
    virtual void assemble_X_for_unit(
        const UnitRun& ur,
        std::size_t T,
        const arma::uvec& T_idxs_for_cohort,
        arma::mat* X_full,
        arma::mat* X_obs,
        std::optional<std::size_t> covariate_index = std::nullopt) const = 0;

    /**
     * @brief Assemble outcomes and covariates together for a unit.
     *
     * @param ur UnitRun describing the row range and unit id.
     * @param T Total outcomes T.
     * @param T_idxs_for_cohort Observed outcome indices for the cohort.
     * @param pos_T_idx_for_cohort Map from outcome index to position within T_idxs_for_cohort.
     * @param Y Output vector (length T_c) to fill.
     * @param X_full Output matrix for full covariates (T x q).
     * @param X_obs Output matrix for observed covariates (T_c x q).
     */
    virtual void assemble_YX_for_unit(
        const UnitRun& ur,
        std::size_t T,
        const arma::uvec& T_idxs_for_cohort,
        const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
        arma::vec& Y,
        arma::mat& X_full,
        arma::mat& X_obs) const = 0;

    /**
     * @brief Assemble auxiliary data matrix (T x d) for a unit.
     *
     * @param ur UnitRun describing the row range and unit id.
     * @param T Total outcomes T.
     * @param A_out Output auxiliary matrix (T x d).
     */
    virtual void assemble_aux_for_unit(
        const UnitRun& ur,
        std::size_t T,
        arma::mat& A_out) const = 0;
};

} // namespace apm

#endif // APM_ABSTRACT_UNBALANCED_PANEL_H


