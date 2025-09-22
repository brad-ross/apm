#pragma once
#ifndef APM_ABSTRACT_UNBALANCED_PANEL_H
#define APM_ABSTRACT_UNBALANCED_PANEL_H

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

struct UnitRun {
    std::size_t start;  // [start, end)
    std::size_t end;
    std::size_t unit;   // 0-based; for synthetic panels may equal cohort index
};

struct CohortBlock {
    std::size_t start;  // [start, end)
    std::size_t end;
    std::size_t cohort; // 0-based
    std::vector<UnitRun> unit_runs;
};

class AbstractUnbalancedPanel {
public:
    virtual ~AbstractUnbalancedPanel() = default;

    // Dimensions
    virtual std::size_t T() const = 0;
    virtual std::size_t q() const = 0;
    virtual std::size_t d() const = 0;
    virtual std::size_t num_units() const = 0;

    // Grouping
    virtual const std::vector<CohortBlock>& cohort_blocks() const = 0;

    // Per-cohort observed outcome indices and position map
    virtual const arma::uvec& T_idx_for_cohort(std::size_t cohort_0b) const = 0;
    virtual const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort(std::size_t cohort_0b) const = 0;

    // Indexed assembly helpers
    virtual void assemble_Y_for_unit(
        const UnitRun& ur,
        const arma::uvec& T_idxs_for_cohort,
        const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
        arma::vec& Y) const = 0;

    virtual void assemble_X_for_unit(
        const UnitRun& ur,
        std::size_t T,
        const arma::uvec& T_idxs_for_cohort,
        arma::mat* X_full,
        arma::mat* X_obs,
        std::optional<std::size_t> covariate_index = std::nullopt) const = 0;

    virtual void assemble_YX_for_unit(
        const UnitRun& ur,
        std::size_t T,
        const arma::uvec& T_idxs_for_cohort,
        const std::unordered_map<int, std::size_t>& pos_T_idx_for_cohort,
        arma::vec& Y,
        arma::mat& X_full,
        arma::mat& X_obs) const = 0;

    virtual void assemble_aux_for_unit(
        const UnitRun& ur,
        std::size_t T,
        arma::mat& A_out) const = 0;
};

} // namespace apm

#endif // APM_ABSTRACT_UNBALANCED_PANEL_H


