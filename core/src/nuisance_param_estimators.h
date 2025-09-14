#pragma once
#ifndef APM_NUISANCE_PARAM_ESTIMATORS_H
#define APM_NUISANCE_PARAM_ESTIMATORS_H

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "bootstrap.h"
#include "cohort_specific_param_structs.h"

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

namespace apm {

class CohortAuxiliaryDataMeanEstimator {
public:
    explicit CohortAuxiliaryDataMeanEstimator(std::size_t T,
                                              std::size_t d,
                                              std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr);

    std::size_t T() const noexcept { return T_; }
    std::size_t d() const noexcept { return d_; }
    bool has_bootstrap() const noexcept { return static_cast<bool>(bootstrap_); }
    std::size_t num_bootstraps() const noexcept { return bootstrap_ ? bootstrap_->n_bootstraps() : 0; }

    // Batch add: eta is N_c x T x d (units x outcomes x aux-cols)
    void add_data(const arma::uvec& unit_idxs, const arma::cube& eta);
    // Single unit add: eta is T x d for the unit
    void add_datum(std::size_t unit_idx, const arma::mat& eta);

    CohortAuxiliaryDataMeanEstimates estimate(std::size_t total_units) const;

private:
    static void validate_data_dimensions(const arma::uvec& unit_idxs, const arma::cube& eta, std::size_t T, std::size_t d);

    std::size_t T_;
    std::size_t d_;
    std::shared_ptr<const WeightedBootstrap> bootstrap_;

    // Running point-estimate aggregates
    double total_weight_;
    arma::mat aux_means_; // T x d

    // Running bootstrap aggregates
    arma::vec  total_boot_weights_; // B
    arma::cube boot_aux_means_;     // T x d x B
};

} // namespace apm

#endif // APM_NUISANCE_PARAM_ESTIMATORS_H


