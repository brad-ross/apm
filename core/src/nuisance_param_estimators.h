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
    explicit CohortAuxiliaryDataMeanEstimator(std::size_t d,
                                              std::shared_ptr<const WeightedBootstrap> bootstrap = nullptr);

    std::size_t d() const noexcept { return d_; }
    bool has_bootstrap() const noexcept { return static_cast<bool>(bootstrap_); }
    std::size_t num_bootstraps() const noexcept { return bootstrap_ ? bootstrap_->n_bootstraps() : 0; }

    // Batch add: A is N_c x T x d (units x outcomes x aux-cols)
    void add_data(const arma::uvec& unit_idxs, const arma::cube& A);
    // Single unit add: A is T x d for the unit
    void add_datum(std::size_t unit_idx, const arma::mat& A);

    double row_count() const noexcept { return row_count_; }
    arma::vec boot_row_counts() const { return boot_row_counts_; }

    CohortAuxiliaryDataMeanEstimates estimate(std::size_t total_rows) const;

private:
    static void validate_data_dimensions(const arma::uvec& unit_idxs, const arma::cube& A, std::size_t d);

    std::size_t d_;
    std::shared_ptr<const WeightedBootstrap> bootstrap_;

    double row_count_;
    arma::vec aux_sum_;    // d x 1
    arma::vec aux_count_;  // d x 1

    arma::vec boot_row_counts_; // B x 1
    arma::mat boot_aux_sum_;    // d x B
    arma::mat boot_aux_weight_; // d x B
};

} // namespace apm

#endif // APM_NUISANCE_PARAM_ESTIMATORS_H


