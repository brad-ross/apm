#pragma once
#ifndef APM_SUMMARIZE_IDENTIFICATION_H
#define APM_SUMMARIZE_IDENTIFICATION_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include "utils.h"            // ObservedOutcomeIndices

namespace apm {

struct IdentificationSummary {
  std::size_t largest_super_cohort_size;            // total units in largest final super cohort
  double       largest_super_cohort_share;          // share of panel units
  std::size_t  min_cohort_size_in_largest_super;    // min combined cohort size within that super cohort
  std::size_t  num_o3_iterations;                   // number of O^3 iterations (length of super_cohort_iterates)
};

std::vector<IdentificationSummary>
summarize_identification(
    const std::vector<ObservedOutcomeIndices>& observed_outcome_indices_vec,
    const std::vector<arma::uvec>& cohort_sizes_vec,
    std::size_t max_model_rank);

IdentificationSummary
summarize_identification(
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank);

} // namespace apm

#endif // APM_SUMMARIZE_IDENTIFICATION_H