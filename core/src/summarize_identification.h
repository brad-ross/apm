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

class InMemoryUnbalancedPanel; // fwd

struct OutcomeClusteringSummary {
  std::size_t largest_super_cohort_size;            // total units in largest final super cohort
  double       largest_super_cohort_share;          // share of panel units
  std::size_t  min_cohort_size_in_largest_super;    // min combined cohort size within that super cohort
  std::size_t  num_o3_iterations;                   // number of O^3 iterations (length of super_cohort_iterates)
};

std::vector<OutcomeClusteringSummary>
summarize_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    const std::vector<arma::uvec>& mappings,
    std::size_t max_model_rank);

OutcomeClusteringSummary
summarize_outcome_clusterings(
    const InMemoryUnbalancedPanel& panel,
    const arma::uvec& mapping,
    std::size_t max_model_rank);

} // namespace apm

#endif // APM_SUMMARIZE_IDENTIFICATION_H


