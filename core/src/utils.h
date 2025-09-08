#pragma once
#ifndef APM_UTILS_H
#define APM_UTILS_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <vector>

namespace apm {

// Central alias for observed outcome indices per cohort (0-based)
using ObservedOutcomeIndices = std::vector<arma::uvec>;

// Count total outcomes T across all cohorts (1 + max index if any, else 0)
arma::uword num_outcomes(const ObservedOutcomeIndices& observed_outcome_indices);

} // namespace apm

#endif // APM_UTILS_H


