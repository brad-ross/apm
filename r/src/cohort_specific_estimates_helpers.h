#pragma once
#ifndef APM_COHORT_SPECIFIC_ESTIMATES_HELPERS_H
#define APM_COHORT_SPECIFIC_ESTIMATES_HELPERS_H

#include <RcppArmadillo.h>
#include "r_utils.h"
#include "../../core/src/est_cohort_specific_params.h"

// Returns raw C++ outputs (no R wrapping). The implementation lives in
// est_cohort_specific_params_bindings.cpp and reuses its internal helpers.
apm::CohortSpecificEstimates cohort_specific_estimates_from_panel_cpp_core(
    SEXP panel_holder_xptr,
    Rcpp::List est_specs,
    SEXP bootstrap_xptr = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in = R_NilValue,
    Rcpp::Nullable<Rcpp::List> cohort_outcomes_to_mask_in = R_NilValue);

// Access observed_outcome_indices (0-based) from a panel holder XPtr
apm::ObservedOutcomeIndices observed_outcome_indices_from_panel_holder(SEXP panel_holder_xptr);

#endif // APM_COHORT_SPECIFIC_ESTIMATES_HELPERS_H