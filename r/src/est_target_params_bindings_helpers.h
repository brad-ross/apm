#pragma once
#ifndef APM_EST_TARGET_PARAMS_BINDINGS_HELPERS_H
#define APM_EST_TARGET_PARAMS_BINDINGS_HELPERS_H

#include <RcppArmadillo.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "r_utils.h"
#include "../../core/src/est_outcome_means.h"
#include "../../core/src/cohort_specific_param_structs.h"

namespace apm {
namespace r_bindings {

std::vector<CohortAuxiliaryDataMeanEstimates> list_to_eta_vec(Rcpp::Nullable<Rcpp::List> maybe_list);

std::unordered_map<std::string, OutcomeMeansEstimates> list_to_ome_map(Rcpp::List ome_by_spec);

std::unordered_map<std::string, std::vector<CohortAuxiliaryDataMeanEstimates>> list_to_eta_map(
    Rcpp::Nullable<Rcpp::List> eta_by_spec);

std::unordered_map<std::string, std::vector<apm::OutcomeMeanSuffStatEstimates>> list_to_stats_map(
    Rcpp::Nullable<Rcpp::List> stats_by_spec);

} // namespace r_bindings
} // namespace apm

#endif // APM_EST_TARGET_PARAMS_BINDINGS_HELPERS_H