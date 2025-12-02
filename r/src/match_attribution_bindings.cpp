#include <RcppArmadillo.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "r_utils.h"
#include "est_target_params_bindings_helpers.h"
#include "../../core/src/target_params/match_attribution.h"
#include "../../core/src/est_outcome_means.h"
#include "../../core/src/cohort_specific_param_structs.h"

// [[Rcpp::depends(RcppArmadillo)]]

using apm::TargetParameterEstimates;
using apm::OutcomeMeansEstimates;
using apm::CohortAuxiliaryDataMeanEstimates;
using apm::r_utils::make_xptr;

// [[Rcpp::export]]
SEXP est_fgw_bipartite_match_outcome_diff_params_cpp(SEXP ome_xptr,
                                                     Rcpp::Nullable<Rcpp::List> stats_xptrs_by_cohort,
                                                     Rcpp::Nullable<Rcpp::List> eta_xptrs_by_cohort,
                                                     int outcome_idx_1,
                                                     int outcome_idx_2,
                                                     Rcpp::List observed_outcome_indices_list)
{
    if (outcome_idx_1 < 1 || outcome_idx_2 < 1) {
        Rcpp::stop("Outcome indices must be >= 1.");
    }
    Rcpp::XPtr<OutcomeMeansEstimates> ome(ome_xptr);
    auto stats_vec = apm::r_utils::list_to_stats_vec(stats_xptrs_by_cohort);
    auto eta_vec = apm::r_bindings::list_to_eta_vec(eta_xptrs_by_cohort);
    apm::ObservedOutcomeIndices ooi = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices_list);
    const std::size_t idx1 = static_cast<std::size_t>(outcome_idx_1 - 1);
    const std::size_t idx2 = static_cast<std::size_t>(outcome_idx_2 - 1);
    std::optional<std::size_t> nt_opt = std::optional<std::size_t>(1);
    TargetParameterEstimates out = apm::est_fgw_bipartite_match_outcome_diff_params(
        *ome,
        stats_vec,
        eta_vec,
        idx1,
        idx2,
        ooi,
        nt_opt);
    return make_xptr(std::move(out));
}

// [[Rcpp::export]]
Rcpp::List est_fgw_bipartite_match_outcome_diff_params_by_spec_cpp(
    Rcpp::List ome_by_spec,
    Rcpp::Nullable<Rcpp::List> stats_by_spec,
    Rcpp::Nullable<Rcpp::List> eta_by_spec,
    int outcome_idx_1,
    int outcome_idx_2,
    Rcpp::List observed_outcome_indices_list)
{
    if (outcome_idx_1 < 1 || outcome_idx_2 < 1) {
        Rcpp::stop("Outcome indices must be >= 1.");
    }

    auto ome_map = apm::r_bindings::list_to_ome_map(ome_by_spec);
    auto eta_map = apm::r_bindings::list_to_eta_map(eta_by_spec);
    auto stats_map = apm::r_bindings::list_to_stats_map(stats_by_spec);

    apm::ObservedOutcomeIndices ooi = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices_list);
    const std::size_t idx1 = static_cast<std::size_t>(outcome_idx_1 - 1);
    const std::size_t idx2 = static_cast<std::size_t>(outcome_idx_2 - 1);
    std::optional<std::size_t> nt_opt = std::optional<std::size_t>(1);

    auto out_map = apm::est_fgw_bipartite_match_outcome_diff_params(
        ome_map,
        stats_map,
        eta_map,
        idx1,
        idx2,
        ooi,
        nt_opt);

    Rcpp::List out(static_cast<int>(out_map.size()));
    Rcpp::CharacterVector names(static_cast<int>(out_map.size()));
    int k = 0;
    for (auto& kv : out_map) {
        names[k] = kv.first;
        out[k] = make_xptr(std::move(kv.second));
        ++k;
    }
    out.attr("names") = names;
    return out;
}


