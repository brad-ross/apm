#include <RcppArmadillo.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

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

namespace {

arma::uvec to_zero_based_uvec(const Rcpp::IntegerVector& idx, const char* arg_name) {
    if (idx.size() == 0) {
        Rcpp::stop("%s must contain at least one index.", arg_name);
    }
    arma::uvec out(static_cast<arma::uword>(idx.size()));
    for (R_xlen_t i = 0; i < idx.size(); ++i) {
        const int value = idx[i];
        if (value == NA_INTEGER || value < 1) {
            Rcpp::stop("%s must contain positive, non-missing indices.", arg_name);
        }
        out[static_cast<arma::uword>(i)] = static_cast<arma::uword>(value - 1);
    }
    return out;
}

std::optional<arma::vec> to_optional_weights(Rcpp::Nullable<Rcpp::NumericVector> maybe_weights) {
    if (maybe_weights.isNull()) {
        return std::nullopt;
    }
    Rcpp::NumericVector weights(maybe_weights);
    arma::vec out(static_cast<arma::uword>(weights.size()));
    for (R_xlen_t i = 0; i < weights.size(); ++i) {
        const double value = weights[i];
        if (!Rcpp::NumericVector::is_na(value)) {
            out[static_cast<arma::uword>(i)] = value;
        } else {
            Rcpp::stop("outcome_weights must not contain NA values.");
        }
    }
    return out;
}

} // namespace

// [[Rcpp::export]]
SEXP est_fgw_bipartite_match_outcome_diff_params_multi_cpp(SEXP ome_xptr,
                                                           Rcpp::Nullable<Rcpp::List> stats_xptrs_by_cohort,
                                                           Rcpp::Nullable<Rcpp::List> eta_xptrs_by_cohort,
                                                           Rcpp::IntegerVector outcome_indices_1,
                                                           Rcpp::IntegerVector outcome_indices_2,
                                                           Rcpp::List observed_outcome_indices_list,
                                                           Rcpp::Nullable<Rcpp::NumericVector> outcome_weights = R_NilValue)
{
    Rcpp::XPtr<OutcomeMeansEstimates> ome(ome_xptr);
    auto stats_vec = apm::r_utils::list_to_stats_vec(stats_xptrs_by_cohort);
    auto eta_vec = apm::r_bindings::list_to_eta_vec(eta_xptrs_by_cohort);
    apm::ObservedOutcomeIndices ooi = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices_list);

    arma::uvec idx1 = to_zero_based_uvec(outcome_indices_1, "outcome_indices_1");
    arma::uvec idx2 = to_zero_based_uvec(outcome_indices_2, "outcome_indices_2");
    std::optional<arma::vec> weights = to_optional_weights(outcome_weights);

    std::optional<std::size_t> nt_opt = std::optional<std::size_t>(1);
    TargetParameterEstimates out = apm::est_fgw_bipartite_match_outcome_diff_params(
        *ome,
        stats_vec,
        eta_vec,
        idx1,
        idx2,
        ooi,
        std::move(weights),
        nt_opt);

    return make_xptr(std::move(out));
}

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

    Rcpp::IntegerVector idx1 = Rcpp::IntegerVector::create(outcome_idx_1);
    Rcpp::IntegerVector idx2 = Rcpp::IntegerVector::create(outcome_idx_2);
    return est_fgw_bipartite_match_outcome_diff_params_multi_cpp(
        ome_xptr,
        stats_xptrs_by_cohort,
        eta_xptrs_by_cohort,
        idx1,
        idx2,
        observed_outcome_indices_list,
        R_NilValue);
}

// [[Rcpp::export]]
Rcpp::List est_fgw_bipartite_match_outcome_diff_params_by_spec_multi_cpp(
    Rcpp::List ome_by_spec,
    Rcpp::Nullable<Rcpp::List> stats_by_spec,
    Rcpp::Nullable<Rcpp::List> eta_by_spec,
    Rcpp::IntegerVector outcome_indices_1,
    Rcpp::IntegerVector outcome_indices_2,
    Rcpp::List observed_outcome_indices_list,
    Rcpp::Nullable<Rcpp::NumericVector> outcome_weights = R_NilValue)
{
    auto ome_map = apm::r_bindings::list_to_ome_map(ome_by_spec);
    auto eta_map = apm::r_bindings::list_to_eta_map(eta_by_spec);
    auto stats_map = apm::r_bindings::list_to_stats_map(stats_by_spec);

    apm::ObservedOutcomeIndices ooi = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices_list);
    arma::uvec idx1 = to_zero_based_uvec(outcome_indices_1, "outcome_indices_1");
    arma::uvec idx2 = to_zero_based_uvec(outcome_indices_2, "outcome_indices_2");
    std::optional<arma::vec> weights = to_optional_weights(outcome_weights);
    std::optional<std::size_t> nt_opt = std::optional<std::size_t>(1);

    auto out_map = apm::est_fgw_bipartite_match_outcome_diff_params(
        ome_map,
        stats_map,
        eta_map,
        idx1,
        idx2,
        ooi,
        std::move(weights),
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

    Rcpp::IntegerVector idx1 = Rcpp::IntegerVector::create(outcome_idx_1);
    Rcpp::IntegerVector idx2 = Rcpp::IntegerVector::create(outcome_idx_2);
    return est_fgw_bipartite_match_outcome_diff_params_by_spec_multi_cpp(
        ome_by_spec,
        stats_by_spec,
        eta_by_spec,
        idx1,
        idx2,
        observed_outcome_indices_list,
        R_NilValue);
}


