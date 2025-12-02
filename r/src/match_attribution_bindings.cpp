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

std::vector<arma::uvec> to_zero_based_groupings(const Rcpp::List& groupings, const char* arg_name) {
    if (groupings.size() < 2) {
        Rcpp::stop("%s must contain at least two groups.", arg_name);
    }
    std::vector<arma::uvec> out;
    out.reserve(static_cast<std::size_t>(groupings.size()));
    for (R_xlen_t i = 0; i < groupings.size(); ++i) {
        if (groupings[i] == R_NilValue) {
            Rcpp::stop("%s elements must be integer vectors.", arg_name);
        }
        Rcpp::IntegerVector idx = groupings[i];
        const std::string elem_name = std::string(arg_name) + "[" + std::to_string(i + 1) + "]";
        out.push_back(to_zero_based_uvec(idx, elem_name.c_str()));
    }
    return out;
}

std::vector<arma::uvec> singleton_groupings_from_indices(const arma::uvec& indices, const char* arg_name) {
    if (indices.n_elem < 2) {
        Rcpp::stop("%s must contain at least two indices.", arg_name);
    }
    std::vector<arma::uvec> out;
    out.reserve(static_cast<std::size_t>(indices.n_elem));
    for (arma::uword i = 0; i < indices.n_elem; ++i) {
        arma::uvec group(1);
        group[0] = indices[i];
        out.push_back(std::move(group));
    }
    return out;
}

std::optional<std::size_t> to_num_threads_opt(int num_threads) {
    if (num_threads < 1) {
        Rcpp::stop("num_threads must be >= 1.");
    }
    return std::optional<std::size_t>(static_cast<std::size_t>(num_threads));
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
    Rcpp::Nullable<Rcpp::List> stats_xptrs_by_cohort,
    Rcpp::Nullable<Rcpp::List> eta_xptrs_by_cohort,
    Rcpp::IntegerVector outcome_indices_1,
    Rcpp::IntegerVector outcome_indices_2,
    Rcpp::List observed_outcome_indices_list,
    Rcpp::Nullable<Rcpp::NumericVector> outcome_weights = R_NilValue)
{
    auto ome_map = apm::r_bindings::list_to_ome_map(ome_by_spec);
    auto eta_vec = apm::r_bindings::list_to_eta_vec(eta_xptrs_by_cohort);
    auto stats_vec = apm::r_utils::list_to_stats_vec(stats_xptrs_by_cohort);

    apm::ObservedOutcomeIndices ooi = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices_list);
    arma::uvec idx1 = to_zero_based_uvec(outcome_indices_1, "outcome_indices_1");
    arma::uvec idx2 = to_zero_based_uvec(outcome_indices_2, "outcome_indices_2");
    std::optional<arma::vec> weights = to_optional_weights(outcome_weights);
    std::optional<std::size_t> nt_opt = std::optional<std::size_t>(1);

    auto out_map = apm::est_fgw_bipartite_match_outcome_diff_params(
        ome_map,
        stats_vec,
        eta_vec,
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
    return est_fgw_bipartite_match_outcome_diff_params_by_spec_multi_cpp(
        ome_by_spec,
        stats_xptrs_by_cohort,
        eta_xptrs_by_cohort,
        idx1,
        idx2,
        observed_outcome_indices_list,
        R_NilValue);
}

// [[Rcpp::export]]
SEXP est_avg_fgw_bipartite_match_outcome_diff_params_multi_cpp(
    SEXP ome_xptr,
    Rcpp::Nullable<Rcpp::List> stats_xptrs_by_cohort,
    Rcpp::Nullable<Rcpp::List> eta_xptrs_by_cohort,
    Rcpp::List outcome_groupings,
    Rcpp::List observed_outcome_indices_list,
    Rcpp::Nullable<Rcpp::NumericVector> outcome_weights = R_NilValue,
    int num_threads = 1)
{
    Rcpp::XPtr<OutcomeMeansEstimates> ome(ome_xptr);
    auto stats_vec = apm::r_utils::list_to_stats_vec(stats_xptrs_by_cohort);
    auto eta_vec = apm::r_bindings::list_to_eta_vec(eta_xptrs_by_cohort);
    apm::ObservedOutcomeIndices ooi = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices_list);
    std::vector<arma::uvec> groupings = to_zero_based_groupings(outcome_groupings, "outcome_groupings");
    std::optional<arma::vec> weights = to_optional_weights(outcome_weights);
    std::optional<std::size_t> nt_opt = to_num_threads_opt(num_threads);

    TargetParameterEstimates out = apm::est_avg_fgw_bipartite_match_outcome_diff_params(
        *ome,
        stats_vec,
        eta_vec,
        groupings,
        ooi,
        std::move(weights),
        nt_opt);

    return make_xptr(std::move(out));
}

// [[Rcpp::export]]
SEXP est_avg_fgw_bipartite_match_outcome_diff_params_cpp(
    SEXP ome_xptr,
    Rcpp::Nullable<Rcpp::List> stats_xptrs_by_cohort,
    Rcpp::Nullable<Rcpp::List> eta_xptrs_by_cohort,
    Rcpp::IntegerVector outcome_indices,
    Rcpp::List observed_outcome_indices_list,
    Rcpp::Nullable<Rcpp::NumericVector> outcome_weights = R_NilValue,
    int num_threads = 1)
{
    Rcpp::XPtr<OutcomeMeansEstimates> ome(ome_xptr);
    auto stats_vec = apm::r_utils::list_to_stats_vec(stats_xptrs_by_cohort);
    auto eta_vec = apm::r_bindings::list_to_eta_vec(eta_xptrs_by_cohort);
    apm::ObservedOutcomeIndices ooi = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices_list);
    arma::uvec indices = to_zero_based_uvec(outcome_indices, "outcome_indices");
    std::vector<arma::uvec> groupings = singleton_groupings_from_indices(indices, "outcome_indices");
    std::optional<arma::vec> weights = to_optional_weights(outcome_weights);
    std::optional<std::size_t> nt_opt = to_num_threads_opt(num_threads);

    TargetParameterEstimates out = apm::est_avg_fgw_bipartite_match_outcome_diff_params(
        *ome,
        stats_vec,
        eta_vec,
        groupings,
        ooi,
        std::move(weights),
        nt_opt);

    return make_xptr(std::move(out));
}

// [[Rcpp::export]]
Rcpp::List est_avg_fgw_bipartite_match_outcome_diff_params_by_spec_multi_cpp(
    Rcpp::List ome_by_spec,
    Rcpp::Nullable<Rcpp::List> stats_xptrs_by_cohort,
    Rcpp::Nullable<Rcpp::List> eta_xptrs_by_cohort,
    Rcpp::List outcome_groupings,
    Rcpp::List observed_outcome_indices_list,
    Rcpp::Nullable<Rcpp::NumericVector> outcome_weights = R_NilValue,
    int num_threads = 1)
{
    auto ome_map = apm::r_bindings::list_to_ome_map(ome_by_spec);
    auto eta_vec = apm::r_bindings::list_to_eta_vec(eta_xptrs_by_cohort);
    auto stats_vec = apm::r_utils::list_to_stats_vec(stats_xptrs_by_cohort);

    apm::ObservedOutcomeIndices ooi = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices_list);
    std::vector<arma::uvec> groupings = to_zero_based_groupings(outcome_groupings, "outcome_groupings");
    std::optional<arma::vec> weights = to_optional_weights(outcome_weights);
    std::optional<std::size_t> nt_opt = to_num_threads_opt(num_threads);

    auto out_map = apm::est_avg_fgw_bipartite_match_outcome_diff_params(
        ome_map,
        stats_vec,
        eta_vec,
        groupings,
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
Rcpp::List est_avg_fgw_bipartite_match_outcome_diff_params_by_spec_cpp(
    Rcpp::List ome_by_spec,
    Rcpp::Nullable<Rcpp::List> stats_xptrs_by_cohort,
    Rcpp::Nullable<Rcpp::List> eta_xptrs_by_cohort,
    Rcpp::IntegerVector outcome_indices,
    Rcpp::List observed_outcome_indices_list,
    Rcpp::Nullable<Rcpp::NumericVector> outcome_weights = R_NilValue,
    int num_threads = 1)
{
    auto ome_map = apm::r_bindings::list_to_ome_map(ome_by_spec);
    auto eta_vec = apm::r_bindings::list_to_eta_vec(eta_xptrs_by_cohort);
    auto stats_vec = apm::r_utils::list_to_stats_vec(stats_xptrs_by_cohort);

    apm::ObservedOutcomeIndices ooi = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices_list);
    arma::uvec indices = to_zero_based_uvec(outcome_indices, "outcome_indices");
    std::vector<arma::uvec> groupings = singleton_groupings_from_indices(indices, "outcome_indices");
    std::optional<arma::vec> weights = to_optional_weights(outcome_weights);
    std::optional<std::size_t> nt_opt = to_num_threads_opt(num_threads);

    auto out_map = apm::est_avg_fgw_bipartite_match_outcome_diff_params(
        ome_map,
        stats_vec,
        eta_vec,
        groupings,
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