#include <RcppArmadillo.h>
#include "r_utils.h"
#include "../../core/src/summarize_identification.h"

// [[Rcpp::depends(RcppArmadillo)]]

namespace {

inline Rcpp::List id_summary_to_r(const apm::IdentificationSummary& s) {
    return Rcpp::List::create(
        Rcpp::Named("largest_super_cohort_size") = static_cast<double>(s.largest_super_cohort_size),
        Rcpp::Named("largest_super_cohort_share") = s.largest_super_cohort_share,
        Rcpp::Named("min_cohort_size_in_largest_super") = static_cast<double>(s.min_cohort_size_in_largest_super),
        Rcpp::Named("num_o3_iterations") = static_cast<double>(s.num_o3_iterations)
    );
}

} // anonymous namespace

//'
//' Summarize identification for one set of cohorts
//'
//' @param observed_outcome_indices A list of integer vectors (1-based) of observed outcomes per cohort
//' @param cohort_sizes An integer vector of cohort sizes (same length as observed_outcome_indices)
//' @param max_model_rank Maximum model rank (r)
//' @return A named list with identification summary fields
//' @export
// [[Rcpp::export]]
Rcpp::List summarize_identification_cpp(
    Rcpp::List observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank)
{
    apm::ObservedOutcomeIndices ooi0 = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    apm::IdentificationSummary s = apm::summarize_identification(ooi0, cohort_sizes, max_model_rank);
    return id_summary_to_r(s);
}

//'
//' Summarize identification for multiple panels
//'
//' @param observed_outcome_indices_list A list of lists as described in summarize_identification
//' @param cohort_sizes_list A list of integer vectors of cohort sizes (one per panel)
//' @param max_model_rank Maximum model rank (r)
//' @return A list of named lists, one per input panel
//' @export
// [[Rcpp::export]]
Rcpp::List summarize_identification_many_cpp(
    Rcpp::List observed_outcome_indices_list,
    Rcpp::List cohort_sizes_list,
    std::size_t max_model_rank)
{
    if (observed_outcome_indices_list.size() != cohort_sizes_list.size()) {
        Rcpp::stop("observed_outcome_indices_list and cohort_sizes_list must have same length");
    }

    const int n = observed_outcome_indices_list.size();
    std::vector<apm::ObservedOutcomeIndices> all_ooi;
    std::vector<arma::uvec> all_sizes;
    all_ooi.reserve(n);
    all_sizes.reserve(n);

    for (int i = 0; i < n; ++i) {
        Rcpp::List ooi_i = observed_outcome_indices_list[i];
        all_ooi.push_back(apm::r_utils::to_cpp_observed_outcome_indices(ooi_i));
        arma::uvec sz_i = Rcpp::as<arma::uvec>(cohort_sizes_list[i]);
        all_sizes.push_back(std::move(sz_i));
    }

    std::vector<apm::IdentificationSummary> v = apm::summarize_identification(all_ooi, all_sizes, max_model_rank);
    Rcpp::List out(n);
    for (int i = 0; i < n; ++i) out[i] = id_summary_to_r(v[static_cast<std::size_t>(i)]);
    return out;
}