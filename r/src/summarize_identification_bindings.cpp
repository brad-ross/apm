#include <RcppArmadillo.h>
#include "r_utils.h"
#include "../../core/src/summarize_identification.h"
#include <optional>

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

// Convert R-facing iteration selector to C++:
// - Positive k is 1-based in R and becomes (k-1) for C++ level indexing
// - Negative values (-1: final, -2: second-to-last, ...) are passed through
// - If NULL, return std::nullopt so caller can select the overload without iter
inline std::optional<int> r_iter_to_cpp(const Rcpp::Nullable<Rcpp::IntegerVector>& iter) {
    if (iter.isNull()) return std::nullopt;
    const int it = Rcpp::as<int>(iter.get());
    return (it > 0) ? std::optional<int>(it - 1) : std::optional<int>(it);
}

} // anonymous namespace

//'
//' Checks if the factors are identified across all cohorts.
//'
//' This function uses the O^3 algorithm to determine if there is sufficient
//' overlap in observed outcomes across all cohorts to uniquely identify all factor 
//' vectors expressed with respect to a common basis. Identification is achieved if 
//' the algorithm terminates with a single super cohort containing all of the 
//' original cohorts.
//'
//' @param observed_outcome_indices A list of integer vectors, where each
//'   vector contains the 1-based indices for the observed outcomes for a cohort.
//' @param r The model rank.
//' @return `TRUE` if the factors are identified, `FALSE` otherwise.
//' @export
// [[Rcpp::export]]
bool aligned_factors_identified(
    Rcpp::List observed_outcome_indices,
    unsigned int r)
{
    apm::ObservedOutcomeIndices ooi0 = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    return apm::aligned_factors_identified(ooi0, r);
}

//'
//' Return the largest super cohort's cohort indices at a given O^3 iteration
//'
//' @param observed_outcome_indices A list of integer vectors (1-based) of observed outcomes per cohort
//' @param cohort_sizes An integer vector of cohort sizes (same length as observed_outcome_indices)
//' @param max_model_rank Maximum model rank (r)
//' @param iter Optional iteration selector, defaults to -1 (final iteration):
//'   Positive k refers to the k-th iteration (1-based). -1 for final, -2 second-to-last, etc. Non-negative clamps.
//' @return An integer vector of 1-based cohort indices belonging to the largest super cohort at the selected iteration
//' @export
// [[Rcpp::export]]
arma::uvec get_largest_super_cohort(
    Rcpp::List observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    Rcpp::Nullable<Rcpp::IntegerVector> iter = R_NilValue)
{
    apm::ObservedOutcomeIndices ooi0 = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    arma::uvec res0;
    std::optional<int> it0 = r_iter_to_cpp(iter);
    if (!it0.has_value()) {
        res0 = apm::get_largest_super_cohort(ooi0, cohort_sizes, max_model_rank);
    } else {
        res0 = apm::get_largest_super_cohort(ooi0, cohort_sizes, max_model_rank, *it0);
    }
    // Return 1-based indices to R
    return res0 + 1;
}

//'
//' Summarize identification for one set of cohorts
//'
//' @param observed_outcome_indices A list of integer vectors (1-based) of observed outcomes per cohort
//' @param cohort_sizes An integer vector of cohort sizes (same length as observed_outcome_indices)
//' @param max_model_rank Maximum model rank (r)
//' @param iter Optional iteration selector, defaults to -1 (final iteration):
//'   Positive k refers to the k-th iteration (1-based). -1 for final, -2 second-to-last, etc. Non-negative clamps.
//' @return A named list with identification summary fields
//' @export
// [[Rcpp::export]]
Rcpp::List summarize_identification_cpp(
    Rcpp::List observed_outcome_indices,
    const arma::uvec& cohort_sizes,
    std::size_t max_model_rank,
    Rcpp::Nullable<Rcpp::IntegerVector> iter = R_NilValue)
{
    apm::ObservedOutcomeIndices ooi0 = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    apm::IdentificationSummary s;
    std::optional<int> it0 = r_iter_to_cpp(iter);
    if (!it0.has_value()) {
        s = apm::summarize_identification(ooi0, cohort_sizes, max_model_rank);
    } else {
        s = apm::summarize_identification(ooi0, cohort_sizes, max_model_rank, *it0);
    }
    return id_summary_to_r(s);
}

//'
//' Summarize identification for multiple panels
//'
//' @param observed_outcome_indices_list A list of lists as described in summarize_identification
//' @param cohort_sizes_list A list of integer vectors of cohort sizes (one per panel)
//' @param max_model_rank Maximum model rank (r)
//' @param iter Optional iteration selector, defaults to -1 (final iteration):
//'   Positive k refers to the k-th iteration (1-based). -1 for final, -2 second-to-last, etc. Non-negative clamps.
//' @return A list of named lists, one per input panel
//' @export
// [[Rcpp::export]]
Rcpp::List summarize_identification_many_cpp(
    Rcpp::List observed_outcome_indices_list,
    Rcpp::List cohort_sizes_list,
    std::size_t max_model_rank,
    Rcpp::Nullable<Rcpp::IntegerVector> iter = R_NilValue)
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

    std::vector<apm::IdentificationSummary> v;
    std::optional<int> it0 = r_iter_to_cpp(iter);
    if (!it0.has_value()) {
        v = apm::summarize_identification(all_ooi, all_sizes, max_model_rank);
    } else {
        v = apm::summarize_identification(all_ooi, all_sizes, max_model_rank, *it0);
    }
    Rcpp::List out(n);
    for (int i = 0; i < n; ++i) out[i] = id_summary_to_r(v[static_cast<std::size_t>(i)]);
    return out;
}