#include <RcppArmadillo.h>
#include "apm.h"

// [[Rcpp::depends(RcppArmadillo)]]

//' Get the version of the apm library
//' @export
// [[Rcpp::export]]
std::string get_version() {
    return apm::get_version();
}

//' Align factors using the Aggregated Projection Matrix method
//'
//' @param cohort_factor_matrices A list of matrices, one for each cohort.
//' @param observed_outcome_indices A list of integer vectors of the same length
//'   as cohort_factor_matrices. Each vector contains the 1-based indices
//'   indicating which time periods were observed for the corresponding cohort.
//' @export
// [[Rcpp::export]]
arma::mat align_factors_using_apm(
    Rcpp::List cohort_factor_matrices,
    Rcpp::List observed_outcome_indices) {

    std::vector<arma::mat> cpp_factor_matrices;
    for (SEXP mat : cohort_factor_matrices) {
        cpp_factor_matrices.push_back(Rcpp::as<arma::mat>(mat));
    }

    std::vector<arma::uvec> cpp_observed_indices;
    for (SEXP vec : observed_outcome_indices) {
        // R is 1-based, C++ is 0-based.
        cpp_observed_indices.push_back(Rcpp::as<arma::uvec>(vec) - 1);
    }

    return apm::align_factors_using_apm(cpp_factor_matrices, cpp_observed_indices);
} 