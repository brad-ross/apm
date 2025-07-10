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
//' @param cohort_weights An optional numeric vector of weights for each cohort.
//'   If not provided, cohorts are weighted equally. The weights are scaled to 
//'   sum to one before use.
//' @export
// [[Rcpp::export]]
arma::mat align_factors_using_apm(
    Rcpp::List cohort_factor_matrices,
    Rcpp::List observed_outcome_indices,
    Rcpp::Nullable<Rcpp::NumericVector> cohort_weights = R_NilValue) {

    std::vector<arma::mat> cpp_factor_matrices;
    for (SEXP mat : cohort_factor_matrices) {
        cpp_factor_matrices.push_back(Rcpp::as<arma::mat>(mat));
    }

    std::vector<arma::uvec> cpp_observed_indices;
    for (SEXP vec : observed_outcome_indices) {
        // R is 1-based, C++ is 0-based.
        cpp_observed_indices.push_back(Rcpp::as<arma::uvec>(vec) - 1);
    }

    if (cohort_weights.isNotNull()) {
        arma::vec cpp_cohort_weights = Rcpp::as<arma::vec>(cohort_weights);
        return apm::align_factors_using_apm(cpp_factor_matrices, cpp_observed_indices, cpp_cohort_weights);
    } 

    return apm::align_factors_using_apm(cpp_factor_matrices, cpp_observed_indices);
} 

//' Impute outcomes for a representative unit
//'
//' @param G A T x r matrix of factors.
//' @param a A q-dimensional vector of covariate coefficients.
//' @param T_c A vector of 1-based indices for the observed time periods for the cohort.
//' @param m_c A vector containing the observed outcomes for the representative unit.
//' @param X_c A T x q matrix containing the values of q covariates corresponding to each outcome.
//' @return A T-dimensional vector containing the estimated outcomes for the representative unit.
//' @export
// [[Rcpp::export]]
arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::vec& a,
    const arma::uvec& T_c,
    const arma::vec& m_c,
    const arma::mat& X_c) {
    
    // R is 1-based, C++ is 0-based.
    return apm::impute_outcomes(G, a, T_c - 1, m_c, X_c);
}

//' Estimate mean outcomes across cohorts
//'
//' @param G A T x r matrix whose rows are estimated factor vectors.
//' @param a A q-dimensional vector of estimated covariate coefficients.
//' @param observed_outcome_indices A list where each element is a vector of 
//'                                 1-based indices for the observed time periods for a cohort.
//' @param m_c_vec A list of arma::vec, where each vector m_c contains the 
//'                observed outcomes for a cohort.
//' @param X_c_vec A list of T x q matrices, where each matrix X_c contains 
//'                the average values of q covariates for each outcome within a cohort.
//' @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
//' @export
// [[Rcpp::export]]
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& a,
    Rcpp::List observed_outcome_indices,
    Rcpp::List m_c_vec,
    Rcpp::List X_c_vec) {

    std::vector<arma::uvec> cpp_observed_indices;
    for (SEXP vec : observed_outcome_indices) {
        // R is 1-based, C++ is 0-based.
        cpp_observed_indices.push_back(Rcpp::as<arma::uvec>(vec) - 1);
    }

    std::vector<arma::vec> cpp_m_c_vec;
    for (SEXP vec : m_c_vec) {
        cpp_m_c_vec.push_back(Rcpp::as<arma::vec>(vec));
    }

    std::vector<arma::mat> cpp_X_c_vec;
    for (SEXP mat : X_c_vec) {
        cpp_X_c_vec.push_back(Rcpp::as<arma::mat>(mat));
    }
    
    return apm::estimate_outcome_means_across_cohorts(G, a, cpp_observed_indices, cpp_m_c_vec, cpp_X_c_vec);
} 