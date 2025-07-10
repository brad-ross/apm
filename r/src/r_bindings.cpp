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
//' This function supports various combinations of factors, fixed effects, and covariates.
//'
//' @param G A T x r matrix of factors.
//' @param T_c A vector of 1-based indices for the observed time periods for the cohort.
//' @param m_c A vector containing the observed outcomes for the representative unit.
//' @param g_0 An optional T-dimensional vector of estimated outcome fixed effects.
//' @param a An optional q-dimensional vector of estimated covariate coefficients.
//' @param X_c An optional T x q matrix of covariates. Required if 'a' is provided.
//' @return A T-dimensional vector containing the estimated outcomes for the representative unit.
//' @export
// [[Rcpp::export]]
arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::uvec& T_c,
    const arma::vec& m_c,
    Rcpp::Nullable<Rcpp::NumericVector> g_0 = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> a = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> X_c = R_NilValue) {
    
    arma::uvec T_c_zero_based = T_c - 1;

    bool has_g0 = g_0.isNotNull();
    bool has_a = a.isNotNull();

    if (has_a && !X_c.isNotNull()) {
        Rcpp::stop("If 'a' is provided, 'X_c' must also be provided.");
    }
    if (!has_a && X_c.isNotNull()) {
        Rcpp::warning("'X_c' is provided but 'a' is not; covariates will be ignored.");
    }

    if (has_g0 && has_a) {
        arma::vec g_0_cpp = Rcpp::as<arma::vec>(g_0);
        arma::vec a_cpp = Rcpp::as<arma::vec>(a);
        arma::mat X_c_cpp = Rcpp::as<arma::mat>(X_c);
        return apm::impute_outcomes(G, g_0_cpp, a_cpp, T_c_zero_based, m_c, X_c_cpp);
    } else if (has_g0) {
        arma::vec g_0_cpp = Rcpp::as<arma::vec>(g_0);
        return apm::impute_outcomes(G, g_0_cpp, T_c_zero_based, m_c);
    } else if (has_a) {
        arma::vec a_cpp = Rcpp::as<arma::vec>(a);
        arma::mat X_c_cpp = Rcpp::as<arma::mat>(X_c);
        return apm::impute_outcomes(G, a_cpp, T_c_zero_based, m_c, X_c_cpp);
    } else {
        return apm::impute_outcomes(G, T_c_zero_based, m_c);
    }
}

//' Estimate mean outcomes across cohorts
//'
//' This function supports various combinations of factors, fixed effects, and covariates.
//'
//' @param G A T x r matrix whose rows are estimated factor vectors.
//' @param observed_outcome_indices A list where each element is a vector of 
//'                                 1-based indices for the observed time periods for a cohort.
//' @param m_c_vec A list of arma::vec, where each vector m_c contains the 
//'                observed outcomes for a cohort.
//' @param g_0 An optional T-dimensional vector of estimated outcome fixed effects.
//' @param a An optional q-dimensional vector of estimated covariate coefficients.
//' @param X_c_vec An optional list of T x q matrices of covariates. Required if 'a' is provided.
//' @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
//' @export
// [[Rcpp::export]]
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    Rcpp::List observed_outcome_indices,
    Rcpp::List m_c_vec,
    Rcpp::Nullable<Rcpp::NumericVector> g_0 = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> a = R_NilValue,
    Rcpp::Nullable<Rcpp::List> X_c_vec = R_NilValue) {

    std::vector<arma::uvec> cpp_observed_indices;
    for (SEXP vec : observed_outcome_indices) {
        // R is 1-based, C++ is 0-based.
        cpp_observed_indices.push_back(Rcpp::as<arma::uvec>(vec) - 1);
    }

    std::vector<arma::vec> cpp_m_c_vec;
    for (SEXP vec : m_c_vec) {
        cpp_m_c_vec.push_back(Rcpp::as<arma::vec>(vec));
    }

    bool has_g0 = g_0.isNotNull();
    bool has_a = a.isNotNull();

    if (has_a && !X_c_vec.isNotNull()) {
        Rcpp::stop("If 'a' is provided, 'X_c_vec' must also be provided.");
    }
     if (!has_a && X_c_vec.isNotNull()) {
        Rcpp::stop("If 'X_c_vec' is provided, 'a' must also be provided.");
    }

    if (has_g0 && has_a) {
        arma::vec g_0_cpp = Rcpp::as<arma::vec>(g_0);
        arma::vec a_cpp = Rcpp::as<arma::vec>(a);
        std::vector<arma::mat> cpp_X_c_vec;
        Rcpp::List r_X_c_vec(X_c_vec);
        for (SEXP mat : r_X_c_vec) {
            cpp_X_c_vec.push_back(Rcpp::as<arma::mat>(mat));
        }
        return apm::estimate_outcome_means_across_cohorts(G, g_0_cpp, a_cpp, cpp_observed_indices, cpp_m_c_vec, cpp_X_c_vec);
    } else if (has_g0) {
        arma::vec g_0_cpp = Rcpp::as<arma::vec>(g_0);
        return apm::estimate_outcome_means_across_cohorts(G, g_0_cpp, cpp_observed_indices, cpp_m_c_vec);
    } else if (has_a) {
        arma::vec a_cpp = Rcpp::as<arma::vec>(a);
        std::vector<arma::mat> cpp_X_c_vec;
        Rcpp::List r_X_c_vec(X_c_vec);
        for (SEXP mat : r_X_c_vec) {
            cpp_X_c_vec.push_back(Rcpp::as<arma::mat>(mat));
        }
        return apm::estimate_outcome_means_across_cohorts(G, a_cpp, cpp_observed_indices, cpp_m_c_vec, cpp_X_c_vec);
    } else {
        return apm::estimate_outcome_means_across_cohorts(G, cpp_observed_indices, cpp_m_c_vec);
    }
} 