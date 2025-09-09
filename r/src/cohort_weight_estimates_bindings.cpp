#include <RcppArmadillo.h>
#include "../../core/src/cohort_specific_param_structs.h"

// [[Rcpp::depends(RcppArmadillo)]]

//==============================================================================
// CohortWeightEstimates accessors (point + bootstrap)
//==============================================================================

// [[Rcpp::export]]
bool cwe_has_bootstrap_cpp(SEXP xp) {
    Rcpp::XPtr<apm::CohortWeightEstimates> p(xp);
    return p->has_bootstrap_replicates();
}

// [[Rcpp::export]]
std::size_t cwe_num_bootstrap_cpp(SEXP xp) {
    Rcpp::XPtr<apm::CohortWeightEstimates> p(xp);
    return p->n_bootstrap_replicates();
}

// [[Rcpp::export]]
arma::vec cwe_point_weights_cpp(SEXP xp) {
    Rcpp::XPtr<apm::CohortWeightEstimates> p(xp);
    return p->cohort_weights;
}

// [[Rcpp::export]]
arma::vec cwe_boot_weights_cpp(SEXP xp, std::size_t b) {
    Rcpp::XPtr<apm::CohortWeightEstimates> p(xp);
    if (b < 1 || b > p->n_bootstrap_replicates()) Rcpp::stop("bootstrap index out of range");
    return p->bootstrap_cohort_weights[b - 1];
}


