#include <RcppArmadillo.h>
#include "../../core/src/est_outcome_means.h"

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
bool ome_has_bootstrap_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeansEstimates> p(xp);
    return p->has_bootstrap_replicates();
}

// [[Rcpp::export]]
int ome_num_bootstrap_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeansEstimates> p(xp);
    return static_cast<int>(p->bootstrap_replicates.size());
}

// [[Rcpp::export]]
Rcpp::NumericMatrix ome_point_means_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeansEstimates> p(xp);
    const arma::mat& M = p->mean_outcomes;
    Rcpp::NumericMatrix out(M.n_rows, M.n_cols);
    std::copy(M.begin(), M.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix ome_boot_means_cpp(SEXP xp, int b1) {
    Rcpp::XPtr<apm::OutcomeMeansEstimates> p(xp);
    int B = static_cast<int>(p->bootstrap_replicates.size());
    if (b1 < 1 || b1 > B) Rcpp::stop("bootstrap index out of range");
    const arma::mat& M = p->bootstrap_replicates[static_cast<std::size_t>(b1 - 1)];
    Rcpp::NumericMatrix out(M.n_rows, M.n_cols);
    std::copy(M.begin(), M.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
int ome_C_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeansEstimates> p(xp);
    return static_cast<int>(p->mean_outcomes.n_rows);
}

// [[Rcpp::export]]
int ome_T_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeansEstimates> p(xp);
    return static_cast<int>(p->mean_outcomes.n_cols);
}


