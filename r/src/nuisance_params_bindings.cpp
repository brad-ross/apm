#include <RcppArmadillo.h>
#include <algorithm>
#include "r_utils.h"
#include "../../core/src/cohort_specific_param_structs.h"

// [[Rcpp::depends(RcppArmadillo)]]

using apm::CohortAuxiliaryDataMeanEstimates;
using apm::CohortAuxiliaryDataMeans;

// Helpers to deref
static CohortAuxiliaryDataMeanEstimates& ref(Rcpp::XPtr<CohortAuxiliaryDataMeanEstimates> xp) { return *xp; }

// Point dims
// [[Rcpp::export]]
Rcpp::IntegerVector caux_point_dims_cpp(SEXP xp_) {
    Rcpp::XPtr<CohortAuxiliaryDataMeanEstimates> xp(xp_);
    const arma::mat& M = ref(xp).estimates.auxiliary_means;
    return Rcpp::IntegerVector::create(static_cast<int>(M.n_rows), static_cast<int>(M.n_cols));
}

// Point pop share
// [[Rcpp::export]]
double caux_point_pop_share_cpp(SEXP xp_) {
    Rcpp::XPtr<CohortAuxiliaryDataMeanEstimates> xp(xp_);
    return ref(xp).estimates.cohort_pop_share;
}

// Point matrix
// [[Rcpp::export]]
Rcpp::NumericMatrix caux_point_aux_means_cpp(SEXP xp_) {
    Rcpp::XPtr<CohortAuxiliaryDataMeanEstimates> xp(xp_);
    const arma::mat& M = ref(xp).estimates.auxiliary_means;
    Rcpp::NumericMatrix out(static_cast<int>(M.n_rows), static_cast<int>(M.n_cols));
    std::copy(M.begin(), M.end(), out.begin());
    return out;
}

// Bootstrap presence and count
// [[Rcpp::export]]
bool caux_has_bootstrap_cpp(SEXP xp_) {
    Rcpp::XPtr<CohortAuxiliaryDataMeanEstimates> xp(xp_);
    return ref(xp).has_bootstrap_replicates();
}

// [[Rcpp::export]]
int caux_num_bootstrap_cpp(SEXP xp_) {
    Rcpp::XPtr<CohortAuxiliaryDataMeanEstimates> xp(xp_);
    return static_cast<int>(ref(xp).n_bootstrap_replicates());
}

// Bootstrap replicate pop share
// [[Rcpp::export]]
double caux_boot_pop_share_cpp(SEXP xp_, int b) {
    Rcpp::XPtr<CohortAuxiliaryDataMeanEstimates> xp(xp_);
    const auto& v = ref(xp).bootstrap_replicates;
    if (b < 1 || b > static_cast<int>(v.size())) Rcpp::stop("bootstrap index out of range");
    return v[static_cast<std::size_t>(b - 1)].cohort_pop_share;
}

// Bootstrap replicate matrix
// [[Rcpp::export]]
Rcpp::NumericMatrix caux_boot_aux_means_cpp(SEXP xp_, int b) {
    Rcpp::XPtr<CohortAuxiliaryDataMeanEstimates> xp(xp_);
    const auto& v = ref(xp).bootstrap_replicates;
    if (b < 1 || b > static_cast<int>(v.size())) Rcpp::stop("bootstrap index out of range");
    const arma::mat& M = v[static_cast<std::size_t>(b - 1)].auxiliary_means;
    Rcpp::NumericMatrix out(static_cast<int>(M.n_rows), static_cast<int>(M.n_cols));
    std::copy(M.begin(), M.end(), out.begin());
    return out;
}


