#include <RcppArmadillo.h>
#include "../../../core/src/factor_model_estimators/FactorModelEstimator.h"
#include "../../../core/src/FactorModelParameters.h"
#include "../../../core/src/bootstrap.h"
#include "../r_utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

//==============================================================================
// Generic estimator operations
//==============================================================================

// [[Rcpp::export]]
void estimator_add_data_cpp(SEXP xp, arma::uvec unit_idxs_1based, const arma::mat& Y, Rcpp::Nullable<arma::cube> X = R_NilValue) {
    Rcpp::XPtr<apm::FactorModelEstimator> est(xp);
    arma::uvec unit_idxs = unit_idxs_1based;
    if (unit_idxs.n_elem > 0) unit_idxs -= 1;
    if (X.isNotNull()) {
        est->add_data(unit_idxs, Y, Rcpp::as<arma::cube>(X));
    } else {
        est->add_data(unit_idxs, Y, arma::cube());
    }
}

// [[Rcpp::export]]
void estimator_add_datum_cpp(SEXP xp, std::size_t unit_idx_1based, const arma::vec& Y, Rcpp::Nullable<arma::mat> X = R_NilValue) {
    Rcpp::XPtr<apm::FactorModelEstimator> est(xp);
    if (unit_idx_1based < 1) Rcpp::stop("unit_idx must be 1-based and >= 1");
    std::size_t unit_idx = unit_idx_1based - 1;
    if (X.isNotNull()) {
        est->add_datum(unit_idx, Y, Rcpp::as<arma::mat>(X));
    } else {
        est->add_datum(unit_idx, Y, arma::mat());
    }
}

// Convenience accessors
// [[Rcpp::export]]
std::size_t estimator_r_cpp(SEXP xp){
    Rcpp::XPtr<apm::FactorModelEstimator> e(xp);
    return e->r();
}
// [[Rcpp::export]]
std::size_t estimator_Tc_cpp(SEXP xp){
    Rcpp::XPtr<apm::FactorModelEstimator> e(xp);
    return e->T_c();
}
// [[Rcpp::export]]
std::size_t estimator_q_cpp(SEXP xp){
    Rcpp::XPtr<apm::FactorModelEstimator> e(xp);
    return e->q();
}
// [[Rcpp::export]]
std::size_t estimator_B_cpp(SEXP xp){
    Rcpp::XPtr<apm::FactorModelEstimator> e(xp);
    return e->num_bootstraps();
}

//==============================================================================
// FactorModelEstimates binding
//==============================================================================

// [[Rcpp::export]]
SEXP estimator_estimate_cpp(SEXP xp) {
    Rcpp::XPtr<apm::FactorModelEstimator> est(xp);
    apm::FactorModelEstimates out = est->estimate();
    auto* heap = new apm::FactorModelEstimates(std::move(out));
    return Rcpp::XPtr<apm::FactorModelEstimates>(heap, true);
}

// Presence and sizes
// [[Rcpp::export]]
bool fme_has_bootstrap_cpp(SEXP fme){
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    return p->has_bootstrap_replicates();
}
// [[Rcpp::export]]
std::size_t fme_num_bootstrap_cpp(SEXP fme){
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    return p->bootstrap_replicates.size();
}
// [[Rcpp::export]]
bool fme_has_g0_cpp(SEXP fme){
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    return p->parameter_estimates.g_0.has_value();
}
// [[Rcpp::export]]
bool fme_has_a_cpp(SEXP fme){
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    return p->parameter_estimates.a.has_value();
}

// Point estimates
// [[Rcpp::export]]
arma::mat fme_point_G_cpp(SEXP fme){
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    return p->parameter_estimates.G;
}
// [[Rcpp::export]]
arma::vec fme_point_g0_cpp(SEXP fme){
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    if(!p->parameter_estimates.g_0) Rcpp::stop("g_0 not present");
    return *(p->parameter_estimates.g_0);
}
// [[Rcpp::export]]
arma::vec fme_point_a_cpp(SEXP fme){
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    if(!p->parameter_estimates.a) Rcpp::stop("a not present");
    return *(p->parameter_estimates.a);
}

// Bootstrap replicates (1-based b)
// [[Rcpp::export]]
arma::mat fme_boot_G_cpp(SEXP fme, std::size_t b) {
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    if (b < 1 || b > p->bootstrap_replicates.size()) Rcpp::stop("bootstrap index out of range");
    return p->bootstrap_replicates[b - 1].G;
}

// [[Rcpp::export]]
arma::vec fme_boot_g0_cpp(SEXP fme, std::size_t b) {
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    if (b < 1 || b > p->bootstrap_replicates.size()) Rcpp::stop("bootstrap index out of range");
    auto& rep = p->bootstrap_replicates[b - 1];
    if (!rep.g_0) Rcpp::stop("g_0 not present in bootstrap replicates");
    return *(rep.g_0);
}

// [[Rcpp::export]]
arma::vec fme_boot_a_cpp(SEXP fme, std::size_t b) {
    Rcpp::XPtr<apm::FactorModelEstimates> p(fme);
    if (b < 1 || b > p->bootstrap_replicates.size()) Rcpp::stop("bootstrap index out of range");
    auto& rep = p->bootstrap_replicates[b - 1];
    if (!rep.a) Rcpp::stop("a not present in bootstrap replicates");
    return *(rep.a);
}


