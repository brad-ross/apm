#include <RcppArmadillo.h>
#include "../../core/src/OutcomeMeanSuffStatEstimator.h"
#include "../../core/src/cohort_specific_param_structs.h"
#include "../../core/src/bootstrap.h"
#include "r_utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

//==============================================================================
// Factory: new OutcomeMeanSuffStatEstimator
//==============================================================================

// [[Rcpp::export]]
SEXP outcome_mean_estimator_new_cpp(std::size_t T_c,
                                    SEXP wb_xptr = R_NilValue,
                                    std::size_t T = 0,
                                    std::size_t q = 0) {
    auto wb = apm::r_utils::xp_to_const_wb_shared(wb_xptr);
    auto* ptr = new apm::OutcomeMeanSuffStatEstimator(T_c, T, q, wb);
    return Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimator>(ptr, true);
}

//==============================================================================
// Streaming operations on OutcomeMeanSuffStatEstimator
//==============================================================================

// [[Rcpp::export]]
void omsse_add_data_cpp(SEXP xp,
                        arma::uvec unit_idxs_1based,
                        const arma::mat& Y,
                        Rcpp::Nullable<arma::cube> X = R_NilValue) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimator> est(xp);
    arma::uvec unit_idxs = unit_idxs_1based;
    if (unit_idxs.n_elem > 0) unit_idxs -= 1;
    if (X.isNotNull()) {
        est->add_data(unit_idxs, Y, Rcpp::as<arma::cube>(X));
    } else {
        est->add_data(unit_idxs, Y, arma::cube());
    }
}

// [[Rcpp::export]]
void omsse_add_datum_cpp(SEXP xp,
                         std::size_t unit_idx_1based,
                         const arma::vec& Y,
                         Rcpp::Nullable<arma::mat> X = R_NilValue) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimator> est(xp);
    if (unit_idx_1based < 1) Rcpp::stop("unit_idx must be 1-based and >= 1");
    std::size_t unit_idx = unit_idx_1based - 1;
    if (X.isNotNull()) {
        est->add_datum(unit_idx, Y, Rcpp::as<arma::mat>(X));
    } else {
        est->add_datum(unit_idx, Y, arma::mat());
    }
}

// Convenience accessors on estimator
// [[Rcpp::export]]
std::size_t omsse_Tc_cpp(SEXP xp) {
    return Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimator>(xp)->T_c();
}
// [[Rcpp::export]]
std::size_t omsse_T_cpp(SEXP xp) {
    return Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimator>(xp)->T();
}
// [[Rcpp::export]]
std::size_t omsse_q_cpp(SEXP xp) {
    return Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimator>(xp)->q();
}
// [[Rcpp::export]]
std::size_t omsse_B_cpp(SEXP xp) {
    return Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimator>(xp)->num_bootstraps();
}

//==============================================================================
// Finalize: estimate() -> OutcomeMeanSuffStatEstimates
//==============================================================================

// [[Rcpp::export]]
SEXP omsse_estimate_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimator> est(xp);
    apm::OutcomeMeanSuffStatEstimates out = est->estimate();
    auto* heap = new apm::OutcomeMeanSuffStatEstimates(std::move(out));
    return Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates>(heap, true);
}

//==============================================================================
// OutcomeMeanSuffStatEstimates accessors (point + bootstrap)
//==============================================================================

// Presence/sizes
// [[Rcpp::export]]
bool omsse_has_bootstrap_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    return p->has_bootstrap_replicates();
}
// [[Rcpp::export]]
std::size_t omsse_num_bootstrap_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    return p->n_bootstrap_replicates();
}
// [[Rcpp::export]]
bool omsse_point_has_covar_means_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    return p->suff_stat_estimates.has_covar_means();
}
// [[Rcpp::export]]
std::size_t omsse_point_Tc_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    return p->suff_stat_estimates.T_c();
}
// [[Rcpp::export]]
std::size_t omsse_point_T_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    return p->suff_stat_estimates.T();
}
// [[Rcpp::export]]
std::size_t omsse_point_q_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    return p->suff_stat_estimates.q();
}

// Point accessors
// [[Rcpp::export]]
arma::vec omsse_point_observed_means_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    return p->suff_stat_estimates.observed_outcome_means;
}
// [[Rcpp::export]]
arma::mat omsse_point_covar_means_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    if (!p->suff_stat_estimates.covar_means) Rcpp::stop("covar_means not present");
    return *(p->suff_stat_estimates.covar_means);
}

// Bootstrap replicate accessors (1-based b)
// [[Rcpp::export]]
arma::vec omsse_boot_observed_means_cpp(SEXP xp, std::size_t b) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    if (b < 1 || b > p->n_bootstrap_replicates()) Rcpp::stop("bootstrap index out of range");
    return p->bootstrap_replicates[b - 1].observed_outcome_means;
}
// [[Rcpp::export]]
arma::mat omsse_boot_covar_means_cpp(SEXP xp, std::size_t b) {
    Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> p(xp);
    if (b < 1 || b > p->n_bootstrap_replicates()) Rcpp::stop("bootstrap index out of range");
    auto& rep = p->bootstrap_replicates[b - 1];
    if (!rep.covar_means) Rcpp::stop("covar_means not present in bootstrap replicates");
    return *(rep.covar_means);
}


