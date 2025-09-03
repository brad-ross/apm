#include <RcppArmadillo.h>
#include "../../core/src/factor_model_parameter_structs.h"
#include "../../core/src/bootstrap.h"
#include "r_utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

// Factory: build OutcomeMeanSufficientStatEstimates from raw data
//' @export
// [[Rcpp::export]]
SEXP outcome_suff_from_data_cpp(
    const arma::mat& outcomes,
    SEXP wb_xptr = R_NilValue,
    Rcpp::Nullable<arma::cube> covars = R_NilValue,
    Rcpp::Nullable<arma::uvec> unit_idxs_1based = R_NilValue) {

    auto wb = apm::r_utils::xp_to_const_wb_shared(wb_xptr);

    arma::uvec unit_idxs;
    if (unit_idxs_1based.isNotNull()) {
        unit_idxs = Rcpp::as<arma::uvec>(unit_idxs_1based);
        if (unit_idxs.n_elem > 0) unit_idxs -= 1; // convert to 0-based
    }

    std::optional<arma::cube> covars_opt = std::nullopt;
    if (covars.isNotNull()) covars_opt = Rcpp::as<arma::cube>(covars);

    apm::OutcomeMeanSufficientStatEstimates est(outcomes, wb, covars_opt, unit_idxs);
    auto* heap = new apm::OutcomeMeanSufficientStatEstimates(std::move(est));
    return Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates>(heap, true);
}

// Presence and sizes on point estimates
//' @export
// [[Rcpp::export]]
bool omsse_has_bootstrap_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    return p->has_bootstrap_replicates();
}

//' @export
// [[Rcpp::export]]
std::size_t omsse_num_bootstrap_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    return p->n_bootstrap_replicates();
}

//' @export
// [[Rcpp::export]]
bool omsse_point_has_covar_means_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    return p->suff_stat_estimates.has_covar_means();
}

//' @export
// [[Rcpp::export]]
std::size_t omsse_point_Tc_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    return p->suff_stat_estimates.T_c();
}

//' @export
// [[Rcpp::export]]
std::size_t omsse_point_T_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    return p->suff_stat_estimates.T();
}

//' @export
// [[Rcpp::export]]
std::size_t omsse_point_q_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    return p->suff_stat_estimates.q();
}

// Point accessors
//' @export
// [[Rcpp::export]]
arma::vec omsse_point_observed_means_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    return p->suff_stat_estimates.observed_outcome_means;
}

//' @export
// [[Rcpp::export]]
arma::mat omsse_point_covar_means_cpp(SEXP xp) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    if (!p->suff_stat_estimates.covar_means) Rcpp::stop("covar_means not present");
    return *(p->suff_stat_estimates.covar_means);
}

// Bootstrap replicate accessors (1-based b)
//' @export
// [[Rcpp::export]]
arma::vec omsse_boot_observed_means_cpp(SEXP xp, std::size_t b) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    if (b < 1 || b > p->bootstrap_replicates.size()) Rcpp::stop("bootstrap index out of range");
    return p->bootstrap_replicates[b - 1].observed_outcome_means;
}

//' @export
// [[Rcpp::export]]
arma::mat omsse_boot_covar_means_cpp(SEXP xp, std::size_t b) {
    Rcpp::XPtr<apm::OutcomeMeanSufficientStatEstimates> p(xp);
    if (b < 1 || b > p->bootstrap_replicates.size()) Rcpp::stop("bootstrap index out of range");
    auto& rep = p->bootstrap_replicates[b - 1];
    if (!rep.covar_means) Rcpp::stop("covar_means not present in bootstrap replicates");
    return *(rep.covar_means);
}


