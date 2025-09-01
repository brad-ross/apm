#include <RcppArmadillo.h>
#include "../../../core/src/factor_model_estimators/pc_estimators.h"
#include "../../../core/src/bootstrap.h"
#include "../r_utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

//==============================================================================
// Estimator factories for PC estimators
//==============================================================================

// [[Rcpp::export]]
SEXP pc_estimator_new_cpp(std::size_t r, std::size_t T_c, SEXP wb_xptr = R_NilValue, std::size_t q = 0) {
    auto wb = apm::r_utils::xp_to_const_wb_shared(wb_xptr);
    auto* ptr = new apm::PCEstimator(r, T_c, wb, q);
    return Rcpp::XPtr<apm::PCEstimator>(ptr, true);
}

// [[Rcpp::export]]
SEXP pc_fe_estimator_new_cpp(std::size_t r, std::size_t T_c, SEXP wb_xptr = R_NilValue, std::size_t q = 0) {
    auto wb = apm::r_utils::xp_to_const_wb_shared(wb_xptr);
    auto* ptr = new apm::PCEstimatorWithFEs(r, T_c, wb, q);
    return Rcpp::XPtr<apm::PCEstimatorWithFEs>(ptr, true);
}


