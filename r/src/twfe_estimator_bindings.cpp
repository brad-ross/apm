#include <RcppArmadillo.h>
#include "../../core/src/factor_model_estimators/twfe_estimator.h"
#include "../../core/src/bootstrap.h"
#include "r_utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

//==============================================================================
// Estimator factories for TWFE estimator
//==============================================================================

// [[Rcpp::export]]
SEXP twfe_estimator_new_cpp(std::size_t T_c, SEXP wb_xptr = R_NilValue, std::size_t q = 0) {
    auto wb = apm::r_utils::xp_to_const_wb_shared(wb_xptr);
    auto* ptr = new apm::TWFEEstimator(T_c, wb, q);
    return Rcpp::XPtr<apm::TWFEEstimator>(ptr, true);
}


