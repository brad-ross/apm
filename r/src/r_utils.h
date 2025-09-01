#ifndef APM_R_UTILS_H
#define APM_R_UTILS_H

#include <RcppArmadillo.h>
#include <vector>
#include <memory>
namespace apm { class WeightedBootstrap; }

// [[Rcpp::depends(RcppArmadillo)]]

namespace apm {
namespace r_utils {

// Converts an R list of 1-based integer vectors to a C++ vector of 0-based arma::uvecs.
std::vector<arma::uvec> to_cpp_observed_outcome_indices(const Rcpp::List& r_list);

// Convert an external pointer holding std::shared_ptr<WeightedBootstrap>
// into a shared_ptr<const WeightedBootstrap>. Returns nullptr if xp is NULL.
std::shared_ptr<const apm::WeightedBootstrap> xp_to_const_wb_shared(SEXP xp);
std::shared_ptr<apm::WeightedBootstrap> xp_to_wb_shared(SEXP xp);

} // namespace r_utils
} // namespace apm

#endif // APM_R_UTILS_H
