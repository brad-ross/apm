#ifndef APM_R_UTILS_H
#define APM_R_UTILS_H

#include <RcppArmadillo.h>
#include <vector>

// [[Rcpp::depends(RcppArmadillo)]]

namespace apm {
namespace r_utils {

// Converts an R list of 1-based integer vectors to a C++ vector of 0-based arma::uvecs.
std::vector<arma::uvec> to_cpp_observed_outcome_indices(const Rcpp::List& r_list);

} // namespace r_utils
} // namespace apm

#endif // APM_R_UTILS_H
