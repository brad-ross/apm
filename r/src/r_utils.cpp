#include "r_utils.h"

namespace apm {
namespace r_utils {

// Converts an R list of 1-based integer vectors to a C++ vector of 0-based arma::uvecs.
std::vector<arma::uvec> to_cpp_observed_outcome_indices(const Rcpp::List& r_list) {
    std::vector<arma::uvec> cpp_vec;
    cpp_vec.reserve(r_list.size());
    for (SEXP vec : r_list) {
        cpp_vec.push_back(Rcpp::as<arma::uvec>(vec) - 1); // R is 1-based, C++ is 0-based.
    }
    return cpp_vec;
}

} // namespace r_utils
} // namespace apm
