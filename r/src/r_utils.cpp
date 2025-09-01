#include "r_utils.h"
#include "../../core/src/bootstrap.h"

namespace apm {
namespace r_utils {

// Converts an R list of 1-based integer vectors to a C++ vector of 0-based arma::uvecs.
apm::ObservedOutcomeIndices to_cpp_observed_outcome_indices(const Rcpp::List& r_list) {
    apm::ObservedOutcomeIndices cpp_vec;
    cpp_vec.reserve(r_list.size());
    for (SEXP vec : r_list) {
        cpp_vec.push_back(Rcpp::as<arma::uvec>(vec) - 1); // R is 1-based, C++ is 0-based.
    }
    return cpp_vec;
}

// Interpret xp as XPtr<std::shared_ptr<WeightedBootstrap>>
std::shared_ptr<const apm::WeightedBootstrap> xp_to_const_wb_shared(SEXP xp) {
    if (xp == R_NilValue) return nullptr;
    Rcpp::XPtr<std::shared_ptr<apm::WeightedBootstrap>> p(xp);
    return *p; // implicit const shared_ptr copy
}

std::shared_ptr<apm::WeightedBootstrap> xp_to_wb_shared(SEXP xp) {
    if (xp == R_NilValue) return nullptr;
    Rcpp::XPtr<std::shared_ptr<apm::WeightedBootstrap>> p(xp);
    return *p; // copy shared_ptr
}

} // namespace r_utils
} // namespace apm
