#ifndef APM_R_UTILS_H
#define APM_R_UTILS_H

#include <RcppArmadillo.h>
#include "../../core/src/utils.h" // ObservedOutcomeIndices
#include "../../core/src/cohort_specific_param_structs.h" // OutcomeMeanSufficientStatistics
#include <unordered_map>
#include <vector>
#include <memory>
#include <utility>
namespace apm { class WeightedBootstrap; }

// [[Rcpp::depends(RcppArmadillo)]]

namespace apm {
namespace r_utils {

// Converts an R list of 1-based integer vectors to a C++ vector of 0-based arma::uvecs.
apm::ObservedOutcomeIndices to_cpp_observed_outcome_indices(const Rcpp::List& r_list);

// Converts C++ ObservedOutcomeIndices (0-based) to R list of 1-based integer vectors
Rcpp::List to_r_observed_outcome_indices(const apm::ObservedOutcomeIndices& cpp_vec);

// Convert an external pointer holding std::shared_ptr<WeightedBootstrap>
// into a shared_ptr<const WeightedBootstrap>. Returns nullptr if xp is NULL.
std::shared_ptr<const apm::WeightedBootstrap> xp_to_const_wb_shared(SEXP xp);
std::shared_ptr<apm::WeightedBootstrap> xp_to_wb_shared(SEXP xp);

// Generic helper to heap-allocate and wrap a C++ object into an owning XPtr.
// Placed in header for templates and inline use across bindings.
template <typename T>
inline Rcpp::XPtr<T> make_xptr(T&& obj) {
    T* heap = new T(std::move(obj));
    return Rcpp::XPtr<T>(heap, true);
}

// Convert masked cohort outcome means map (0-based cohort ids) to an R named list
// names = 1-based cohort ids; values = numeric vectors of masked observed outcome means per cohort.
Rcpp::List masked_means_to_r_list(const std::unordered_map<int, apm::OutcomeMeanSufficientStatistics>& masked);

} // namespace r_utils
} // namespace apm

#endif // APM_R_UTILS_H
