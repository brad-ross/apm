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

Rcpp::List to_r_observed_outcome_indices(const apm::ObservedOutcomeIndices& cpp_vec) {
    Rcpp::List out(static_cast<int>(cpp_vec.size()));
    for (std::size_t i = 0; i < cpp_vec.size(); ++i) {
        const arma::uvec& idx0 = cpp_vec[i];
        Rcpp::IntegerVector idx1(static_cast<int>(idx0.n_elem));
        for (arma::uword k = 0; k < idx0.n_elem; ++k) idx1[k] = static_cast<int>(idx0[k] + 1);
        out[static_cast<int>(i)] = idx1;
    }
    return out;
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

Rcpp::List masked_means_to_r_list(const std::unordered_map<int, apm::OutcomeMeanSufficientStatistics>& masked) {
    Rcpp::List masked_means(static_cast<int>(masked.size()));
    Rcpp::CharacterVector keys(static_cast<int>(masked.size()));
    int i = 0;
    for (const auto& kv : masked) {
        int cohort1 = kv.first + 1;
        const apm::OutcomeMeanSufficientStatistics& s = kv.second;
        Rcpp::NumericVector mu(static_cast<int>(s.observed_outcome_means.n_elem));
        for (arma::uword k = 0; k < s.observed_outcome_means.n_elem; ++k) mu[k] = s.observed_outcome_means[k];
        masked_means[i] = mu;
        keys[i] = std::to_string(cohort1);
        ++i;
    }
    masked_means.attr("names") = keys;
    return masked_means;
}

} // namespace r_utils
} // namespace apm
