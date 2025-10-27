#ifndef APM_R_UTILS_H
#define APM_R_UTILS_H

#include <RcppArmadillo.h>
#include "../../core/src/utils.h" // ObservedOutcomeIndices
#include "../../core/src/outcome_imputation.h" // ImputationOptions, AccelMethod
#include "../../core/src/cohort_specific_param_structs.h" // OutcomeMeanSufficientStatistics
#include "../../core/src/est_cohort_specific_params.h" // EstimatorSpecification, CohortOutcomeMask
#include "../../core/src/panels/InMemoryUnbalancedPanel.h" // InMemoryUnbalancedPanel
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

// Convert est_specs: named list → unordered_map<string, EstimatorSpecification>
std::unordered_map<std::string, apm::EstimatorSpecification> to_cpp_specs(const Rcpp::List& est_specs_r);

// Convert mask R list (names = cohort ids 1-based, values = integer vectors 1-based outcomes) to C++ 0-based
apm::CohortOutcomeMask to_cpp_mask(Rcpp::Nullable<Rcpp::List> mask_in);

// Resolve num_threads optional parameter (returns optional value flag and size)
std::pair<bool, std::size_t> resolve_num_threads(Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in);

// Parse ImputationOptions from an optional R list with fields:
// tol (double), max_iters (int), method (string: "irons-tuck" or "none"),
// grand_period (int), grand_k (int), stabilize_after (int), extra_proj (int).
// Missing fields keep defaults from ImputationOptions().
apm::ImputationOptions imputation_options_from_r_list(Rcpp::Nullable<Rcpp::List> imputation_options_in);

// OutcomeMeanSuffStatEstimates helpers
// Convert a nullable R list of XPtr<OutcomeMeanSuffStatEstimates> to a C++ vector
std::vector<apm::OutcomeMeanSuffStatEstimates> list_to_stats_vec(Rcpp::Nullable<Rcpp::List> maybe_list);

// Extract point sufficient statistics from a vector of OutcomeMeanSuffStatEstimates
std::vector<apm::OutcomeMeanSufficientStatistics> point_stats_from_estimates(const std::vector<apm::OutcomeMeanSuffStatEstimates>& v);

// Panel holder shared across bindings
struct PanelHolder {
    Rcpp::IntegerVector unit_idx;
    Rcpp::IntegerVector cohort_id;
    Rcpp::IntegerVector outcome_idx;
    Rcpp::NumericVector y;
    std::vector<Rcpp::NumericVector> covars;
    std::vector<Rcpp::NumericVector> aux;
    std::vector<const double*> covar_ptrs;
    std::vector<const double*> aux_ptrs;
    apm::InMemoryUnbalancedPanel panel;

    static std::vector<const double*> to_ptrs(const std::vector<Rcpp::NumericVector>& cols) {
        std::vector<const double*> out;
        out.reserve(cols.size());
        for (const auto& v : cols) out.push_back(REAL(v));
        return out;
    }

    PanelHolder(Rcpp::IntegerVector unit_idx_,
                Rcpp::IntegerVector cohort_id_,
                Rcpp::IntegerVector outcome_idx_,
                Rcpp::NumericVector y_,
                std::vector<Rcpp::NumericVector> covars_,
                std::vector<Rcpp::NumericVector> aux_,
                const apm::ObservedOutcomeIndices& ooi0b,
                std::optional<std::size_t> num_units_opt = std::nullopt)
        : unit_idx(unit_idx_)
        , cohort_id(cohort_id_)
        , outcome_idx(outcome_idx_)
        , y(y_)
        , covars(std::move(covars_))
        , aux(std::move(aux_))
        , covar_ptrs(to_ptrs(covars))
        , aux_ptrs(to_ptrs(aux))
        , panel(
            INTEGER(unit_idx), INTEGER(cohort_id), INTEGER(outcome_idx), REAL(y),
            covar_ptrs, aux_ptrs,
            static_cast<std::size_t>(unit_idx.size()),
            ooi0b,
            /*one_indexed=*/true,
            num_units_opt)
    {}
};

// Access underlying panel reference from a panel holder XPtr
const apm::InMemoryUnbalancedPanel& panel_ref_from_panel_holder(SEXP panel_holder_xptr);

// Access observed_outcome_indices (0-based) from a panel holder XPtr
apm::ObservedOutcomeIndices observed_outcome_indices_from_panel_holder(SEXP panel_holder_xptr);

} // namespace r_utils
} // namespace apm

#endif // APM_R_UTILS_H
