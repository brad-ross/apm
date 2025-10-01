#include <RcppArmadillo.h>
#include <set>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstddef>
#include <optional>
#include "../../core/src/est_cohort_specific_params.h"
#include "r_utils.h"
#include "cohort_specific_estimates_helpers.h"
#include "../../core/src/panels/InMemoryUnbalancedPanel.h"

// [[Rcpp::depends(RcppArmadillo)]]

// Convert est_specs: named list → unordered_map<string, EstimatorSpecification>
// Each element is a list with fields: factor_model_estimator (string), include_outcome_fes (logical), r (integer),
// optional cohort_weighting (string: "equal" or "by_size")
static bool has_valid_names(const Rcpp::List& lst) {
    SEXP nmSxp = Rf_getAttrib(lst, R_NamesSymbol);
    if (nmSxp == R_NilValue) return false;
    Rcpp::CharacterVector nm(nmSxp);
    if (nm.size() != lst.size()) return false;
    for (int i = 0; i < nm.size(); ++i) {
        Rcpp::String s = nm[i];
        if (s == NA_STRING) return false;
        const char* cptr = s.get_cstring();
        if (cptr == nullptr || *cptr == '\0') return false;
    }
    return true;
}

static std::unordered_map<std::string, apm::EstimatorSpecification>
to_cpp_specs(const Rcpp::List& est_specs_r) {
    std::unordered_map<std::string, apm::EstimatorSpecification> out;
    const bool has_names = has_valid_names(est_specs_r);
    Rcpp::CharacterVector nm;
    if (has_names) {
        SEXP nmSxp = Rf_getAttrib(est_specs_r, R_NamesSymbol);
        nm = Rcpp::CharacterVector(nmSxp);
    }
    for (int i = 0; i < est_specs_r.size(); ++i) {
        Rcpp::List sp = est_specs_r[i];
        apm::EstimatorSpecification csp;
        csp.factor_model_estimator = Rcpp::as<std::string>(sp["factor_model_estimator"]);
        csp.include_outcome_fes = Rcpp::as<bool>(sp["include_outcome_fes"]);
        csp.r = static_cast<std::size_t>(Rcpp::as<int>(sp["r"]));
        if (sp.containsElementNamed("cohort_weighting")) {
            csp.cohort_weighting = Rcpp::as<std::string>(sp["cohort_weighting"]);
        }
        std::string key = has_names ? std::string(Rcpp::as<std::string>(nm[i])) : std::string("spec_") + std::to_string(i + 1);
        out.emplace(std::move(key), std::move(csp));
    }
    return out;
}


// Extract covariate columns as numeric vectors and raw pointers
struct CovariateColumns {
    std::vector<Rcpp::NumericVector> cols_r;
    std::vector<const double*> ptrs;
};

static std::vector<Rcpp::NumericVector> grab_numeric_cols(Rcpp::DataFrame processed_panel, Rcpp::CharacterVector cols, std::size_t n_rows) {
    std::vector<Rcpp::NumericVector> out;
    out.reserve(cols.size());
    for (int j = 0; j < cols.size(); ++j) {
        Rcpp::NumericVector cj = processed_panel[Rcpp::as<std::string>(cols[j])];
        if (cj.size() != static_cast<int>(n_rows)) Rcpp::stop("Column has inconsistent length");
        out.push_back(std::move(cj));
    }
    return out;
}

// Extract auxiliary columns as numeric vectors and raw pointers (same as covariates)
// Build PanelHolder XPtr from processed_panel
// [[Rcpp::export]]
SEXP build_R_panel_holder_cpp(Rcpp::DataFrame processed_panel,
                              Rcpp::List observed_outcome_indices,
                              const std::string& outcome_value_col,
                              Rcpp::CharacterVector covar_cols,
                              Rcpp::CharacterVector auxiliary_cols,
                              Rcpp::Nullable<Rcpp::IntegerVector> num_units_in = R_NilValue) {
    Rcpp::IntegerVector unit_idx_r = processed_panel["unit_idx"];
    Rcpp::IntegerVector cohort_id_r = processed_panel["cohort_id"];
    Rcpp::IntegerVector outcome_idx_r = processed_panel["outcome_idx"];
    Rcpp::NumericVector y_r = processed_panel[outcome_value_col];
    const std::size_t n_rows = static_cast<std::size_t>(unit_idx_r.size());
    if (cohort_id_r.size() != static_cast<int>(n_rows) || outcome_idx_r.size() != static_cast<int>(n_rows) || y_r.size() != static_cast<int>(n_rows)) {
        Rcpp::stop("Processed panel columns have inconsistent lengths");
    }

    apm::ObservedOutcomeIndices ooi0b = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    auto covs = grab_numeric_cols(processed_panel, covar_cols, n_rows);
    auto aux = grab_numeric_cols(processed_panel, auxiliary_cols, n_rows);

    std::optional<std::size_t> num_units_opt = std::nullopt;
    if (num_units_in.isNotNull()) {
        Rcpp::IntegerVector nu(num_units_in);
        if (nu.size() > 0 && !Rcpp::IntegerVector::is_na(nu[0]) && nu[0] > 0) {
            num_units_opt = static_cast<std::size_t>(nu[0]);
        }
    }

    auto* holder = new apm::r_utils::PanelHolder(unit_idx_r, cohort_id_r, outcome_idx_r, y_r, std::move(covs), std::move(aux), ooi0b, num_units_opt);
    return Rcpp::XPtr<apm::r_utils::PanelHolder>(holder, true);
}

// Resolve num_threads optional parameter (returns optional value flag and size)

// Convert core results to R list; move into heap allocations; source spec names from result maps
static Rcpp::List build_return_list(apm::CohortSpecificEstimates&& ests) {
    // Prepare outputs that do not depend on spec names
    Rcpp::List out_oms(ests.cohort_outcome_mean_ests.size());
    for (std::size_t c = 0; c < ests.cohort_outcome_mean_ests.size(); ++c) {
        auto* heap = new apm::OutcomeMeanSuffStatEstimates(std::move(ests.cohort_outcome_mean_ests[c]));
        out_oms[static_cast<int>(c)] = Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates>(heap, true);
    }

    Rcpp::List out_aux;
    const bool has_aux_means = !ests.cohort_auxiliary_means.empty();
    if (has_aux_means) {
        out_aux = Rcpp::List(ests.cohort_auxiliary_means.size());
        for (std::size_t c = 0; c < ests.cohort_auxiliary_means.size(); ++c) {
            auto* heapA = new apm::CohortAuxiliaryDataMeanEstimates(std::move(ests.cohort_auxiliary_means[c]));
            out_aux[static_cast<int>(c)] = Rcpp::XPtr<apm::CohortAuxiliaryDataMeanEstimates>(heapA, true);
        }
    }

    // Build factor and weight outputs together, using the union of spec names from both maps
    auto& factor_map = ests.cohort_specific_factor_ests;
    auto& weight_map = ests.cohort_weights;

    std::set<std::string> union_keys;
    for (const auto& kv : factor_map) union_keys.insert(kv.first);
    for (const auto& kw : weight_map) union_keys.insert(kw.first);

    Rcpp::List out_factor(static_cast<int>(union_keys.size()));
    Rcpp::List out_weights(static_cast<int>(union_keys.size()));
    Rcpp::CharacterVector spec_names(static_cast<int>(union_keys.size()));

    int idx = 0;
    for (const auto& key : union_keys) {
        auto fit = factor_map.find(key);
        auto wit = weight_map.find(key);
        if (fit == factor_map.end()) {
            Rcpp::stop("Spec key present in cohort weights but missing in factor estimates: " + key);
        }
        if (wit == weight_map.end()) {
            Rcpp::stop("Spec key present in factor estimates but missing in cohort weights: " + key);
        }

        auto& vec = fit->second; // vector<FactorModelEstimates>

        // Per-cohort factor estimates (move each element into heap)
        Rcpp::List per_cohort(vec.size());
        for (std::size_t c = 0; c < vec.size(); ++c) {
            auto* heap = new apm::FactorModelEstimates(std::move(vec[c]));
            per_cohort[static_cast<int>(c)] = Rcpp::XPtr<apm::FactorModelEstimates>(heap, true);
        }
        out_factor[idx] = per_cohort;

        // Weights object for this spec (move into heap)
        auto* heapW = new apm::CohortWeightEstimates(std::move(wit->second));
        out_weights[idx] = Rcpp::XPtr<apm::CohortWeightEstimates>(heapW, true);

        spec_names[idx] = key;
        ++idx;
    }

    out_factor.attr("names") = spec_names;
    out_weights.attr("names") = spec_names;

    Rcpp::List res = Rcpp::List::create(
        Rcpp::Named("cohort_specific_factor_ests") = out_factor,
        Rcpp::Named("cohort_outcome_means") = out_oms,
        Rcpp::Named("cohort_weights") = out_weights
    );
    if (has_aux_means) {
        res.push_back(out_aux, "cohort_auxiliary_means");
    }
    // Attach masked fields if present
    if (ests.masked_observed_outcome_indices.has_value()) {
        res.push_back(apm::r_utils::to_r_observed_outcome_indices(*ests.masked_observed_outcome_indices),
                      "masked_observed_outcome_indices");
    }
    if (!ests.masked_cohort_outcome_means.empty()) {
        res.push_back(apm::r_utils::masked_means_to_r_list(ests.masked_cohort_outcome_means),
                      "masked_cohort_outcome_means");
    }
    return res;
}

// Returns raw C++ outputs (no R wrapping), called by exported wrapper
apm::CohortSpecificEstimates cohort_specific_estimates_from_panel_cpp_core(
	SEXP panel_holder_xptr,
	Rcpp::List est_specs,
	SEXP bootstrap_xptr,
	Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in,
	Rcpp::Nullable<Rcpp::List> cohort_outcomes_to_mask_in)
{
	Rcpp::XPtr<apm::r_utils::PanelHolder> ph(panel_holder_xptr);
	auto cpp_specs = to_cpp_specs(est_specs);
	auto wb = apm::r_utils::xp_to_const_wb_shared(bootstrap_xptr);
	auto [has_threads, nt] = apm::r_utils::resolve_num_threads(num_threads_in);

	apm::CohortOutcomeMask mask = apm::r_utils::to_cpp_mask(cohort_outcomes_to_mask_in);

	std::optional<std::size_t> nt_opt = has_threads ? std::optional<std::size_t>(nt) : std::nullopt;
	return apm::estimate_cohort_specific_params_from_internal_panel_rep(
		ph->panel,
		cpp_specs,
		wb,
		nt_opt,
		mask
	);
}

// [[Rcpp::export]]
Rcpp::List est_cohort_specific_params_from_panel_cpp(SEXP panel_holder_xptr,
                                                    Rcpp::List est_specs,
                                                    SEXP bootstrap_xptr = R_NilValue,
                                                    Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in = R_NilValue,
                                                    Rcpp::Nullable<Rcpp::List> cohort_outcomes_to_mask_in = R_NilValue) {
    apm::CohortSpecificEstimates ests = cohort_specific_estimates_from_panel_cpp_core(
        panel_holder_xptr,
        est_specs,
        bootstrap_xptr,
        num_threads_in,
        cohort_outcomes_to_mask_in
    );

    return build_return_list(std::move(ests));
}