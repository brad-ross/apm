#include <RcppArmadillo.h>
#include <set>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstddef>
#include "../../core/src/est_cohort_specific_params.h"
#include "r_utils.h"
#include "cohort_specific_estimates_helpers.h"

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

// (removed) ordered_spec_names: we now source names from the result maps

// Convert mask R list (names = cohort ids 1-based, values = integer vectors 1-based outcomes) to C++ 0-based
static apm::CohortOutcomeMask to_cpp_mask(Rcpp::Nullable<Rcpp::List> mask_in) {
    apm::CohortOutcomeMask out;
    if (mask_in.isNull()) return out;
    Rcpp::List L(mask_in);
    if (L.size() == 0) return out;
    Rcpp::CharacterVector nms = Rcpp::as<Rcpp::CharacterVector>(L.names());
    for (int i = 0; i < L.size(); ++i) {
        std::string s = Rcpp::as<std::string>(nms[i]);
        int cohort1 = std::stoi(s);
        int cohort0 = cohort1 - 1;
        Rcpp::IntegerVector v = L[i];
        arma::uvec vv(static_cast<arma::uword>(v.size()));
        for (int j = 0; j < v.size(); ++j) {
            if (Rcpp::IntegerVector::is_na(v[j]) || v[j] <= 0) Rcpp::stop("mask outcome ids must be positive integers");
            vv[static_cast<arma::uword>(j)] = static_cast<arma::uword>(v[j] - 1);
        }
        out.emplace(cohort0, std::move(vv));
    }
    return out;
}

// Extract 1-based index columns and y from processed_panel and return 0-based vectors and y pointer
struct PanelRawColumns {
    std::vector<int> unit_idx0;
    std::vector<int> cohort_id0;
    std::vector<int> outcome_idx0;
    const int* unit_ptr;
    const int* cohort_ptr;
    const int* outcome_ptr;
    const double* y_ptr;
    std::size_t n_rows;
};

static PanelRawColumns extract_panel_columns_0b(Rcpp::DataFrame processed_panel, const std::string& outcome_value_col) {
    Rcpp::IntegerVector unit_idx_r = processed_panel["unit_idx"];
    Rcpp::IntegerVector cohort_id_r = processed_panel["cohort_id"];
    Rcpp::IntegerVector outcome_idx_r = processed_panel["outcome_idx"];
    Rcpp::NumericVector y_r = processed_panel[outcome_value_col];

    const std::size_t n_rows = static_cast<std::size_t>(unit_idx_r.size());
    if (cohort_id_r.size() != static_cast<int>(n_rows) || outcome_idx_r.size() != static_cast<int>(n_rows) || y_r.size() != static_cast<int>(n_rows)) {
        Rcpp::stop("Processed panel columns have inconsistent lengths");
    }

    std::vector<int> unit_idx0(n_rows), cohort_id0(n_rows), outcome_idx0(n_rows);
    for (std::size_t i = 0; i < n_rows; ++i) {
        int u = unit_idx_r[i], c = cohort_id_r[i], o = outcome_idx_r[i];
        if (u <= 0 || c <= 0 || o <= 0) Rcpp::stop("Indices in processed_panel must be positive 1-based integers");
        unit_idx0[i] = u - 1;
        cohort_id0[i] = c - 1;
        outcome_idx0[i] = o - 1;
    }

    PanelRawColumns out{std::move(unit_idx0), std::move(cohort_id0), std::move(outcome_idx0), nullptr, nullptr, nullptr, REAL(y_r), n_rows};
    out.unit_ptr = out.unit_idx0.data();
    out.cohort_ptr = out.cohort_id0.data();
    out.outcome_ptr = out.outcome_idx0.data();
    return out;
}

// Extract covariate columns as numeric vectors and raw pointers
struct CovariateColumns {
    std::vector<Rcpp::NumericVector> cols_r;
    std::vector<const double*> ptrs;
};

static CovariateColumns extract_covariate_columns(Rcpp::DataFrame processed_panel, Rcpp::CharacterVector covar_cols, std::size_t n_rows) {
    CovariateColumns out;
    out.cols_r.reserve(covar_cols.size());
    out.ptrs.reserve(covar_cols.size());
    for (int j = 0; j < covar_cols.size(); ++j) {
        Rcpp::NumericVector cj = processed_panel[Rcpp::as<std::string>(covar_cols[j])];
        if (cj.size() != static_cast<int>(n_rows)) Rcpp::stop("Covariate column has inconsistent length");
        out.ptrs.push_back(REAL(cj));
        out.cols_r.push_back(std::move(cj));
    }
    return out;
}

// Extract auxiliary columns as numeric vectors and raw pointers (same as covariates)
static CovariateColumns extract_auxiliary_columns(Rcpp::DataFrame processed_panel, Rcpp::CharacterVector aux_cols, std::size_t n_rows) {
    CovariateColumns out;
    out.cols_r.reserve(aux_cols.size());
    out.ptrs.reserve(aux_cols.size());
    for (int j = 0; j < aux_cols.size(); ++j) {
        Rcpp::NumericVector cj = processed_panel[Rcpp::as<std::string>(aux_cols[j])];
        if (cj.size() != static_cast<int>(n_rows)) Rcpp::stop("Auxiliary column has inconsistent length");
        out.ptrs.push_back(REAL(cj));
        out.cols_r.push_back(std::move(cj));
    }
    return out;
}

// Resolve num_threads optional parameter (returns optional value flag and size)
static std::pair<bool, std::size_t> resolve_num_threads(Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in) {
    if (num_threads_in.isNotNull()) {
        Rcpp::IntegerVector nt(num_threads_in);
        std::size_t num_threads = 1;
        if (nt.size() > 0 && !Rcpp::IntegerVector::is_na(nt[0])) {
            num_threads = static_cast<std::size_t>(std::max(1, static_cast<int>(nt[0])));
        }
        return {true, num_threads};
    }
    return {false, 0};
}

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
        Rcpp::List masked_means(ests.masked_cohort_outcome_means.size());
        Rcpp::CharacterVector keys(ests.masked_cohort_outcome_means.size());
        int i = 0;
        for (const auto& kv : ests.masked_cohort_outcome_means) {
            int cohort1 = kv.first + 1;
            const auto& s = kv.second;
            Rcpp::NumericVector mu(s.observed_outcome_means.n_elem);
            for (arma::uword k = 0; k < s.observed_outcome_means.n_elem; ++k) mu[k] = s.observed_outcome_means[k];
            masked_means[i] = mu;
            keys[i] = std::to_string(cohort1);
            ++i;
        }
        masked_means.attr("names") = keys;
        res.push_back(masked_means, "masked_cohort_outcome_means");
    }
    return res;
}

// Returns raw C++ outputs (no R wrapping)
apm::CohortSpecificEstimates cohort_specific_estimates_from_panel_cpp_core(
	Rcpp::DataFrame processed_panel,
	Rcpp::List observed_outcome_indices, // 1-based
	const std::string& outcome_value_col,
	Rcpp::CharacterVector covar_cols,
	Rcpp::CharacterVector auxiliary_cols,
	Rcpp::List est_specs,
	SEXP bootstrap_xptr,
	Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in,
	Rcpp::Nullable<Rcpp::List> cohort_outcomes_to_mask_in)
{
	// Extract columns and pointers
	PanelRawColumns cols = extract_panel_columns_0b(processed_panel, outcome_value_col);
	CovariateColumns covs = extract_covariate_columns(processed_panel, covar_cols, cols.n_rows);
	CovariateColumns auxs = extract_auxiliary_columns(processed_panel, auxiliary_cols, cols.n_rows);

	// Convert inputs
	apm::ObservedOutcomeIndices obs_idx_0b = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
	auto cpp_specs = to_cpp_specs(est_specs);
	auto wb = apm::r_utils::xp_to_const_wb_shared(bootstrap_xptr);
	auto [has_threads, nt] = resolve_num_threads(num_threads_in);

	apm::CohortOutcomeMask mask = to_cpp_mask(cohort_outcomes_to_mask_in);

	// Call core with optional num_threads (NULL -> std::nullopt)
	std::optional<std::size_t> nt_opt = has_threads ? std::optional<std::size_t>(nt) : std::nullopt;
	return apm::estimate_cohort_specific_params_from_raw(
		cols.unit_ptr,
		cols.cohort_ptr,
		cols.outcome_ptr,
		cols.y_ptr,
		covs.ptrs,
		auxs.ptrs,
		cols.n_rows,
		cpp_specs,
		obs_idx_0b,
		wb,
		nt_opt,
		mask
	);
}

// [[Rcpp::export]]
Rcpp::List est_cohort_specific_params_from_panel_cpp(Rcpp::DataFrame processed_panel,
                                                    Rcpp::List observed_outcome_indices, // 1-based
                                                    const std::string& outcome_value_col,
                                                    Rcpp::CharacterVector covar_cols,
                                                    Rcpp::CharacterVector auxiliary_cols,
                                                    Rcpp::List est_specs,
                                                    SEXP bootstrap_xptr = R_NilValue,
                                                    Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in = R_NilValue,
                                                    Rcpp::Nullable<Rcpp::List> cohort_outcomes_to_mask_in = R_NilValue) {
    apm::CohortSpecificEstimates ests = cohort_specific_estimates_from_panel_cpp_core(
        processed_panel,
        observed_outcome_indices,
        outcome_value_col,
        covar_cols,
        auxiliary_cols,
        est_specs,
        bootstrap_xptr,
        num_threads_in,
        cohort_outcomes_to_mask_in
    );

    return build_return_list(std::move(ests));
}

