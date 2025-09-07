#include <RcppArmadillo.h>
#include "../../core/src/est_cohort_specific_params.h"
#include "../../core/src/apm_core.h"
#include "r_utils.h"

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

// Determine the ordered output spec names: preserve names(est_specs) when set; else synthesize
static Rcpp::CharacterVector ordered_spec_names(const Rcpp::List& est_specs_r) {
    if (has_valid_names(est_specs_r)) {
        SEXP nmSxp = Rf_getAttrib(est_specs_r, R_NamesSymbol);
        return Rcpp::CharacterVector(nmSxp);
    }
    Rcpp::CharacterVector out(est_specs_r.size());
    for (int i = 0; i < est_specs_r.size(); ++i) out[i] = std::string("spec_") + std::to_string(i + 1);
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

// Convert core results to R list using the given spec name order
static Rcpp::List build_return_list(const apm::CohortSpecificEstimates& ests, const Rcpp::CharacterVector& spec_names) {
    Rcpp::List out_factor(spec_names.size());
    out_factor.attr("names") = spec_names;
    for (int i = 0; i < spec_names.size(); ++i) {
        std::string key = Rcpp::as<std::string>(spec_names[i]);
        auto it = ests.cohort_specific_factor_ests.find(key);
        if (it == ests.cohort_specific_factor_ests.end()) {
            Rcpp::stop("Spec key not found in result: " + key);
        }
        const auto& vec = it->second;
        Rcpp::List per_cohort(vec.size());
        for (std::size_t c = 0; c < vec.size(); ++c) {
            auto* heap = new apm::FactorModelEstimates(vec[c]);
            per_cohort[static_cast<int>(c)] = Rcpp::XPtr<apm::FactorModelEstimates>(heap, true);
        }
        out_factor[i] = per_cohort;
    }

    Rcpp::List out_oms(ests.cohort_outcome_mean_ests.size());
    for (std::size_t c = 0; c < ests.cohort_outcome_mean_ests.size(); ++c) {
        auto* heap = new apm::OutcomeMeanSuffStatEstimates(ests.cohort_outcome_mean_ests[c]);
        out_oms[static_cast<int>(c)] = Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates>(heap, true);
    }

    // Cohort auxiliary means (as external pointers to C++ objects)
    Rcpp::List out_aux(ests.cohort_auxiliary_means.size());
    for (std::size_t c = 0; c < ests.cohort_auxiliary_means.size(); ++c) {
        auto* heapA = new apm::CohortAuxiliaryDataMeanEstimates(ests.cohort_auxiliary_means[c]);
        out_aux[static_cast<int>(c)] = Rcpp::XPtr<apm::CohortAuxiliaryDataMeanEstimates>(heapA, true);
    }

    // Cohort weights per spec (as external pointers to C++ objects)
    Rcpp::List out_weights(spec_names.size());
    out_weights.attr("names") = spec_names;
    for (int i = 0; i < spec_names.size(); ++i) {
        std::string key = Rcpp::as<std::string>(spec_names[i]);
        auto wit = ests.cohort_weights.find(key);
        if (wit == ests.cohort_weights.end()) {
            Rcpp::stop("Spec key not found in cohort_weights: " + key);
        }
        auto* heapW = new apm::CohortWeightEstimates(wit->second);
        out_weights[i] = Rcpp::XPtr<apm::CohortWeightEstimates>(heapW, true);
    }

    return Rcpp::List::create(
        Rcpp::Named("cohort_specific_factor_ests") = out_factor,
        Rcpp::Named("cohort_outcome_means") = out_oms,
        Rcpp::Named("cohort_weights") = out_weights,
        Rcpp::Named("cohort_auxiliary_means") = out_aux
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
                                                    Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in = R_NilValue) {
    // Extract columns and pointers
    PanelRawColumns cols = extract_panel_columns_0b(processed_panel, outcome_value_col);
    CovariateColumns covs = extract_covariate_columns(processed_panel, covar_cols, cols.n_rows);
    CovariateColumns auxs = extract_auxiliary_columns(processed_panel, auxiliary_cols, cols.n_rows);

    // Convert inputs
    apm::ObservedOutcomeIndices obs_idx_0b = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    auto cpp_specs = to_cpp_specs(est_specs);
    auto spec_names = ordered_spec_names(est_specs);
    auto wb = apm::r_utils::xp_to_const_wb_shared(bootstrap_xptr);
    auto [has_threads, nt] = resolve_num_threads(num_threads_in);

    // Call core
    apm::CohortSpecificEstimates ests;
    if (has_threads) {
        ests = apm::estimate_cohort_specific_params_from_raw(
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
            nt
        );
    } else {
        ests = apm::estimate_cohort_specific_params_from_raw(
            cols.unit_ptr,
            cols.cohort_ptr,
            cols.outcome_ptr,
            cols.y_ptr,
            covs.ptrs,
            auxs.ptrs,
            cols.n_rows,
            cpp_specs,
            obs_idx_0b,
            wb
        );
    }

    return build_return_list(ests, spec_names);
}

