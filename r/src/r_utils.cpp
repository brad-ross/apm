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

// Convert est_specs: named list → unordered_map<string, EstimatorSpecification>
std::unordered_map<std::string, apm::EstimatorSpecification> to_cpp_specs(const Rcpp::List& est_specs_r) {
    std::unordered_map<std::string, apm::EstimatorSpecification> out;
    SEXP nmSxp = Rf_getAttrib(est_specs_r, R_NamesSymbol);
    const bool has_names = (nmSxp != R_NilValue);
    Rcpp::CharacterVector nm;
    if (has_names) nm = Rcpp::CharacterVector(nmSxp);
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

// Convert mask R list (names = cohort ids 1-based, values = integer vectors 1-based outcomes) to C++ 0-based
apm::CohortOutcomeMask to_cpp_mask(Rcpp::Nullable<Rcpp::List> mask_in) {
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

std::pair<bool, std::size_t> resolve_num_threads(Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in) {
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

std::vector<apm::OutcomeMeanSuffStatEstimates> list_to_stats_vec(Rcpp::Nullable<Rcpp::List> maybe_list) {
    std::vector<apm::OutcomeMeanSuffStatEstimates> out;
    if (maybe_list.isNotNull()) {
        Rcpp::List L(maybe_list);
        out.reserve(L.size());
        for (int i = 0; i < L.size(); ++i) {
            if (Rf_isNull(L[i])) continue;
            Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> xp(L[i]);
            out.push_back(*xp);
        }
    }
    return out;
}

std::vector<apm::OutcomeMeanSufficientStatistics> point_stats_from_estimates(const std::vector<apm::OutcomeMeanSuffStatEstimates>& v) {
    std::vector<apm::OutcomeMeanSufficientStatistics> out;
    out.reserve(v.size());
    for (const auto& e : v) out.push_back(e.suff_stat_estimates);
    return out;
}

const apm::InMemoryUnbalancedPanel& panel_ref_from_panel_holder(SEXP panel_holder_xptr) {
    Rcpp::XPtr<PanelHolder> ph(panel_holder_xptr);
    return ph->panel;
}

apm::ObservedOutcomeIndices observed_outcome_indices_from_panel_holder(SEXP panel_holder_xptr) {
    Rcpp::XPtr<PanelHolder> ph(panel_holder_xptr);
    return ph->panel.observed_outcome_indices();
}

apm::ImputationOptions imputation_options_from_r_list(Rcpp::Nullable<Rcpp::List> imputation_options_in) {
    apm::ImputationOptions opts;
    if (imputation_options_in.isNotNull()) {
        Rcpp::List L(imputation_options_in);
        if (L.containsElementNamed("tol")) opts.tol = Rcpp::as<double>(L["tol"]);
        if (L.containsElementNamed("max_iters")) {
            int mi = Rcpp::as<int>(L["max_iters"]);
            if (mi < 0) mi = 0;
            opts.max_iters = static_cast<std::size_t>(mi);
        }
        if (L.containsElementNamed("method")) {
            std::string m = Rcpp::as<std::string>(L["method"]);
            for (auto &ch : m) ch = static_cast<char>(::tolower(ch));
            if (m == "none") opts.method = apm::AccelMethod::None; else opts.method = apm::AccelMethod::IronsTuck;
        }
        if (L.containsElementNamed("grand_period")) { int v = Rcpp::as<int>(L["grand_period"]); if (v < 0) v = 0; opts.grand_period = static_cast<std::size_t>(v); }
        if (L.containsElementNamed("grand_k")) { int v = Rcpp::as<int>(L["grand_k"]); if (v < 0) v = 0; opts.grand_k = static_cast<std::size_t>(v); }
        if (L.containsElementNamed("stabilize_after")) { int v = Rcpp::as<int>(L["stabilize_after"]); if (v < 0) v = 0; opts.stabilize_after = static_cast<std::size_t>(v); }
        if (L.containsElementNamed("extra_proj")) { int v = Rcpp::as<int>(L["extra_proj"]); if (v < 0) v = 0; opts.extra_proj = static_cast<std::size_t>(v); }

        // Solver selection ("fixed-point" or "lsmr"). Defaults to FixedPoint.
        if (L.containsElementNamed("solver")) {
            std::string s = Rcpp::as<std::string>(L["solver"]);
            for (auto &ch : s) ch = static_cast<char>(::tolower(ch));
            if (s == "lsmr") {
                opts.solver = apm::ImputationSolver::LSMR;
            } else if (s == "fixed-point" || s == "fixed_point" || s == "fixedpoint" || s == "fixed") {
                opts.solver = apm::ImputationSolver::FixedPoint;
            }
        }

        // LSMR-specific options
        if (L.containsElementNamed("lsmr_diagonal_precond")) {
            opts.lsmr_diagonal_precond = Rcpp::as<bool>(L["lsmr_diagonal_precond"]);
        }
        if (L.containsElementNamed("lsmr_num_diag_approx_draws")) {
            int v = Rcpp::as<int>(L["lsmr_num_diag_approx_draws"]);
            if (v < 0) v = 0;
            opts.lsmr_num_diag_approx_draws = static_cast<std::size_t>(v);
        }
        if (L.containsElementNamed("lsmr_homotopy_iters")) {
            int v = Rcpp::as<int>(L["lsmr_homotopy_iters"]);
            if (v < 0) v = 0;
            opts.lsmr_homotopy_iters = static_cast<std::size_t>(v);
        }

        // LSMR solver numeric controls
        if (L.containsElementNamed("lsmr_atol")) {
            opts.lsmr_atol = Rcpp::as<double>(L["lsmr_atol"]);
        }
        if (L.containsElementNamed("lsmr_btol")) {
            opts.lsmr_btol = Rcpp::as<double>(L["lsmr_btol"]);
        }
        if (L.containsElementNamed("lsmr_conlim")) {
            opts.lsmr_conlim = Rcpp::as<double>(L["lsmr_conlim"]);
        }
        if (L.containsElementNamed("lsmr_max_iters")) {
            int v = Rcpp::as<int>(L["lsmr_max_iters"]);
            if (v < 0) v = 0;
            opts.lsmr_max_iters = static_cast<std::size_t>(v);
        }
        if (L.containsElementNamed("lsmr_lambda")) {
            opts.lsmr_lambda = Rcpp::as<double>(L["lsmr_lambda"]);
        }
    }
    return opts;
}

} // namespace r_utils
} // namespace apm
