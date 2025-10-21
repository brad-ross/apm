#include <RcppArmadillo.h>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

#include "r_utils.h"
#include "../../core/src/target_params/est_target_params.h"
#include "../../core/src/est_outcome_means.h"
#include "../../core/src/panels/InMemoryUnbalancedPanel.h"
#include "../../core/src/cohort_specific_param_structs.h"
#include "cohort_specific_estimates_helpers.h"
#include "../../core/src/outcome_imputation.h"

// [[Rcpp::depends(RcppArmadillo)]]

using apm::TargetParameterEstimates;
using apm::OutcomeMeansEstimates;
using apm::CohortAuxiliaryDataMeanEstimates;
using apm::CohortAuxiliaryDataMeans;
using apm::r_utils::make_xptr;

namespace {

std::vector<CohortAuxiliaryDataMeanEstimates> list_to_eta_vec(Rcpp::Nullable<Rcpp::List> maybe_list) {
    std::vector<CohortAuxiliaryDataMeanEstimates> out;
    if (maybe_list.isNotNull()) {
        Rcpp::List L(maybe_list);
        out.reserve(L.size());
        for (int i = 0; i < L.size(); ++i) {
            if (Rf_isNull(L[i])) continue;
            Rcpp::XPtr<CohortAuxiliaryDataMeanEstimates> xp(L[i]);
            out.push_back(*xp);
        }
    }
    return out;
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

apm::TargetFn make_target_fn(Rcpp::Function r_fn) {
    return [r_fn](const arma::mat& Y,
                  const std::vector<apm::OutcomeMeanSufficientStatistics>& stats_all,
                  const std::vector<CohortAuxiliaryDataMeans>& eta_all) -> arma::vec {
        Rcpp::NumericMatrix Y_r(Y.n_rows, Y.n_cols);
        std::copy(Y.begin(), Y.end(), Y_r.begin());
        const int C = static_cast<int>(eta_all.size());
        Rcpp::NumericVector shares(C);
        Rcpp::List observed_means_list(C);
        Rcpp::List covar_means_list(C);
        Rcpp::List T_list(C);
        Rcpp::List q_list(C);
        Rcpp::List eta_list(C);
        for (int c = 0; c < C; ++c) {
            shares[c] = stats_all[c].cohort_pop_share;
            observed_means_list[c] = Rcpp::NumericVector(stats_all[c].observed_outcome_means.begin(), stats_all[c].observed_outcome_means.end());
            if (stats_all[c].covar_means.has_value()) {
                const arma::mat& XM = *(stats_all[c].covar_means);
                Rcpp::NumericMatrix XMr(XM.n_rows, XM.n_cols);
                std::copy(XM.begin(), XM.end(), XMr.begin());
                covar_means_list[c] = XMr;
            } else {
                covar_means_list[c] = R_NilValue;
            }
            T_list[c] = static_cast<int>(stats_all[c].T());
            q_list[c] = static_cast<int>(stats_all[c].q());
            const arma::mat& M = eta_all[c].auxiliary_means;
            Rcpp::NumericMatrix Mr(M.n_rows, M.n_cols);
            std::copy(M.begin(), M.end(), Mr.begin());
            eta_list[c] = Mr;
        }
        Rcpp::RObject res = r_fn(Y_r, shares, observed_means_list, covar_means_list, eta_list);
        return Rcpp::as<arma::vec>(res);
    };
}

} // anonymous namespace

// Compute from single spec
// [[Rcpp::export]]
SEXP est_target_params_cpp(SEXP ome_xptr,
                          Rcpp::Nullable<Rcpp::List> stats_xptrs_by_cohort,
                          Rcpp::Nullable<Rcpp::List> eta_xptrs_by_cohort,
                          Rcpp::Function r_fn) {
    Rcpp::XPtr<OutcomeMeansEstimates> ome(ome_xptr);
    auto stats_vec = list_to_stats_vec(stats_xptrs_by_cohort);
    auto eta_vec = list_to_eta_vec(eta_xptrs_by_cohort);
    apm::TargetFn cb = make_target_fn(r_fn);
    // NOTE: Calling R from multiple threads is unsafe. We therefore force single-threaded
    // execution (num_threads = 1) for target param estimation invoked via R bindings.
    std::optional<std::size_t> nt_opt = std::optional<std::size_t>(1);
    TargetParameterEstimates out = apm::est_target_params(*ome, stats_vec, eta_vec, cb, nt_opt);
    return make_xptr(std::move(out));
}

// Accessors aligned with other *Estimates
// [[Rcpp::export]]
bool tpe_has_bootstrap_cpp(SEXP xp_) {
    Rcpp::XPtr<TargetParameterEstimates> xp(xp_);
    return xp->has_bootstrap_replicates();
}

// [[Rcpp::export]]
int tpe_num_bootstrap_cpp(SEXP xp_) {
    Rcpp::XPtr<TargetParameterEstimates> xp(xp_);
    return static_cast<int>(xp->n_bootstrap_replicates());
}

// [[Rcpp::export]]
int tpe_p_cpp(SEXP xp_) {
    Rcpp::XPtr<TargetParameterEstimates> xp(xp_);
    return static_cast<int>(xp->p());
}

// [[Rcpp::export]]
Rcpp::NumericVector tpe_point_params_cpp(SEXP xp_) {
    Rcpp::XPtr<TargetParameterEstimates> xp(xp_);
    const arma::vec& v = xp->point;
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector tpe_boot_params_cpp(SEXP xp_, int b1) {
    Rcpp::XPtr<TargetParameterEstimates> xp(xp_);
    int B = static_cast<int>(xp->bootstrap_replicates.n_cols);
    if (b1 < 1 || b1 > B) Rcpp::stop("bootstrap index out of range");
    arma::vec v = xp->bootstrap_replicates.col(static_cast<arma::uword>(b1 - 1));
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix tpe_boot_params_matrix_cpp(SEXP xp_) {
    Rcpp::XPtr<TargetParameterEstimates> xp(xp_);
    const std::size_t B = xp->n_bootstrap_replicates();
    const std::size_t p = xp->p();
    // Directly wrap the arma::mat as R matrix
    const arma::mat& M = xp->bootstrap_replicates;
    if (static_cast<std::size_t>(M.n_rows) != p) Rcpp::stop("Unexpected bootstrap matrix n_rows vs p.");
    Rcpp::NumericMatrix out(static_cast<int>(M.n_rows), static_cast<int>(M.n_cols));
    std::copy(M.begin(), M.end(), out.begin());
    return out;
}

// By-spec variant: named list of XPtr<TargetParameterEstimates>
// [[Rcpp::export]]
Rcpp::List est_target_params_by_spec_cpp(Rcpp::List ome_by_spec,
                                         Rcpp::Nullable<Rcpp::List> stats_by_spec,
                                         Rcpp::Nullable<Rcpp::List> eta_by_spec,
                                         Rcpp::Function r_fn) {
    // Build maps
std::unordered_map<std::string, OutcomeMeansEstimates> ome_map;
    {
        Rcpp::CharacterVector nms = ome_by_spec.names();
        for (int i = 0; i < ome_by_spec.size(); ++i) {
            std::string key = Rcpp::as<std::string>(nms[i]);
            Rcpp::XPtr<OutcomeMeansEstimates> xp(ome_by_spec[i]);
            ome_map.emplace(std::move(key), *xp);
        }
    }

    std::unordered_map<std::string, std::vector<CohortAuxiliaryDataMeanEstimates>> eta_map;
    std::unordered_map<std::string, std::vector<apm::OutcomeMeanSuffStatEstimates>> stats_map;
    if (eta_by_spec.isNotNull()) {
        Rcpp::List L(eta_by_spec);
        Rcpp::CharacterVector nms = L.names();
        for (int i = 0; i < L.size(); ++i) {
            std::string key = Rcpp::as<std::string>(nms[i]);
            std::vector<CohortAuxiliaryDataMeanEstimates> eta_vec = list_to_eta_vec(Rcpp::List(L[i]));
            eta_map.emplace(std::move(key), std::move(eta_vec));
        }
    }
    if (stats_by_spec.isNotNull()) {
        Rcpp::List L(stats_by_spec);
        Rcpp::CharacterVector nms = L.names();
        for (int i = 0; i < L.size(); ++i) {
            std::string key = Rcpp::as<std::string>(nms[i]);
            std::vector<apm::OutcomeMeanSuffStatEstimates> stats_vec = list_to_stats_vec(Rcpp::List(L[i]));
            stats_map.emplace(std::move(key), std::move(stats_vec));
        }
    }

    apm::TargetFn cb = make_target_fn(r_fn);
    // NOTE: Calling R from multiple threads is unsafe. We therefore force single-threaded
    // execution (num_threads = 1) for target param estimation invoked via R bindings.
    std::optional<std::size_t> nt_opt = std::optional<std::size_t>(1); // single-thread for R safety
    auto out_map = apm::est_target_params(ome_map, stats_map, eta_map, cb, nt_opt);

    Rcpp::List out(static_cast<int>(out_map.size()));
    Rcpp::CharacterVector names(static_cast<int>(out_map.size()));
    int k = 0;
    for (auto& kv : out_map) {
        names[k] = kv.first;
        out[k] = make_xptr(std::move(kv.second));
        ++k;
    }
    out.attr("names") = names;
    return out;
}

//------------------------------------------------------------------------------
// End-to-end: estimate target parameter components from panel (by spec)
//------------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List est_target_param_components_from_panel_cpp(
    SEXP panel_holder_xptr,
    Rcpp::List est_specs,
    SEXP bootstrap_xptr = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in = R_NilValue,
    Rcpp::Nullable<Rcpp::List> cohort_outcomes_to_mask_in = R_NilValue,
    bool est_outcome_means_via_imputation = true,
    Rcpp::Nullable<Rcpp::NumericVector> imputation_tol_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> imputation_max_iters_in = R_NilValue,
    Rcpp::Nullable<Rcpp::String> imputation_fixed_point_method_in = R_NilValue)
{
    const apm::InMemoryUnbalancedPanel& panel = apm::r_utils::panel_ref_from_panel_holder(panel_holder_xptr);
    auto cpp_specs = apm::r_utils::to_cpp_specs(est_specs);
    auto wb = apm::r_utils::xp_to_const_wb_shared(bootstrap_xptr);
    std::optional<std::size_t> nt_opt = std::nullopt; {
        auto p = apm::r_utils::resolve_num_threads(num_threads_in);
        if (p.first) nt_opt = p.second;
    }
    apm::CohortOutcomeMask mask = apm::r_utils::to_cpp_mask(cohort_outcomes_to_mask_in);

    double tol = apm::DEFAULT_TOL;
    if (imputation_tol_in.isNotNull()) {
        Rcpp::NumericVector v(imputation_tol_in);
        if (v.size() > 0) tol = static_cast<double>(v[0]);
    }
    std::size_t max_iters = apm::DEFAULT_MAX_ITERS;
    if (imputation_max_iters_in.isNotNull()) {
        Rcpp::IntegerVector iv(imputation_max_iters_in);
        if (iv.size() > 0) {
            int mi = iv[0];
            if (mi < 0) mi = 0;
            max_iters = static_cast<std::size_t>(mi);
        }
    }
    std::string fixed_point_method = apm::DEFAULT_FP_METHOD;
    if (imputation_fixed_point_method_in.isNotNull()) {
        Rcpp::String s = Rcpp::String(imputation_fixed_point_method_in.get());
        fixed_point_method = std::string(s.get_cstring());
    }

    apm::TargetParamComponents comps = apm::est_target_param_components_from_panel(
        panel, cpp_specs, wb, nt_opt, mask, est_outcome_means_via_imputation, tol, max_iters, fixed_point_method);

    Rcpp::List ome_out(static_cast<int>(comps.outcome_means_by_spec.size()));
    Rcpp::CharacterVector names(static_cast<int>(comps.outcome_means_by_spec.size()));
    int k = 0;
    for (auto& kv : comps.outcome_means_by_spec) {
        names[k] = kv.first;
        ome_out[k] = make_xptr(std::move(kv.second));
        ++k;
    }
    ome_out.attr("names") = names;

    Rcpp::RObject aux_out = R_NilValue;
    if (!comps.cohort_auxiliary_means.empty()) {
        Rcpp::List aux_list(static_cast<int>(comps.cohort_auxiliary_means.size()));
        for (int i = 0; i < static_cast<int>(comps.cohort_auxiliary_means.size()); ++i) {
            aux_list[i] = make_xptr(apm::CohortAuxiliaryDataMeanEstimates(comps.cohort_auxiliary_means[static_cast<std::size_t>(i)]));
        }
        aux_out = aux_list;
    }

    Rcpp::List final(2);
    final["outcome_means"] = ome_out;
    final["auxiliary_means"] = aux_out;
    if (comps.masked_observed_outcome_indices.has_value()) {
        final.push_back(apm::r_utils::to_r_observed_outcome_indices(*comps.masked_observed_outcome_indices),
                        "masked_observed_outcome_indices");
    }
    if (!comps.masked_cohort_outcome_means.empty()) {
        final.push_back(apm::r_utils::masked_means_to_r_list(comps.masked_cohort_outcome_means),
                        "masked_cohort_outcome_means");
    }
    return final;
}

//------------------------------------------------------------------------------
// Inference for target parameters given panel
//------------------------------------------------------------------------------

// [[Rcpp::export]]
SEXP target_param_inference_cpp(SEXP tpe_xptr,
                                SEXP panel_holder_xptr,
                                double sig_level = 0.05) {
    Rcpp::XPtr<apm::TargetParameterEstimates> tpe(tpe_xptr);
    const apm::InMemoryUnbalancedPanel& panel = apm::r_utils::panel_ref_from_panel_holder(panel_holder_xptr);
    apm::SimultaneousInferenceResults res = apm::target_param_inference(*tpe, panel, sig_level);
    return apm::r_utils::make_xptr(std::move(res));
}

// By-spec inference: named list of XPtr<TargetParameterEstimates> -> named list of XPtr<SimultaneousInferenceResults>
// [[Rcpp::export]]
Rcpp::List target_param_inference_by_spec_cpp(Rcpp::List tpe_by_spec,
                                              SEXP panel_holder_xptr,
                                              double sig_level = 0.05) {
    // Build input map<string, TargetParameterEstimates>
    std::unordered_map<std::string, apm::TargetParameterEstimates> ests_map;
    {
        Rcpp::CharacterVector nms = tpe_by_spec.names();
        for (int i = 0; i < tpe_by_spec.size(); ++i) {
            std::string key = Rcpp::as<std::string>(nms[i]);
            Rcpp::XPtr<apm::TargetParameterEstimates> xp(tpe_by_spec[i]);
            ests_map.emplace(std::move(key), *xp);
        }
    }

    // Panel ref
    const apm::InMemoryUnbalancedPanel& panel = apm::r_utils::panel_ref_from_panel_holder(panel_holder_xptr);

    // Delegate to core overload
    auto res_map = apm::target_param_inference(ests_map, panel, sig_level);

    // Return named list of XPtr<SimultaneousInferenceResults>
    Rcpp::List out(static_cast<int>(res_map.size()));
    Rcpp::CharacterVector names(static_cast<int>(res_map.size()));
    int k = 0;
    for (auto& kv : res_map) {
        names[k] = kv.first;
        out[k] = apm::r_utils::make_xptr(std::move(kv.second));
        ++k;
    }
    out.attr("names") = names;
    return out;
}