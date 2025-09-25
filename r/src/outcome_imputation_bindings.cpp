#include <RcppArmadillo.h>
#include <unordered_map>
#include <string>
#include "r_utils.h"
#include "cohort_specific_estimates_helpers.h"
#include "../../core/src/outcome_imputation.h"
#include "../../core/src/cohort_specific_param_structs.h"

using apm::r_utils::make_xptr;

// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// Compute imputation components (point + bootstrap) for one estimator spec.
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
SEXP comp_imputation_components_cpp(
    SEXP panel_holder_xptr,
    SEXP factor_model_estimates_xptr,
    Rcpp::Nullable<Rcpp::List> cohort_outcome_mean_suff_stat_ests = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> unit_weights = R_NilValue,
    Rcpp::Nullable<Rcpp::List> effective_observed_outcome_indices = R_NilValue,
    double tol = 1e-10,
    std::size_t max_iters = 1000,
    const std::string& fixed_point_method = "irons-tuck",
    Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in = R_NilValue)
{
    const apm::InMemoryUnbalancedPanel& panel = apm::r_utils::panel_ref_from_panel_holder(panel_holder_xptr);
    Rcpp::XPtr<apm::FactorModelEstimates> fptr(factor_model_estimates_xptr);

    // Optional args
    std::optional<arma::vec> unit_w_opt = std::nullopt;
    if (unit_weights.isNotNull()) {
        arma::vec w = Rcpp::as<arma::vec>(unit_weights);
        if (w.n_elem > 0) unit_w_opt = std::move(w);
    }

    std::optional<apm::ObservedOutcomeIndices> eff_ooi_opt = std::nullopt;
    if (effective_observed_outcome_indices.isNotNull()) {
        apm::ObservedOutcomeIndices ooi0b = apm::r_utils::to_cpp_observed_outcome_indices(Rcpp::List(effective_observed_outcome_indices));
        eff_ooi_opt = std::move(ooi0b);
    }

    auto stats_vec = apm::r_utils::list_to_stats_vec(cohort_outcome_mean_suff_stat_ests);

    // Resolve num_threads
    std::optional<std::size_t> nt_opt = std::nullopt; {
        auto p = apm::r_utils::resolve_num_threads(num_threads_in);
        if (p.first) nt_opt = p.second;
    }

    apm::FactorModelEstimates out = apm::comp_imputation_components(
        panel,
        *fptr,
        stats_vec,
        unit_w_opt,
        eff_ooi_opt,
        tol,
        max_iters,
        fixed_point_method,
        nt_opt);

    return make_xptr(std::move(out));
}

// -----------------------------------------------------------------------------
// Named-list variant: factor_model_estimates_by_spec is named list of XPtr<FactorModelEstimates>.
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List comp_imputation_components_by_spec_cpp(
    SEXP panel_holder_xptr,
    Rcpp::List factor_model_estimates_by_spec,
    Rcpp::Nullable<Rcpp::List> cohort_outcome_mean_suff_stat_ests = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> unit_weights = R_NilValue,
    Rcpp::Nullable<Rcpp::List> effective_observed_outcome_indices = R_NilValue,
    double tol = 1e-10,
    std::size_t max_iters = 1000,
    const std::string& fixed_point_method = "irons-tuck",
    Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in = R_NilValue)
{
    const apm::InMemoryUnbalancedPanel& panel = apm::r_utils::panel_ref_from_panel_holder(panel_holder_xptr);

    std::unordered_map<std::string, apm::FactorModelEstimates> fmap;
    {
        Rcpp::CharacterVector nms = factor_model_estimates_by_spec.names();
        for (int i = 0; i < factor_model_estimates_by_spec.size(); ++i) {
            std::string key = Rcpp::as<std::string>(nms[i]);
            Rcpp::XPtr<apm::FactorModelEstimates> xp(factor_model_estimates_by_spec[i]);
            fmap.emplace(std::move(key), *xp);
        }
    }

    // Optional args
    std::optional<arma::vec> unit_w_opt = std::nullopt;
    if (unit_weights.isNotNull()) {
        arma::vec w = Rcpp::as<arma::vec>(unit_weights);
        if (w.n_elem > 0) unit_w_opt = std::move(w);
    }

    std::optional<apm::ObservedOutcomeIndices> eff_ooi_opt = std::nullopt;
    if (effective_observed_outcome_indices.isNotNull()) {
        apm::ObservedOutcomeIndices ooi0b = apm::r_utils::to_cpp_observed_outcome_indices(Rcpp::List(effective_observed_outcome_indices));
        eff_ooi_opt = std::move(ooi0b);
    }

    auto stats_vec = apm::r_utils::list_to_stats_vec(cohort_outcome_mean_suff_stat_ests);

    // Resolve num_threads
    std::optional<std::size_t> nt_opt = std::nullopt; {
        auto p = apm::r_utils::resolve_num_threads(num_threads_in);
        if (p.first) nt_opt = p.second;
    }

    auto out_map = apm::comp_imputation_components(
        panel,
        fmap,
        stats_vec,
        unit_w_opt,
        eff_ooi_opt,
        tol,
        max_iters,
        fixed_point_method,
        nt_opt);

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


