#include <RcppArmadillo.h>
#include <unordered_map>
#include <string>
#include <set>
#include "r_utils.h"
#include "cohort_specific_estimates_helpers.h"
#include "../../core/src/est_outcome_means.h"
#include "../../core/src/utils.h"
#include "../../core/src/agg_cohort_specific_factor_model_params.h"

// [[Rcpp::depends(RcppArmadillo)]]

using apm::r_utils::make_xptr;

//------------------------------------------------------------------------------
// Outcome mean estimation
//------------------------------------------------------------------------------

// [[Rcpp::export]]
SEXP estimate_outcome_means_across_cohorts_cpp(
    SEXP factor_model_estimates_xptr,
    Rcpp::List observed_outcome_indices,
    Rcpp::List suff_stat_estimates_xptrs) {

    Rcpp::XPtr<apm::FactorModelEstimates> fptr(factor_model_estimates_xptr);

    std::vector<apm::OutcomeMeanSuffStatEstimates> suff_vec;
    suff_vec.reserve(suff_stat_estimates_xptrs.size());
    for (int i = 0; i < suff_stat_estimates_xptrs.size(); ++i) {
        Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> xp(suff_stat_estimates_xptrs[i]);
        suff_vec.push_back(*xp);
    }

    apm::ObservedOutcomeIndices obs_idx = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    apm::OutcomeMeansEstimates out = apm::estimate_outcome_means_across_cohorts(*fptr, obs_idx, suff_vec);
    return make_xptr(std::move(out));
}

// [[Rcpp::export]]
Rcpp::List estimate_outcome_means_across_cohorts_by_spec_cpp(
    Rcpp::List factor_model_estimates_by_spec, // named list of XPtr<FactorModelEstimates>
    Rcpp::List observed_outcome_indices,
    Rcpp::List suff_stat_estimates_xptrs) {

    std::unordered_map<std::string, apm::FactorModelEstimates> fmap;
    {
        Rcpp::CharacterVector nms = factor_model_estimates_by_spec.names();
        for (int i = 0; i < factor_model_estimates_by_spec.size(); ++i) {
            std::string key = Rcpp::as<std::string>(nms[i]);
            Rcpp::XPtr<apm::FactorModelEstimates> xp(factor_model_estimates_by_spec[i]);
            fmap.emplace(std::move(key), *xp);
        }
    }

    std::vector<apm::OutcomeMeanSuffStatEstimates> suff_vec;
    suff_vec.reserve(suff_stat_estimates_xptrs.size());
    for (int i = 0; i < suff_stat_estimates_xptrs.size(); ++i) {
        Rcpp::XPtr<apm::OutcomeMeanSuffStatEstimates> xp(suff_stat_estimates_xptrs[i]);
        suff_vec.push_back(*xp);
    }

    apm::ObservedOutcomeIndices obs_idx = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    auto out_map = apm::estimate_outcome_means_across_cohorts(fmap, obs_idx, suff_vec);

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
    Rcpp::Nullable<Rcpp::List> cohort_outcomes_to_mask_in = R_NilValue)
{
    // 1) Cohort-specific estimates (raw C++) via panel-holder core
    apm::CohortSpecificEstimates ests = cohort_specific_estimates_from_panel_cpp_core(
        panel_holder_xptr,
        est_specs,
        bootstrap_xptr,
        num_threads_in,
        cohort_outcomes_to_mask_in
    );

    // 2) Observed outcome indices (prefer masked if present)
    const apm::ObservedOutcomeIndices obs_idx_panel = observed_outcome_indices_from_panel_holder(panel_holder_xptr);
    const apm::ObservedOutcomeIndices& obs_idx_eff = ests.masked_observed_outcome_indices.has_value()
        ? *ests.masked_observed_outcome_indices
        : obs_idx_panel;

    // 3) Aggregate factor model params by spec (use obs_idx_eff)
    std::unordered_map<std::string, apm::FactorModelEstimates> agg_by_spec =
        apm::aggregate_cohort_specific_factor_model_params(
            ests.cohort_specific_factor_ests,
            obs_idx_eff,
            ests.cohort_weights);

    // 4) Estimate outcome means by spec (use obs_idx_eff)
    std::unordered_map<std::string, apm::OutcomeMeansEstimates> ome_by_spec =
        apm::estimate_outcome_means_across_cohorts(
            agg_by_spec,
            obs_idx_eff,
            ests.cohort_outcome_mean_ests);

    // 5) Build return list with outcome_means (by spec) and optional auxiliary_means (by cohort)
    Rcpp::List ome_out(static_cast<int>(ome_by_spec.size()));
    Rcpp::CharacterVector names(static_cast<int>(ome_by_spec.size()));
    int k = 0;
    for (auto& kv : ome_by_spec) {
        names[k] = kv.first;
        ome_out[k] = make_xptr(std::move(kv.second));
        ++k;
    }
    ome_out.attr("names") = names;

    Rcpp::RObject aux_out = R_NilValue;
    if (!ests.cohort_auxiliary_means.empty()) {
        Rcpp::List aux_list(static_cast<int>(ests.cohort_auxiliary_means.size()));
        for (int i = 0; i < static_cast<int>(ests.cohort_auxiliary_means.size()); ++i) {
            aux_list[i] = make_xptr(apm::CohortAuxiliaryDataMeanEstimates(ests.cohort_auxiliary_means[static_cast<std::size_t>(i)]));
        }
        aux_out = aux_list;
    }

    // Build final list with required fields
    Rcpp::List final(2);
    final["outcome_means"] = ome_out;
    final["auxiliary_means"] = aux_out;

    // Attach masked outputs if present (mirror cohort-specific bindings)
    if (ests.masked_observed_outcome_indices.has_value()) {
        final.push_back(apm::r_utils::to_r_observed_outcome_indices(*ests.masked_observed_outcome_indices),
                        "masked_observed_outcome_indices");
    }
    if (!ests.masked_cohort_outcome_means.empty()) {
        final.push_back(apm::r_utils::masked_means_to_r_list(ests.masked_cohort_outcome_means),
                        "masked_cohort_outcome_means");
    }

    return final;
}

//------------------------------------------------------------------------------
// oneTBB helpers
//------------------------------------------------------------------------------

// [[Rcpp::export]]
int get_cpp_default_concurrency() { return static_cast<int>(apm::get_cpp_default_concurrency()); }

//------------------------------------------------------------------------------
// Simple factories to aid tests
//------------------------------------------------------------------------------

// [[Rcpp::export]]
SEXP make_factor_model_estimates_cpp(const arma::mat& G,
                                     Rcpp::Nullable<arma::vec> g0 = R_NilValue,
                                     Rcpp::Nullable<arma::vec> a = R_NilValue,
                                     int B = 0) {
    std::optional<arma::vec> g0_opt;
    std::optional<arma::vec> a_opt;
    if (g0.isNotNull()) g0_opt = Rcpp::as<arma::vec>(g0);
    if (a.isNotNull()) a_opt = Rcpp::as<arma::vec>(a);
    apm::FactorModelParameters point(G, g0_opt, a_opt);
    std::vector<apm::FactorModelParameters> boots;
    if (B > 0) boots = std::vector<apm::FactorModelParameters>(static_cast<std::size_t>(B), point);
    apm::FactorModelEstimates out(std::move(point), std::move(boots));
    return make_xptr(std::move(out));
}

// [[Rcpp::export]]
SEXP make_outcome_mean_suff_stat_estimates_cpp(const arma::vec& observed_means,
                                               Rcpp::Nullable<arma::mat> covar_means = R_NilValue,
                                               int B = 0) {
    std::optional<arma::mat> X_opt;
    if (covar_means.isNotNull()) X_opt = Rcpp::as<arma::mat>(covar_means);
    apm::OutcomeMeanSufficientStatistics point(observed_means, X_opt);
    std::vector<apm::OutcomeMeanSufficientStatistics> boots;
    if (B > 0) boots = std::vector<apm::OutcomeMeanSufficientStatistics>(static_cast<std::size_t>(B), point);
    apm::OutcomeMeanSuffStatEstimates out(std::move(point), std::move(boots));
    return make_xptr(std::move(out));
}

// [[Rcpp::export]]
SEXP make_cohort_weight_estimates_cpp(const arma::vec& cohort_weights,
                                      Rcpp::Nullable<Rcpp::List> bootstrap_weights = R_NilValue) {
    apm::CohortWeightEstimates w;
    w.cohort_weights = cohort_weights;
    if (bootstrap_weights.isNotNull()) {
        Rcpp::List L(bootstrap_weights);
        w.bootstrap_cohort_weights.reserve(L.size());
        for (int i = 0; i < L.size(); ++i) {
            w.bootstrap_cohort_weights.push_back(Rcpp::as<arma::vec>(L[i]));
        }
    }
    auto* heap = new apm::CohortWeightEstimates(std::move(w));
    return Rcpp::XPtr<apm::CohortWeightEstimates>(heap, true);
}


