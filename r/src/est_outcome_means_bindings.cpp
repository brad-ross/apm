#include <RcppArmadillo.h>
#include <unordered_map>
#include <string>
#include <set>
#include "r_utils.h"
#include "../../core/src/est_outcome_means.h"

// [[Rcpp::depends(RcppArmadillo)]]

namespace {

template <typename T>
Rcpp::XPtr<T> make_xptr(T&& obj) {
    T* heap = new T(std::move(obj));
    return Rcpp::XPtr<T>(heap, true);
}

} // anonymous namespace

//------------------------------------------------------------------------------
// Aggregators
//------------------------------------------------------------------------------

// [[Rcpp::export]]
SEXP aggregate_factor_model_params_cpp(
    Rcpp::List per_cohort_factor_estimates_xptrs,
    Rcpp::List observed_outcome_indices, // 1-based
    SEXP cohort_weights_xptr) {

    std::vector<apm::FactorModelEstimates> vec;
    vec.reserve(per_cohort_factor_estimates_xptrs.size());
    for (int i = 0; i < per_cohort_factor_estimates_xptrs.size(); ++i) {
        Rcpp::XPtr<apm::FactorModelEstimates> xp(per_cohort_factor_estimates_xptrs[i]);
        vec.push_back(*xp);
    }

    apm::ObservedOutcomeIndices obs_idx = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    Rcpp::XPtr<apm::CohortWeightEstimates> wptr(cohort_weights_xptr);
    apm::FactorModelEstimates out = apm::aggregate_cohort_specific_factor_model_params(
        vec, obs_idx, *wptr);
    return make_xptr(std::move(out));
}

// [[Rcpp::export]]
Rcpp::List aggregate_factor_model_params_by_spec_cpp(
    Rcpp::List cohort_specific_factor_ests, // named list: each value is list of XPtr<FactorModelEstimates>
    Rcpp::List observed_outcome_indices,    // 1-based
    Rcpp::List cohort_weights_by_spec       // named list: each value is XPtr<CohortWeightEstimates>
) {
    // Build factor map
    std::unordered_map<std::string, std::vector<apm::FactorModelEstimates>> fact_map;
    {
        Rcpp::CharacterVector nms = cohort_specific_factor_ests.names();
        for (int i = 0; i < cohort_specific_factor_ests.size(); ++i) {
            std::string key = Rcpp::as<std::string>(nms[i]);
            Rcpp::List per_cohort = cohort_specific_factor_ests[i];
            std::vector<apm::FactorModelEstimates> vec;
            vec.reserve(per_cohort.size());
            for (int j = 0; j < per_cohort.size(); ++j) {
                Rcpp::XPtr<apm::FactorModelEstimates> xp(per_cohort[j]);
                vec.push_back(*xp);
            }
            fact_map.emplace(std::move(key), std::move(vec));
        }
    }

    // Build weight map
    std::unordered_map<std::string, apm::CohortWeightEstimates> w_map;
    {
        Rcpp::CharacterVector nms = cohort_weights_by_spec.names();
        for (int i = 0; i < cohort_weights_by_spec.size(); ++i) {
            std::string key = Rcpp::as<std::string>(nms[i]);
            Rcpp::XPtr<apm::CohortWeightEstimates> xp(cohort_weights_by_spec[i]);
            w_map.emplace(std::move(key), *xp);
        }
    }

    // Union of keys and symmetric validation (mirror build_return_list)
    std::set<std::string> union_keys;
    for (const auto& kv : fact_map) union_keys.insert(kv.first);
    for (const auto& kw : w_map) union_keys.insert(kw.first);

    for (const auto& key : union_keys) {
        if (fact_map.find(key) == fact_map.end()) {
            Rcpp::stop("Spec key present in cohort weights but missing in factor estimates: " + key);
        }
        if (w_map.find(key) == w_map.end()) {
            Rcpp::stop("Spec key present in factor estimates but missing in cohort weights: " + key);
        }
    }

    apm::ObservedOutcomeIndices obs_idx = apm::r_utils::to_cpp_observed_outcome_indices(observed_outcome_indices);
    auto out_map = apm::aggregate_cohort_specific_factor_model_params(fact_map, obs_idx, w_map);

    // Return named list of XPtr<FactorModelEstimates>
    Rcpp::List out(static_cast<int>(out_map.size()));
    Rcpp::CharacterVector names(static_cast<int>(out_map.size()));
    int k = 0;
    for (const auto& kv : out_map) {
        names[k] = kv.first;
        out[k] = make_xptr(apm::FactorModelEstimates(kv.second));
        ++k;
    }
    out.attr("names") = names;
    return out;
}

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


