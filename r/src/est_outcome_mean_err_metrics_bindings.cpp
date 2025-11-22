#include <RcppArmadillo.h>
#include "r_utils.h"
#include "../../core/src/target_params/est_outcome_mean_err_metrics.h"

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::DataFrame est_masked_outcome_mean_err_metrics_cpp(Rcpp::List comps_list) {
    // Convert R list back into TargetParamComponents
    apm::TargetParamComponents comps = apm::r_utils::target_param_components_from_r_list(comps_list);

    // Compute metrics
    auto metrics_map = apm::est_masked_outcome_mean_err_metrics(comps);

    // Determine spec names for consistent iteration order
    std::vector<std::string> spec_names;
    spec_names.reserve(comps.outcome_means_by_spec.size());
    for (const auto& kv : comps.outcome_means_by_spec) spec_names.push_back(kv.first);

    const std::size_t P = metrics_map.size();
    const std::size_t S = spec_names.size();
    const int n = static_cast<int>(P * S);

    Rcpp::IntegerVector cohort(n), outcome(n);
    Rcpp::CharacterVector spec(n);
    Rcpp::NumericVector bias(n), se(n), rmse(n), cohort_pop_share(n);

    int i = 0;
    for (const auto& kv : metrics_map) {
        const std::size_t c0 = kv.first.first;
        const std::size_t t0 = kv.first.second;
        const apm::OutcomeMeanErrMetrics& m = kv.second;

        for (const auto& sname : spec_names) {
            cohort[i] = static_cast<int>(c0 + 1);  // 1-based for R
            outcome[i] = static_cast<int>(t0 + 1);
            spec[i] = sname.c_str();
            cohort_pop_share[i] = m.cohort_pop_share;

            auto itb = m.bias_by_spec.find(sname);
            auto its = m.se_by_spec.find(sname);
            auto itr = m.rmse_by_spec.find(sname);

            bias[i] = (itb != m.bias_by_spec.end() ? itb->second : NA_REAL);
            se[i] = (its != m.se_by_spec.end() ? its->second : NA_REAL);
            rmse[i] = (itr != m.rmse_by_spec.end() ? itr->second : NA_REAL);
            ++i;
        }
    }

    return Rcpp::DataFrame::create(
        Rcpp::Named("cohort") = cohort,
        Rcpp::Named("outcome") = outcome,
        Rcpp::Named("spec") = spec,
        Rcpp::Named("bias") = bias,
        Rcpp::Named("se") = se,
        Rcpp::Named("rmse") = rmse,
        Rcpp::Named("cohort_pop_share") = cohort_pop_share,
        Rcpp::Named("stringsAsFactors") = false
    );
}


