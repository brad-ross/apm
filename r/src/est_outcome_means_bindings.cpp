#include <RcppArmadillo.h>
#include <unordered_map>
#include <string>
#include <set>
#include "r_utils.h"
#include "../../core/src/est_outcome_means.h"
#include "../../core/src/utils.h"

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
// oneTBB helpers
//------------------------------------------------------------------------------

//' Get the default C++ concurrency level
//'
//' Returns the default number of threads used by the C++ backend for parallel
//' computations. This value is typically determined by the oneTBB library
//' and corresponds to the number of hardware threads available on the system.
//'
//' @return An integer indicating the default number of threads for C++
//'   parallel operations.
//' @seealso \code{\link{set_apm_threads}} for setting the number of threads.
//' @examples
//' get_cpp_default_concurrency()
//' @export
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


