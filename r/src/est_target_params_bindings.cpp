#include <RcppArmadillo.h>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

#include "r_utils.h"
#include "../../core/src/target_params/est_target_params.h"
#include "../../core/src/est_outcome_means.h"
#include "../../core/src/cohort_specific_param_structs.h"

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

apm::TargetFn make_target_fn(Rcpp::Function r_fn) {
    return [r_fn](const arma::mat& Y,
                  const std::vector<CohortAuxiliaryDataMeans>& eta_all) -> arma::vec {
        Rcpp::NumericMatrix Y_r(Y.n_rows, Y.n_cols);
        std::copy(Y.begin(), Y.end(), Y_r.begin());
        Rcpp::NumericVector shares(static_cast<int>(eta_all.size()));
        Rcpp::List eta_list(static_cast<int>(eta_all.size()));
        for (std::size_t c = 0; c < eta_all.size(); ++c) {
            shares[static_cast<int>(c)] = eta_all[c].cohort_pop_share;
            const arma::mat& M = eta_all[c].auxiliary_means;
            Rcpp::NumericMatrix Mr(M.n_rows, M.n_cols);
            std::copy(M.begin(), M.end(), Mr.begin());
            eta_list[static_cast<int>(c)] = Mr;
        }
        Rcpp::RObject res = r_fn(Y_r, shares, eta_list);
        return Rcpp::as<arma::vec>(res);
    };
}

} // anonymous namespace

// Compute from single spec
// [[Rcpp::export]]
SEXP est_target_params_cpp(SEXP ome_xptr,
                          Rcpp::Nullable<Rcpp::List> eta_xptrs_by_cohort,
                          Rcpp::Function r_fn) {
    Rcpp::XPtr<OutcomeMeansEstimates> ome(ome_xptr);
    auto eta_vec = list_to_eta_vec(eta_xptrs_by_cohort);
    apm::TargetFn cb = make_target_fn(r_fn);
    // NOTE: Calling R from multiple threads is unsafe. We therefore force single-threaded
    // execution (num_threads = 1) for target param estimation invoked via R bindings.
    std::optional<std::size_t> nt_opt = std::optional<std::size_t>(1);
    TargetParameterEstimates out = apm::est_target_params(*ome, eta_vec, cb, nt_opt);
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
    int B = static_cast<int>(xp->bootstrap_replicates.size());
    if (b1 < 1 || b1 > B) Rcpp::stop("bootstrap index out of range");
    const arma::vec& v = xp->bootstrap_replicates[static_cast<std::size_t>(b1 - 1)];
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix tpe_boot_params_matrix_cpp(SEXP xp_) {
    Rcpp::XPtr<TargetParameterEstimates> xp(xp_);
    const std::size_t B = xp->n_bootstrap_replicates();
    const std::size_t p = xp->p();
    Rcpp::NumericMatrix out(static_cast<int>(p), static_cast<int>(B));
    for (std::size_t b = 0; b < B; ++b) {
        const arma::vec& v = xp->bootstrap_replicates[b];
        if (v.n_elem != p) Rcpp::stop("Inconsistent p across bootstrap replicates.");
        for (std::size_t i = 0; i < p; ++i) out(static_cast<int>(i), static_cast<int>(b)) = v(i);
    }
    return out;
}

// By-spec variant: named list of XPtr<TargetParameterEstimates>
// [[Rcpp::export]]
Rcpp::List est_target_params_by_spec_cpp(Rcpp::List ome_by_spec,
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
    if (eta_by_spec.isNotNull()) {
        Rcpp::List L(eta_by_spec);
        Rcpp::CharacterVector nms = L.names();
        for (int i = 0; i < L.size(); ++i) {
            std::string key = Rcpp::as<std::string>(nms[i]);
            std::vector<CohortAuxiliaryDataMeanEstimates> eta_vec = list_to_eta_vec(Rcpp::List(L[i]));
            eta_map.emplace(std::move(key), std::move(eta_vec));
        }
    }

    apm::TargetFn cb = make_target_fn(r_fn);
    // NOTE: Calling R from multiple threads is unsafe. We therefore force single-threaded
    // execution (num_threads = 1) for target param estimation invoked via R bindings.
    std::optional<std::size_t> nt_opt = std::optional<std::size_t>(1); // single-thread for R safety
    auto out_map = apm::est_target_params(ome_map, eta_map, cb, nt_opt);

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


