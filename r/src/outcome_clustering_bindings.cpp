#include <RcppArmadillo.h>
#include <optional>
#include <string>
#include <vector>

#include "r_utils.h"
#include "../../core/src/outcome_clustering/cluster_outcomes.h"
#include "../../core/src/panels/InMemoryUnbalancedPanel.h"

// [[Rcpp::depends(RcppArmadillo)]]

namespace {

inline std::optional<std::size_t> parse_optional_size(Rcpp::Nullable<Rcpp::IntegerVector> maybe_int) {
    if (maybe_int.isNotNull()) {
        Rcpp::IntegerVector v(maybe_int);
        if (v.size() > 0 && !Rcpp::IntegerVector::is_na(v[0]) && v[0] > 0) {
            return static_cast<std::size_t>(v[0]);
        }
    }
    return std::nullopt;
}

inline std::optional<uint64_t> parse_optional_u64(Rcpp::Nullable<Rcpp::NumericVector> maybe_num) {
    if (maybe_num.isNotNull()) {
        Rcpp::NumericVector v(maybe_num);
        if (v.size() > 0 && !Rcpp::NumericVector::is_na(v[0])) {
            double d = v[0];
            if (Rcpp::traits::is_finite<REALSXP>(d)) {
                if (d < 0.0) Rcpp::stop("seed must be non-negative");
                return static_cast<uint64_t>(d);
            }
        }
    }
    return std::nullopt;
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Single-k mapping: returns 1-based cluster ids as IntegerVector length T
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::IntegerVector comp_outcome_clustering_cpp(
    SEXP panel_holder_xptr,
    int grid_size,
    int k,
    Rcpp::Nullable<Rcpp::IntegerVector> n_inits_in = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> seed_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in = R_NilValue)
{
    const apm::InMemoryUnbalancedPanel& panel = apm::r_utils::panel_ref_from_panel_holder(panel_holder_xptr);

    std::optional<std::size_t> n_inits = parse_optional_size(n_inits_in);
    std::optional<uint64_t> seed = parse_optional_u64(seed_in);
    std::optional<std::size_t> num_threads = parse_optional_size(num_threads_in);

    arma::uvec mapping0 = apm::comp_outcome_clustering(
        panel,
        static_cast<std::size_t>(grid_size),
        static_cast<std::size_t>(k),
        n_inits,
        seed,
        num_threads);

    const std::size_t T = static_cast<std::size_t>(mapping0.n_elem);
    Rcpp::IntegerVector out(static_cast<int>(T));
    for (std::size_t i = 0; i < T; ++i) {
        out[static_cast<int>(i)] = static_cast<int>(mapping0(static_cast<arma::uword>(i)) + 1u);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Multi-k mapping: returns T x (#k) IntegerMatrix, colnames = as.character(k)
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::IntegerMatrix comp_outcome_clusterings_cpp(
    SEXP panel_holder_xptr,
    int grid_size,
    int min_k,
    int max_k,
    Rcpp::Nullable<Rcpp::IntegerVector> n_inits_in = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> seed_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> num_threads_in = R_NilValue)
{
    const apm::InMemoryUnbalancedPanel& panel = apm::r_utils::panel_ref_from_panel_holder(panel_holder_xptr);

    std::optional<std::size_t> n_inits = parse_optional_size(n_inits_in);
    std::optional<uint64_t> seed = parse_optional_u64(seed_in);
    std::optional<std::size_t> num_threads = parse_optional_size(num_threads_in);

    std::vector<arma::uvec> mappings = apm::comp_outcome_clusterings(
        panel,
        static_cast<std::size_t>(grid_size),
        static_cast<std::size_t>(min_k),
        static_cast<std::size_t>(max_k),
        n_inits,
        seed,
        num_threads);

    const std::size_t T = panel.T();
    const std::size_t K = static_cast<std::size_t>(mappings.size());
    Rcpp::IntegerMatrix out(static_cast<int>(T), static_cast<int>(K));

    for (std::size_t j = 0; j < K; ++j) {
        const arma::uvec& m = mappings[j];
        if (m.n_elem != static_cast<arma::uword>(T)) {
            Rcpp::stop("Internal error: mapping length does not match T");
        }
        for (std::size_t i = 0; i < T; ++i) {
            out(static_cast<int>(i), static_cast<int>(j)) = static_cast<int>(m(static_cast<arma::uword>(i)) + 1u);
        }
    }

    Rcpp::CharacterVector cn(static_cast<int>(K));
    for (std::size_t j = 0; j < K; ++j) {
        cn[static_cast<int>(j)] = std::to_string(static_cast<std::size_t>(min_k) + j);
    }
    out.attr("dimnames") = Rcpp::List::create(R_NilValue, cn);
    return out;
}

// -----------------------------------------------------------------------------
// Combine outcomes mapping into new cohorts; returns 1-based OOI and sizes
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List get_new_cohorts_from_combining_outcomes_cpp(
    SEXP panel_holder_xptr,
    Rcpp::IntegerVector old_to_new_outcome_1b)
{
    const apm::InMemoryUnbalancedPanel& panel = apm::r_utils::panel_ref_from_panel_holder(panel_holder_xptr);

    apm::ObservedOutcomeIndices ooi0b = apm::r_utils::observed_outcome_indices_from_panel_holder(panel_holder_xptr);
    const std::size_t T = panel.T();

    if (old_to_new_outcome_1b.size() != static_cast<int>(T)) {
        Rcpp::stop("old_to_new_outcome must have length equal to number of outcomes T");
    }

    arma::uvec old_to_new_0b(static_cast<arma::uword>(T));
    for (std::size_t t = 0; t < T; ++t) {
        int v = old_to_new_outcome_1b[static_cast<int>(t)];
        if (Rcpp::IntegerVector::is_na(v)) Rcpp::stop("old_to_new_outcome contains NA");
        if (v < 1) Rcpp::stop("old_to_new_outcome must be >= 1");
        old_to_new_0b(static_cast<arma::uword>(t)) = static_cast<arma::uword>(v - 1);
    }

    arma::uvec cohort_sizes = panel.get_cohort_sizes();

    auto res = apm::get_new_cohorts_from_combining_outcomes(ooi0b, cohort_sizes, old_to_new_0b);

    Rcpp::List ooi1b = apm::r_utils::to_r_observed_outcome_indices(res.first);
    Rcpp::IntegerVector sizes(static_cast<int>(res.second.n_elem));
    for (arma::uword i = 0; i < res.second.n_elem; ++i) sizes[static_cast<int>(i)] = static_cast<int>(res.second(i));

    return Rcpp::List::create(
        Rcpp::Named("observed_outcome_indices") = ooi1b,
        Rcpp::Named("cohort_sizes") = sizes
    );
}