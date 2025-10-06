#include <RcppArmadillo.h>
#include "../../core/src/bootstrap.h"
#include "r_utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

// Factory returning an external pointer to a shared_ptr<WeightedBootstrap>
// [[Rcpp::export]]
SEXP get_weighted_bootstrap_ptr_cpp(std::size_t N, std::size_t B, const std::string& type, Rcpp::Nullable<Rcpp::NumericVector> seed = R_NilValue) {
    std::uint64_t s = 0;
    if (seed.isNotNull()) {
        Rcpp::NumericVector v(seed);
        if (v.size() > 0 && !Rcpp::NumericVector::is_na(v[0])) {
            s = static_cast<std::uint64_t>(v[0]);
        }
    }

    std::string t = type;
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);

    std::shared_ptr<apm::WeightedBootstrap> ptr;
    if (t == "multinomial") {
        ptr = std::make_shared<apm::MultinomialBootstrap>(N, B, s);
    } else if (t == "bayesian") {
        ptr = std::make_shared<apm::BayesianBootstrap>(N, B, s);
    } else {
        Rcpp::stop("type must be either 'multinomial' or 'bayesian'");
    }

    // XPtr holds shared_ptr; R GC will delete the shared_ptr, managing last-owner deletion
    return Rcpp::XPtr<std::shared_ptr<apm::WeightedBootstrap>>(new std::shared_ptr<apm::WeightedBootstrap>(std::move(ptr)), true);
}

// Accessor wrappers operating on the pointer
// [[Rcpp::export]]
std::size_t wb_n_obs_cpp(SEXP xp) {
    Rcpp::XPtr<std::shared_ptr<apm::WeightedBootstrap>> p(xp);
    return (*p)->n_obs();
}

// [[Rcpp::export]]
std::size_t wb_n_bootstraps_cpp(SEXP xp) {
    Rcpp::XPtr<std::shared_ptr<apm::WeightedBootstrap>> p(xp);
    return (*p)->n_bootstraps();
}

// [[Rcpp::export]]
arma::vec wb_draw_cpp(SEXP xp, std::size_t b) {
    Rcpp::XPtr<std::shared_ptr<apm::WeightedBootstrap>> p(xp);
    // R is 1-based; validate and convert to 0-based
    if (b < 1 || b > (*p)->n_bootstraps()) {
        std::string msg = std::string("draw index out of range (1..") + std::to_string((*p)->n_bootstraps()) + ")";
        Rcpp::stop(msg);
    }
    return (*p)->draw(b - 1);
}

// [[Rcpp::export]]
arma::vec wb_obs_cpp(SEXP xp, std::size_t i) {
    Rcpp::XPtr<std::shared_ptr<apm::WeightedBootstrap>> p(xp);
    // R is 1-based; validate and convert to 0-based
    if (i < 1 || i > (*p)->n_obs()) {
        std::string msg = std::string("observation index out of range (1..") + std::to_string((*p)->n_obs()) + ")";
        Rcpp::stop(msg);
    }
    return (*p)->obs(i - 1);
}

// [[Rcpp::export]]
arma::mat wb_obs_rows_cpp(SEXP xp, const arma::uvec& idx) {
    Rcpp::XPtr<std::shared_ptr<apm::WeightedBootstrap>> p(xp);
    // R is 1-based; validate and convert to 0-based
    if (idx.n_elem > 0) {
        if (idx.min() < 1 || idx.max() > (*p)->n_obs()) {
            std::string msg = std::string("observation indices out of range (1..") + std::to_string((*p)->n_obs()) + ")";
            Rcpp::stop(msg);
        }
    }
    arma::uvec zero_based = idx;
    if (zero_based.n_elem > 0) {
        zero_based -= 1;
    }
    return (*p)->obs(zero_based);
}

// [[Rcpp::export]]
arma::mat wb_weights_cpp(SEXP xp) {
    Rcpp::XPtr<std::shared_ptr<apm::WeightedBootstrap>> p(xp);
    return (*p)->weights();
}

//------------------------------------------------------------------------------
// Bootstrap-based simultaneous inference bindings
//------------------------------------------------------------------------------

// [[Rcpp::export]]
SEXP get_bootstrap_inference_cpp(Rcpp::NumericVector point_ests,
                                 Rcpp::NumericMatrix bootstrap_replicates,
                                 std::size_t N,
                                 double sig_level = 0.05) {
    arma::vec theta(point_ests.begin(), point_ests.size(), /*copy_aux_mem=*/false);
    arma::mat boots(bootstrap_replicates.begin(),
                    bootstrap_replicates.nrow(),
                    bootstrap_replicates.ncol(),
                    /*copy_aux_mem=*/false);
    apm::SimultaneousInferenceResults res =
        apm::get_bootstrap_inference(theta, boots, N, sig_level);
    return apm::r_utils::make_xptr(std::move(res));
}

// Accessors for SimultaneousInferenceResults

// [[Rcpp::export]]
Rcpp::NumericVector sir_point_cpp(SEXP xp) {
    Rcpp::XPtr<apm::SimultaneousInferenceResults> p(xp);
    const arma::vec& v = p->point_ests;
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector sir_pointwise_t_cpp(SEXP xp) {
    Rcpp::XPtr<apm::SimultaneousInferenceResults> p(xp);
    const arma::vec& v = p->pointwise_t_stats;
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector sir_pointwise_p_cpp(SEXP xp) {
    Rcpp::XPtr<apm::SimultaneousInferenceResults> p(xp);
    const arma::vec& v = p->pointwise_p_vals;
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
double sir_sig_level_cpp(SEXP xp) {
    Rcpp::XPtr<apm::SimultaneousInferenceResults> p(xp);
    return p->sig_level;
}

// [[Rcpp::export]]
Rcpp::NumericVector sir_ci_lb_cpp(SEXP xp) {
    Rcpp::XPtr<apm::SimultaneousInferenceResults> p(xp);
    const arma::vec& v = p->ci_lb;
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector sir_ci_ub_cpp(SEXP xp) {
    Rcpp::XPtr<apm::SimultaneousInferenceResults> p(xp);
    const arma::vec& v = p->ci_ub;
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector sir_cb_lb_cpp(SEXP xp) {
    Rcpp::XPtr<apm::SimultaneousInferenceResults> p(xp);
    const arma::vec& v = p->cb_lb;
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector sir_cb_ub_cpp(SEXP xp) {
    Rcpp::XPtr<apm::SimultaneousInferenceResults> p(xp);
    const arma::vec& v = p->cb_ub;
    Rcpp::NumericVector out(v.n_elem);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}