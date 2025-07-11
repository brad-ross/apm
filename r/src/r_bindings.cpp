#include <RcppArmadillo.h>
#include "apm.h"

// [[Rcpp::depends(RcppArmadillo)]]

namespace { // Anonymous namespace for R binding helpers

// Converts an R list of 1-based integer vectors to a C++ vector of 0-based arma::uvecs.
std::vector<arma::uvec> to_cpp_observed_outcome_indices(const Rcpp::List& r_list) {
    std::vector<arma::uvec> cpp_vec;
    cpp_vec.reserve(r_list.size());
    for (SEXP vec : r_list) {
        cpp_vec.push_back(Rcpp::as<arma::uvec>(vec) - 1); // R is 1-based, C++ is 0-based.
    }
    return cpp_vec;
}

} // anonymous namespace

//' Get the version of the apm library
//' @export
// [[Rcpp::export]]
std::string get_version() {
    return apm::get_version();
}

//' Computes an aligned matrix of factor vectors from cohort-specific ones.
//'
//' This function constructs an Aggregated Projection Matrix (APM) from cohort-specific 
//' factor matrices and then returns an orthonormal basis for the null space of the APM, 
//' which also serves as a basis for the column space of the matrix whose rows are the 
//' factor vectors corresponding to each outcome.
//'
//' @param cohort_factor_matrices A list of matrices, one for each cohort, 
//' where the rows of the matrix corresponding to a given cohort contain the factor vectors for the observed outcomes for that cohort.
//' Each matrix must have the same number of columns, equal to the rank of the factor model.
//' @param observed_outcome_indices A list of integer vectors of the same length
//'   as cohort_factor_matrices. Each vector contains the 1-based indices
//'   indicating which outcomes were observed for the corresponding cohort. The number of indices
//'   must match the number of rows in the cohort's factor matrix.
//' @param cohort_weights An optional numeric vector of weights for each cohort.
//'   If not provided, cohorts are weighted equally. The weights are scaled to 
//'   sum to one.
//' @return A matrix whose columns form an orthonormal basis for the null space
//'         of the aggregated projection matrix.
//' @export
// [[Rcpp::export]]
arma::mat align_factors_using_apm(
    Rcpp::List cohort_factor_matrices,
    Rcpp::List observed_outcome_indices,
    Rcpp::Nullable<Rcpp::NumericVector> cohort_weights = R_NilValue) {

    std::vector<arma::mat> cpp_factor_matrices;
    for (SEXP mat : cohort_factor_matrices) {
        cpp_factor_matrices.push_back(Rcpp::as<arma::mat>(mat));
    }

    std::vector<arma::uvec> cpp_observed_outcome_indices = to_cpp_observed_outcome_indices(observed_outcome_indices);

    if (cohort_weights.isNotNull()) {
        arma::vec cpp_cohort_weights = Rcpp::as<arma::vec>(cohort_weights);
        return apm::align_factors_using_apm(cpp_factor_matrices, cpp_observed_outcome_indices, cpp_cohort_weights);
    } 

    return apm::align_factors_using_apm(cpp_factor_matrices, cpp_observed_outcome_indices);
} 

//' Aggregates cohort-specific covariate coefficient estimates.
//'
//' This function takes a vector of cohort-specific covariate coefficient estimates
//' and returns their average.
//'
//' @param a_c_vec A list of numeric vectors, where each vector contains cohort-specific
//'   covariate coefficient estimates.
//' @return A numeric vector containing the aggregated covariate coefficient estimates.
//' @export
// [[Rcpp::export]]
arma::vec aggregate_cohort_specific_covariate_coefs(Rcpp::List a_c_vec) {
    std::vector<arma::vec> cpp_a_c_vec;
    cpp_a_c_vec.reserve(a_c_vec.size());
    for (SEXP vec : a_c_vec) {
        cpp_a_c_vec.push_back(Rcpp::as<arma::vec>(vec));
    }
    return apm::aggregate_cohort_specific_covariate_coefs(cpp_a_c_vec);
}

//' Aggregates cohort-specific outcome fixed effect estimates.
//'
//' This function computes the average of outcome fixed effect estimates across cohorts
//' for each outcome.
//'
//' @param g_0_c_vec A list of numeric vectors, where each vector contains cohort-specific estimates of outcome
//'   fixed effects for the observed outcomes in those cohorts.
//' @param observed_outcome_indices A list of integer vectors, where each
//'   vector contains the 1-based indices of observed outcomes for a cohort.
//' @return A numeric vector containing the aggregated outcome fixed effect
//'   estimates for all outcomes.
//' @export
// [[Rcpp::export]]
arma::vec aggregate_cohort_specific_outcome_fes(
    Rcpp::List g_0_c_vec,
    Rcpp::List observed_outcome_indices) {
    
    std::vector<arma::vec> cpp_g_0_c_vec;
    cpp_g_0_c_vec.reserve(g_0_c_vec.size());
    for (SEXP vec : g_0_c_vec) {
        cpp_g_0_c_vec.push_back(Rcpp::as<arma::vec>(vec));
    }

    std::vector<arma::uvec> cpp_observed_outcome_indices = to_cpp_observed_outcome_indices(observed_outcome_indices);

    return apm::aggregate_cohort_specific_outcome_fes(cpp_g_0_c_vec, cpp_observed_outcome_indices);
} 

//' Estimates outcomes for a representative unit.
//'
//' This function supports various combinations of factors, fixed effects, and covariates.
//'
//' @param G A T x r matrix of estimated factors.
//' @param T_c A vector of 1-based indices for the observed outcomes for the cohort.
//' @param m_c A vector containing the observed outcomes for the representative unit.
//' @param g_0 An optional T-dimensional vector of estimated outcome fixed effects.
//' @param a An optional q-dimensional vector of estimated covariate coefficients.
//' @param X_c An optional T x q matrix containing the values of q covariates corresponding to each outcome for the representative unit.
//' @return A T-dimensional vector containing the estimated outcomes for the representative unit.
//' @export
// [[Rcpp::export]]
arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::uvec& T_c,
    const arma::vec& m_c,
    Rcpp::Nullable<Rcpp::NumericVector> g_0 = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> a = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> X_c = R_NilValue) {
    
    arma::uvec T_c_zero_based = T_c - 1;

    bool has_g0 = g_0.isNotNull();
    bool has_a = a.isNotNull();

    if (has_a && !X_c.isNotNull()) {
        Rcpp::stop("If 'a' is provided, 'X_c' must also be provided.");
    }
    if (!has_a && X_c.isNotNull()) {
        Rcpp::warning("'X_c' is provided but 'a' is not; covariates will be ignored.");
    }

    if (has_g0 && has_a) {
        arma::vec g_0_cpp = Rcpp::as<arma::vec>(g_0);
        arma::vec a_cpp = Rcpp::as<arma::vec>(a);
        arma::mat X_c_cpp = Rcpp::as<arma::mat>(X_c);
        return apm::impute_outcomes(G, g_0_cpp, a_cpp, T_c_zero_based, m_c, X_c_cpp);
    } else if (has_g0) {
        arma::vec g_0_cpp = Rcpp::as<arma::vec>(g_0);
        return apm::impute_outcomes(G, g_0_cpp, T_c_zero_based, m_c);
    } else if (has_a) {
        arma::vec a_cpp = Rcpp::as<arma::vec>(a);
        arma::mat X_c_cpp = Rcpp::as<arma::mat>(X_c);
        return apm::impute_outcomes(G, a_cpp, T_c_zero_based, m_c, X_c_cpp);
    } else {
        return apm::impute_outcomes(G, T_c_zero_based, m_c);
    }
}

//' Estimates mean outcomes for each cohort.
//'
//' This function supports various combinations of factors, fixed effects, and covariates.
//'
//' @param G A T x r matrix whose rows are estimated factor vectors.
//' @param observed_outcome_indices A list where each element is a vector of 
//'                                 1-based indices for the observed outcomes for a cohort.
//' @param m_c_vec A list of numeric vectors, where each vector m_c contains the 
//'                observed outcomes for a cohort.
//' @param g_0 An optional T-dimensional vector of estimated outcome fixed effects.
//' @param a An optional q-dimensional vector of estimated covariate coefficients.
//' @param X_c_vec An optional list of T x q matrices, where each matrix X_c contains the average values of q covariates for each outcome within a cohort.
//' @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
//' @export
// [[Rcpp::export]]
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    Rcpp::List observed_outcome_indices,
    Rcpp::List m_c_vec,
    Rcpp::Nullable<Rcpp::NumericVector> g_0 = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> a = R_NilValue,
    Rcpp::Nullable<Rcpp::List> X_c_vec = R_NilValue) {

    std::vector<arma::uvec> cpp_observed_outcome_indices = to_cpp_observed_outcome_indices(observed_outcome_indices);

    std::vector<arma::vec> cpp_m_c_vec;
    for (SEXP vec : m_c_vec) {
        cpp_m_c_vec.push_back(Rcpp::as<arma::vec>(vec));
    }

    bool has_g0 = g_0.isNotNull();
    bool has_a = a.isNotNull();

    if (has_a && !X_c_vec.isNotNull()) {
        Rcpp::stop("If 'a' is provided, 'X_c_vec' must also be provided.");
    }
     if (!has_a && X_c_vec.isNotNull()) {
        Rcpp::stop("If 'X_c_vec' is provided, 'a' must also be provided.");
    }

    if (has_g0 && has_a) {
        arma::vec g_0_cpp = Rcpp::as<arma::vec>(g_0);
        arma::vec a_cpp = Rcpp::as<arma::vec>(a);
        std::vector<arma::mat> cpp_X_c_vec;
        Rcpp::List r_X_c_vec(X_c_vec);
        for (SEXP mat : r_X_c_vec) {
            cpp_X_c_vec.push_back(Rcpp::as<arma::mat>(mat));
        }
        return apm::estimate_outcome_means_across_cohorts(G, g_0_cpp, a_cpp, cpp_observed_outcome_indices, cpp_m_c_vec, cpp_X_c_vec);
    } else if (has_g0) {
        arma::vec g_0_cpp = Rcpp::as<arma::vec>(g_0);
        return apm::estimate_outcome_means_across_cohorts(G, g_0_cpp, cpp_observed_outcome_indices, cpp_m_c_vec);
    } else if (has_a) {
        arma::vec a_cpp = Rcpp::as<arma::vec>(a);
        std::vector<arma::mat> cpp_X_c_vec;
        Rcpp::List r_X_c_vec(X_c_vec);
        for (SEXP mat : r_X_c_vec) {
            cpp_X_c_vec.push_back(Rcpp::as<arma::mat>(mat));
        }
        return apm::estimate_outcome_means_across_cohorts(G, a_cpp, cpp_observed_outcome_indices, cpp_m_c_vec, cpp_X_c_vec);
    } else {
        return apm::estimate_outcome_means_across_cohorts(G, cpp_observed_outcome_indices, cpp_m_c_vec);
    }
} 

//' Implements the Observed Outcome Overlap (O^3) algorithm to assess factor identification.
//'
//' This function iteratively groups cohorts based on the overlap of their
//' observed outcomes. Two super cohorts are merged if the number of their
//' shared outcomes meets or exceeds the model rank, `r`. The process
//' continues until no more cohorts can be merged.
//'
//' @param observed_outcome_indices A list of integer vectors, where each
//'   vector contains the 1-based indices corresponding to the observed outcomes for a cohort.
//' @param r The model rank, used as the minimum overlap threshold for merging cohorts.
//' @return A list of lists of sets, representing the super cohorts at each
//'   iteration of the algorithm. Each list of super cohorts is a list of integer vectors
//'   corresponding to the original cohorts together forming a super cohort.
//' @export
// [[Rcpp::export]]
Rcpp::List o3_algorithm(
    Rcpp::List observed_outcome_indices,
    unsigned int r) {

    std::vector<arma::uvec> cpp_observed_outcome_indices = to_cpp_observed_outcome_indices(observed_outcome_indices);

    auto iterations = apm::o3_algorithm(cpp_observed_outcome_indices, r);

    // Convert back to R list structure with 1-based indexing
    Rcpp::List all_iterations_list;
    for (const auto& iteration : iterations) {
        Rcpp::List current_iteration_list;
        for (const auto& super_cohort_set : iteration) {
            std::vector<arma::uword> sc_vec(super_cohort_set.begin(), super_cohort_set.end());
            // Rcpp converts arma::uvec to a numeric vector, add 1 for 1-based index
            current_iteration_list.push_back(arma::uvec(sc_vec) + 1);
        }
        all_iterations_list.push_back(current_iteration_list);
    }

    return all_iterations_list;
}

//' Checks if the factors are identified across all cohorts.
//'
//' This function uses the O^3 algorithm to determine if there is sufficient
//' overlap in observed outcomes across all cohorts to uniquely identify all factor 
//' vectors expressed with respect to a common basis. Identification is achieved if 
//' the algorithm terminates with a single super cohort containing all of the 
//' original cohorts.
//'
//' @param observed_outcome_indices A list of integer vectors, where each
//'   vector contains the 1-based indices for the observed outcomes for a cohort.
//' @param r The model rank.
//' @return `TRUE` if the factors are identified, `FALSE` otherwise.
//' @export
// [[Rcpp::export]]
bool aligned_factors_identified(
    Rcpp::List observed_outcome_indices,
    unsigned int r) {
    
    std::vector<arma::uvec> cpp_observed_outcome_indices = to_cpp_observed_outcome_indices(observed_outcome_indices);

    return apm::aligned_factors_identified(cpp_observed_outcome_indices, r);
} 