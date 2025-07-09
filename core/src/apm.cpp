#include "apm.h"
#include "linear_algebra_utils.h"
#include <stdexcept>
#include <vector>

namespace { // Anonymous namespace for helper functions

struct ProblemDimensions {
    arma::uword r; // Number of factors
    arma::uword T; // Total number of time periods
};

ProblemDimensions get_problem_dimensions(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const std::vector<arma::uvec>& observed_outcome_indices) {
    
    if (cohort_factor_matrices.size() != observed_outcome_indices.size() || cohort_factor_matrices.empty()) {
        throw std::invalid_argument("There must be at least one cohort's factor matrix and observed outcome indices, and the number of cohort factor matrices must match the number of observed outcome indices.");
    }

    const arma::uword r = cohort_factor_matrices[0].n_cols;
    arma::uword max_idx = 0;
    for (size_t c = 0; c < cohort_factor_matrices.size(); ++c) {
        if (cohort_factor_matrices[c].n_cols != r) {
            throw std::invalid_argument("All factor matrices must have the same number of columns.");
        }
        if (cohort_factor_matrices[c].n_rows != observed_outcome_indices[c].n_elem) {
            throw std::invalid_argument("The number of rows in a cohort-specific factor matrix must match the number of observed outcomes for that cohort.");
        }
        if (!observed_outcome_indices[c].empty()) {
            max_idx = std::max(max_idx, observed_outcome_indices[c].max());
        }
    }

    const arma::uword T = max_idx + 1;
    return {r, T};
}

arma::mat compute_aggregated_projection_matrix(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const ProblemDimensions& dims) {

    arma::mat agg_proj_mat(dims.T, dims.T, arma::fill::zeros);
    for (size_t c = 0; c < cohort_factor_matrices.size(); ++c) {
        const arma::mat& G_c = cohort_factor_matrices[c];
        const arma::uvec& T_c = observed_outcome_indices[c];

        if (T_c.empty()) {
            continue;
        }
        
        arma::mat padded_G_c(dims.T, dims.r, arma::fill::zeros);
        padded_G_c.rows(T_c) = G_c;
        
        arma::mat P_c = apm::internal::projection_matrix(padded_G_c);
        
        arma::mat P_c_perp = -P_c;
        for (const arma::uword& idx : T_c) {
            P_c_perp(idx, idx) += 1.0;
        }
        
        agg_proj_mat += P_c_perp / cohort_factor_matrices.size();
    }

    return agg_proj_mat;
}

} // anonymous namespace

namespace apm {

std::string get_version() {
    return "0.1.0";
}

arma::mat align_factors_using_apm(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const std::vector<arma::uvec>& observed_outcome_indices) {

    const auto dims = get_problem_dimensions(cohort_factor_matrices, observed_outcome_indices);

    arma::mat agg_proj_mat = compute_aggregated_projection_matrix(
        cohort_factor_matrices, observed_outcome_indices, dims);

    return arma::null(agg_proj_mat);
}

} // namespace apm