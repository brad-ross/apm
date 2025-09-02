#include "apm_core.h"
#include "linear_algebra_utils.h"
#include <stdexcept>
#include <vector>
#include <numeric>
#include <set>
#include <algorithm>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

namespace { // Anonymous namespace for helper functions

struct ProblemDimensions {
    arma::uword r; // Number of factors
    arma::uword T; // Total number of time periods
    arma::uword C; // Number of cohorts
};

arma::uword get_num_outcomes(
    const std::vector<arma::uvec>& observed_outcome_indices) {
    
    arma::uword max_idx = 0;
    bool has_observations = false;
    for (const auto& T_c : observed_outcome_indices) {
        if (!T_c.empty()) {
            has_observations = true;
            max_idx = std::max(max_idx, T_c.max());
        }
    }

    return has_observations ? max_idx + 1 : 0;
}

ProblemDimensions get_problem_dimensions(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const apm::ObservedOutcomeIndices& observed_outcome_indices) {
    
    if (cohort_factor_matrices.size() != observed_outcome_indices.size() || cohort_factor_matrices.empty()) {
        throw std::invalid_argument("There must be at least one cohort's factor matrix and observed outcome indices, and the number of cohort factor matrices must match the number of observed outcome indices.");
    }
    const arma::uword C = cohort_factor_matrices.size();

    const arma::uword r = cohort_factor_matrices[0].n_cols;

    for (arma::uword c = 0; c < C; ++c) {
        if (cohort_factor_matrices[c].n_cols != r) {
            throw std::invalid_argument("All factor matrices must have the same number of columns.");
        }
        if (cohort_factor_matrices[c].n_rows != observed_outcome_indices[c].n_elem) {
            throw std::invalid_argument("The number of rows in a cohort-specific factor matrix must match the number of observed outcomes for that cohort.");
        }
    }

    const arma::uword T = get_num_outcomes(observed_outcome_indices);

    return {r, T, C};
}

arma::vec process_weights(
    const arma::vec& cohort_weights,
    const arma::uword C) {

    if (cohort_weights.n_elem != C) {
        throw std::invalid_argument("The number of cohort weights must match the number of cohorts.");
    }

    if (arma::any(cohort_weights < 0)) {
        throw std::invalid_argument("Cohort weights cannot be negative.");
    }
    
    const double sum_weights = arma::sum(cohort_weights);

    if (sum_weights == 0.0) {
        // All weights are zero, which is not valid for creating a weighted average.
        throw std::invalid_argument("At least one cohort weight must be positive.");
    }
    
    return cohort_weights / sum_weights;
}

arma::mat compute_aggregated_projection_matrix(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const apm::ObservedOutcomeIndices& observed_outcome_indices,
    const ProblemDimensions& dims,
    const arma::vec& cohort_weights) {
    arma::mat agg_proj_mat(dims.T, dims.T, arma::fill::zeros);
    for (arma::uword c = 0; c < dims.C; ++c) {
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
        
        agg_proj_mat += P_c_perp * cohort_weights(c);
    }

    return agg_proj_mat;
}

arma::mat compute_bridge_functions(
    const arma::mat& G,
    const arma::uvec& T_c) {
    
    const arma::mat G_c = G.rows(T_c);
    return apm::internal::multi_min_norm_solve(G_c.t(), G.t()).t();
}

// Consolidated validator and resolver for imputation arguments
struct ResolvedImputationArgs {
    const arma::mat& G;
    const arma::vec* g0;       // nullptr if absent
    const arma::vec* a;        // nullptr if absent
    const arma::mat* X;        // nullptr if absent
    const arma::uvec& T_c;
    const arma::vec& m_c;
};

ResolvedImputationArgs resolve_imputation_args(
    const apm::FactorModelParameters& params,
    const arma::uvec& T_c,
    const apm::OutcomeMeanSufficientStatistics& stats) {

    const arma::mat& G = params.G;

    const bool has_g0 = params.has_fixed_effects();
    const bool has_a  = params.has_covariate_coefs();
    const bool has_X  = stats.has_covar_means();
    if (has_a != has_X) {
        throw std::invalid_argument("impute_outcomes: covariate information mismatch between parameters and sufficient statistics.");
    }

    const arma::vec* g0_ptr = nullptr;
    if (has_g0) {
        const arma::vec& g0 = *(params.g_0);
        g0_ptr = &g0;
    }

    const arma::vec* a_ptr = nullptr;
    const arma::mat* X_ptr = nullptr;
    if (has_a) {
        const arma::vec& a = *(params.a);
        const arma::mat& X = *(stats.covar_means);
        a_ptr = &a;
        X_ptr = &X;
    }

    return ResolvedImputationArgs{G, g0_ptr, a_ptr, X_ptr, T_c, stats.observed_outcome_means};
}

// Consolidated validation helpers for outcome mean estimation
void validate_g0_length(const arma::vec& g_0, arma::uword T) {
    if (g_0.n_elem != T) {
        throw std::invalid_argument("The number of elements in g_0 must match the number of rows in G.");
    }
}

void validate_m_c(const arma::vec& m_c, const arma::uvec& T_c) {
    if (m_c.n_elem != T_c.n_elem) {
        throw std::invalid_argument("The number of observed outcomes for each cohort must match the number of outcomes in the m_c vector.");
    }
}

void validate_X_c(const arma::mat& X_c, arma::uword T, arma::uword q) {
    if (X_c.n_rows != T) {
        throw std::invalid_argument("The number of rows in X_c must match the number of rows in G.");
    }
    if (X_c.n_cols != q) {
        throw std::invalid_argument("The number of columns in X_c must match the number of elements in a.");
    }
}

// Define the graph type using Boost Graph Library
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

// Helper to compute the union of observed outcomes for a super cohort
std::set<arma::uword> get_observed_outcomes_for_super_cohort(
    const std::set<arma::uword>& super_cohort,
    const apm::ObservedOutcomeIndices& observed_outcome_indices) {
    
    std::set<arma::uword> all_indices;
    for (const auto& cohort_idx : super_cohort) {
        const arma::uvec& outcomes = observed_outcome_indices.at(cohort_idx);
        all_indices.insert(outcomes.begin(), outcomes.end());
    }

    return all_indices;
}

Graph construct_o3_graph(
    const std::vector<std::set<arma::uword>>& super_cohorts,
    const apm::ObservedOutcomeIndices& observed_outcome_indices,
    unsigned int r) {
    
    const arma::uword num_super_cohorts = super_cohorts.size();
    Graph o3_graph(num_super_cohorts);

    std::vector<std::set<arma::uword>> super_cohort_outcomes(num_super_cohorts);
    for(arma::uword m = 0; m < num_super_cohorts; ++m) {
        super_cohort_outcomes[m] = get_observed_outcomes_for_super_cohort(super_cohorts[m], observed_outcome_indices);
    }

    for (arma::uword m1 = 0; m1 < num_super_cohorts; ++m1) {
        for (arma::uword m2 = m1 + 1; m2 < num_super_cohorts; ++m2) {
            std::set<arma::uword> intersection;
            std::set_intersection(
                super_cohort_outcomes[m1].begin(), super_cohort_outcomes[m1].end(),
                super_cohort_outcomes[m2].begin(), super_cohort_outcomes[m2].end(),
                std::inserter(intersection, intersection.begin())
            );

            if (intersection.size() >= r) {
                boost::add_edge(m1, m2, o3_graph);
            }
        }
    }

    return o3_graph;
}

// Collect cohort-specific parameter vectors (G, optional g_0, optional a)
struct CohortSpecificCollections {
    std::vector<arma::mat> G_list;
    std::vector<arma::vec> g0_list;
    std::vector<arma::vec> a_list;
};

CohortSpecificCollections extract_cohort_specific_collections(
    const std::vector<apm::FactorModelParameters>& cohort_specific_factor_model_params) {
    CohortSpecificCollections out;
    const arma::uword C = cohort_specific_factor_model_params.size();
    if (C == 0) {
        return out;
    }

    out.G_list.reserve(C);
    out.g0_list.reserve(C);
    out.a_list.reserve(C);

    const bool ref_has_g0 = cohort_specific_factor_model_params[0].has_fixed_effects();
    const bool ref_has_a  = cohort_specific_factor_model_params[0].has_covariate_coefs();

    for (arma::uword c = 0; c < C; ++c) {
        const auto& params = cohort_specific_factor_model_params[c];
        if (c > 0) {
            if (params.has_fixed_effects() != ref_has_g0) {
                throw std::invalid_argument("All or none of the FactorModelParameters must include g_0.");
            }
            if (params.has_covariate_coefs() != ref_has_a) {
                throw std::invalid_argument("All or none of the FactorModelParameters must include a.");
            }
        }

        out.G_list.push_back(params.G);
        if (ref_has_g0) {
            out.g0_list.push_back(*(params.g_0));
        }
        if (ref_has_a) {
            out.a_list.push_back(*(params.a));
        }
    }

    return out;
}

} // anonymous namespace

namespace apm {

std::string get_version() {
    return "0.1.0";
}

//==============================================================================
// Aggregators of Cohort-Specific Parameter Estimates
//==============================================================================

arma::mat align_factors_using_apm(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::vec& cohort_weights) {
    const auto dims = get_problem_dimensions(cohort_factor_matrices, observed_outcome_indices);

    arma::vec weights = cohort_weights;
    if (weights.n_elem == 0) {
        weights = arma::vec(dims.C, arma::fill::ones);
    }
    const auto effective_weights = process_weights(weights, dims.C);

    arma::mat agg_proj_mat = compute_aggregated_projection_matrix(
        cohort_factor_matrices, 
        observed_outcome_indices, 
        dims, 
        effective_weights
    );

    return arma::null(agg_proj_mat);
}

arma::vec aggregate_cohort_specific_outcome_fes(
    const std::vector<arma::vec>& g_0_c_vec,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::vec& cohort_weights) {

    if (g_0_c_vec.size() != observed_outcome_indices.size()) {
        throw std::invalid_argument("Number of fixed effect vectors must match number of observed outcome index vectors.");
    }
    
    const arma::uword T = get_num_outcomes(observed_outcome_indices);

    if (T == 0) {
        return arma::vec();
    }

    arma::vec g_0_sum(T, arma::fill::zeros);
    arma::vec g_0_weight_sum(T, arma::fill::zeros);

    const arma::uword C = g_0_c_vec.size();
    arma::vec weights = cohort_weights;
    if (weights.n_elem == 0) {
        weights = arma::vec(C, arma::fill::ones);
    }
    const arma::vec effective_weights = process_weights(weights, C);

    for (size_t c = 0; c < g_0_c_vec.size(); ++c) {
        const arma::vec& g_0_c = g_0_c_vec[c];
        const arma::uvec& T_c = observed_outcome_indices[c];
        
        if (g_0_c.n_elem != T_c.n_elem) {
            throw std::invalid_argument("Length of fixed effect vector does not match number of observed outcomes for a cohort.");
        }

        for (arma::uword i = 0; i < T_c.n_elem; ++i) {
            const arma::uword t = T_c(i);
            const double w = effective_weights(c);
            g_0_sum(t) += w * g_0_c(i);
            g_0_weight_sum(t) += w;
        }
    }

    arma::vec g_0(T, arma::fill::zeros);
    for (arma::uword t = 0; t < T; ++t) {
        if (g_0_weight_sum(t) > 0) {
            g_0(t) = g_0_sum(t) / g_0_weight_sum(t);
        }
    }

    return g_0;
}

arma::vec aggregate_cohort_specific_covariate_coefs(
    const std::vector<arma::vec>& a_c_vec,
    const arma::vec& cohort_weights) {
    if (a_c_vec.empty()) {
        return arma::vec();
    }

    const arma::uword C = a_c_vec.size();
    const arma::uword q = a_c_vec[0].n_elem;
    arma::vec mean_a(q, arma::fill::zeros);

    arma::vec weights = cohort_weights;
    if (weights.n_elem == 0) {
        weights = arma::vec(C, arma::fill::ones);
    }
    const arma::vec effective_weights = process_weights(weights, C);

    for (arma::uword c = 0; c < C; ++c) {
        const auto& a_c = a_c_vec[c];
        if (a_c.n_elem != q) {
            throw std::invalid_argument("All covariate coefficient vectors must have the same length.");
        }
        mean_a += a_c * effective_weights(c);
    }

    return mean_a;
}

FactorModelParameters aggregate_cohort_specific_factor_model_params(
    const std::vector<FactorModelParameters>& cohort_specific_factor_model_params,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::vec& cohort_weights) {
    const arma::uword C = cohort_specific_factor_model_params.size();
    if (C == 0) {
        return FactorModelParameters();
    }
    if (observed_outcome_indices.size() != C) {
        throw std::invalid_argument("Number of cohorts in parameters must match observed_outcome_indices.");
    }

    const auto collections = extract_cohort_specific_collections(cohort_specific_factor_model_params);

    arma::mat aggregated_G = align_factors_using_apm(collections.G_list, observed_outcome_indices, cohort_weights);

    std::optional<arma::vec> aggregated_g_0;
    if (!collections.g0_list.empty()) {
        aggregated_g_0 = aggregate_cohort_specific_outcome_fes(collections.g0_list, observed_outcome_indices, cohort_weights);
    }

    std::optional<arma::vec> aggregated_a;
    if (!collections.a_list.empty()) {
        aggregated_a = aggregate_cohort_specific_covariate_coefs(collections.a_list, cohort_weights);
    }

    return FactorModelParameters(std::move(aggregated_G), std::move(aggregated_g_0), std::move(aggregated_a));
}

FactorModelEstimates aggregate_cohort_specific_factor_model_params(
    const std::vector<FactorModelEstimates>& cohort_specific_factor_model_param_ests,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::vec& cohort_weights,
    const std::vector<arma::vec>& bootstrap_cohort_weights) {
    
    const std::size_t C = cohort_specific_factor_model_param_ests.size();

    // Aggregate point estimates
    std::vector<FactorModelParameters> point_params;
    point_params.reserve(C);
    for (const auto& est : cohort_specific_factor_model_param_ests) {
        point_params.push_back(est.parameter_estimates);
    }

    FactorModelParameters agg_point = aggregate_cohort_specific_factor_model_params(
        point_params, observed_outcome_indices, cohort_weights);

    // Determine number of bootstrap replicates from first cohort (0 if none)
    std::size_t B = cohort_specific_factor_model_param_ests.empty()
        ? 0
        : cohort_specific_factor_model_param_ests.front().bootstrap_replicates.size();
    
    if (bootstrap_cohort_weights.size() != B) {
        throw std::invalid_argument("bootstrap_cohort_weights must have length equal to number of bootstrap replicates.");
    }

    std::vector<FactorModelParameters> agg_bootstrap;
    agg_bootstrap.reserve(B);

    for (std::size_t b = 0; b < B; ++b) {
        std::vector<FactorModelParameters> params_b;
        params_b.reserve(C);
        for (const auto& est : cohort_specific_factor_model_param_ests) {
            if (b == 0) {
                if (est.bootstrap_replicates.size() != B) {
                    throw std::invalid_argument("All cohorts must have the same number of bootstrap replicates.");
                }
            }
            params_b.push_back(est.bootstrap_replicates[b]);
        }
        
        agg_bootstrap.push_back(
            aggregate_cohort_specific_factor_model_params(
                params_b, observed_outcome_indices, bootstrap_cohort_weights[b]));
    }

    return FactorModelEstimates(std::move(agg_point), std::move(agg_bootstrap));
}

//==============================================================================
// Outcome Imputation
//==============================================================================

arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::vec& g_0,
    const arma::vec& a,
    const arma::uvec& T_c,
    const arma::vec& m_c,
    const arma::mat& X_c) {
    
    const arma::mat B_c = compute_bridge_functions(G, T_c);
    const arma::vec g_0_obs = g_0.rows(T_c);
    const arma::mat X_c_obs = X_c.rows(T_c);
    
    return B_c * (m_c - g_0_obs - X_c_obs * a) + g_0 + X_c * a;
}

arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::vec& g_0,
    const arma::uvec& T_c,
    const arma::vec& m_c) {

    const arma::mat B_c = compute_bridge_functions(G, T_c);
    const arma::vec g_0_obs = g_0.rows(T_c);
    return B_c * (m_c - g_0_obs) + g_0;
}

arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::vec& a,
    const arma::uvec& T_c,
    const arma::vec& m_c,
    const arma::mat& X_c) {
    
    const arma::mat B_c = compute_bridge_functions(G, T_c);
    const arma::mat X_c_obs = X_c.rows(T_c);
    return B_c * (m_c - X_c_obs * a) + X_c * a;
}

arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::uvec& T_c,
    const arma::vec& m_c) {
    
    const arma::mat B_c = compute_bridge_functions(G, T_c);
    return B_c * m_c;
}

arma::vec impute_outcomes(
    const FactorModelParameters& factor_model_parameters,
    const arma::uvec& T_c,
    const OutcomeMeanSufficientStatistics& outcome_mean_suff_stats) {

    const ResolvedImputationArgs args = resolve_imputation_args(factor_model_parameters, T_c, outcome_mean_suff_stats);

    if (args.g0 && args.a) {
        return impute_outcomes(args.G, *args.g0, *args.a, args.T_c, args.m_c, *args.X);
    }
    if (args.g0 && !args.a) {
        return impute_outcomes(args.G, *args.g0, args.T_c, args.m_c);
    }
    if (!args.g0 && args.a) {
        return impute_outcomes(args.G, *args.a, args.T_c, args.m_c, *args.X);
    }
    return impute_outcomes(args.G, args.T_c, args.m_c);
}

//==============================================================================
// Outcome Mean Estimation Across Cohorts
//==============================================================================

arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& g_0,
    const arma::vec& a,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec,
    const std::vector<arma::mat>& X_c_vec) {

    const arma::uword T = G.n_rows;
    const arma::uword C = observed_outcome_indices.size();
    if (m_c_vec.size() != C || X_c_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }
    validate_g0_length(g_0, T);

    arma::mat m(C, T, arma::fill::zeros);
    for (arma::uword c = 0; c < C; ++c) {
        const arma::uvec& T_c = observed_outcome_indices[c];
        validate_m_c(m_c_vec[c], T_c);
        validate_X_c(X_c_vec[c], T, a.n_elem);
        m.row(c) = impute_outcomes(G, g_0, a, T_c, m_c_vec[c], X_c_vec[c]).t();
    }
    return m;
}

arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& g_0,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec) {
    
    const arma::uword T = G.n_rows;
    const arma::uword C = observed_outcome_indices.size();
    if (m_c_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }
    validate_g0_length(g_0, T);

    arma::mat m(C, T, arma::fill::zeros);
    for (arma::uword c = 0; c < C; ++c) {
        const arma::uvec& T_c = observed_outcome_indices[c];
        validate_m_c(m_c_vec[c], T_c);
        m.row(c) = impute_outcomes(G, g_0, T_c, m_c_vec[c]).t();
    }
    return m;
}

arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& a,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec,
    const std::vector<arma::mat>& X_c_vec) {

    const arma::uword T = G.n_rows;
    const arma::uword C = observed_outcome_indices.size();
    if (m_c_vec.size() != C || X_c_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }

    arma::mat m(C, T, arma::fill::zeros);
    for (arma::uword c = 0; c < C; ++c) {
        const arma::uvec& T_c = observed_outcome_indices[c];
        validate_m_c(m_c_vec[c], T_c);
        validate_X_c(X_c_vec[c], T, a.n_elem);
        m.row(c) = impute_outcomes(G, a, T_c, m_c_vec[c], X_c_vec[c]).t();
    }
    return m;
}

arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec) {

    const arma::uword T = G.n_rows;
    const arma::uword C = observed_outcome_indices.size();
    if (m_c_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }

    arma::mat m(C, T, arma::fill::zeros);
    for (arma::uword c = 0; c < C; ++c) {
        const arma::uvec& T_c = observed_outcome_indices[c];
        validate_m_c(m_c_vec[c], T_c);
        m.row(c) = impute_outcomes(G, T_c, m_c_vec[c]).t();
    }
    return m;
}

arma::mat estimate_outcome_means_across_cohorts(
    const FactorModelParameters& factor_model_parameters,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<OutcomeMeanSufficientStatistics>& suff_stats_vec) {

    const arma::mat& G = factor_model_parameters.G;
    const arma::uword C = observed_outcome_indices.size();
    if (suff_stats_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }

    const arma::uword T = G.n_rows;
    if (factor_model_parameters.has_fixed_effects()) {
        validate_g0_length(*(factor_model_parameters.g_0), T);
    }

    const bool has_a = factor_model_parameters.has_covariate_coefs();

    arma::mat m(C, T, arma::fill::zeros);
    for (arma::uword c = 0; c < C; ++c) {
        const arma::uvec& T_c = observed_outcome_indices[c];
        const auto& stats = suff_stats_vec[c];
        validate_m_c(stats.observed_outcome_means, T_c);
        if (has_a != stats.has_covar_means()) {
            throw std::invalid_argument("Covariate presence mismatch between parameters and sufficient statistics.");
        }
        if (has_a) {
            validate_X_c(*(stats.covar_means), T, static_cast<arma::uword>(factor_model_parameters.a->n_elem));
        }
        m.row(c) = impute_outcomes(factor_model_parameters, T_c, stats).t();
    }

    return m;
}

//==============================================================================
// Identification Verification Via the O^3 Algorithm
//==============================================================================

std::vector<std::vector<std::set<arma::uword>>> o3_algorithm(
    const ObservedOutcomeIndices& observed_outcome_indices,
    unsigned int r) {
    
    const arma::uword C = observed_outcome_indices.size();

    // Iteration 0: Initial super cohorts are just the individual cohorts
    std::vector<std::set<arma::uword>> current_super_cohorts(C);
    for (arma::uword c = 0; c < C; ++c) {
        current_super_cohorts[c] = {c};
    }

    std::vector<std::vector<std::set<arma::uword>>> all_iterations_super_cohorts;
    
    // Repeat until convergence
    while (true) {
        const arma::uword num_super_cohorts = current_super_cohorts.size();
        
        // 1. Construct O3 graph
        Graph o3_graph = construct_o3_graph(current_super_cohorts, observed_outcome_indices, r);

        // 2. Find connected components
        std::vector<int> component(num_super_cohorts);
        const unsigned int num_components = boost::connected_components(o3_graph, &component[0]);

        // 3. Check for convergence
        if (num_components == num_super_cohorts) {
            break;
        }

        // 4. Form new super cohorts
        std::vector<std::set<arma::uword>> next_super_cohorts(num_components);
        for (arma::uword m = 0; m < num_super_cohorts; ++m) {
            next_super_cohorts[component[m]].insert(
                current_super_cohorts[m].begin(),
                current_super_cohorts[m].end()
            );
        }

        current_super_cohorts = next_super_cohorts;
        all_iterations_super_cohorts.push_back(current_super_cohorts);
    }

    return all_iterations_super_cohorts;
}

bool aligned_factors_identified(
    const ObservedOutcomeIndices& observed_outcome_indices,
    unsigned int r) {

    const arma::uword C = observed_outcome_indices.size();

    if (C <= 1) {
        return true;
    }

    auto super_cohort_iterations = o3_algorithm(observed_outcome_indices, r);

    // If no iterations were recorded, it means the cohorts did not merge at all.
    // Since C > 1, this means it's not identified.
    if (super_cohort_iterations.empty()) {
        return false;
    }
    
    // Check the final state of super-cohorts (the last element of the iterations vector).
    const auto& final_super_cohorts = super_cohort_iterations.back();

    // Identification requires that all cohorts merge into a single super-cohort.
    // This means the final list of super-cohorts has size 1, and that single
    // super-cohort contains all C original cohorts.
    return final_super_cohorts.size() == 1 && final_super_cohorts[0].size() == C;
}

} // namespace apm