#include "apm.h"
#include "linear_algebra_utils.h"
#include <stdexcept>
#include <vector>
#include <numeric>
#include <set>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

namespace { // Anonymous namespace for helper functions

struct ProblemDimensions {
    arma::uword r; // Number of factors
    arma::uword T; // Total number of time periods
    arma::uword C; // Number of cohorts
};

ProblemDimensions get_problem_dimensions(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const std::vector<arma::uvec>& observed_outcome_indices) {
    
    if (cohort_factor_matrices.size() != observed_outcome_indices.size() || cohort_factor_matrices.empty()) {
        throw std::invalid_argument("There must be at least one cohort's factor matrix and observed outcome indices, and the number of cohort factor matrices must match the number of observed outcome indices.");
    }
    const arma::uword C = cohort_factor_matrices.size();

    const arma::uword r = cohort_factor_matrices[0].n_cols;

    arma::uword max_idx = 0;
    for (arma::uword c = 0; c < C; ++c) {
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
    const std::vector<arma::uvec>& observed_outcome_indices,
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

// Define the graph type using Boost Graph Library
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

// Helper to compute the union of observed outcomes for a super cohort
std::set<arma::uword> get_observed_outcomes_for_super_cohort(
    const std::set<arma::uword>& super_cohort,
    const std::vector<arma::uvec>& observed_outcome_indices) {
    
    std::set<arma::uword> all_indices;
    for (const auto& cohort_idx : super_cohort) {
        const arma::uvec& outcomes = observed_outcome_indices.at(cohort_idx);
        all_indices.insert(outcomes.begin(), outcomes.end());
    }

    return all_indices;
}

Graph construct_o3_graph(
    const std::vector<std::set<arma::uword>>& super_cohorts,
    const std::vector<arma::uvec>& observed_outcome_indices,
    arma::uword r) {
    
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

} // anonymous namespace

namespace apm {

std::string get_version() {
    return "0.1.0";
}

arma::mat align_factors_using_apm(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const std::vector<arma::uvec>& observed_outcome_indices) {
    // Default to equal weights across cohorts.
    arma::vec cohort_weights(cohort_factor_matrices.size(), arma::fill::ones);

    return align_factors_using_apm(cohort_factor_matrices, observed_outcome_indices, cohort_weights);
}

arma::mat align_factors_using_apm(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const arma::vec& cohort_weights) {
    const auto dims = get_problem_dimensions(cohort_factor_matrices, observed_outcome_indices);

    const auto processed_weights = process_weights(cohort_weights, dims.C);

    arma::mat agg_proj_mat = compute_aggregated_projection_matrix(
        cohort_factor_matrices, 
        observed_outcome_indices, 
        dims, 
        processed_weights
    );

    return arma::null(agg_proj_mat);
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

//==============================================================================
// Identification Verification Via the O^3 Algorithm
//==============================================================================

std::vector<std::vector<std::set<arma::uword>>> o3_algorithm(
    const std::vector<arma::uvec>& observed_outcome_indices,
    arma::uword r) {
    
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
        int num_components = boost::connected_components(o3_graph, &component[0]);

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

//==============================================================================
// Outcome Mean Estimation Across Cohorts
//==============================================================================

arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& g_0,
    const arma::vec& a,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec,
    const std::vector<arma::mat>& X_c_vec) {

    const arma::uword T = G.n_rows;
    const arma::uword C = observed_outcome_indices.size();

    if (g_0.n_elem != T) {
        throw std::invalid_argument("The number of elements in g_0 must match the number of rows in G.");
    }
    if (observed_outcome_indices.size() != C || m_c_vec.size() != C || X_c_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }

    arma::mat m(C, T, arma::fill::zeros);
    for (arma::uword c = 0; c < C; ++c) {
        if (observed_outcome_indices[c].size() != m_c_vec[c].n_elem) {
            throw std::invalid_argument("The number of observed outcomes for each cohort must match the number of outcomes in the m_c vector.");
        }
        if (X_c_vec[c].n_rows != T) {
            throw std::invalid_argument("The number of rows in X_c must match the number of rows in G.");
        }
        if (X_c_vec[c].n_cols != a.n_elem) {
            throw std::invalid_argument("The number of columns in X_c must match the number of elements in a.");
        }

        const arma::uvec& T_c = observed_outcome_indices[c];
        m.row(c) = impute_outcomes(G, g_0, a, T_c, m_c_vec[c], X_c_vec[c]).t();
    }
    return m;
}

arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& g_0,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec) {
    
    const arma::uword T = G.n_rows;
    const arma::uword C = observed_outcome_indices.size();

    if (g_0.n_elem != T) {
        throw std::invalid_argument("The number of elements in g_0 must match the number of rows in G.");
    }
    if (observed_outcome_indices.size() != C || m_c_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }

    arma::mat m(C, T, arma::fill::zeros);
    for (arma::uword c = 0; c < C; ++c) {
        if (observed_outcome_indices[c].size() != m_c_vec[c].n_elem) {
            throw std::invalid_argument("The number of observed outcomes for each cohort must match the number of outcomes in the m_c vector.");
        }

        const arma::uvec& T_c = observed_outcome_indices[c];
        m.row(c) = impute_outcomes(G, g_0, T_c, m_c_vec[c]).t();
    }
    return m;
}

arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& a,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec,
    const std::vector<arma::mat>& X_c_vec) {

    const arma::uword T = G.n_rows;
    const arma::uword C = observed_outcome_indices.size();

    if (observed_outcome_indices.size() != C || m_c_vec.size() != C || X_c_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }

    arma::mat m(C, T, arma::fill::zeros);
    for (arma::uword c = 0; c < C; ++c) {
        if (observed_outcome_indices[c].size() != m_c_vec[c].n_elem) {
            throw std::invalid_argument("The number of observed outcomes for each cohort must match the number of outcomes in the m_c vector.");
        }
        if (X_c_vec[c].n_rows != T) {
            throw std::invalid_argument("The number of rows in X_c must match the number of rows in G.");
        }

        const arma::uvec& T_c = observed_outcome_indices[c];
        m.row(c) = impute_outcomes(G, a, T_c, m_c_vec[c], X_c_vec[c]).t();
    }
    return m;
}

arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec) {

    const arma::uword T = G.n_rows;
    const arma::uword C = observed_outcome_indices.size();

    if (observed_outcome_indices.size() != C || m_c_vec.size() != C) {
        throw std::invalid_argument("Input vectors must have a size equal to the number of cohorts.");
    }

    arma::mat m(C, T, arma::fill::zeros);
    for (arma::uword c = 0; c < C; ++c) {
        if (observed_outcome_indices[c].size() != m_c_vec[c].n_elem) {
            throw std::invalid_argument("The number of observed outcomes for each cohort must match the number of outcomes in the m_c vector.");
        }

        const arma::uvec& T_c = observed_outcome_indices[c];
        m.row(c) = impute_outcomes(G, T_c, m_c_vec[c]).t();
    }
    return m;
}

} // namespace apm