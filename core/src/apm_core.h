#ifndef APM_H
#define APM_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif
#include <string>
#include <vector>
#include <set>
#include "factor_model_parameter_structs.h"

namespace apm {

// Alias representing the type of observed_outcome_indices throughout the codebase
using ObservedOutcomeIndices = std::vector<arma::uvec>;

/**
 * @brief Get library version information
 * 
 * @return Version string
 */
std::string get_version();

//==============================================================================
// Aggregators of Cohort-Specific Parameter Estimates
//==============================================================================

/**
 * @brief Computes an aligned matrix of factor vectors from cohort-specific ones (optionally weighted).
 *
 * Constructs the Aggregated Projection Matrix (APM) from cohort-specific factor matrices
 * and returns an orthonormal basis for its null space. If cohort_weights is omitted or
 * empty, equal weights are used across cohorts.
 *
 * @param cohort_factor_matrices A vector of matrices, one for each cohort.
 * @param observed_outcome_indices Observed outcomes per cohort (0-based indices).
 * @param cohort_weights Optional weights per cohort; scaled to sum to one if provided.
 * @return A matrix whose columns form an orthonormal basis for the null space of the APM.
 */
arma::mat align_factors_using_apm(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::vec& cohort_weights = arma::vec());

/**
 * @brief Aggregates cohort-specific outcome fixed effect estimates.
 *
 * This function computes the average of outcome fixed effect estimates across cohorts
 * for each outcome.
 *
 * @param g_0_c_vec A vector of arma::vec, where each vector contains cohort-specific estimates of outcome
 *                  fixed effects for the observed outcomes in those cohorts.
 * @param observed_outcome_indices A vector of arma::uvec, where each uvec contains
 *                                 the 0-indexed indices of observed outcomes for a cohort.
 * @return An arma::vec containing the aggregated outcome fixed effect estimates for all outcomes.
 */
arma::vec aggregate_cohort_specific_outcome_fes(
    const std::vector<arma::vec>& g_0_c_vec,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::vec& cohort_weights = arma::vec());

/**
 * @brief Aggregates cohort-specific covariate coefficient estimates.
 *
 * This function takes a vector of cohort-specific covariate coefficient estimates
 * and returns their (optionally weighted) average.
 *
 * @param a_c_vec A vector of arma::vec, where each vector contains cohort-specific
 *                covariate coefficient estimates.
 * @return An arma::vec containing the aggregated covariate coefficient estimates.
 */
arma::vec aggregate_cohort_specific_covariate_coefs(
    const std::vector<arma::vec>& a_c_vec,
    const arma::vec& cohort_weights = arma::vec());

/**
 * @brief Aggregate cohort-specific parameter estimates into a unified set.
 *
 * Given per-cohort FactorModelParameters and observed outcome indices, aligns
 * factor matrices across cohorts and aggregates optional fixed effects and
 * covariate coefficients when present for all cohorts.
 *
 * @param cohort_specific_factor_model_params Vector of per-cohort parameters.
 * @param observed_outcome_indices Observed outcome indices per cohort (0-based).
 * @param cohort_weights Optional cohort weights; scaled to sum to one if provided.
 * @return Aggregated FactorModelParameters (G and optional g_0, a).
 */
FactorModelParameters aggregate_cohort_specific_factor_model_params(
    const std::vector<FactorModelParameters>& cohort_specific_factor_model_params,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const arma::vec& cohort_weights = arma::vec());

//==============================================================================
// Outcome Imputation
//==============================================================================

/**
 * @brief Estimates outcomes for a representative unit (factors, fixed effects, and covariates).
 * @param G A T x r matrix of estimated factors.
 * @param g_0 A T-dimensional vector of estimated outcome fixed effects.
 * @param a A q-dimensional vector of estimated covariate coefficients.
 * @param T_c A vector of indices for the observed outcomes for the cohort.
 * @param m_c A vector containing the observed outcomes for the representative unit.
 * @param X_c A T x q matrix containing the values of q covariates corresponding to each outcome for the representative unit.
 * @return A T-dimensional vector containing the estimated outcomes for the representative unit.
 */
arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::vec& g_0,
    const arma::vec& a,
    const arma::uvec& T_c,
    const arma::vec& m_c,
    const arma::mat& X_c);

/**
 * @brief Estimates outcomes for a representative unit (factors and fixed effects).
 * @param G A T x r matrix of estimated factors.
 * @param g_0 A T-dimensional vector of estimated outcome fixed effects.
 * @param T_c A vector of indices for the observed outcomes for the cohort.
 * @param m_c A vector containing the observed outcomes for the representative unit.
 * @return A T-dimensional vector containing the estimated outcomes for the representative unit.
 */
arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::vec& g_0,
    const arma::uvec& T_c,
    const arma::vec& m_c);

/**
 * @brief Estimates outcomes for a representative unit (factors and covariates).
 * @param G A T x r matrix of estimated factors.
 * @param a A q-dimensional vector of estimated covariate coefficients.
 * @param T_c A vector of indices for the observed outcomes for the cohort.
 * @param m_c A vector containing the observed outcomes for the representative unit.
 * @param X_c A T x q matrix containing the values of q covariates corresponding to each outcome for the representative unit.
 * @return A T-dimensional vector containing the estimated outcomes for the representative unit.
 */
arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::vec& a,
    const arma::uvec& T_c,
    const arma::vec& m_c,
    const arma::mat& X_c);

/**
 * @brief Estimates outcomes for a representative unit (factors only).
 * @param G A T x r matrix of estimated factors.
 * @param T_c A vector of indices for the observed outcomes for the cohort.
 * @param m_c A vector containing the observed outcomes for the representative unit.
 * @return A T-dimensional vector containing the estimated outcomes for the representative unit.
 */
arma::vec impute_outcomes(
    const arma::mat& G,
    const arma::uvec& T_c,
    const arma::vec& m_c);

//==============================================================================
// Outcome Mean Estimation Across Cohorts
//==============================================================================

/**
 * @brief Estimates mean outcomes for each cohort (factors, fixed effects, and covariates).
 * @param G A T x r matrix whose rows are estimated factor vectors.
 * @param g_0 A T-dimensional vector of estimated outcome fixed effects.
 * @param a A q-dimensional vector of estimated covariate coefficients.
 * @param observed_outcome_indices A vector where each element is a vector of indices for the observed outcomes for a cohort.
 * @param m_c_vec A vector of arma::vec, where each vector m_c contains the observed outcomes for a cohort.
 * @param X_c_vec A vector of T x q matrices, where each matrix X_c contains the average values of q covariates for each outcome within a cohort.
 * @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
 */
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& g_0,
    const arma::vec& a,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec,
    const std::vector<arma::mat>& X_c_vec);

/**
 * @brief Estimates mean outcomes for each cohort (factors and fixed effects).
 * @param G A T x r matrix whose rows are estimated factor vectors.
 * @param g_0 A T-dimensional vector of estimated outcome fixed effects.
 * @param observed_outcome_indices A vector where each element is a vector of indices for the observed outcomes for a cohort.
 * @param m_c_vec A vector of arma::vec, where each vector m_c contains the observed outcomes for a cohort.
 * @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
 */
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& g_0,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec);

/**
 * @brief Estimates mean outcomes for each cohort (factors and covariates).
 * @param G A T x r matrix whose rows are estimated factor vectors.
 * @param a A q-dimensional vector of estimated covariate coefficients.
 * @param observed_outcome_indices A vector where each element is a vector of indices for the observed outcomes for a cohort.
 * @param m_c_vec A vector of arma::vec, where each vector m_c contains the observed outcomes for a cohort.
 * @param X_c_vec A vector of T x q matrices, where each matrix X_c contains the average values of q covariates for each outcome within a cohort.
 * @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
 */
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& a,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec,
    const std::vector<arma::mat>& X_c_vec);

/**
 * @brief Estimates mean outcomes for each cohort (factors only).
 * @param G A T x r matrix whose rows are estimated factor vectors.
 * @param observed_outcome_indices A vector where each element is a vector of indices for the observed outcomes for a cohort.
 * @param m_c_vec A vector of arma::vec, where each vector m_c contains the observed outcomes for a cohort.
 * @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
 */
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const ObservedOutcomeIndices& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec);

//==============================================================================
// Identification Verification Via the O^3 Algorithm
//==============================================================================

/**
 * @brief Implements the Observed Outcome Overlap (O^3) algorithm to assess factor identification.
 *
 * This algorithm iteratively groups cohorts based on the overlap of their
 * observed outcomes. Two super cohorts are merged if the number of their
 * shared outcomes meets or exceeds the model rank, `r`. The process
 * continues until no more cohorts can be merged.
 *
 * @param observed_outcome_indices A vector where each element is a vector of
 *                                 indices corresponding to the observed outcomes for a cohort.
 * @param r The model rank, used as the minimum overlap threshold for merging cohorts.
 * @return A vector containing the lists of super cohorts at each iteration of the
 *         algorithm. Each list of super cohorts is a vector of sets of indices 
 *         corresponding to the original cohorts together forming a super cohort.
 */
std::vector<std::vector<std::set<arma::uword>>> o3_algorithm(
    const ObservedOutcomeIndices& observed_outcome_indices,
    unsigned int r);

/**
 * @brief Checks if the factors are identified across all cohorts.
 *
 * This function uses the O^3 algorithm to determine if there is sufficient
 * overlap in observed outcomes across all cohorts to uniquely identify all factor 
 * vectors expressed with respect to a common basis. Identification is achieved if 
 * the algorithm terminates with a single super cohort containing all of the 
 * original cohorts.
 *
 * @param observed_outcome_indices A vector where each element is a vector of
 *                                 indices for the observed outcomes for a cohort.
 * @param r The model rank.
 * @return `true` if the factors are identified, `false` otherwise.
 */
bool aligned_factors_identified(
    const ObservedOutcomeIndices& observed_outcome_indices,
    unsigned int r);

} // namespace apm

#endif // APM_H