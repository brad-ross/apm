#ifndef APM_H
#define APM_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif
#include <string>
#include <vector>

namespace apm {

/**
 * @brief Get library version information
 * 
 * @return Version string
 */
std::string get_version();

/**
 * @brief Computes an aligned matrix of factor vectors from cohort-specific ones.
 *
 * This function constructs an Aggregated Projection Matrix from cohort-specific data and 
 * then returns an orthonormal basis for the null space of this matrix, which 
 * also serves as a basis for the column space of the matrix whose rows are the 
 * factor vectors corresponding to each outcome.
 *
 * @param cohort_factor_matrices A vector of matrices, one for each cohort, 
 * where the rows of the matrix corresponding to a given cohort contain the factor vectors corresponding to the observed outcomes for that cohort.
 * Each matrix must have the same number of columns equal to the rank 
 * of the factor model.
 * @param observed_outcome_indices A vector of the same length as cohort_factor_matrices.
 *                                 Each element is a vector of indices indicating
 *                                 which time periods were observed for the
 *                                 corresponding cohort. The number of indices
 *                                 must match the number of rows in the cohort's
 *                                 factor matrix.
 * @return A matrix whose columns form an orthonormal basis for the null space
 *         of the aggregated projection matrix.
 */
arma::mat align_factors_using_apm(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const std::vector<arma::uvec>& observed_outcome_indices);

/**
 * @brief Computes an aligned matrix of factor vectors from cohort-specific ones using cohort-specific weights.
 *
 * This function constructs an Aggregated Projection Matrix from cohort-specific data and 
 * then returns an orthonormal basis for the null space of this matrix, which 
 * also serves as a basis for the column space of the matrix whose rows are the 
 * factor vectors corresponding to each outcome.
 *
 * @param cohort_factor_matrices A vector of matrices, one for each cohort, 
 * where the rows of the matrix corresponding to a given cohort contain the factor vectors corresponding to the observed outcomes for that cohort.
 * Each matrix must have the same number of columns equal to the rank 
 * of the factor model.
 * @param observed_outcome_indices A vector of the same length as cohort_factor_matrices.
 *                                 Each element is a vector of indices indicating
 *                                 which time periods were observed for the
 *                                 corresponding cohort. The number of indices
 *                                 must match the number of rows in the cohort's
 *                                 factor matrix.
 * @param cohort_weights A vector of weights for each cohort. The weights are scaled to sum to one.
 * @return A matrix whose columns form an orthonormal basis for the null space
 *         of the aggregated projection matrix.
 */
arma::mat align_factors_using_apm(
    const std::vector<arma::mat>& cohort_factor_matrices,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const arma::vec& cohort_weights);

//==============================================================================
// Outcome Imputation
//==============================================================================

/**
 * @brief Estimates outcomes for a representative unit (factors, fixed effects, and covariates).
 * @param G A T x r matrix of estimated factors.
 * @param g_0 A T-dimensional vector of estimated outcome fixed effects.
 * @param a A q-dimensional vector of estimated covariate coefficients.
 * @param T_c A vector of indices for the observed time periods for the cohort.
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
 * @param T_c A vector of indices for the observed time periods for the cohort.
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
 * @param T_c A vector of indices for the observed time periods for the cohort.
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
 * @param T_c A vector of indices for the observed time periods for the cohort.
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
 * @param observed_outcome_indices A vector where each element is a vector of indices for the observed time periods for a cohort.
 * @param m_c_vec A vector of arma::vec, where each vector m_c contains the observed outcomes for a cohort.
 * @param X_c_vec A vector of T x q matrices, where each matrix X_c contains the average values of q covariates for each outcome within a cohort.
 * @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
 */
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& g_0,
    const arma::vec& a,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec,
    const std::vector<arma::mat>& X_c_vec);

/**
 * @brief Estimates mean outcomes for each cohort (factors and fixed effects).
 * @param G A T x r matrix whose rows are estimated factor vectors.
 * @param g_0 A T-dimensional vector of estimated outcome fixed effects.
 * @param observed_outcome_indices A vector where each element is a vector of indices for the observed time periods for a cohort.
 * @param m_c_vec A vector of arma::vec, where each vector m_c contains the observed outcomes for a cohort.
 * @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
 */
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& g_0,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec);

/**
 * @brief Estimates mean outcomes for each cohort (factors and covariates).
 * @param G A T x r matrix whose rows are estimated factor vectors.
 * @param a A q-dimensional vector of estimated covariate coefficients.
 * @param observed_outcome_indices A vector where each element is a vector of indices for the observed time periods for a cohort.
 * @param m_c_vec A vector of arma::vec, where each vector m_c contains the observed outcomes for a cohort.
 * @param X_c_vec A vector of T x q matrices, where each matrix X_c contains the average values of q covariates for each outcome within a cohort.
 * @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
 */
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const arma::vec& a,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec,
    const std::vector<arma::mat>& X_c_vec);

/**
 * @brief Estimates mean outcomes for each cohort (factors only).
 * @param G A T x r matrix whose rows are estimated factor vectors.
 * @param observed_outcome_indices A vector where each element is a vector of indices for the observed time periods for a cohort.
 * @param m_c_vec A vector of arma::vec, where each vector m_c contains the observed outcomes for a cohort.
 * @return A C x T matrix where each row c contains the estimated T mean outcomes for cohort c.
 */
arma::mat estimate_outcome_means_across_cohorts(
    const arma::mat& G,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::vec>& m_c_vec);

} // namespace apm

#endif // APM_H