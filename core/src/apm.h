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

} // namespace apm

#endif // APM_H