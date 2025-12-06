#ifndef APM_ONLINE_ACCUMULATORS_H
#define APM_ONLINE_ACCUMULATORS_H

#include <utility>

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

namespace apm { namespace stats {

/**
 * @brief Online combination of (row-weighted) outcome means: Y is N x T_c, w is length N.
 *
 * @param Y N x T_c matrix of outcomes for the batch.
 * @param w Length-N nonnegative weights for rows in the batch.
 * @param mean_prev Running mean vector from prior batches (length T_c).
 * @param total_prev Running total weight from prior batches.
 * @return Pair {updated mean vector, updated total weight}.
 */
std::pair<arma::vec,double> online_weighted_mean(
    const arma::mat& Y,
    const arma::vec& w,
    const arma::vec& mean_prev,
    double total_prev);

/**
 * @brief Online combination of (row-weighted) covariate means across an N x T x q cube.
 *
 * @param X N x T x q cube of covariates for the batch.
 * @param w Length-N nonnegative weights for slices in the batch.
 * @param means_prev Running covariate means from prior batches (T x q).
 * @param total_prev Running total weight from prior batches.
 * @return Pair {updated T x q means, updated total weight}.
 */
std::pair<arma::mat,double> online_weighted_means_over_cube(
    const arma::cube& X,
    const arma::vec& w,
    const arma::mat& means_prev,
    double total_prev);

/**
 * @brief Online combination of (row-weighted) second moment E[YY'].
 *
 * Useful for PC estimators; Y is N x T_c, w is length N.
 *
 * @param Y N x T_c matrix of outcomes for the batch.
 * @param w Length-N nonnegative weights for matrices in the batch.
 * @param second_prev Running second-moment matrix from prior batches (T_c x T_c).
 * @param total_prev Running total weight from prior batches.
 * @return Pair {updated second-moment matrix, updated total weight}.
 */
std::pair<arma::mat,double> online_weighted_second_moment(
    const arma::mat& Y,
    const arma::vec& w,
    const arma::mat& second_prev,
    double total_prev);

}} // namespace apm::stats

#endif // APM_ONLINE_ACCUMULATORS_H


