#ifndef APM_ONLINE_ACCUMULATORS_H
#define APM_ONLINE_ACCUMULATORS_H

#include <utility>

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

namespace apm { namespace stats {

// Online combine of (row-weighted) outcome means: Y is N x T_c, w is length N
std::pair<arma::vec,double> online_weighted_mean(
    const arma::mat& Y,
    const arma::vec& w,
    const arma::vec& mean_prev,
    double total_prev);

// Online combine of (row-weighted) covariate means across an N x T x q cube
// Returns updated T x q means and new total weight
std::pair<arma::mat,double> online_weighted_means_over_cube(
    const arma::cube& X,
    const arma::vec& w,
    const arma::mat& means_prev,
    double total_prev);

// Online combine of (row-weighted) second moment E[YY']
// Useful for PC estimators; Y is N x T_c, w is length N
std::pair<arma::mat,double> online_weighted_second_moment(
    const arma::mat& Y,
    const arma::vec& w,
    const arma::mat& second_prev,
    double total_prev);

}} // namespace apm::stats

#endif // APM_ONLINE_ACCUMULATORS_H


