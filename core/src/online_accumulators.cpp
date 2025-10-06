#include "online_accumulators.h"

namespace apm { namespace stats {

std::pair<arma::vec,double> online_weighted_mean(
    const arma::mat& Y,
    const arma::vec& w,
    const arma::vec& mean_prev,
    double total_prev)
{
    const double batch_w = arma::accu(w);
    if (batch_w == 0.0) return {mean_prev, total_prev};
    arma::rowvec weighted_sum = w.t() * Y;           // 1 x T_c
    arma::vec batch_mean = (weighted_sum / batch_w).t(); // T_c
    const double total = total_prev + batch_w;
    if (total == 0.0) return {mean_prev, 0.0};
    const double rel = batch_w / total;
    arma::vec combined = rel * batch_mean + (1.0 - rel) * mean_prev;
    return {std::move(combined), total};
}

std::pair<arma::mat,double> online_weighted_means_over_cube(
    const arma::cube& X,
    const arma::vec& w,
    const arma::mat& means_prev,
    double total_prev)
{
    const double batch_w = arma::accu(w);
    if (batch_w == 0.0) return {means_prev, total_prev};
    const arma::uword T = X.n_cols, q = X.n_slices;
    arma::mat batch_means(T, q, arma::fill::zeros);
    for (arma::uword k = 0; k < q; ++k) {
        arma::rowvec weighted_sum = w.t() * X.slice(k);   // 1 x T
        batch_means.col(k) = (weighted_sum / batch_w).t(); // T x 1
    }
    const double total = total_prev + batch_w;
    if (total == 0.0) return {means_prev, 0.0};
    const double rel = batch_w / total;
    arma::mat combined = rel * batch_means + (1.0 - rel) * means_prev;
    return {std::move(combined), total};
}

std::pair<arma::mat,double> online_weighted_second_moment(
    const arma::mat& Y,
    const arma::vec& w,
    const arma::mat& second_prev,
    double total_prev)
{
    const double batch_w = arma::accu(w);
    if (batch_w == 0.0) return {second_prev, total_prev};
    arma::mat WY = Y.each_col() % w;                   // N x T_c
    arma::mat batch_second = (Y.t() * WY) / batch_w;   // T_c x T_c
    const double total = total_prev + batch_w;
    if (total == 0.0) return {second_prev, 0.0};
    const double rel = batch_w / total;
    arma::mat combined = rel * batch_second + (1.0 - rel) * second_prev;
    return {std::move(combined), total};
}

}} // namespace apm::stats


