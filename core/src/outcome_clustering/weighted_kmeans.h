#pragma once
#ifndef APM_WEIGHTED_KMEANS_H
#define APM_WEIGHTED_KMEANS_H

//==============================================================================
// Weighted k-means routine used by outcome clustering.
//==============================================================================

#include <cstddef>
#include <utility>
#include <optional>
#include <cstdint>

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

namespace apm {

/**
 * @brief Run weighted k-means on rows of data.
 *
 * @param data Matrix with observations in rows and features in columns.
 * @param weights Nonnegative weights per observation (length n_rows).
 * @param k Number of clusters.
 * @param seed Optional RNG seed.
 * @return Pair {objective value, row assignment vector}.
 */
std::pair<double, arma::Row<size_t>> kmeans_weighted(const arma::mat& data,
    const arma::vec& weights,
    std::size_t k,
    std::optional<uint64_t> seed);

} // namespace apm

#endif // APM_WEIGHTED_KMEANS_H