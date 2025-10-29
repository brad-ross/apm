#pragma once
#ifndef APM_WEIGHTED_KMEANS_H
#define APM_WEIGHTED_KMEANS_H

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

std::pair<double, arma::Row<size_t>> kmeans_weighted(const arma::mat& data,
    const arma::vec& weights,
    std::size_t k,
    std::optional<uint64_t> seed);

} // namespace apm

#endif // APM_WEIGHTED_KMEANS_H