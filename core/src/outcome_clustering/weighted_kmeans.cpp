#include "outcome_clustering/weighted_kmeans.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

// mlpack support removed

namespace apm {

namespace {

inline void seed_rng_if_requested(const std::optional<uint64_t>& seed) {
    if (!seed.has_value()) return;
    arma::arma_rng::set_seed(static_cast<arma::uword>(*seed));
}


inline double weighted_sse(const arma::mat& data,
                           const arma::vec& weights,
                           const arma::Row<size_t>& assignments,
                           const arma::mat& centers)
{
    const arma::uword T = data.n_cols;
    double sse = 0.0;
    for (arma::uword i = 0; i < T; ++i) {
        const arma::uword a = static_cast<arma::uword>(assignments(i));
        const double wi = weights(i);
        if (wi == 0.0) continue;
        const arma::vec diff = data.col(i) - centers.col(a);
        sse += wi * arma::dot(diff, diff);
    }
    return sse;
}

arma::mat kpp_init(const arma::mat& data, std::size_t k, std::optional<uint64_t> seed)
{
    seed_rng_if_requested(seed);
    const arma::uword T = data.n_cols;
    arma::uvec centers_idx(static_cast<arma::uword>(k));
    centers_idx(0) = (T <= 1 ? 0u : static_cast<arma::uword>(std::floor(arma::randu() * T)));

    arma::vec min_d2(T);
    for (arma::uword i = 0; i < T; ++i) {
        min_d2(i) = arma::accu(arma::square(data.col(i) - data.col(centers_idx(0))));
    }
    for (arma::uword c = 1; c < static_cast<arma::uword>(k); ++c) {
        const double sum_d2 = arma::accu(min_d2);
        double r = arma::randu() * (sum_d2 > 0.0 ? sum_d2 : 1.0);
        arma::uword pick = 0;
        for (; pick < T; ++pick) {
            r -= min_d2(pick);
            if (r <= 0.0) break;
        }
        if (pick >= T) pick = T - 1;
        centers_idx(c) = pick;
        for (arma::uword i = 0; i < T; ++i) {
            const double d2 = arma::accu(arma::square(data.col(i) - data.col(pick)));
            if (c == 1 || d2 < min_d2(i)) min_d2(i) = d2;
        }
    }
    return data.cols(centers_idx);
}

void weighted_lloyd(const arma::mat& data,
                           const arma::vec& weights,
                           arma::mat& centers,
                           arma::Row<size_t>& assignments,
                           std::size_t max_iter = 100,
                           double tol = 1e-8)
{
    (void)tol;
    const arma::uword T = data.n_cols;
    const arma::uword k = centers.n_cols;
    assignments.set_size(T);
    arma::Row<size_t> prev(T); prev.fill(std::numeric_limits<size_t>::max());

    for (std::size_t it = 0; it < max_iter; ++it) {
        for (arma::uword i = 0; i < T; ++i) {
            arma::vec d = arma::sum(arma::square(centers.each_col() - data.col(i)), 0).t();
            assignments(i) = static_cast<size_t>(d.index_min());
        }
        if (arma::all(arma::conv_to<arma::uvec>::from(assignments == prev))) break;
        prev = assignments;

        centers.zeros();
        arma::vec denom(k, arma::fill::zeros);
        for (arma::uword i = 0; i < T; ++i) {
            const arma::uword c = static_cast<arma::uword>(assignments(i));
            const double w = weights(i);
            if (w == 0.0) continue;
            centers.col(c) += data.col(i) * w;
            denom(c) += w;
        }
        for (arma::uword c = 0; c < k; ++c) {
            if (denom(c) > 0.0) centers.col(c) /= denom(c);
        }
    }
}

} // anonymous namespace

std::pair<double, arma::Row<size_t>> kmeans_weighted(const arma::mat& data,
                                                            const arma::vec& weights,
                                                            std::size_t k,
                                                            std::optional<uint64_t> seed)
{
    arma::mat centers = kpp_init(data, k, seed);
    arma::Row<size_t> assignments;
    weighted_lloyd(data, weights, centers, assignments);
    double sse = 0.0;
    for (arma::uword i = 0; i < data.n_cols; ++i) {
        const arma::uword a = static_cast<arma::uword>(assignments(i));
        const double w = weights(i);
        if (w == 0.0) continue;
        const arma::vec diff = data.col(i) - centers.col(a);
        sse += w * arma::dot(diff, diff);
    }
    return {sse, std::move(assignments)};
}

} // namespace apm