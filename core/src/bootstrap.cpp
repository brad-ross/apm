#include "bootstrap.h"
#include <random>
#include <sstream>

namespace {

inline void validate_bootstrap_sizes(std::size_t N, std::size_t B, const char* which) {
    if (N == 0 || B == 0) {
        std::ostringstream oss;
        oss << which << ": N>0 and B>0 required.";
        throw std::invalid_argument(oss.str());
    }
}

} // anonymous namespace

namespace apm {

//--------- WeightedBootstrap (base) ---------

WeightedBootstrap::WeightedBootstrap(arma::mat weights)
    : weights_(std::move(weights)) {
    validate_nonempty(weights_);
    validate_nonnegative(weights_);
    normalize_columns(weights_);
}

arma::vec WeightedBootstrap::draw(std::size_t b) const {
    validate_index(b, n_bootstraps(), "draw index");
    return weights_.col(b);
}

arma::vec WeightedBootstrap::obs(std::size_t i) const {
    validate_index(i, n_obs(), "observation index");
    return weights_.row(i).t();
}

arma::mat WeightedBootstrap::obs(const arma::uvec& idx) const {
    validate_indices(idx, n_obs(), "observation indices");
    return weights_.rows(idx);
}

arma::mat WeightedBootstrap::obs(const std::vector<std::size_t>& idx) const {
    arma::uvec aidx(idx.size());
    for (std::size_t k = 0; k < idx.size(); ++k) aidx[k] = static_cast<arma::uword>(idx[k]);
    validate_indices(aidx, n_obs(), "observation indices");
    return weights_.rows(aidx);
}

std::string WeightedBootstrap::desc() const {
    std::ostringstream oss;
    oss << "WeightedBootstrap[N=" << n_obs() << ", B=" << n_bootstraps() << "]";
    return oss.str();
}

void WeightedBootstrap::normalize_columns(arma::mat& W) {
    const arma::rowvec col_sums = arma::sum(W, 0);
    // Create a "safe" version of column sums that replaces zeros with ones
    // to avoid division-by-zero. This does not change the result for nonzero
    // columns and only affects degenerate columns (which should not occur
    // with our weight generators), where normalization would be undefined.
    arma::rowvec safe = col_sums;
    for (arma::uword j = 0; j < safe.n_elem; ++j) {
        if (safe[j] == 0.0) safe[j] = 1.0;
    }
    // Broadcast divide: each row is divided elementwise by the 1xB vector
    // so W(i,j) becomes W(i,j) / safe(j), i.e., column-wise normalization.
    W.each_row() /= safe;
}

void WeightedBootstrap::validate_nonempty(const arma::mat& W) {
    if (W.n_rows == 0 || W.n_cols == 0) {
        throw std::invalid_argument("WeightedBootstrap: weights must be non-empty (N>0, B>0).");
    }
}

void WeightedBootstrap::validate_nonnegative(const arma::mat& W) {
    if (W.min() < 0.0) {
        throw std::invalid_argument("WeightedBootstrap: weights must be nonnegative.");
    }
}

void WeightedBootstrap::validate_index(std::size_t idx, std::size_t lim, const char* what) {
    if (idx >= lim) {
        std::ostringstream oss;
        oss << "WeightedBootstrap: " << what << " out of range [0," << (lim ? lim - 1 : 0) << "]";
        throw std::out_of_range(oss.str());
    }
}

void WeightedBootstrap::validate_indices(const arma::uvec& idx, std::size_t lim, const char* what) {
    for (arma::uword k = 0; k < idx.n_elem; ++k) {
        if (idx[k] >= lim) {
            std::ostringstream oss;
            oss << "WeightedBootstrap: " << what << " contain out-of-range index " << idx[k]
                << " (valid [0," << (lim ? lim - 1 : 0) << "])";
            throw std::out_of_range(oss.str());
        }
    }
}

//--------- MultinomialBootstrap ---------

MultinomialBootstrap::MultinomialBootstrap(std::size_t N, std::size_t B, std::uint64_t seed)
    : WeightedBootstrap([&]() {
          validate_bootstrap_sizes(N, B, "MultinomialBootstrap");
          arma::mat W(N, B, arma::fill::zeros);

          std::mt19937_64 gen(seed ? seed : std::random_device{}());
          std::uniform_int_distribution<std::size_t> unif(0, N - 1);

          for (std::size_t b = 0; b < B; ++b) {
              arma::Col<double> counts(N, arma::fill::zeros);
              for (std::size_t t = 0; t < N; ++t) {
                  const std::size_t j = unif(gen);
                  counts[j] += 1.0;
              }
              W.col(b) = counts / static_cast<double>(N);
          }
          return W;
      }()) {}

//--------- BayesianBootstrap ---------

BayesianBootstrap::BayesianBootstrap(std::size_t N, std::size_t B, std::uint64_t seed)
    : WeightedBootstrap([&]() {
          validate_bootstrap_sizes(N, B, "BayesianBootstrap");
          // When building for R (via RcppArmadillo), Armadillo's RNG is wired to
          // R's RNG. Seeding must be done from R (set.seed). Avoid calling
          // arma::arma_rng::set_seed* to prevent warnings and ensure correctness.
          // Outside of R builds, we seed Armadillo's RNG directly.
          #ifndef USING_R
          if (seed) {
              arma::arma_rng::set_seed(seed);
          } else {
              arma::arma_rng::set_seed_random();
          }
          #endif

          arma::mat W(N, B, arma::fill::none);
          for (std::size_t b = 0; b < B; ++b) {
              arma::vec g = arma::randg<arma::vec>(N, arma::distr_param(1.0, 1.0));
              const double s = arma::accu(g);
              W.col(b) = (s > 0.0) ? (g / s) : (arma::vec(N, arma::fill::ones) / static_cast<double>(N));
          }
          return W;
      }()) {}

} // namespace apm


