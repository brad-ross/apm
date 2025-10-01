#include "bootstrap.h"
#include <random>
#include <sstream>
#include <cmath>

namespace {

inline void validate_bootstrap_sizes(std::size_t N, std::size_t B, const char* which) {
    if (N == 0 || B == 0) {
        std::ostringstream oss;
        oss << which << ": N>0 and B>0 required.";
        throw std::invalid_argument(oss.str());
    }
}

// Robust IQR scale of standard normal: qnorm(0.75) - qnorm(0.25) = 2 * 0.6744897501960817
constexpr double kNormalIQR = 1.3489795003921634;

inline void validate_bootstrap_inference_args(
    const arma::vec& point_ests,
    const arma::mat& boot,
    std::size_t N,
    double sig_level
) {
    if (point_ests.n_elem == 0) {
        throw std::invalid_argument("get_bootstrap_inference: point_ests must be non-empty.");
    }
    if (boot.n_rows != point_ests.n_elem) {
        throw std::invalid_argument("get_bootstrap_inference: bootstrap_replicates must have p rows.");
    }
    if (boot.n_cols == 0) {
        throw std::invalid_argument("get_bootstrap_inference: bootstrap_replicates must have B>0 columns.");
    }
    if (N == 0) {
        throw std::invalid_argument("get_bootstrap_inference: N must be > 0.");
    }
    if (!(sig_level > 0.0 && sig_level < 1.0)) {
        throw std::invalid_argument("get_bootstrap_inference: sig_level must be in (0,1).");
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

//--------- Bootstrap Inference ---------

SimultaneousInferenceResults get_bootstrap_inference(
    const arma::vec& point_ests,
    const arma::mat& bootstrap_replicates,
    std::size_t N,
    double sig_level
) {
    validate_bootstrap_inference_args(point_ests, bootstrap_replicates, N, sig_level);

    const double sqrtN = std::sqrt(static_cast<double>(N));
    const std::size_t B = bootstrap_replicates.n_cols;

    // z_stats: sqrt(N) * (boot - theta), broadcasting theta along columns
    arma::mat z_stats = sqrtN * (bootstrap_replicates.each_col() - point_ests);

    // Rowwise robust scale via IQR / IQR(N(0,1))
    arma::vec q25 = arma::quantile(z_stats, arma::vec{0.25}, 1); // p x 1, dim=1 for rowwise
    arma::vec q75 = arma::quantile(z_stats, arma::vec{0.75}, 1); // p x 1
    arma::vec row_sd = (q75 - q25) / kNormalIQR;

    // Guard against zero/negative scales for division
    arma::vec scale = row_sd;
    scale.transform([](double x) {
        return (x > 0.0) ? x : 1.0;
    });

    // Pointwise t-statistics: point_ests / row_sd
    arma::vec pointwise_t_stats = point_ests / scale;

    // |t|-stats per row, per draw
    arma::mat abs_t_stats = arma::abs(z_stats.each_col() / scale);

    // Pointwise p-values: for each parameter, count how many bootstrap |t|'s exceed observed |t|
    arma::vec pointwise_p_vals(point_ests.n_elem);
    arma::vec abs_pointwise_t = arma::abs(pointwise_t_stats);
    for (arma::uword i = 0; i < point_ests.n_elem; ++i) {
        double count = 0.0;
        for (arma::uword b = 0; b < B; ++b) {
            if (abs_t_stats(i, b) > abs_pointwise_t(i)) {
                count += 1.0;
            }
        }
        pointwise_p_vals(i) = (count + 1.0) / (static_cast<double>(B) + 1.0);
    }

    // Per-parameter critical values: (1 - sig_level) quantile across draws
    arma::vec ci_crit_vals = arma::quantile(abs_t_stats, arma::vec{1.0 - sig_level}, 1); // dim=1 for rowwise

    // Pointwise CI half-widths and bounds
    arma::vec ci_half = (ci_crit_vals % row_sd) / sqrtN;
    arma::vec ci_lb = point_ests - ci_half;
    arma::vec ci_ub = point_ests + ci_half;

    // Kolmogorov-Smirnov stats across parameters for each draw (column-wise max over rows)
    arma::vec ks_stats = arma::max(abs_t_stats, 0).t(); // B x 1, dim=0 for column-wise max

    // Simultaneous critical value and band
    arma::vec simult_crit_val_vec = arma::quantile(ks_stats, arma::vec{1.0 - sig_level});
    const double simult_crit_val = simult_crit_val_vec(0);
    arma::vec cb_half = (row_sd * simult_crit_val) / sqrtN;
    arma::vec cb_lb = point_ests - cb_half;
    arma::vec cb_ub = point_ests + cb_half;

    return SimultaneousInferenceResults{
        point_ests,
        pointwise_t_stats,
        pointwise_p_vals,
        sig_level,
        ci_lb,
        ci_ub,
        cb_lb,
        cb_ub
    };
}

} // namespace apm


