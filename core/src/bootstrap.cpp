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

// Compute bootstrap pointwise p-values with +1/(B+1) correction.
// abs_t_stats is p x B with entries |t_i^*,m|; abs_obs is p-vector of |t_i|.
inline arma::vec bootstrap_pointwise_pvals(const arma::mat& abs_t_stats,
                                           const arma::vec& abs_obs) {
    const std::size_t p = abs_t_stats.n_rows;
    const std::size_t B = abs_t_stats.n_cols;
    arma::vec pvals(p);
    for (arma::uword i = 0; i < p; ++i) {
        double thr = abs_obs(i);
        double count = 0.0;
        for (arma::uword b = 0; b < B; ++b) {
            if (abs_t_stats(i, b) >= thr) {
                count += 1.0;
            }
        }
        pvals(i) = (count + 1.0) / (static_cast<double>(B) + 1.0);
    }
    return pvals;
}

// Compute Romano–Wolf stepdown adjusted p-values from absolute t-statistics.
// abs_t_stats is p x B with entries |t_i^*,m|; abs_obs is p-vector of |t_i|.
// Returns p-vector of adjusted p-values, in the original hypothesis order.
inline arma::vec romano_wolf_stepdown_pvals(const arma::mat& abs_t_stats,
                                            const arma::vec& abs_obs) {
    const std::size_t p = abs_t_stats.n_rows;
    const std::size_t B = abs_t_stats.n_cols;

    // Order hypotheses by decreasing observed |t|
    arma::uvec order = arma::sort_index(abs_obs, "descend");
    arma::vec abs_obs_sorted = abs_obs(order);
    arma::mat abs_t_ordered = abs_t_stats.rows(order);

    // Suffix maxima across remaining hypotheses for each draw
    arma::mat suffix_max(p, B, arma::fill::none);
    suffix_max.row(p - 1) = abs_t_ordered.row(p - 1);
    for (arma::sword j = static_cast<arma::sword>(p) - 2; j >= 0; --j) {
        suffix_max.row(j) = arma::max(abs_t_ordered.row(j), suffix_max.row(j + 1));
    }

    // Initial p-values with +1 / (B + 1) correction, using shared helper (>= threshold)
    arma::vec p_init = bootstrap_pointwise_pvals(suffix_max, abs_obs_sorted);

    // Monotonicity enforcement
    arma::vec p_adj = p_init;
    for (arma::uword j = 1; j < p; ++j) {
        if (p_adj(j) < p_adj(j - 1)) p_adj(j) = p_adj(j - 1);
    }

    // Map back to original order
    arma::vec out(p);
    for (arma::uword j = 0; j < p; ++j) {
        out(order(j)) = p_adj(j);
    }
    return out;
}

// Construct confidence intervals given critical values (vector), row scales, and sqrtN.
// crit_vals can be a p-vector of per-parameter critical values (pointwise) or a
// repeated scalar expanded by the caller (simultaneous).
inline void constr_conf_intervals(const arma::vec& point_ests,
                                  const arma::vec& crit_vals,
                                  const arma::vec& row_sd,
                                  double sqrtN,
                                  arma::vec& lb,
                                  arma::vec& ub) {
    arma::vec half = (crit_vals % row_sd) / sqrtN;
    lb = point_ests - half;
    ub = point_ests + half;
}

// Construct pointwise confidence intervals using rowwise quantiles of abs t-stats
inline void constr_pointwise_conf_band(const arma::mat& abs_t_stats,
                                       double sqrtN,
                                       double sig_level,
                                       const arma::vec& row_sd,
                                       const arma::vec& point_ests,
                                       arma::vec& ci_lb,
                                       arma::vec& ci_ub) {
    arma::vec ci_crit_vals = arma::quantile(abs_t_stats, arma::vec{1.0 - sig_level}, 1); // dim=1 rowwise
    constr_conf_intervals(point_ests, ci_crit_vals, row_sd, sqrtN, ci_lb, ci_ub);
}

// Construct simultaneous confidence band using KS critical value from abs t-stats
// Inputs: abs_t_stats (p x B), sqrtN, sig_level in (0,1), row_sd (p), point_ests (p)
// Outputs: cb_lb, cb_ub (both p)
inline void constr_simult_conf_band(const arma::mat& abs_t_stats,
                                    double sqrtN,
                                    double sig_level,
                                    const arma::vec& row_sd,
                                    const arma::vec& point_ests,
                                    arma::vec& cb_lb,
                                    arma::vec& cb_ub) {
    // Kolmogorov-Smirnov stats across parameters for each draw (column-wise max over rows)
    arma::vec ks_stats = arma::max(abs_t_stats, 0).t(); // B x 1, dim=0 for column-wise max
    // Simultaneous critical value replicated to p-length vector
    arma::vec simult_crit_val_vec = arma::quantile(ks_stats, arma::vec{1.0 - sig_level});
    arma::vec crit_vals = arma::repmat(simult_crit_val_vec, point_ests.n_elem, 1);
    constr_conf_intervals(point_ests, crit_vals, row_sd, sqrtN, cb_lb, cb_ub);
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

    // Pointwise t-statistics: sqrt(N) * point_ests / row_sd
    arma::vec pointwise_t_stats = (sqrtN * point_ests) / scale;

    // |t|-stats per row, per draw
    arma::mat abs_t_stats = arma::abs(z_stats.each_col() / scale);

    // Pointwise p-values (bootstrap-based)
    arma::vec abs_pointwise_t = arma::abs(pointwise_t_stats);
    arma::vec pointwise_p_vals = bootstrap_pointwise_pvals(abs_t_stats, abs_pointwise_t);

    // Pointwise confidence intervals
    arma::vec ci_lb, ci_ub;
    constr_pointwise_conf_band(abs_t_stats, sqrtN, sig_level, row_sd, point_ests, ci_lb, ci_ub);

    // Romano–Wolf stepdown adjusted p-values (simultaneous p-values)
    arma::vec abs_obs = arma::abs(pointwise_t_stats);
    arma::vec simult_p_vals = romano_wolf_stepdown_pvals(abs_t_stats, abs_obs);

    arma::vec cb_lb, cb_ub;
    constr_simult_conf_band(abs_t_stats, sqrtN, sig_level, row_sd, point_ests, cb_lb, cb_ub);

    return SimultaneousInferenceResults{
        point_ests,
        pointwise_t_stats,
        pointwise_p_vals,
        sig_level,
        ci_lb,
        ci_ub,
        simult_p_vals,
        cb_lb,
        cb_ub
    };
}

} // namespace apm