#pragma once
#include <cstdint>
#include <optional>
#include <random>
#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

namespace apm {
// Call on the main thread before dispatching any parallel initializations.
inline std::uint64_t resolve_random_seed(std::optional<std::uint64_t> seed) {
    if (seed.has_value()) return *seed;
#ifdef USING_R
    // RcppArmadillo routes this draw through R's RNG, respecting set.seed().
    return static_cast<std::uint64_t>(arma::randu() * 9007199254740992.0);
#else
    return std::mt19937_64(std::random_device{}())();
#endif
}
} // namespace apm
