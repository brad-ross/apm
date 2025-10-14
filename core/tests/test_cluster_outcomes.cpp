#include <gtest/gtest.h>
#include <armadillo>
#include <limits>

#include "cluster_outcomes.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "utils.h"

TEST(ClusterOutcomesTest, CompOutcomeDists_SimpleDiscrete) {
    using apm::InMemoryUnbalancedPanel;

    const int T = 3;
    const int K = 4;
    const std::size_t G = 3;

    // One cohort with all outcomes observed: {0,1,2}
    apm::ObservedOutcomeIndices ooi;
    ooi.emplace_back(arma::uvec({0, 1, 2}));

    std::vector<int> unit_idx;
    std::vector<int> cohort_id;
    std::vector<int> outcome_idx;
    std::vector<double> y;

    unit_idx.reserve(static_cast<std::size_t>(K * T));
    cohort_id.reserve(static_cast<std::size_t>(K * T));
    outcome_idx.reserve(static_cast<std::size_t>(K * T));
    y.reserve(static_cast<std::size_t>(K * T));

    for (int u = 0; u < K; ++u) {
        for (int t = 0; t < T; ++t) {
            unit_idx.push_back(u);
            cohort_id.push_back(0);    // single cohort 0
            outcome_idx.push_back(t);  // outcomes 0..2
            y.push_back(static_cast<double>(t)); // discrete: 0,1,2
        }
    }

    std::vector<const double*> covar_cols;     // q=0
    std::vector<const double*> auxiliary_cols; // d=0

    InMemoryUnbalancedPanel panel(
        unit_idx.data(),
        cohort_id.data(),
        outcome_idx.data(),
        y.data(),
        covar_cols,
        auxiliary_cols,
        y.size(),
        ooi,
        /*one_indexed=*/false);

    auto res = apm::test::comp_outcome_dists_test(panel, G);
    const arma::mat& cdfs = res.first;   // T x G
    const arma::uvec& counts = res.second;

    ASSERT_EQ(static_cast<arma::uword>(T), cdfs.n_rows);
    ASSERT_EQ(static_cast<arma::uword>(G), cdfs.n_cols);

    // Expected CDF rows
    arma::rowvec r0 = {1.0, 1.0, 1.0};
    arma::rowvec r1 = {0.0, 1.0, 1.0};
    arma::rowvec r2 = {0.0, 0.0, 1.0};

    EXPECT_TRUE(arma::approx_equal(cdfs.row(0), r0, "absdiff", 1e-12));
    EXPECT_TRUE(arma::approx_equal(cdfs.row(1), r1, "absdiff", 1e-12));
    EXPECT_TRUE(arma::approx_equal(cdfs.row(2), r2, "absdiff", 1e-12));

    // Counts per outcome
    EXPECT_EQ(counts(0), static_cast<arma::uword>(K));
    EXPECT_EQ(counts(1), static_cast<arma::uword>(K));
    EXPECT_EQ(counts(2), static_cast<arma::uword>(K));
}

TEST(ClusterOutcomesTest, CompOutcomeDists_IgnoresNaNsAndCountsCorrectly) {
    using apm::InMemoryUnbalancedPanel;

    const int T = 3;
    const int K = 4;
    const std::size_t G = 3;

    apm::ObservedOutcomeIndices ooi;
    ooi.emplace_back(arma::uvec({0, 1, 2}));

    std::vector<int> unit_idx;
    std::vector<int> cohort_id;
    std::vector<int> outcome_idx;
    std::vector<double> y;
    unit_idx.reserve(static_cast<std::size_t>(K * T));
    cohort_id.reserve(static_cast<std::size_t>(K * T));
    outcome_idx.reserve(static_cast<std::size_t>(K * T));
    y.reserve(static_cast<std::size_t>(K * T));

    for (int u = 0; u < K; ++u) {
        for (int t = 0; t < T; ++t) {
            unit_idx.push_back(u);
            cohort_id.push_back(0);
            outcome_idx.push_back(t);
            double val = static_cast<double>(t);
            if (t == 1 && u == 0) val = std::numeric_limits<double>::quiet_NaN(); // one missing for outcome 1
            y.push_back(val);
        }
    }

    std::vector<const double*> covar_cols;
    std::vector<const double*> auxiliary_cols;
    InMemoryUnbalancedPanel panel(
        unit_idx.data(),
        cohort_id.data(),
        outcome_idx.data(),
        y.data(),
        covar_cols,
        auxiliary_cols,
        y.size(),
        ooi,
        /*one_indexed=*/false);

    auto res = apm::test::comp_outcome_dists_test(panel, G);
    const arma::mat& cdfs = res.first;
    const arma::uvec& counts = res.second;

    arma::rowvec r0 = {1.0, 1.0, 1.0};
    arma::rowvec r1 = {0.0, 1.0, 1.0};
    arma::rowvec r2 = {0.0, 0.0, 1.0};

    EXPECT_TRUE(arma::approx_equal(cdfs.row(0), r0, "absdiff", 1e-12));
    EXPECT_TRUE(arma::approx_equal(cdfs.row(1), r1, "absdiff", 1e-12));
    EXPECT_TRUE(arma::approx_equal(cdfs.row(2), r2, "absdiff", 1e-12));

    EXPECT_EQ(counts(0), static_cast<arma::uword>(K));
    EXPECT_EQ(counts(1), static_cast<arma::uword>(K - 1)); // one NaN ignored
    EXPECT_EQ(counts(2), static_cast<arma::uword>(K));
}


