#include <gtest/gtest.h>
#include <armadillo>
#include <limits>
#include <cstdint>
#include <optional>

#include "outcome_clustering/cluster_outcomes.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "utils.h"
// no direct mlpack random include needed; API now accepts seed

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



TEST(ClusterOutcomesTest, KMeansAssignments_TwoClusters_SingleAndRange) {
    using apm::InMemoryUnbalancedPanel;

    const int T = 4;           // outcomes 0..3
    const int K = 5;           // units per outcome
    const std::size_t G = 3;   // grid size

    apm::ObservedOutcomeIndices ooi;
    ooi.emplace_back(arma::uvec({0, 1, 2, 3}));

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
            // outcomes 0,1 near 0.0; outcomes 2,3 near 10.0
            double val = (t <= 1) ? 0.0 : 10.0;
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

    const uint64_t seed = 123ULL;

    // single-k
    arma::uvec labels_single = apm::comp_outcome_clustering(panel, G, static_cast<std::size_t>(2), std::optional<uint64_t>(seed));
    EXPECT_EQ(labels_single.n_elem, static_cast<arma::uword>(T));
    EXPECT_TRUE(labels_single(0) == labels_single(1));
    EXPECT_TRUE(labels_single(2) == labels_single(3));
    EXPECT_NE(labels_single(0), labels_single(2));

    // range with the same k
    auto maps = apm::comp_outcome_clusterings(panel, G, static_cast<std::size_t>(2), static_cast<std::size_t>(2), seed);
    ASSERT_EQ(maps.size(), 1u);
    const arma::uvec& labels_range = maps[0];
    EXPECT_TRUE(arma::all(labels_single == labels_range));
}

TEST(ClusterOutcomesTest, KMeansAssignments_KRangeAndEmptyClusters) {
    using apm::InMemoryUnbalancedPanel;

    const int T = 4;
    const int K = 5;
    const std::size_t G = 3;

    apm::ObservedOutcomeIndices ooi;
    ooi.emplace_back(arma::uvec({0, 1, 2, 3}));

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
            double val = (t <= 1) ? 0.0 : 10.0;
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

    const uint64_t seed = 123ULL;
    auto maps = apm::comp_outcome_clusterings(panel, G, static_cast<std::size_t>(1), static_cast<std::size_t>(3), seed);
    ASSERT_EQ(maps.size(), 3u);

    const arma::uvec& k1 = maps[0];
    const arma::uvec& k2 = maps[1];
    const arma::uvec& k3 = maps[2];

    // k=1: all same label
    EXPECT_EQ(arma::unique(k1).eval().n_elem, 1u);

    // k=3: still only 2 groups present due to identical feature rows per group
    EXPECT_EQ(arma::unique(k3).eval().n_elem, 2u);
    EXPECT_TRUE(k3(0) == k3(1));
    EXPECT_TRUE(k3(2) == k3(3));
    EXPECT_NE(k3(0), k3(2));
}

TEST(ClusterOutcomesTest, KMeansConsistency_SingleVsRangeSameSeed) {
    using apm::InMemoryUnbalancedPanel;

    const int T = 4;
    const int K = 5;
    const std::size_t G = 3;

    apm::ObservedOutcomeIndices ooi;
    ooi.emplace_back(arma::uvec({0, 1, 2, 3}));

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
            double val = (t <= 1) ? 0.0 : 10.0;
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

    const uint64_t seed = 777ULL;

    // Single-k results
    arma::uvec k2_single = apm::comp_outcome_clustering(panel, G, static_cast<std::size_t>(2), std::optional<uint64_t>(seed));
    arma::uvec k3_single = apm::comp_outcome_clustering(panel, G, static_cast<std::size_t>(3), std::optional<uint64_t>(seed));

    // Range results
    auto maps = apm::comp_outcome_clusterings(panel, G, static_cast<std::size_t>(2), static_cast<std::size_t>(3), seed);
    ASSERT_EQ(maps.size(), 2u);
    const arma::uvec& k2_range = maps[0];
    const arma::uvec& k3_range = maps[1];

    EXPECT_TRUE(arma::all(k2_single == k2_range));
    EXPECT_TRUE(arma::all(k3_single == k3_range));
}