#include <gtest/gtest.h>
#include <vector>
#include <set>
#include <armadillo>
#include "summarize_identification.h"

// Helpers to build simple ObservedOutcomeIndices inputs
static std::vector<arma::uvec> make_ooi(std::initializer_list<std::initializer_list<arma::uword>> lists) {
    std::vector<arma::uvec> out;
    out.reserve(lists.size());
    for (const auto &lst : lists) {
        arma::uvec v(lst.size());
        arma::uword i = 0;
        for (arma::uword x : lst) v(i++) = x;
        out.push_back(v);
    }
    return out;
}

TEST(SummarizeIdentificationTest, SinglePanel_NoMerges_FinalIteration) {
    // Equivalent to R test: sizes <- c(50L, 40L), ooi <- list(c(1L,3L), c(2L,3L)), r=2
    // Note: C++ is 0-indexed for outcome indices; R used 1-based.
    std::vector<arma::uvec> ooi = make_ooi({{0,2}, {1,2}});
    arma::uvec sizes = {50u, 40u};
    const std::size_t r = 2;

    apm::IdentificationSummary out = apm::summarize_identification(ooi, sizes, r);

    EXPECT_GE(out.largest_super_cohort_size, static_cast<std::size_t>(0));
    EXPECT_GE(out.min_cohort_size_in_largest_super, static_cast<std::size_t>(0));
    EXPECT_GE(out.largest_super_cohort_share, 0.0);
    EXPECT_LE(out.largest_super_cohort_share, 1.0);
    EXPECT_GE(out.num_outcomes_in_largest_super_cohort, static_cast<std::size_t>(0));
    // With r=2 and these cohorts, no merges; only initial state is present
    EXPECT_EQ(out.num_o3_iterations, static_cast<std::size_t>(1));
    EXPECT_EQ(out.largest_super_cohort_size, static_cast<std::size_t>(50));
    EXPECT_EQ(out.min_cohort_size_in_largest_super, static_cast<std::size_t>(50));
    EXPECT_NEAR(out.largest_super_cohort_share, 50.0/90.0, 1e-12);
    // Largest super cohort is the larger single cohort {0,2}; union outcomes count is 2
    EXPECT_EQ(out.num_outcomes_in_largest_super_cohort, static_cast<std::size_t>(2));
    EXPECT_DOUBLE_EQ(out.total_outcome_weight_in_largest_super_cohort, 2.0);
    EXPECT_NEAR(out.share_outcomes_in_largest_super_cohort, 2.0/3.0, 1e-12);
    EXPECT_NEAR(out.share_outcome_weight_in_largest_super_cohort, 2.0/3.0, 1e-12);
}

TEST(SummarizeIdentificationTest, ManyPanels_ListInputs_NoMerges) {
    // R analog: ooi_list <- list(list(c(1L), c(2L,3L))); size_list <- list(c(10L,25L)); r=2
    std::vector<std::vector<arma::uvec>> ooi_list = { make_ooi({{0}, {1,2}}) };
    std::vector<arma::uvec> size_list = { arma::uvec{10u, 25u} };
    const std::size_t r = 2;

    std::vector<apm::IdentificationSummary> res = apm::summarize_identification(ooi_list, size_list, r);
    ASSERT_EQ(res.size(), static_cast<std::size_t>(1));
    const auto &el = res[0];
    EXPECT_EQ(el.num_o3_iterations, static_cast<std::size_t>(1));
    EXPECT_EQ(el.largest_super_cohort_size, static_cast<std::size_t>(25));
    EXPECT_EQ(el.min_cohort_size_in_largest_super, static_cast<std::size_t>(25));
    EXPECT_NEAR(el.largest_super_cohort_share, 25.0/35.0, 1e-12);
    // Largest super cohort is the larger single cohort {1,2}; union outcomes count is 2
    EXPECT_EQ(el.num_outcomes_in_largest_super_cohort, static_cast<std::size_t>(2));
    EXPECT_DOUBLE_EQ(el.total_outcome_weight_in_largest_super_cohort, 2.0);
    EXPECT_NEAR(el.share_outcomes_in_largest_super_cohort, 2.0/3.0, 1e-12);
    EXPECT_NEAR(el.share_outcome_weight_in_largest_super_cohort, 2.0/3.0, 1e-12);
}

// Move aligned_factors_identified tests here from test_apm_core.cpp
TEST(SummarizeIdentificationTest, AlignedFactorsIdentified_StaircasePattern) {
    std::vector<arma::uvec> ooi = make_ooi({{0,1,2}, {1,2,3}, {2,3,4}});
    unsigned int r = 2;
    ASSERT_TRUE(apm::aligned_factors_identified(ooi, r));
}

TEST(SummarizeIdentificationTest, AlignedFactorsIdentified_NonContiguous_TwoSteps) {
    std::vector<arma::uvec> ooi = make_ooi({{0,1,2}, {1,2,3}, {0,3,4}});
    unsigned int r = 2;
    ASSERT_TRUE(apm::aligned_factors_identified(ooi, r));
}

TEST(SummarizeIdentificationTest, AlignedFactorsNotIdentified_OneIteration) {
    std::vector<arma::uvec> ooi = make_ooi({{0,1}, {1,2,3}, {2,3,4}});
    unsigned int r = 2;
    ASSERT_FALSE(apm::aligned_factors_identified(ooi, r));
}

TEST(SummarizeIdentificationTest, AlignedFactorsNotIdentified_NoMerges) {
    std::vector<arma::uvec> ooi = make_ooi({{0,1}, {1,2}, {2,3,4}});
    unsigned int r = 2;
    ASSERT_FALSE(apm::aligned_factors_identified(ooi, r));
}

TEST(SummarizeIdentificationTest, CountOutcomesWithRankOverlap_BasicScenario) {
    std::vector<arma::uvec> ooi = make_ooi({{0,1,2}, {1,2,3}, {4}});
    const std::size_t rank = 2;

    arma::vec counts = apm::count_outcomes_with_rank_overlap_per_cohort(ooi, rank);
    ASSERT_EQ(counts.n_elem, static_cast<arma::uword>(3));
    EXPECT_DOUBLE_EQ(counts(0), 4.0);
    EXPECT_DOUBLE_EQ(counts(1), 4.0);
    // The focal cohort always contributes its own outcomes even when it has
    // fewer than `rank` observed outcomes.
    EXPECT_DOUBLE_EQ(counts(2), 1.0);
}

TEST(SummarizeIdentificationTest, CountOutcomesWithRankOverlap_RankZeroIncludesAll) {
    std::vector<arma::uvec> ooi = make_ooi({{0}, {1,2}, {2,3}});
    const std::size_t rank = 0;

    arma::vec counts = apm::count_outcomes_with_rank_overlap_per_cohort(ooi, rank);
    ASSERT_EQ(counts.n_elem, static_cast<arma::uword>(3));
    EXPECT_DOUBLE_EQ(counts(0), 4.0);
    EXPECT_DOUBLE_EQ(counts(1), 4.0);
    EXPECT_DOUBLE_EQ(counts(2), 4.0);
}

TEST(SummarizeIdentificationTest, WeightedSummariesAndCounts) {
    std::vector<arma::uvec> ooi = make_ooi({{0,2}, {1,2}});
    arma::uvec sizes = {50u, 40u};
    arma::vec weights = {1.0, 5.0, 10.0};
    const std::size_t r = 2;

    apm::IdentificationSummary out = apm::summarize_identification(ooi, sizes, r, -1, weights);
    EXPECT_EQ(out.num_outcomes_in_largest_super_cohort, static_cast<std::size_t>(2));
    EXPECT_DOUBLE_EQ(out.total_outcome_weight_in_largest_super_cohort, 11.0);
    EXPECT_NEAR(out.share_outcomes_in_largest_super_cohort, 2.0/3.0, 1e-12);
    EXPECT_NEAR(out.share_outcome_weight_in_largest_super_cohort, 11.0/16.0, 1e-12);

    arma::vec weighted_counts = apm::count_outcomes_with_rank_overlap_per_cohort(ooi, 1, weights);
    ASSERT_EQ(weighted_counts.n_elem, static_cast<arma::uword>(2));
    EXPECT_DOUBLE_EQ(weighted_counts(0), 16.0);
    EXPECT_DOUBLE_EQ(weighted_counts(1), 16.0);
}