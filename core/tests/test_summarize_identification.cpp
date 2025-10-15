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
    // With r=2 and these cohorts, no merges; only initial state is present
    EXPECT_EQ(out.num_o3_iterations, static_cast<std::size_t>(1));
    EXPECT_EQ(out.largest_super_cohort_size, static_cast<std::size_t>(50));
    EXPECT_EQ(out.min_cohort_size_in_largest_super, static_cast<std::size_t>(50));
    EXPECT_NEAR(out.largest_super_cohort_share, 50.0/90.0, 1e-12);
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


