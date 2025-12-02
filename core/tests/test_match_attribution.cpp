#include <gtest/gtest.h>

#include "target_params/match_attribution.h"

using namespace apm;

TEST(MatchAttribution, FactoryComputesExpectedRatio) {
    ObservedOutcomeIndices ooi(2);
    ooi[0] = arma::uvec({0, 1});
    ooi[1] = arma::uvec({1, 2});

    auto fn = get_fgw_bipartite_match_outcome_diff_params_fn(0, 2, ooi);

    arma::mat Y = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    std::vector<OutcomeMeanSufficientStatistics> stats(2);
    stats[0].cohort_pop_share = 0.4;
    stats[1].cohort_pop_share = 0.6;
    std::vector<CohortAuxiliaryDataMeans> eta;

    arma::vec result = fn(Y, stats, eta);
    ASSERT_EQ(result.n_elem, 2u);

    // obs_diff = 1 - 6 = -5; pop_diff = 2.8 - 4.8 = -2 -> ratio = 0.4
    EXPECT_NEAR(result[0], 0.4, 1e-10);
    EXPECT_NEAR(result[1], 0.6, 1e-10);
    EXPECT_NEAR(result[0] + result[1], 1.0, 1e-10);
}

TEST(MatchAttribution, ThrowsOnUnobservedOutcome) {
    ObservedOutcomeIndices ooi(1);
    ooi[0] = arma::uvec({0, 1});
    EXPECT_THROW(get_fgw_bipartite_match_outcome_diff_params_fn(0, 5, ooi), std::invalid_argument);
}

