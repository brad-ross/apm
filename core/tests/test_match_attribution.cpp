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

TEST(MatchAttribution, HandlesOutcomeGroupsWithWeights) {
    ObservedOutcomeIndices ooi(2);
    ooi[0] = arma::uvec({0, 1});
    ooi[1] = arma::uvec({1, 2});

    arma::mat Y = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    std::vector<OutcomeMeanSufficientStatistics> stats(2);
    stats[0].cohort_pop_share = 0.4;
    stats[1].cohort_pop_share = 0.6;
    std::vector<CohortAuxiliaryDataMeans> eta;

    arma::uvec group_a({0, 1});
    arma::uvec group_b({1, 2});
    arma::vec outcome_weights({1.0, 2.0, 3.0});

    auto fn = get_fgw_bipartite_match_outcome_diff_params_fn(group_a, group_b, ooi, outcome_weights);
    arma::vec result = fn(Y, stats, eta);

    // Cohort-level contributions:
    // Cohort 0 (pop share 0.4) observes outcomes {0,1}
    //   Sum weights*values group A = 1*1 + 2*2 = 5, weight sum = 3 -> avg = 5/3
    //   Sum weights*values group B = 2*2 = 4, weight sum = 2 -> avg = 2
    // Cohort 1 (pop share 0.6) observes outcomes {1,2}
    //   Group A uses only outcome 1: sum = 2*5 = 10, weight sum = 2 -> avg = 5
    //   Group B uses outcomes {1,2}: sum = 2*5 + 3*6 = 28, weight sum = 5 -> avg = 28/5
    const double obs_group_a = (0.4 * (5.0 / 3.0) + 0.6 * 5.0) / (0.4 + 0.6);
    const double obs_group_b = (0.4 * 2.0 + 0.6 * (28.0 / 5.0)) / (0.4 + 0.6);

    const double pop_group_a_num =
        stats[0].cohort_pop_share * (outcome_weights[0] * Y(0, 0) + outcome_weights[1] * Y(0, 1)) +
        stats[1].cohort_pop_share * (outcome_weights[0] * Y(1, 0) + outcome_weights[1] * Y(1, 1));
    const double pop_group_a_den = (stats[0].cohort_pop_share + stats[1].cohort_pop_share) *
                                   (outcome_weights[0] + outcome_weights[1]);
    const double pop_group_a = pop_group_a_num / pop_group_a_den;

    const double pop_group_b_num =
        stats[0].cohort_pop_share * (outcome_weights[1] * Y(0, 1) + outcome_weights[2] * Y(0, 2)) +
        stats[1].cohort_pop_share * (outcome_weights[1] * Y(1, 1) + outcome_weights[2] * Y(1, 2));
    const double pop_group_b_den = (stats[0].cohort_pop_share + stats[1].cohort_pop_share) *
                                   (outcome_weights[1] + outcome_weights[2]);
    const double pop_group_b = pop_group_b_num / pop_group_b_den;

    const double expected_ratio = (pop_group_a - pop_group_b) / (obs_group_a - obs_group_b);

    EXPECT_NEAR(result[0], expected_ratio, 1e-10);
    EXPECT_NEAR(result[1], 1.0 - expected_ratio, 1e-10);
}

