#include <gtest/gtest.h>
#include <armadillo>
#include <unordered_map>
#include <vector>
#include <string>

#include "target_params/est_target_params.h"
#include "target_params/est_outcome_mean_err_metrics.h"

using namespace apm;

// Helper: build OutcomeMeansEstimates with B bootstrap draws and specified
// values for selected (cohort, outcome) cells.
static OutcomeMeansEstimates make_ome(std::size_t C,
                                      std::size_t T,
                                      const std::vector<std::tuple<std::size_t,std::size_t,std::vector<double>>>& cell_draws,
                                      std::size_t B) {
    arma::mat point(C, T, arma::fill::zeros);
    std::vector<arma::mat> boots;
    boots.reserve(B);
    for (std::size_t b = 0; b < B; ++b) {
        boots.emplace_back(C, T, arma::fill::zeros);
    }
    for (const auto& tup : cell_draws) {
        std::size_t c = std::get<0>(tup);
        std::size_t t = std::get<1>(tup);
        const std::vector<double>& draws = std::get<2>(tup);
        EXPECT_EQ(draws.size(), B);
        for (std::size_t b = 0; b < B; ++b) {
            boots[b](static_cast<arma::uword>(c), static_cast<arma::uword>(t)) = draws[b];
        }
    }
    return OutcomeMeansEstimates(std::move(point), std::move(boots));
}

TEST(EstMaskedOutcomeMeanErrMetrics, ComputesBiasSeRmseForMaskedPairs) {
    const std::size_t C = 2, T = 4, B = 3;

    // Mask: cohort 1 -> outcome 2; cohort 0 -> outcome 3 (absolute indices)
    arma::uvec mask_c1(1); mask_c1(0) = 2;
    arma::uvec mask_c0(1); mask_c0(0) = 3;
    CohortOutcomeMask mask; mask.emplace(1, mask_c1); mask.emplace(0, mask_c0);

    // Truths in the same relative order as mask indices
    OutcomeMeanSufficientStatistics s1(arma::vec({10.0}), std::nullopt); // (1,2)
    OutcomeMeanSufficientStatistics s0(arma::vec({ 4.0}), std::nullopt); // (0,3)
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_means;
    masked_means.emplace(1, s1);
    masked_means.emplace(0, s0);

    // Bootstrap draws: (1,2) -> [11,9,10]; (0,3) -> [5,5,3]
    std::vector<std::tuple<std::size_t,std::size_t,std::vector<double>>> defs = {
        {1, 2, {11.0, 9.0, 10.0}},
        {0, 3, { 5.0, 5.0,  3.0}}
    };
    OutcomeMeansEstimates omeA = make_ome(C, T, defs, B);
    OutcomeMeansEstimates omeB = make_ome(C, T, defs, B);
    std::unordered_map<std::string, OutcomeMeansEstimates> ome_map;
    ome_map.emplace("specA", omeA);
    ome_map.emplace("specB", omeB);

    OutcomeMeanSufficientStatistics ss0(arma::vec(0), std::nullopt); ss0.cohort_pop_share = 0.4;
    OutcomeMeanSufficientStatistics ss1(arma::vec(0), std::nullopt); ss1.cohort_pop_share = 0.6;
    std::vector<OutcomeMeanSuffStatEstimates> cohort_stats { OutcomeMeanSuffStatEstimates(ss0), OutcomeMeanSuffStatEstimates(ss1) };

    TargetParamComponents comps;
    comps.outcome_means_by_spec = std::move(ome_map);
    comps.cohort_outcome_mean_ests = std::move(cohort_stats);
    comps.masked_cohort_outcome_means = std::move(masked_means);
    comps.cohort_outcome_mask = mask;

    auto out = est_masked_outcome_mean_err_metrics(comps);

    // (1,2): truth=10, draws=[11,9,10] => bias=0, se=1, rmse=sqrt(2/3)
    {
        auto it = out.find({1,2});
        ASSERT_TRUE(it != out.end());
        const auto& m = it->second;
        EXPECT_NEAR(m.cohort_pop_share, 0.6, 1e-12);
        for (const auto& sname : {std::string("specA"), std::string("specB")}) {
            EXPECT_NEAR(m.bias_by_spec.at(sname), 0.0, 1e-12);
            EXPECT_NEAR(m.se_by_spec.at(sname), 1.0, 1e-12);
            EXPECT_NEAR(m.rmse_by_spec.at(sname), std::sqrt(2.0/3.0), 1e-12);
        }
    }
    // (0,3): truth=4, draws=[5,5,3] => bias=1/3, se=sqrt(17/18), rmse=1
    {
        auto it = out.find({0,3});
        ASSERT_TRUE(it != out.end());
        const auto& m = it->second;
        EXPECT_NEAR(m.cohort_pop_share, 0.4, 1e-12);
        for (const auto& sname : {std::string("specA"), std::string("specB")}) {
            EXPECT_NEAR(m.bias_by_spec.at(sname), 1.0/3.0, 1e-12);
            EXPECT_NEAR(m.se_by_spec.at(sname), std::sqrt(17.0/18.0), 1e-12);
            EXPECT_NEAR(m.rmse_by_spec.at(sname), 1.0, 1e-12);
        }
    }
}

TEST(EstMaskedOutcomeMeanErrMetrics, ThrowsWhenSpecMissingBootstrap) {
    const std::size_t C = 1, T = 2;
    arma::uvec mask_c0(1); mask_c0(0) = 1;
    CohortOutcomeMask mask; mask.emplace(0, mask_c0);

    OutcomeMeanSufficientStatistics sm(arma::vec({2.0}), std::nullopt);
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_means; masked_means.emplace(0, sm);

    arma::mat point(C, T, arma::fill::zeros);
    OutcomeMeansEstimates omeA(point, {}); // no bootstrap
    std::unordered_map<std::string, OutcomeMeansEstimates> ome_map; ome_map.emplace("specA", omeA);

    OutcomeMeanSufficientStatistics ss0(arma::vec(0), std::nullopt); ss0.cohort_pop_share = 1.0;
    std::vector<OutcomeMeanSuffStatEstimates> cohort_stats { OutcomeMeanSuffStatEstimates(ss0) };

    TargetParamComponents comps;
    comps.outcome_means_by_spec = std::move(ome_map);
    comps.cohort_outcome_mean_ests = std::move(cohort_stats);
    comps.masked_cohort_outcome_means = std::move(masked_means);
    comps.cohort_outcome_mask = mask;

    EXPECT_THROW(est_masked_outcome_mean_err_metrics(comps), std::invalid_argument);
}

TEST(EstMaskedOutcomeMeanErrMetrics, ThrowsOnMaskLengthMismatch) {
    const std::size_t C = 1, T = 3, B = 2;

    arma::uvec mask_c0(2); mask_c0(0) = 0; mask_c0(1) = 2; // length 2
    CohortOutcomeMask mask; mask.emplace(0, mask_c0);

    OutcomeMeanSufficientStatistics sm(arma::vec({3.0}), std::nullopt); // length 1 (mismatch)
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_means; masked_means.emplace(0, sm);

    std::vector<std::tuple<std::size_t,std::size_t,std::vector<double>>> defs = {
        {0,0,{3.0,3.0}}, {0,2,{3.0,3.0}}
    };
    OutcomeMeansEstimates ome = make_ome(C, T, defs, B);
    std::unordered_map<std::string, OutcomeMeansEstimates> ome_map; ome_map.emplace("spec", ome);

    OutcomeMeanSufficientStatistics ss0(arma::vec(0), std::nullopt); ss0.cohort_pop_share = 1.0;
    std::vector<OutcomeMeanSuffStatEstimates> cohort_stats { OutcomeMeanSuffStatEstimates(ss0) };

    TargetParamComponents comps;
    comps.outcome_means_by_spec = std::move(ome_map);
    comps.cohort_outcome_mean_ests = std::move(cohort_stats);
    comps.masked_cohort_outcome_means = std::move(masked_means);
    comps.cohort_outcome_mask = mask;

    EXPECT_THROW(est_masked_outcome_mean_err_metrics(comps), std::invalid_argument);
}

TEST(EstMaskedOutcomeMeanErrMetrics, ThrowsOnOutcomeIndexOutOfRange) {
    const std::size_t C = 1, T = 2, B = 2;

    arma::uvec mask_c0(1); mask_c0(0) = 10; // out of range
    CohortOutcomeMask mask; mask.emplace(0, mask_c0);

    OutcomeMeanSufficientStatistics sm(arma::vec({1.0}), std::nullopt);
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_means; masked_means.emplace(0, sm);

    std::vector<std::tuple<std::size_t,std::size_t,std::vector<double>>> defs; // none required
    OutcomeMeansEstimates ome = make_ome(C, T, defs, B);
    std::unordered_map<std::string, OutcomeMeansEstimates> ome_map; ome_map.emplace("spec", ome);

    OutcomeMeanSufficientStatistics ss0(arma::vec(0), std::nullopt); ss0.cohort_pop_share = 1.0;
    std::vector<OutcomeMeanSuffStatEstimates> cohort_stats { OutcomeMeanSuffStatEstimates(ss0) };

    TargetParamComponents comps;
    comps.outcome_means_by_spec = std::move(ome_map);
    comps.cohort_outcome_mean_ests = std::move(cohort_stats);
    comps.masked_cohort_outcome_means = std::move(masked_means);
    comps.cohort_outcome_mask = mask;

    EXPECT_THROW(est_masked_outcome_mean_err_metrics(comps), std::invalid_argument);
}


