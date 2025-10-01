#include <gtest/gtest.h>
#include <vector>
#include <unordered_map>
#include <stdexcept>

#include "est_outcome_means.h"
#include "linear_algebra_utils.h"
#include "test_helpers.h"

// Tests focused on functions in est_outcome_means.{h,cpp} that use *Estimates types

TEST(EstOutcomeMeanTest, EstimateMeans_EstimatesWithBootstrap_AllComponents) {
    auto data = setup_estimation_test_data();

    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0] + data.X_c_vec[0] * data.a + data.g_0).t();
    true_m.row(1) = (data.G * data.l_c[1] + data.X_c_vec[1] * data.a + data.g_0).t();

    apm::FactorModelParameters params_point(data.G, data.g_0, data.a);

    const std::size_t B = 2;
    auto suff_stats_point = make_suff_stats_vec(true_m, data.observed_outcome_indices, data.X_c_vec);
    auto suff_est_vec = duplicate_bootstrap_suff(suff_stats_point, B);

    // Duplicate bootstrap replicates identical to the point parameters
    apm::FactorModelEstimates param_estimates = duplicate_bootstrap(params_point, B);

    apm::OutcomeMeansEstimates out = apm::estimate_outcome_means_across_cohorts(
        param_estimates, data.observed_outcome_indices, suff_est_vec);

    ASSERT_TRUE(arma::approx_equal(out.mean_outcomes, true_m, "absdiff", 1e-9));
    ASSERT_EQ(out.bootstrap_replicates.size(), B);
    for (std::size_t b = 0; b < B; ++b) {
        ASSERT_TRUE(arma::approx_equal(out.bootstrap_replicates[b], out.mean_outcomes, "absdiff", 1e-9));
    }
}

TEST(EstOutcomeMeanTest, EstimateMeans_EstimatesWithBootstrap_MismatchThrows) {
    auto data = setup_estimation_test_data();

    std::vector<apm::FactorModelParameters> param_boot;
    param_boot.emplace_back(data.G, data.g_0 + 0.1, data.a);
    param_boot.emplace_back(data.G, data.g_0 + 0.2, data.a);
    apm::FactorModelEstimates param_estimates(apm::FactorModelParameters(data.G, data.g_0, data.a),
                                              std::move(param_boot));

    std::vector<apm::OutcomeMeanSuffStatEstimates> suff_est_vec;
    suff_est_vec.reserve(data.C);
    for (arma::uword c = 0; c < data.C; ++c) {
        arma::mat X_c = data.X_c_vec[c];
        arma::vec m_c(data.observed_outcome_indices[c].n_elem, arma::fill::zeros);
        apm::OutcomeMeanSufficientStatistics stats_point(m_c, X_c);
        std::vector<apm::OutcomeMeanSufficientStatistics> boots;
        if (c == 0) {
            boots.push_back(stats_point);
        } else {
            boots.push_back(stats_point);
            boots.push_back(stats_point);
        }
        suff_est_vec.emplace_back(stats_point, std::move(boots));
    }

    EXPECT_THROW(
        (void)apm::estimate_outcome_means_across_cohorts(param_estimates,
                                                         data.observed_outcome_indices,
                                                         suff_est_vec),
        std::invalid_argument);
}


TEST(EstOutcomeMeanTest, EstimateMeans_MapOverSpecs_Succeeds) {
    auto data = setup_estimation_test_data();

    arma::mat true_m(data.C, data.T);
    true_m.row(0) = (data.G * data.l_c[0] + data.X_c_vec[0] * data.a + data.g_0).t();
    true_m.row(1) = (data.G * data.l_c[1] + data.X_c_vec[1] * data.a + data.g_0).t();

    apm::FactorModelParameters params_point(data.G, data.g_0, data.a);

    const std::size_t B = 2;
    auto suff_stats_point = make_suff_stats_vec(true_m, data.observed_outcome_indices, data.X_c_vec);
    auto suff_est_vec = duplicate_bootstrap_suff(suff_stats_point, B);

    apm::FactorModelEstimates param_estimates = duplicate_bootstrap(params_point, B);

    std::unordered_map<std::string, apm::FactorModelEstimates> spec_map;
    spec_map.emplace("specA", param_estimates);
    spec_map.emplace("specB", param_estimates);

    auto out_map = apm::estimate_outcome_means_across_cohorts(
        spec_map, data.observed_outcome_indices, suff_est_vec);

    ASSERT_EQ(out_map.size(), 2U);
    for (const auto& kv : out_map) {
        const auto& out = kv.second;
        ASSERT_TRUE(arma::approx_equal(out.mean_outcomes, true_m, "absdiff", 1e-9));
        ASSERT_EQ(out.bootstrap_replicates.size(), B);
        for (std::size_t b = 0; b < B; ++b) {
            ASSERT_TRUE(arma::approx_equal(out.bootstrap_replicates[b], out.mean_outcomes, "absdiff", 1e-9));
        }
    }
}


