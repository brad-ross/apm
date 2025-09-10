#include <gtest/gtest.h>
#include <vector>
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

TEST(EstOutcomeMeanTest, AggregateFactorModelParams_WithBootstrap_AllParams) {
    // True factors (T x r)
    arma::mat true_factors = {
        {0.1, 0.6},
        {0.2, 0.7},
        {0.3, 0.8},
        {0.4, 0.9},
        {0.5, 1.0}
    };

    std::vector<arma::uvec> observed_outcome_indices = {
        {0, 1, 2},
        {1, 2, 3},
        {2, 3, 4}
    };

    const size_t C = observed_outcome_indices.size();
    const arma::uword r = true_factors.n_cols;
    const arma::uword q = 2;
    std::vector<arma::mat> rotation_matrices = generate_rotation_matrices(C, r);

    arma::vec g0_true = {0.10, 0.20, 0.30, 0.40, 0.50};

    // Cohort weights for point and bootstrap (use same per-draw)
    arma::vec cohort_weights = arma::vec(C, arma::fill::ones);

    // Build cohort-specific parameter estimates with B identical bootstrap replicates
    const std::size_t B = 2;
    std::vector<apm::FactorModelEstimates> cohort_estimates;
    cohort_estimates.reserve(C);

    std::vector<arma::vec> a_c_vec;
    a_c_vec.reserve(C);

    for (size_t c = 0; c < C; ++c) {
        arma::mat G_c = true_factors.rows(observed_outcome_indices[c]) * rotation_matrices[c];
        arma::vec g0_c = g0_true.elem(observed_outcome_indices[c]);
        arma::vec a_c(q);
        a_c(0) = 0.5 + 0.1 * static_cast<double>(c);
        a_c(1) = 1.0 + 0.2 * static_cast<double>(c);
        a_c_vec.push_back(a_c);

        apm::FactorModelParameters point(G_c, g0_c, a_c);
        std::vector<apm::FactorModelParameters> boot(B, point); // identical replicates
        cohort_estimates.emplace_back(std::move(point), std::move(boot));
    }

    // Bootstrap cohort weights: B vectors of length C; use ones
    std::vector<arma::vec> bootstrap_cohort_weights(B, arma::vec(C, arma::fill::ones));

    apm::FactorModelEstimates agg = apm::aggregate_cohort_specific_factor_model_params(
        cohort_estimates, observed_outcome_indices, cohort_weights, bootstrap_cohort_weights);

    // Check aligned G subspace matches true_factors
    arma::mat proj_aligned = apm::internal::projection_matrix(agg.parameter_estimates.G);
    arma::mat proj_true = apm::internal::projection_matrix(true_factors);
    ASSERT_TRUE(arma::approx_equal(proj_aligned, proj_true, "absdiff", 1e-9));

    // Check g_0 equals the true g0_true (since each cohort uses same per-t value)
    ASSERT_TRUE(agg.parameter_estimates.g_0.has_value());
    ASSERT_EQ(agg.parameter_estimates.g_0->n_elem, g0_true.n_elem);
    ASSERT_TRUE(arma::approx_equal(*(agg.parameter_estimates.g_0), g0_true, "absdiff", 1e-12));

    // Weighted average for a with equal weights equals the simple mean of a_c across cohorts
    ASSERT_TRUE(agg.parameter_estimates.a.has_value());
    arma::vec expected_a(q, arma::fill::zeros);
    for (size_t c = 0; c < C; ++c) {
        expected_a += a_c_vec[c] * (1.0 / static_cast<double>(C));
    }
    ASSERT_TRUE(arma::approx_equal(*(agg.parameter_estimates.a), expected_a, "absdiff", 1e-12));

    // Bootstrap replicates should be identical to point since inputs were identical per draw
    ASSERT_TRUE(agg.has_bootstrap_replicates());
    ASSERT_EQ(agg.bootstrap_replicates.size(), B);
    for (std::size_t b = 0; b < B; ++b) {
        const auto& rep = agg.bootstrap_replicates[b];
        arma::mat proj_rep = apm::internal::projection_matrix(rep.G);
        ASSERT_TRUE(arma::approx_equal(proj_rep, proj_true, "absdiff", 1e-9));
        ASSERT_TRUE(rep.g_0.has_value());
        ASSERT_TRUE(arma::approx_equal(*(rep.g_0), g0_true, "absdiff", 1e-12));
        ASSERT_TRUE(rep.a.has_value());
        ASSERT_TRUE(arma::approx_equal(*(rep.a), expected_a, "absdiff", 1e-12));
    }
}


