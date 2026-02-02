#include <gtest/gtest.h>
#include <armadillo>
#include <memory>

#include "bootstrap.h"
#include "OutcomeMeanSuffStatEstimator.h"

namespace {

// Deterministic bootstrap for tests
class TestBootstrap : public apm::WeightedBootstrap {
public:
    explicit TestBootstrap(const arma::mat& W) : apm::WeightedBootstrap(W) {}
};

} // anonymous namespace

//==============================================================================
// OutcomeSuffStatEstimator tests (means only, no covariates)
//==============================================================================

TEST(OutcomeSuffStatEstimatorTest, SingleBatch_NoBootstrap_OutcomeMeans) {
    const std::size_t T_c = 3;
    apm::OutcomeMeanSuffStatEstimator est(T_c);

    arma::mat Y = {
        {1.0, 2.0, 3.0},
        {3.0, 4.0, 5.0}
    };
    arma::uvec unit_idxs = {0, 1};

    est.add_data(unit_idxs, Y);
    apm::OutcomeMeanSuffStatEstimates out = est.estimate(/*total_units=*/2);

    arma::vec expected_mu = {2.0, 3.0, 4.0};
    ASSERT_TRUE(arma::approx_equal(out.suff_stat_estimates.observed_outcome_means, expected_mu, "absdiff", 1e-12));
    EXPECT_FALSE(out.suff_stat_estimates.has_covar_means());
    EXPECT_TRUE(out.bootstrap_replicates.empty());
}

TEST(OutcomeSuffStatEstimatorTest, SplitBatch_Invariance) {
    const std::size_t T_c = 2;
    apm::OutcomeMeanSuffStatEstimator est(T_c);

    arma::mat Y1 = {
        {1.0, 0.0},
        {1.0, 0.0}
    };
    arma::uvec idx1 = {0, 1};

    arma::mat Y2 = {
        {0.0, 2.0},
        {0.0, 2.0}
    };
    arma::uvec idx2 = {2, 3};

    est.add_data(idx1, Y1);
    est.add_data(idx2, Y2);

    apm::OutcomeMeanSuffStatEstimates out = est.estimate(/*total_units=*/4);
    arma::vec expected_mu = {0.5, 1.0};
    ASSERT_TRUE(arma::approx_equal(out.suff_stat_estimates.observed_outcome_means, expected_mu, "absdiff", 1e-12));
}

TEST(OutcomeSuffStatEstimatorTest, AddDatum_Equivalence) {
    const std::size_t T_c = 2;
    apm::OutcomeMeanSuffStatEstimator est(T_c);

    arma::mat Y = {
        {1.0, 0.0},
        {1.0, 0.0},
        {0.0, 2.0},
        {0.0, 2.0}
    };

    for (std::size_t i = 0; i < 4; ++i) {
        est.add_datum(i, Y.row(static_cast<arma::uword>(i)).t());
    }

    apm::OutcomeMeanSuffStatEstimates out = est.estimate(/*total_units=*/4);
    arma::vec expected_mu = {0.5, 1.0};
    ASSERT_TRUE(arma::approx_equal(out.suff_stat_estimates.observed_outcome_means, expected_mu, "absdiff", 1e-12));
}

TEST(OutcomeSuffStatEstimatorTest, Bootstrap_DeterministicReplicates) {
    const std::size_t T_c = 2, B = 2;

    arma::mat Y = {
        {1.0, 0.0},
        {1.0, 0.0},
        {0.0, 2.0},
        {0.0, 2.0}
    };
    arma::uvec unit_idxs = {0, 1, 2, 3};

    // Columns sum to 1; b=0 uses first two rows, b=1 uses last two
    arma::mat W(4, 2, arma::fill::zeros);
    W(0, 0) = 0.5; W(1, 0) = 0.5;
    W(2, 1) = 0.5; W(3, 1) = 0.5;
    auto boot = std::make_shared<TestBootstrap>(W);

    apm::OutcomeMeanSuffStatEstimator est(T_c, /*T=*/0, /*q=*/0, boot);
    est.add_data(unit_idxs, Y);
    apm::OutcomeMeanSuffStatEstimates out = est.estimate(/*total_units=*/4);

    ASSERT_EQ(out.bootstrap_replicates.size(), B);

    arma::vec expected_point = {0.5, 1.0};
    ASSERT_TRUE(arma::approx_equal(out.suff_stat_estimates.observed_outcome_means, expected_point, "absdiff", 1e-12));

    arma::vec mu_b0 = {1.0, 0.0};
    arma::vec mu_b1 = {0.0, 2.0};
    ASSERT_TRUE(arma::approx_equal(out.bootstrap_replicates[0].observed_outcome_means, mu_b0, "absdiff", 1e-12));
    ASSERT_TRUE(arma::approx_equal(out.bootstrap_replicates[1].observed_outcome_means, mu_b1, "absdiff", 1e-12));
}

//==============================================================================
// OutcomeSuffStatEstimator tests (with covariates T x q, T may differ from T_c)
//==============================================================================

TEST(OutcomeSuffStatEstimatorTest, CovariateMeans_ComputedAndDimensions) {
    const std::size_t T_c = 2, T = 3, q = 2;
    apm::OutcomeMeanSuffStatEstimator est(T_c, T, q);

    arma::mat Y = {
        {1.0, 2.0},
        {3.0, 4.0}
    }; // N=2
    arma::uvec unit_idxs = {0, 1};

    // Build X: N x T x q, with simple patterns
    arma::cube X(2, T, q, arma::fill::zeros);
    // slice 0: values [[1,2,3],[3,4,5]] -> mean col-wise = [2,3,4]
    X.slice(0).row(0) = arma::rowvec({1.0, 2.0, 3.0});
    X.slice(0).row(1) = arma::rowvec({3.0, 4.0, 5.0});
    // slice 1: values [[10,20,30],[30,40,50]] -> mean = [20,30,40]
    X.slice(1).row(0) = arma::rowvec({10.0, 20.0, 30.0});
    X.slice(1).row(1) = arma::rowvec({30.0, 40.0, 50.0});

    est.add_data(unit_idxs, Y, X);
    apm::OutcomeMeanSuffStatEstimates out = est.estimate(/*total_units=*/2);

    ASSERT_TRUE(out.suff_stat_estimates.has_covar_means());
    const arma::mat& cm = *(out.suff_stat_estimates.covar_means);
    ASSERT_EQ(cm.n_rows, T);
    ASSERT_EQ(cm.n_cols, q);

    arma::mat expected_cm(T, q);
    expected_cm.col(0) = arma::vec({2.0, 3.0, 4.0});
    expected_cm.col(1) = arma::vec({20.0, 30.0, 40.0});
    ASSERT_TRUE(arma::approx_equal(cm, expected_cm, "absdiff", 1e-12));
}

TEST(OutcomeSuffStatEstimatorTest, InputValidation_XRequirements) {
    const std::size_t T_c = 2, T = 3, q = 1;
    arma::mat Y = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    arma::uvec unit_idxs = {0, 1};

    // q==0 -> X must be empty
    {
        apm::OutcomeMeanSuffStatEstimator est_q0(T_c);
        arma::cube X_nonempty(2, T, 1, arma::fill::ones);
        EXPECT_THROW(est_q0.add_data(unit_idxs, Y, X_nonempty), std::invalid_argument);
    }

    // q>0 -> X must be N x T x q
    {
        apm::OutcomeMeanSuffStatEstimator est(T_c, T, q);
        arma::cube X_good(2, T, q, arma::fill::ones);
        EXPECT_NO_THROW(est.add_data(unit_idxs, Y, X_good));
    }
    {
        apm::OutcomeMeanSuffStatEstimator est(T_c, T, q);
        arma::cube X_wrong_rows(3, T, q, arma::fill::ones);
        EXPECT_THROW(est.add_data(unit_idxs, Y, X_wrong_rows), std::invalid_argument);
    }
    {
        apm::OutcomeMeanSuffStatEstimator est(T_c, T, q);
        arma::cube X_wrong_cols(2, T + 1, q, arma::fill::ones);
        EXPECT_THROW(est.add_data(unit_idxs, Y, X_wrong_cols), std::invalid_argument);
    }
    {
        apm::OutcomeMeanSuffStatEstimator est(T_c, T, q);
        arma::cube X_wrong_slices(2, T, q + 1, arma::fill::ones);
        EXPECT_THROW(est.add_data(unit_idxs, Y, X_wrong_slices), std::invalid_argument);
    }
}


