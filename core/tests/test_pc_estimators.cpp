#include <gtest/gtest.h>
#include <armadillo>
#include <memory>

#include "bootstrap.h"
#include "linear_algebra_utils.h"
#include "factor_model_estimators/pc_estimators.h"

namespace {

// Deterministic bootstrap for tests
class TestBootstrap : public apm::WeightedBootstrap {
public:
    explicit TestBootstrap(const arma::mat& W) : apm::WeightedBootstrap(W) {}
};

arma::mat proj(const arma::mat& X) {
    return apm::internal::projection_matrix(X);
}

void expect_same_subspace(const arma::mat& G1, const arma::mat& G2, double tol = 1e-9) {
    arma::mat P1 = proj(G1);
    arma::mat P2 = proj(G2);
    ASSERT_TRUE(arma::approx_equal(P1, P2, "absdiff", tol))
        << "P1:\n" << P1 << "\nP2:\n" << P2;
}

} // anonymous namespace

//==============================================================================
// PCEstimator tests (no fixed effects)
//==============================================================================

TEST(PCEstimatorTest, SingleBatch_NoBootstrap_PrincipalDirection) {
    const std::size_t T_c = 2, r = 1;
    apm::PCEstimator est(r, T_c);

    arma::mat Y = {
        {1.0, 0.0},
        {1.0, 0.0},
        {0.0, 2.0},
        {0.0, 2.0}
    };
    arma::uvec unit_idxs = {0, 1, 2, 3};

    est.add_data(unit_idxs, Y);
    apm::FactorModelEstimates out = est.estimate();

    arma::mat expected_G(2, 1, arma::fill::zeros); // span(e2)
    expected_G(1, 0) = 1.0;
    expect_same_subspace(out.parameter_estimates.G, expected_G);
    EXPECT_TRUE(out.bootstrap_replicates.empty());
}

TEST(PCEstimatorTest, SplitBatch_Invariance) {
    const std::size_t T_c = 2, r = 1;
    apm::PCEstimator est(r, T_c);

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

    apm::FactorModelEstimates out = est.estimate();
    arma::mat expected_G(2, 1, arma::fill::zeros);
    expected_G(1, 0) = 1.0;
    expect_same_subspace(out.parameter_estimates.G, expected_G);
}

TEST(PCEstimatorTest, AddDatum_Equivalence) {
    const std::size_t T_c = 2, r = 1;
    apm::PCEstimator est(r, T_c);

    arma::mat Y = {
        {1.0, 0.0},
        {1.0, 0.0},
        {0.0, 2.0},
        {0.0, 2.0}
    };

    for (std::size_t i = 0; i < 4; ++i) {
        est.add_datum(i, Y.row(static_cast<arma::uword>(i)).t());
    }

    apm::FactorModelEstimates out = est.estimate();
    arma::mat expected_G(2, 1, arma::fill::zeros);
    expected_G(1, 0) = 1.0;
    expect_same_subspace(out.parameter_estimates.G, expected_G);
}

TEST(PCEstimatorTest, Bootstrap_DeterministicReplicates) {
    const std::size_t T_c = 2, r = 1, B = 2;

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

    apm::PCEstimator est(r, T_c, boot);
    est.add_data(unit_idxs, Y);
    apm::FactorModelEstimates out = est.estimate();

    ASSERT_EQ(out.bootstrap_replicates.size(), B);

    arma::mat expected_point(2, 1, arma::fill::zeros); // span(e2)
    expected_point(1, 0) = 1.0;
    expect_same_subspace(out.parameter_estimates.G, expected_point);

    arma::mat e1(2, 1, arma::fill::zeros);
    e1(0, 0) = 1.0;
    arma::mat e2(2, 1, arma::fill::zeros);
    e2(1, 0) = 1.0;
    expect_same_subspace(out.bootstrap_replicates[0].G, e1);
    expect_same_subspace(out.bootstrap_replicates[1].G, e2);
}

TEST(PCEstimatorTest, QHandling_DimensionsEnforced_CovariatesIgnored) {
    const std::size_t T_c = 2, r = 1, q = 1;

    arma::mat Y = {
        {1.0, 0.0},
        {1.0, 0.0},
        {0.0, 2.0},
        {0.0, 2.0}
    };
    arma::uvec unit_idxs = {0, 1, 2, 3};

    // Two runs with different X but same expected G
    arma::cube X1(4, T_c, q, arma::fill::ones);
    arma::cube X2(4, T_c, q, arma::fill::zeros);

    apm::PCEstimator est1(r, T_c, nullptr, q);
    est1.add_data(unit_idxs, Y, X1);
    apm::FactorModelEstimates out1 = est1.estimate();

    apm::PCEstimator est2(r, T_c, nullptr, q);
    est2.add_data(unit_idxs, Y, X2);
    apm::FactorModelEstimates out2 = est2.estimate();

    expect_same_subspace(out1.parameter_estimates.G, out2.parameter_estimates.G);

    // Invalid shapes should throw
    apm::PCEstimator est_bad_q0(r, T_c); // q==0
    arma::cube X_bad_q0(4, T_c, 1, arma::fill::ones);
    EXPECT_THROW(est_bad_q0.add_data(unit_idxs, Y, X_bad_q0), std::invalid_argument);

    apm::PCEstimator est_bad_dims(r, T_c, nullptr, q);
    arma::cube X_wrong_slices(4, T_c, 2, arma::fill::ones);
    EXPECT_THROW(est_bad_dims.add_data(unit_idxs, Y, X_wrong_slices), std::invalid_argument);
    arma::cube X_wrong_rows(5, T_c, q, arma::fill::ones);
    EXPECT_THROW(est_bad_dims.add_data(unit_idxs, Y, X_wrong_rows), std::invalid_argument);
}

TEST(PCEstimatorTest, RankZero_CornerCase) {
    const std::size_t T_c = 2, r = 0;
    apm::PCEstimator est(r, T_c);

    arma::mat Y = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    arma::uvec unit_idxs = {0, 1};

    est.add_data(unit_idxs, Y);
    apm::FactorModelEstimates out = est.estimate();

    EXPECT_EQ(out.parameter_estimates.G.n_rows, T_c);
    EXPECT_EQ(out.parameter_estimates.G.n_cols, 0u);
    EXPECT_FALSE(out.parameter_estimates.has_fixed_effects());
    EXPECT_TRUE(out.bootstrap_replicates.empty());
}

TEST(PCEstimatorTest, InputValidationAndConstructorChecks) {
    // r > T_c should throw on construction
    EXPECT_THROW(apm::PCEstimator bad(3, 2), std::invalid_argument);

    const std::size_t T_c = 2, r = 1;
    apm::PCEstimator est(r, T_c);

    arma::uvec unit_idxs = {0, 1};
    arma::mat Y_wrong_cols = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}; // N x 3, but T_c=2
    EXPECT_THROW(est.add_data(unit_idxs, Y_wrong_cols), std::invalid_argument);

    arma::mat Y = {{1.0, 2.0}, {3.0, 4.0}};
    arma::cube X_nonempty(2, T_c, 1, arma::fill::ones);
    EXPECT_THROW(est.add_data(unit_idxs, Y, X_nonempty), std::invalid_argument); // q==0 but X non-empty
}

//==============================================================================
// PCEstimatorWithFEs tests (subtract means)
//==============================================================================

TEST(PCEstimatorWithFEsTest, SingleBatch_MeansAndCovariance_NoBootstrap) {
    const std::size_t T_c = 2, r = 1;
    apm::PCEstimatorWithFEs est(r, T_c);

    arma::mat Y = {
        {1.0, 2.0},
        {1.0, 0.0},
        {0.0, 2.0},
        {0.0, 0.0}
    };
    arma::uvec unit_idxs = {0, 1, 2, 3};

    est.add_data(unit_idxs, Y);
    apm::FactorModelEstimates out = est.estimate();

    ASSERT_TRUE(out.parameter_estimates.has_fixed_effects());
    arma::vec expected_mu = {0.5, 1.0};
    ASSERT_TRUE(arma::approx_equal(*(out.parameter_estimates.g_0), expected_mu, "absdiff", 1e-12))
        << "mu:\n" << *(out.parameter_estimates.g_0);

    arma::mat expected_G(2, 1, arma::fill::zeros); // span(e2)
    expected_G(1, 0) = 1.0;
    expect_same_subspace(out.parameter_estimates.G, expected_G);
}

TEST(PCEstimatorWithFEsTest, SplitBatch_Invariance) {
    const std::size_t T_c = 2, r = 1;
    apm::PCEstimatorWithFEs est(r, T_c);

    arma::mat Y1 = {
        {1.0, 2.0},
        {1.0, 0.0}
    };
    arma::uvec idx1 = {0, 1};

    arma::mat Y2 = {
        {0.0, 2.0},
        {0.0, 0.0}
    };
    arma::uvec idx2 = {2, 3};

    est.add_data(idx1, Y1);
    est.add_data(idx2, Y2);

    apm::FactorModelEstimates out = est.estimate();

    arma::vec expected_mu = {0.5, 1.0};
    ASSERT_TRUE(out.parameter_estimates.has_fixed_effects());
    ASSERT_TRUE(arma::approx_equal(*(out.parameter_estimates.g_0), expected_mu, "absdiff", 1e-12));

    arma::mat expected_G(2, 1, arma::fill::zeros);
    expected_G(1, 0) = 1.0;
    expect_same_subspace(out.parameter_estimates.G, expected_G);
}

TEST(PCEstimatorWithFEsTest, AddDatum_Equivalence) {
    const std::size_t T_c = 2, r = 1;
    apm::PCEstimatorWithFEs est(r, T_c);

    arma::mat Y = {
        {1.0, 2.0},
        {1.0, 0.0},
        {0.0, 2.0},
        {0.0, 0.0}
    };

    for (std::size_t i = 0; i < 4; ++i) {
        est.add_datum(i, Y.row(static_cast<arma::uword>(i)).t());
    }

    apm::FactorModelEstimates out = est.estimate();
    arma::vec expected_mu = {0.5, 1.0};
    ASSERT_TRUE(out.parameter_estimates.has_fixed_effects());
    ASSERT_TRUE(arma::approx_equal(*(out.parameter_estimates.g_0), expected_mu, "absdiff", 1e-12));

    arma::mat expected_G(2, 1, arma::fill::zeros);
    expected_G(1, 0) = 1.0;
    expect_same_subspace(out.parameter_estimates.G, expected_G);
}

TEST(PCEstimatorWithFEsTest, Bootstrap_DeterministicReplicates) {
    const std::size_t T_c = 2, r = 1, B = 2;

    arma::mat Y = {
        {1.0, 2.0},
        {1.0, 0.0},
        {0.0, 2.0},
        {0.0, 0.0}
    };
    arma::uvec unit_idxs = {0, 1, 2, 3};

    arma::mat W(4, 2, arma::fill::zeros);
    W(0, 0) = 0.5; W(1, 0) = 0.5; // first two rows only
    W(2, 1) = 0.5; W(3, 1) = 0.5; // last two rows only
    auto boot = std::make_shared<TestBootstrap>(W);

    apm::PCEstimatorWithFEs est(r, T_c, boot);
    est.add_data(unit_idxs, Y);
    apm::FactorModelEstimates out = est.estimate();

    ASSERT_EQ(out.bootstrap_replicates.size(), B);

    // Point estimate
    arma::vec mu = {0.5, 1.0};
    ASSERT_TRUE(out.parameter_estimates.has_fixed_effects());
    ASSERT_TRUE(arma::approx_equal(*(out.parameter_estimates.g_0), mu, "absdiff", 1e-12));
    arma::mat e2(2, 1, arma::fill::zeros);
    e2(1, 0) = 1.0;
    expect_same_subspace(out.parameter_estimates.G, e2);

    // Replicates
    arma::vec mu_b0 = {1.0, 1.0};
    arma::vec mu_b1 = {0.0, 1.0};
    ASSERT_TRUE(out.bootstrap_replicates[0].has_fixed_effects());
    ASSERT_TRUE(out.bootstrap_replicates[1].has_fixed_effects());
    ASSERT_TRUE(arma::approx_equal(*(out.bootstrap_replicates[0].g_0), mu_b0, "absdiff", 1e-12));
    ASSERT_TRUE(arma::approx_equal(*(out.bootstrap_replicates[1].g_0), mu_b1, "absdiff", 1e-12));
    expect_same_subspace(out.bootstrap_replicates[0].G, e2);
    expect_same_subspace(out.bootstrap_replicates[1].G, e2);
}

TEST(PCEstimatorWithFEsTest, QHandling_DimensionsEnforced_CovariatesIgnored) {
    const std::size_t T_c = 2, r = 1, q = 1;

    arma::mat Y = {
        {1.0, 2.0},
        {1.0, 0.0},
        {0.0, 2.0},
        {0.0, 0.0}
    };
    arma::uvec unit_idxs = {0, 1, 2, 3};

    arma::cube X1(4, T_c, q, arma::fill::ones);
    arma::cube X2(4, T_c, q, arma::fill::zeros);

    apm::PCEstimatorWithFEs est1(r, T_c, nullptr, q);
    est1.add_data(unit_idxs, Y, X1);
    apm::FactorModelEstimates out1 = est1.estimate();

    apm::PCEstimatorWithFEs est2(r, T_c, nullptr, q);
    est2.add_data(unit_idxs, Y, X2);
    apm::FactorModelEstimates out2 = est2.estimate();

    // Means equal and G subspace equal
    ASSERT_TRUE(out1.parameter_estimates.has_fixed_effects());
    ASSERT_TRUE(out2.parameter_estimates.has_fixed_effects());
    ASSERT_TRUE(arma::approx_equal(*(out1.parameter_estimates.g_0), *(out2.parameter_estimates.g_0), "absdiff", 1e-12));
    expect_same_subspace(out1.parameter_estimates.G, out2.parameter_estimates.G);

    // Invalid shapes should throw
    apm::PCEstimatorWithFEs est_bad_q0(r, T_c); // q==0
    arma::cube X_bad_q0(4, T_c, 1, arma::fill::ones);
    EXPECT_THROW(est_bad_q0.add_data(unit_idxs, Y, X_bad_q0), std::invalid_argument);

    apm::PCEstimatorWithFEs est_bad_dims(r, T_c, nullptr, q);
    arma::cube X_wrong_slices(4, T_c, 2, arma::fill::ones);
    EXPECT_THROW(est_bad_dims.add_data(unit_idxs, Y, X_wrong_slices), std::invalid_argument);
    arma::cube X_wrong_rows(5, T_c, q, arma::fill::ones);
    EXPECT_THROW(est_bad_dims.add_data(unit_idxs, Y, X_wrong_rows), std::invalid_argument);
}

TEST(PCEstimatorWithFEsTest, RankZero_CornerCase) {
    const std::size_t T_c = 2, r = 0;
    apm::PCEstimatorWithFEs est(r, T_c);

    arma::mat Y = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    arma::uvec unit_idxs = {0, 1};

    est.add_data(unit_idxs, Y);
    apm::FactorModelEstimates out = est.estimate();

    EXPECT_EQ(out.parameter_estimates.G.n_rows, T_c);
    EXPECT_EQ(out.parameter_estimates.G.n_cols, 0u);
    ASSERT_TRUE(out.parameter_estimates.has_fixed_effects());
    // Mean should be the column-wise average
    arma::vec expected_mu = {2.0, 3.0};
    ASSERT_TRUE(arma::approx_equal(*(out.parameter_estimates.g_0), expected_mu, "absdiff", 1e-12));
    EXPECT_TRUE(out.bootstrap_replicates.empty());
}


