#include <gtest/gtest.h>
#include <armadillo>
#include <memory>

#include "bootstrap.h"
#include "factor_model_estimators/twfe_estimator.h"
#include "est_cohort_specific_params.h"
#include "panels/InMemoryUnbalancedPanel.h"

// Deterministic bootstrap for tests
namespace {
class TestBootstrap : public apm::WeightedBootstrap {
public:
	explicit TestBootstrap(const arma::mat& W) : apm::WeightedBootstrap(W) {}
};
} // namespace

TEST(TWFEEstimatorTest, SingleBatch_NoBootstrap_GOnes_MeanFE_NoCovars) {
	const std::size_t T_c = 3;
	apm::TWFEEstimator est(T_c);

	// Y is N x T_c (rows = units)
	arma::mat Y = {
		{1.0, 2.0, 3.0},
		{3.0, 2.0, 1.0},
		{2.0, 2.0, 2.0}
	};
	arma::uvec unit_idxs = {0, 1, 2};

	est.add_data(unit_idxs, Y);
	apm::FactorModelEstimates out = est.estimate();

	// G is all ones (T_c x 1)
	ASSERT_EQ(out.parameter_estimates.G.n_rows, T_c);
	ASSERT_EQ(out.parameter_estimates.G.n_cols, 1u);
	ASSERT_TRUE(arma::approx_equal(out.parameter_estimates.G, arma::ones<arma::mat>(T_c, 1), "absdiff", 0.0));

	// g_0 equals column-wise means of Y
	ASSERT_TRUE(out.parameter_estimates.has_fixed_effects());
	arma::vec mu(T_c);
	for (arma::uword j = 0; j < T_c; ++j) mu(j) = arma::mean(Y.col(j));
	ASSERT_TRUE(arma::approx_equal(*(out.parameter_estimates.g_0), mu, "absdiff", 1e-12));

	// no covariate coefs when q==0
	EXPECT_FALSE(out.parameter_estimates.has_covariate_coefs());
	EXPECT_TRUE(out.bootstrap_replicates.empty());
}

TEST(TWFEEstimatorTest, SplitBatch_And_AddDatum_Equivalence) {
	const std::size_t T_c = 2;
	apm::TWFEEstimator est_batch(T_c);
	apm::TWFEEstimator est_datum(T_c);

	arma::mat Y = {
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
		{7.0, 8.0}
	};
	arma::uvec idx1 = {0, 1};
	arma::uvec idx2 = {2, 3};

	est_batch.add_data(idx1, Y.rows(0, 1));
	est_batch.add_data(idx2, Y.rows(2, 3));

	for (std::size_t i = 0; i < 4; ++i) {
		est_datum.add_datum(i, Y.row(static_cast<arma::uword>(i)).t());
	}

	auto out1 = est_batch.estimate();
	auto out2 = est_datum.estimate();

	ASSERT_TRUE(out1.parameter_estimates.has_fixed_effects());
	ASSERT_TRUE(out2.parameter_estimates.has_fixed_effects());
	ASSERT_TRUE(arma::approx_equal(*(out1.parameter_estimates.g_0), *(out2.parameter_estimates.g_0), "absdiff", 1e-12));
	ASSERT_TRUE(arma::approx_equal(out1.parameter_estimates.G, out2.parameter_estimates.G, "absdiff", 0.0));
}

TEST(TWFEEstimatorTest, Bootstrap_DeterministicReplicates) {
	const std::size_t T_c = 2, B = 2;

	arma::mat Y = {
		{1.0, 0.0},
		{1.0, 0.0},
		{0.0, 2.0},
		{0.0, 2.0}
	};
	arma::uvec unit_idxs = {0, 1, 2, 3};

	// Columns sum to 1; b=0 uses first two rows, b=1 uses last two rows
	arma::mat W(4, 2, arma::fill::zeros);
	W(0, 0) = 0.5; W(1, 0) = 0.5;
	W(2, 1) = 0.5; W(3, 1) = 0.5;
	auto boot = std::make_shared<TestBootstrap>(W);

	apm::TWFEEstimator est(T_c, boot);
	est.add_data(unit_idxs, Y);
	auto out = est.estimate();

	ASSERT_EQ(out.bootstrap_replicates.size(), B);
	// Point g_0 should be average across all rows
	arma::vec mu_point = arma::mean(Y, 0).t();
	ASSERT_TRUE(out.parameter_estimates.has_fixed_effects());
	ASSERT_TRUE(arma::approx_equal(*(out.parameter_estimates.g_0), mu_point, "absdiff", 1e-12));

	// Replicate 0: first two rows; replicate 1: last two rows
	arma::vec mu_b0 = arma::mean(Y.rows(0, 1), 0).t();
	arma::vec mu_b1 = arma::mean(Y.rows(2, 3), 0).t();
	ASSERT_TRUE(out.bootstrap_replicates[0].has_fixed_effects());
	ASSERT_TRUE(out.bootstrap_replicates[1].has_fixed_effects());
	ASSERT_TRUE(arma::approx_equal(*(out.bootstrap_replicates[0].g_0), mu_b0, "absdiff", 1e-12));
	ASSERT_TRUE(arma::approx_equal(*(out.bootstrap_replicates[1].g_0), mu_b1, "absdiff", 1e-12));

	// G is always ones
	ASSERT_TRUE(arma::approx_equal(out.parameter_estimates.G, arma::ones<arma::mat>(T_c, 1), "absdiff", 0.0));
	ASSERT_TRUE(arma::approx_equal(out.bootstrap_replicates[0].G, arma::ones<arma::mat>(T_c, 1), "absdiff", 0.0));
	ASSERT_TRUE(arma::approx_equal(out.bootstrap_replicates[1].G, arma::ones<arma::mat>(T_c, 1), "absdiff", 0.0));
}

TEST(TWFEEstimatorTest, QHandling_DimensionsEnforced_ZeroCoeffsPresentWhenQPositive) {
	const std::size_t T_c = 3, q = 2;

	arma::mat Y = {
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0}
	};
	arma::uvec unit_idxs = {0, 1};

	// Correct X shape: N x T_c x q
	arma::cube X(2, T_c, q, arma::fill::ones);

	apm::TWFEEstimator est(T_c, nullptr, q);
	est.add_data(unit_idxs, Y, X);
	auto out = est.estimate();

	// 'a' should be present (zeros of length q)
	ASSERT_TRUE(out.parameter_estimates.has_covariate_coefs());
	ASSERT_EQ(out.parameter_estimates.q(), q);
	ASSERT_TRUE(arma::approx_equal(*(out.parameter_estimates.a), arma::zeros<arma::vec>(q), "absdiff", 0.0));

	// If q==0 and X non-empty → error
	apm::TWFEEstimator est_bad_q0(T_c);
	arma::cube X_bad(2, T_c, 1, arma::fill::ones);
	EXPECT_THROW(est_bad_q0.add_data(unit_idxs, Y, X_bad), std::invalid_argument);
}

TEST(TWFEEstimatorTest, FactoryGuards_RAndIncludeFEsConstraints) {
	// Minimal 1-cohort panel just to trigger factory construction
	arma::uword T = 3, T_c = 3;
	std::vector<int> unit_idx = {0, 1};
	std::vector<int> cohort_id = {0, 0};
	std::vector<int> outcome_idx = {0, 1};
	std::vector<double> y = {1.0, 2.0};
	std::vector<const double*> covar_cols; std::vector<const double*> aux_cols;
	apm::ObservedOutcomeIndices ooi;
	ooi.resize(1);
	ooi[0] = arma::uvec{0, 1, 2};
	apm::InMemoryUnbalancedPanel panel(unit_idx.data(), cohort_id.data(), outcome_idx.data(), y.data(),
		covar_cols, aux_cols, y.size(), ooi, /*one_indexed=*/false);

	// include_outcome_fes must be true
	std::unordered_map<std::string, apm::EstimatorSpecification> specs_bad_fes;
	specs_bad_fes.emplace("twfe", apm::EstimatorSpecification{"twfe", false, 1});
	EXPECT_THROW(apm::estimate_cohort_specific_params_from_internal_panel_rep(panel, specs_bad_fes, nullptr, std::nullopt, apm::CohortOutcomeMask()), std::invalid_argument);

	// r must be 1
	std::unordered_map<std::string, apm::EstimatorSpecification> specs_bad_r;
	specs_bad_r.emplace("twfe", apm::EstimatorSpecification{"twfe", true, 2});
	EXPECT_THROW(apm::estimate_cohort_specific_params_from_internal_panel_rep(panel, specs_bad_r, nullptr, std::nullopt, apm::CohortOutcomeMask()), std::invalid_argument);
}
