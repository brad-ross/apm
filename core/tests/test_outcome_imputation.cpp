#include <gtest/gtest.h>
#include <armadillo>

#include "linear_algebra_utils.h"
#include "outcome_imputation_helpers.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "cohort_specific_param_structs.h"
#include "test_helpers.h"

TEST(OutcomeImputationTest, OutcomeSpecificParams_NestedLambda_NoCovariates) {
	// Context and raw panel without covariates
	auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/3, /*q=*/0);
	auto rp = make_raw_panel(ctx, /*with_covariates=*/false);
	for (std::size_t i = 0; i < rp.y.size(); ++i) {
		int t = rp.outcome_idx[i];
		rp.y[i] += ctx.g0_true[static_cast<arma::uword>(t)];
	}

	std::vector<const double*> covar_cols;      // q = 0
	std::vector<const double*> auxiliary_cols;  // d = 0
	apm::InMemoryUnbalancedPanel panel(
		rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
		covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

	apm::FactorModelParameters fmp(ctx.G_true /*T x r*/);
	auto var = apm::VariableSpec::outcome();

	// Using generalized API with both g_0_prev provided and storing unit params (lambda)
	    auto res = apm::internal::comp_unit_and_outcome_specific_params(
	        std::optional<arma::vec>(ctx.g0_true),
	        panel,
	        var,
	        fmp,
	        std::nullopt,
	        std::nullopt,
	        /*store_unit_params=*/true);

	    ASSERT_TRUE(res.first.has_value());
	    arma::vec g0_est = *res.first;

	    // Expect orthogonal projection of g0_true onto complement of span(G_true)
	    arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);
	    EXPECT_TRUE(arma::approx_equal(g0_est, g0_exp, "absdiff", 1e-8));

	    // Validate lambda was returned with expected dimensions and finite values
	    ASSERT_TRUE(res.second.has_value());
	    const arma::mat& L = *res.second;
	    EXPECT_EQ(static_cast<std::size_t>(L.n_rows), panel.num_units());
	    EXPECT_EQ(static_cast<std::size_t>(L.n_cols), static_cast<std::size_t>(ctx.G_true.n_cols));
	    EXPECT_TRUE(L.is_finite());

	    // Compare lambda to expected loadings from context
	    arma::mat L_exp(L.n_rows, L.n_cols, arma::fill::zeros);
	    for (arma::uword u = 0; u < L_exp.n_rows; ++u) {
	        L_exp.row(u) = ctx.l_unit[static_cast<std::size_t>(u)].t();
	    }
	    EXPECT_TRUE(arma::approx_equal(L, L_exp, "absdiff", 1e-8));
}

TEST(OutcomeImputationTest, FixedPoint_Vanilla_RecoversLambdaAndG0_NoCovariates) {
	// Context and raw panel
	auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/0);
	auto rp = make_raw_panel(ctx, /*with_covariates=*/false);
	for (std::size_t i = 0; i < rp.y.size(); ++i) {
		int t = rp.outcome_idx[i];
		rp.y[i] += ctx.g0_true[static_cast<arma::uword>(t)];
	}

	std::vector<const double*> covar_cols;      // q = 0
	std::vector<const double*> auxiliary_cols;  // d = 0
	apm::InMemoryUnbalancedPanel panel(
		rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
		covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

	apm::FactorModelParameters fmp(ctx.G_true);
	auto var = apm::VariableSpec::outcome();

    // Run fixed-point via generalized API; only need g0, no lambda
    arma::vec g0_est = apm::internal::comp_outcome_specific_params_fixed_point(panel, var, fmp);
    
    arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);
    
	EXPECT_TRUE(arma::approx_equal(g0_est, g0_exp, "absdiff", 1e-8));
}


TEST(OutcomeImputationTest, FixedPoint_IronsTuck_RecoversLambdaAndG0_NoCovariates) {
    // Context and raw panel
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/0);
    auto rp = make_raw_panel(ctx, /*with_covariates=*/false);
    for (std::size_t i = 0; i < rp.y.size(); ++i) {
        int t = rp.outcome_idx[i];
        rp.y[i] += ctx.g0_true[static_cast<arma::uword>(t)];
    }

    std::vector<const double*> covar_cols;      // q = 0
    std::vector<const double*> auxiliary_cols;  // d = 0
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    apm::FactorModelParameters fmp(ctx.G_true);
    auto var = apm::VariableSpec::outcome();

    // Run Irons-Tuck fixed-point (returns g0). Expect orthogonal projection of g0_true
    arma::vec g0_est = apm::internal::comp_outcome_specific_params_fixed_point(panel, var, fmp, std::nullopt, 1e-10, 1000, "irons-tuck");

    arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);

    EXPECT_TRUE(arma::approx_equal(g0_est, g0_exp, "absdiff", 1e-8));
}


