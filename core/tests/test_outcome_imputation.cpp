#include <gtest/gtest.h>
#include <armadillo>

#include "outcome_imputation.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "cohort_specific_param_structs.h"
#include "test_helpers.h"

TEST(OutcomeImputationTest, RecoversLambdaAndG0_NoCovariates) {
	// 1) Context and raw panel without covariates
	auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/3, /*q=*/0);
	auto rp = make_raw_panel(ctx, /*with_covariates=*/false);

	// 2) Inject fixed effects into y: y += g0_true[outcome]
	for (std::size_t i = 0; i < rp.y.size(); ++i) {
		int t = rp.outcome_idx[i];
		rp.y[i] += ctx.g0_true[static_cast<arma::uword>(t)];
	}

	// 3) Build panel
	std::vector<const double*> covar_cols;      // q = 0
	std::vector<const double*> auxiliary_cols;  // d = 0
	apm::InMemoryUnbalancedPanel panel(
		rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
		covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

	// 4) True parameters and variable spec
	apm::FactorModelParameters fmp(ctx.G_true /*T x r*/);
	auto var = apm::VariableSpec::outcome();

	// 5) True loadings as N x r
	arma::uword N = static_cast<arma::uword>(ctx.l_unit.size());
	arma::mat L_true(N, ctx.r, arma::fill::zeros);
	for (arma::uword u = 0; u < N; ++u) L_true.row(u) = ctx.l_unit[static_cast<std::size_t>(u)].t();

	// 6) Recover lambda with true g0, then recover g0 with true lambda
	arma::mat lambda = apm::comp_unit_specific_params(ctx.g0_true, panel, var, fmp);
	EXPECT_TRUE(arma::approx_equal(lambda, L_true, "absdiff", 1e-12));

	arma::vec g0_est = apm::comp_outcome_specific_params(L_true, panel, var, fmp);
	EXPECT_TRUE(arma::approx_equal(g0_est, ctx.g0_true, "absdiff", 1e-12));
}


