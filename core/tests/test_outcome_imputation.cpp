#include <gtest/gtest.h>
#include <armadillo>

#include "linear_algebra_utils.h"
#include "outcome_imputation.h"
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

	// Using nested lambda computation with true g0
	arma::vec g0_est = apm::comp_outcome_specific_params(ctx.g0_true, panel, var, fmp);
    
	// Expect orthogonal projection of g0_true onto complement of span(G_true)
    arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);

	EXPECT_TRUE(arma::approx_equal(g0_est, g0_exp, "absdiff", 1e-8));
}

TEST(OutcomeImputationTest, FixedPoint_RecoversLambdaAndG0_NoCovariates) {
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

    // Run fixed-point (returns g0). Expect orthogonal projection of g0_true
	arma::vec g0_est = apm::comp_unit_and_outcome_specific_params_vanilla_fixed_point(panel, var, fmp);
    
    arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);
    
	EXPECT_TRUE(arma::approx_equal(g0_est, g0_exp, "absdiff", 1e-8));
}


