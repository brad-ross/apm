#include <gtest/gtest.h>
#include <armadillo>

#include "linear_algebra_utils.h"
#include "outcome_imputation_helpers.h"
#include "outcome_imputation.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "est_cohort_specific_params.h"
#include "cohort_specific_param_structs.h"
#include "test_helpers.h"

TEST(OutcomeImputationTest, OutcomeSpecificParams_NestedLambda_NoCovariates) {
    // Context and raw panel without covariates; include fixed effects in data generation
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/3, /*q=*/0, /*with_covariates=*/false, /*with_fixed_effects=*/true);
    auto rp = make_raw_panel(ctx);

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
		L_exp.row(u) = ctx.l_unit[static_cast<std::size_t>(u)].t(); // + gamma.t();
	}
	EXPECT_TRUE(arma::approx_equal(L, L_exp, "absdiff", 1e-8));
}

TEST(OutcomeImputationTest, FixedPoint_Vanilla_RecoversLambdaAndG0_NoCovariates) {
    // Context and raw panel; include fixed effects in data generation
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/0, /*with_covariates=*/false, /*with_fixed_effects=*/true);
    auto rp = make_raw_panel(ctx);

	std::vector<const double*> covar_cols;      // q = 0
	std::vector<const double*> auxiliary_cols;  // d = 0
	apm::InMemoryUnbalancedPanel panel(
		rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
		covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

	apm::FactorModelParameters fmp(ctx.G_true);
	auto var = apm::VariableSpec::outcome();

    // Run fixed-point via generalized API; only need g0, no lambda
    arma::vec g0_est = apm::internal::comp_outcome_specific_params_fixed_point(panel, var, fmp, std::nullopt, std::nullopt, 1e-10, 1000, "vanilla");
    
    arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);
    
	EXPECT_TRUE(arma::approx_equal(g0_est, g0_exp, "absdiff", 1e-8));
}

TEST(OutcomeImputationTest, FixedPoint_IronsTuck_RecoversLambdaAndG0_NoCovariates) {
    // Context and raw panel; include fixed effects in data generation
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/0, /*with_covariates=*/false, /*with_fixed_effects=*/true);
    auto rp = make_raw_panel(ctx);

    std::vector<const double*> covar_cols;      // q = 0
    std::vector<const double*> auxiliary_cols;  // d = 0
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    arma::vec a_dim(static_cast<arma::uword>(ctx.q), arma::fill::zeros);
    apm::FactorModelParameters fmp(ctx.G_true, std::nullopt, a_dim);
    auto var = apm::VariableSpec::outcome();

    // Run Irons-Tuck fixed-point (returns g0). Expect orthogonal projection of g0_true
    arma::vec g0_est = apm::internal::comp_outcome_specific_params_fixed_point(panel, var, fmp, std::nullopt, std::nullopt, 1e-10, 1000, "irons-tuck");

    arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);

    EXPECT_TRUE(arma::approx_equal(g0_est, g0_exp, "absdiff", 1e-8));
}


TEST(OutcomeImputationTest, CompCovarCoefs_RecoversAlpha_NoFixedEffects) {
    // Context with covariates (alpha present); raw panel includes X * alpha in Y
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/2, /*with_covariates=*/true);
    auto rp = make_raw_panel(ctx);

    // Build panel with covariate columns
    std::vector<const double*> covar_cols;
    covar_cols.push_back(rp.cov1.data());
    covar_cols.push_back(rp.cov2.data());
    std::vector<const double*> auxiliary_cols; // none

    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    arma::vec a_dim(static_cast<arma::uword>(ctx.q), arma::fill::zeros);
    apm::FactorModelParameters fmp(ctx.G_true, std::nullopt, a_dim);

    // Estimate alpha without fixed effects
    arma::vec alpha = apm::internal::comp_covar_coefs(panel, fmp, std::nullopt, {}, std::nullopt, std::nullopt);

    ASSERT_EQ(static_cast<std::size_t>(alpha.n_elem), static_cast<std::size_t>(ctx.q));
    EXPECT_TRUE(alpha.is_finite());
    ASSERT_EQ(static_cast<std::size_t>(ctx.a_true.n_elem), static_cast<std::size_t>(ctx.q));
    EXPECT_TRUE(arma::approx_equal(alpha, ctx.a_true, "absdiff", 1e-5));
}

TEST(OutcomeImputationTest, CompCovarCoefs_RecoversAlpha_WithFixedEffects) {
    // Context with covariates and fixed effects included in data generation
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/2, /*with_covariates=*/true, /*with_fixed_effects=*/true);
    auto rp = make_raw_panel(ctx);

    // Build panel with covariate columns
    std::vector<const double*> covar_cols;
    covar_cols.push_back(rp.cov1.data());
    covar_cols.push_back(rp.cov2.data());
    std::vector<const double*> auxiliary_cols; // none

    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

	arma::vec a_dim(static_cast<arma::uword>(ctx.q), arma::fill::zeros);
    apm::FactorModelParameters fmp(ctx.G_true, ctx.g0_true, a_dim);

    // Compute FE vectors via fixed-point
    auto var_y = apm::VariableSpec::outcome();
    arma::vec g0_init = apm::internal::comp_outcome_specific_params_fixed_point(panel, var_y, fmp, std::nullopt);

    std::vector<arma::vec> g0_init_covars(static_cast<std::size_t>(ctx.q));
    for (std::size_t j = 0; j < static_cast<std::size_t>(ctx.q); ++j) {
        auto var_xj = apm::VariableSpec::covariate(j);
        g0_init_covars[j] = apm::internal::comp_outcome_specific_params_fixed_point(panel, var_xj, fmp, std::nullopt);
    }

    // Estimate alpha with FE residualization
    arma::vec alpha = apm::internal::comp_covar_coefs(panel, fmp, g0_init, g0_init_covars, std::nullopt, std::nullopt);

    ASSERT_EQ(static_cast<std::size_t>(alpha.n_elem), static_cast<std::size_t>(ctx.q));
    EXPECT_TRUE(alpha.is_finite());
    ASSERT_EQ(static_cast<std::size_t>(ctx.a_true.n_elem), static_cast<std::size_t>(ctx.q));
    EXPECT_TRUE(arma::approx_equal(alpha, ctx.a_true, "absdiff", 1e-5));
}

TEST(OutcomeImputationTest, ImputationComponents_CovarsAndFixedEffects) {
    // Context with covariates and fixed effects included in data generation
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/2, /*with_covariates=*/true, /*with_fixed_effects=*/true);
    auto rp = make_raw_panel(ctx);

    std::vector<const double*> covar_cols{rp.cov1.data(), rp.cov2.data()};
    std::vector<const double*> auxiliary_cols;
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    // Signal presence of FE and covariates via FactorModelParameters
    arma::vec a_dim(static_cast<arma::uword>(ctx.q), arma::fill::zeros);
    apm::FactorModelParameters fmp(ctx.G_true, ctx.g0_true, a_dim);

    apm::FactorModelParameters out = apm::comp_imputation_components(panel, fmp, /*cohort_outcome_mean_suff_stats=*/{}, std::nullopt);

    // Check G span unchanged
    expect_same_subspace(out.G, ctx.G_true);
    // Check alpha recovered
    ASSERT_TRUE(out.a.has_value());
    EXPECT_TRUE(arma::approx_equal(*out.a, ctx.a_true, "absdiff", 1e-4));
    // Check g0 recovered up to orthogonal projection to rows of G
    arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);
    ASSERT_TRUE(out.g_0.has_value());
    EXPECT_TRUE(arma::approx_equal(*out.g_0, g0_exp, "absdiff", 1e-6));
    // Check lambda close to true unit loadings adjusted for g0 projection
    ASSERT_TRUE(out.L.has_value());
    const arma::mat& L = *out.L;
    ASSERT_EQ(static_cast<std::size_t>(L.n_rows), ctx.l_unit.size());
    arma::vec gamma = apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);
    for (arma::uword u = 0; u < L.n_rows; ++u) {
        arma::rowvec L_exp = ctx.l_unit[static_cast<std::size_t>(u)].t() + gamma.t();
        EXPECT_TRUE(arma::approx_equal(L.row(u), L_exp, "absdiff", 1e-5));
    }
}

TEST(OutcomeImputationTest, ImputationComponents_CovarsOnly) {
    // Context with covariates only; no g0 term
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/2, /*with_covariates=*/true);
    auto rp = make_raw_panel(ctx);

    std::vector<const double*> covar_cols{rp.cov1.data(), rp.cov2.data()};
    std::vector<const double*> auxiliary_cols;
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    // Signal covariates present but no FE
    arma::vec a_dim(static_cast<arma::uword>(ctx.q), arma::fill::zeros);
    apm::FactorModelParameters fmp(ctx.G_true, std::nullopt, a_dim);

    apm::FactorModelParameters out = apm::comp_imputation_components(panel, fmp, /*cohort_outcome_mean_suff_stats=*/{}, std::nullopt);

    // Check G span unchanged
    expect_same_subspace(out.G, ctx.G_true);
    // Check alpha recovered
    ASSERT_TRUE(out.a.has_value());
    EXPECT_TRUE(arma::approx_equal(*out.a, ctx.a_true, "absdiff", 1e-4));
    // No g0
    EXPECT_FALSE(out.g_0.has_value());
    // Lambda close to true
    ASSERT_TRUE(out.L.has_value());
    const arma::mat& L = *out.L;
    for (arma::uword u = 0; u < L.n_rows; ++u) {
        EXPECT_TRUE(arma::approx_equal(L.row(u), ctx.l_unit[static_cast<std::size_t>(u)].t(), "absdiff", 1e-5));
    }
}

TEST(OutcomeImputationTest, ImputationComponents_FixedEffectsOnly) {
    // No covariates; fixed effects included in data generation
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/0, /*with_covariates=*/false, /*with_fixed_effects=*/true);
    auto rp = make_raw_panel(ctx);

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols;
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    // FE present, no covariates
    apm::FactorModelParameters fmp(ctx.G_true, ctx.g0_true, std::nullopt);

    apm::FactorModelParameters out = apm::comp_imputation_components(panel, fmp, /*cohort_outcome_mean_suff_stats=*/{}, std::nullopt);

    expect_same_subspace(out.G, ctx.G_true);
    EXPECT_FALSE(out.a.has_value());
    arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);
    ASSERT_TRUE(out.g_0.has_value());
    EXPECT_TRUE(arma::approx_equal(*out.g_0, g0_exp, "absdiff", 1e-6));
    ASSERT_TRUE(out.L.has_value());
}

TEST(OutcomeImputationTest, ImputationComponents_AllOnesFactors_CovarsAndFixedEffects) {
    // r=1, G all ones to trigger comp_lambda_i shortcut
    arma::uword T = 6, r = 1, T_c = 3, units_per = 4, q = 2;
    auto ctx = make_staircase_panel_context(T, r, T_c, units_per, q, /*with_covariates=*/true, /*with_fixed_effects=*/true);
    ctx.G_true = arma::ones<arma::mat>(T, r);
    auto rp = make_raw_panel(ctx);

    std::vector<const double*> covar_cols{rp.cov1.data(), rp.cov2.data()};
    std::vector<const double*> auxiliary_cols;
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    arma::vec a_dim(static_cast<arma::uword>(q), arma::fill::zeros);
    apm::FactorModelParameters fmp(ctx.G_true, ctx.g0_true, a_dim);

    apm::FactorModelParameters out = apm::comp_imputation_components(panel, fmp, /*cohort_outcome_mean_suff_stats=*/{}, std::nullopt);

    // Check G
    ASSERT_EQ(out.G.n_rows, T);
    ASSERT_EQ(out.G.n_cols, r);
    // Validate L via per-unit mean residual on each unit run
    ASSERT_TRUE(out.L.has_value());
    const arma::mat& L = *out.L; // N x 1
    for (arma::uword u = 0; u < L.n_rows; ++u) {
        // Compute expected lambda as mean_t (Y_it - g0_t - X_it * a)
        std::vector<double> vals;
        for (std::size_t i = 0; i < rp.y.size(); ++i) {
            if (static_cast<std::size_t>(rp.unit_idx[i]) == static_cast<std::size_t>(u) && std::isfinite(rp.y[i])) {
                int t = rp.outcome_idx[i];
                double x1 = rp.cov1[i];
                double x2 = rp.cov2[i];
                double resid = rp.y[i];
                if (out.g_0.has_value()) resid -= (*out.g_0)[static_cast<arma::uword>(t)];
                if (out.a.has_value()) resid -= (*out.a)(0) * x1 + (*out.a)(1) * x2;
                vals.push_back(resid);
            }
        }
        double mean_resid = 0.0;
        for (double v : vals) mean_resid += v;
        mean_resid = vals.empty() ? 0.0 : (mean_resid / static_cast<double>(vals.size()));
        EXPECT_NEAR(L(static_cast<arma::uword>(u), 0), mean_resid, 1e-6);
    }
}

TEST(OutcomeImputationTest, ImputationComponents_WithCohortSuff_CovarsAndFixedEffects) {
	// Context with covariates and fixed effects; multiple units per cohort
	auto ctx = make_staircase_panel_context(/*T=*/8, /*r=*/2, /*T_c=*/3, /*units_per=*/5, /*q=*/2, /*with_covariates=*/true, /*with_fixed_effects=*/true);
	auto rp = make_raw_panel(ctx);

	std::vector<const double*> covar_cols{rp.cov1.data(), rp.cov2.data()};
	std::vector<const double*> auxiliary_cols;  // d = 0
	apm::InMemoryUnbalancedPanel panel(
		rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
		covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

	// Signal presence of FE and covariates via FactorModelParameters
	apm::FactorModelParameters fmp(ctx.G_true, ctx.g0_true, ctx.a_true);

	// Precompute cohort mean loadings matrix (C x r)
	arma::mat mean_L(static_cast<arma::uword>(ctx.C), ctx.r, arma::fill::zeros);
	for (std::size_t c = 0; c < static_cast<std::size_t>(ctx.C); ++c) {
		std::size_t start_u = c * static_cast<std::size_t>(ctx.units_per);
		arma::vec mean_l(ctx.r, arma::fill::zeros);
		for (std::size_t u = 0; u < static_cast<std::size_t>(ctx.units_per); ++u) {
			mean_l += ctx.l_unit[start_u + u];
		}
		mean_l /= static_cast<double>(ctx.units_per);
		mean_L.row(static_cast<arma::uword>(c)) = mean_l.t();
	}

	// Compute cohort-level sufficient statistics using the core cohort-specific estimator
	std::unordered_map<std::string, apm::EstimatorSpecification> specs;
	specs.emplace("pca_fe", apm::EstimatorSpecification{"principal_components", true, static_cast<std::size_t>(ctx.r)});
	apm::CohortSpecificEstimates cse = apm::estimate_cohort_specific_params_from_internal_panel_rep(
		panel, specs, /*bootstrap=*/nullptr, /*num_threads=*/std::nullopt, /*mask=*/apm::CohortOutcomeMask());

	std::vector<apm::OutcomeMeanSufficientStatistics> suff_stats;
	suff_stats.reserve(static_cast<std::size_t>(ctx.C));
	for (std::size_t c = 0; c < static_cast<std::size_t>(ctx.C); ++c) {
		suff_stats.push_back(cse.cohort_outcome_mean_ests[c].suff_stat_estimates);
	}

	// Compute imputation components using cohort-level sufficient stats
	apm::FactorModelParameters out = apm::comp_imputation_components(
		panel,
		fmp,
		suff_stats);

	// G passed through
	expect_same_subspace(out.G, ctx.G_true);
	// Alpha recovered
	ASSERT_TRUE(out.a.has_value());
	EXPECT_TRUE(arma::approx_equal(*out.a, ctx.a_true, "absdiff", 1e-4));
	// g0 recovered up to orthogonal projection to rows of G
	ASSERT_TRUE(out.g_0.has_value());
	arma::vec g0_exp = ctx.g0_true - ctx.G_true * apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);
	EXPECT_TRUE(arma::approx_equal(*out.g_0, g0_exp, "absdiff", 1e-6));
	// L should be C x r and close to cohort mean loadings adjusted by gamma from FE
	ASSERT_TRUE(out.L.has_value());
	const arma::mat& L = *out.L;
	ASSERT_EQ(static_cast<std::size_t>(L.n_rows), static_cast<std::size_t>(ctx.C));
	ASSERT_EQ(static_cast<std::size_t>(L.n_cols), static_cast<std::size_t>(ctx.r));
	arma::vec gamma = apm::internal::min_norm_solve(ctx.G_true, ctx.g0_true);
	for (std::size_t c = 0; c < static_cast<std::size_t>(ctx.C); ++c) {
		arma::rowvec L_exp = mean_L.row(static_cast<arma::uword>(c)) + gamma.t();
		EXPECT_TRUE(arma::approx_equal(L.row(static_cast<arma::uword>(c)), L_exp, "absdiff", 1e-5));
	}
}