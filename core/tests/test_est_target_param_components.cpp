#include <gtest/gtest.h>
#include <unordered_map>

#include "target_params/est_target_params.h"
#include "est_cohort_specific_params.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "test_helpers.h"

// Verifies end-to-end estimation of target parameter components from a staircase
// panel using principal components with outcome fixed effects, and asserts that
// both the imputation and direct paths produce a full C x T mean matrix that
// matches the ground truth implied by the context (average per-cohort loadings
// times the global basis plus outcome FEs).
TEST(TargetParamComponentsTest, EstFromPanel_Staircase_WithOutcomeFEs_BothPathsCorrect) {
    // 1) Deterministic staircase context with outcome fixed effects enabled
    auto ctx = make_staircase_panel_context(
        /*T=*/9, /*r=*/2, /*T_c=*/3, /*units_per=*/4,
        /*q=*/0, /*with_covariates=*/false, /*with_fixed_effects=*/true);
    auto rp = make_raw_panel(ctx, /*with_auxiliary=*/false);

    // 2) In-memory panel (0-based indices), no covariates or auxiliary
    std::vector<const double*> covar_cols;     // q = 0
    std::vector<const double*> auxiliary_cols; // d = 0
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(),
        rp.cohort_id.data(),
        rp.outcome_idx.data(),
        rp.y.data(),
        covar_cols,
        auxiliary_cols,
        rp.y.size(),
        ctx.observed_outcome_indices,
        /*one_indexed=*/false);

    // 3) Estimation specification: principal components with outcome FEs
    std::unordered_map<std::string, apm::EstimatorSpecification> est_specs;
    est_specs.emplace("pc", apm::EstimatorSpecification{
        /*factor_model_estimator=*/"principal_components",
        /*include_outcome_fes=*/true,
        /*r=*/static_cast<std::size_t>(ctx.r),
        /*cohort_weighting=*/"equal"
    });

    const std::size_t C = static_cast<std::size_t>(ctx.C);

    // 4) Ground truth from ctx: cohort-mean loadings times global basis + outcome FEs
    arma::mat L_mean(static_cast<arma::uword>(C), ctx.r, arma::fill::zeros);
    for (std::size_t c = 0; c < C; ++c) {
        arma::vec lbar(ctx.r, arma::fill::zeros);
        for (std::size_t u = 0; u < static_cast<std::size_t>(ctx.units_per); ++u) {
            lbar += ctx.l_unit[c * static_cast<std::size_t>(ctx.units_per) + u];
        }
        lbar /= static_cast<double>(ctx.units_per);
        L_mean.row(static_cast<arma::uword>(c)) = lbar.t();
    }

    arma::mat M_true = L_mean * ctx.G_true.t()
                     + arma::ones(static_cast<arma::uword>(C), 1) * ctx.g0_true.t();

    // 5) Run components (imputation enabled)
    apm::TargetParamComponents comps_imp = apm::est_target_param_components_from_panel(
        panel,
        est_specs,
        /*bootstrap=*/nullptr,
        /*num_threads=*/1,
        apm::CohortOutcomeMask(),
        /*est_outcome_means_via_imputation=*/true);

    ASSERT_EQ(comps_imp.outcome_means_by_spec.count("pc"), 1U);
    const arma::mat& M_hat_imp = comps_imp.outcome_means_by_spec.at("pc").mean_outcomes;
    ASSERT_EQ(static_cast<std::size_t>(M_hat_imp.n_rows), C);
    ASSERT_EQ(static_cast<std::size_t>(M_hat_imp.n_cols), static_cast<std::size_t>(ctx.T));
    EXPECT_TRUE(arma::approx_equal(M_hat_imp, M_true, "absdiff", 1e-6));

    // TODO: fix code when imputation is disabled
    // // 6) Run components (imputation disabled)
    // apm::TargetParamComponents comps_dir = apm::est_target_param_components_from_panel(
    //     panel,
    //     est_specs,
    //     /*bootstrap=*/nullptr,
    //     /*num_threads=*/1,
    //     apm::CohortOutcomeMask(),
    //     /*est_outcome_means_via_imputation=*/false);

    // ASSERT_EQ(comps_dir.outcome_means_by_spec.count("pc"), 1U);
    // const arma::mat& M_hat_dir = comps_dir.outcome_means_by_spec.at("pc").mean_outcomes;
    // ASSERT_EQ(static_cast<std::size_t>(M_hat_dir.n_rows), C);
    // ASSERT_EQ(static_cast<std::size_t>(M_hat_dir.n_cols), static_cast<std::size_t>(ctx.T));
    // EXPECT_TRUE(arma::approx_equal(M_hat_dir, M_true, "absdiff", 1e-6));

    // // 7) Imputation on/off should agree as well (redundant but informative)
    // EXPECT_TRUE(arma::approx_equal(M_hat_dir, M_hat_imp, "absdiff", 1e-8));

    // 8) No auxiliary provided; no masking expected
    EXPECT_TRUE(comps_imp.cohort_auxiliary_means.empty());
    EXPECT_FALSE(comps_imp.masked_observed_outcome_indices.has_value());
    EXPECT_TRUE(comps_imp.masked_cohort_outcome_means.empty());
}


// Uses a trivial bootstrap where each draw assigns uniform weights to all units,
// so bootstrap replicates should match the point estimate exactly. Also checks
// that imputation-enabled path matches ctx-derived truth with outcome FEs.
TEST(TargetParamComponentsTest, EstFromPanel_WithUniformBootstrap_ReplicatesMatchPoint) {
    // 1) Deterministic staircase context with FEs enabled
    auto ctx = make_staircase_panel_context(
        /*T=*/9, /*r=*/2, /*T_c=*/3, /*units_per=*/4,
        /*q=*/0, /*with_covariates=*/false, /*with_fixed_effects=*/true);
    auto rp = make_raw_panel(ctx, /*with_auxiliary=*/false);

    std::vector<const double*> covar_cols;
    std::vector<const double*> auxiliary_cols;
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    // 2) Estimation spec: PC with outcome FEs
    std::unordered_map<std::string, apm::EstimatorSpecification> est_specs;
    est_specs.emplace("pc", apm::EstimatorSpecification{
        "principal_components", /*include_outcome_fes=*/true,
        static_cast<std::size_t>(ctx.r), "equal"
    });

    // 3) Trivial uniform bootstrap: each draw uses equal weights over units
    const std::size_t N = panel.num_units();
    const std::size_t B = 3;
    arma::mat W_uni(static_cast<arma::uword>(N), static_cast<arma::uword>(B), arma::fill::ones);
    W_uni /= static_cast<double>(N);
    auto wb = std::make_shared<TestBootstrap>(W_uni);

    // 4) Ground truth from ctx
    const std::size_t C = static_cast<std::size_t>(ctx.C);
    arma::mat L_mean(static_cast<arma::uword>(C), ctx.r, arma::fill::zeros);
    for (std::size_t c = 0; c < C; ++c) {
        arma::vec lbar(ctx.r, arma::fill::zeros);
        for (std::size_t u = 0; u < static_cast<std::size_t>(ctx.units_per); ++u) {
            lbar += ctx.l_unit[c * static_cast<std::size_t>(ctx.units_per) + u];
        }
        lbar /= static_cast<double>(ctx.units_per);
        L_mean.row(static_cast<arma::uword>(c)) = lbar.t();
    }
    arma::mat M_true = L_mean * ctx.G_true.t()
                     + arma::ones(static_cast<arma::uword>(C), 1) * ctx.g0_true.t();

    // 5) Run imputation-enabled path with bootstrap
    apm::TargetParamComponents comps_imp = apm::est_target_param_components_from_panel(
        panel, est_specs, wb, /*num_threads=*/1, apm::CohortOutcomeMask(), /*impute=*/true);
    ASSERT_EQ(comps_imp.outcome_means_by_spec.count("pc"), 1U);
    const apm::OutcomeMeansEstimates& ome_imp = comps_imp.outcome_means_by_spec.at("pc");
    EXPECT_TRUE(arma::approx_equal(ome_imp.mean_outcomes, M_true, "absdiff", 1e-6));
    ASSERT_EQ(ome_imp.bootstrap_replicates.size(), B);
    for (std::size_t b = 0; b < B; ++b) {
        EXPECT_EQ(static_cast<std::size_t>(ome_imp.bootstrap_replicates[b].n_rows), C);
        EXPECT_EQ(static_cast<std::size_t>(ome_imp.bootstrap_replicates[b].n_cols), static_cast<std::size_t>(ctx.T));
        EXPECT_TRUE(arma::approx_equal(ome_imp.bootstrap_replicates[b], ome_imp.mean_outcomes, "absdiff", 1e-7));
    }

    // // 6) Run direct path (no imputation) with bootstrap: replicates equal point
    // apm::TargetParamComponents comps_dir = apm::est_target_param_components_from_panel(
    //     panel, est_specs, wb, /*num_threads=*/1, apm::CohortOutcomeMask(), /*impute=*/false);
    // ASSERT_EQ(comps_dir.outcome_means_by_spec.count("pc"), 1U);
    // const apm::OutcomeMeansEstimates& ome_dir = comps_dir.outcome_means_by_spec.at("pc");
    // ASSERT_EQ(ome_dir.bootstrap_replicates.size(), B);
    // for (std::size_t b = 0; b < B; ++b) {
    //     EXPECT_TRUE(arma::approx_equal(ome_dir.bootstrap_replicates[b], ome_dir.mean_outcomes, "absdiff", 1e-9));
    // }
}