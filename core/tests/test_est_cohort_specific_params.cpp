#include <gtest/gtest.h>
#include <armadillo>
#include <memory>
#include <unordered_map>
#include <limits>

#include "est_cohort_specific_params.h"
#include "panels/InMemoryUnbalancedPanel.h"
#include "apm_core.h"
#include "linear_algebra_utils.h"
#include "bootstrap.h"
#include "utils.h"
#include "test_helpers.h"

namespace {

class TestBootstrap : public apm::WeightedBootstrap {
public:
    explicit TestBootstrap(const arma::mat& W) : apm::WeightedBootstrap(W) {}
};

} // namespace

TEST(CohortSpecificRawTest, InvalidEstimatorNameThrows) {
    // Generic staircase setup
    auto ctx = make_staircase_panel_context(/*T=*/5, /*r=*/2, /*T_c=*/3);
    auto rp = make_raw_panel(ctx);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("bad", apm::EstimatorSpecification{"not_supported", false, 1});

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols; // d=0
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);
    EXPECT_THROW(
        (void)apm::estimate_cohort_specific_params_from_internal_panel_rep(
            panel, specs, /*bootstrap=*/nullptr, /*num_threads=*/std::nullopt, /*mask=*/apm::CohortOutcomeMask()),
        std::invalid_argument);
}

TEST(CohortSpecificRawTest, RGreaterThanTcThrows) {
    auto ctx = make_staircase_panel_context(/*T=*/5, /*r=*/2, /*T_c=*/3);
    auto rp = make_raw_panel(ctx);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, /*r=*/static_cast<std::size_t>(ctx.T_c + 1)});

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols; // d=0
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);
    EXPECT_THROW(
        (void)apm::estimate_cohort_specific_params_from_internal_panel_rep(
            panel, specs, /*bootstrap=*/nullptr, /*num_threads=*/std::nullopt, /*mask=*/apm::CohortOutcomeMask()),
        std::invalid_argument);
}

TEST(CohortSpecificRawTest, IntegratesEstimators_NoCovariates) {
    auto ctx = make_staircase_panel_context(/*T=*/5, /*r=*/2, /*T_c=*/3);
    auto rp = make_raw_panel(ctx);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, ctx.r});
    specs.emplace("pca_fe", apm::EstimatorSpecification{"principal_components", true, ctx.r});

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols; // d=0
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);
    apm::CohortSpecificEstimates out = apm::estimate_cohort_specific_params_from_internal_panel_rep(
        panel, specs, /*bootstrap=*/nullptr, /*num_threads=*/std::nullopt, /*mask=*/apm::CohortOutcomeMask());

    ASSERT_EQ(out.cohort_specific_factor_ests.size(), 2u);
    ASSERT_EQ(out.cohort_outcome_mean_ests.size(), ctx.C);

    const auto& pca_vec = out.cohort_specific_factor_ests.at("pca");
    const auto& pca_fe_vec = out.cohort_specific_factor_ests.at("pca_fe");
    ASSERT_EQ(pca_vec.size(), ctx.C);
    ASSERT_EQ(pca_fe_vec.size(), ctx.C);

    double largest_diff = 0.0;
    for (std::size_t c = 0; c < ctx.C; ++c) {
        const auto& est_no_fe = pca_vec[c].parameter_estimates;
        const auto& est_fe    = pca_fe_vec[c].parameter_estimates;
        EXPECT_EQ(est_no_fe.G.n_rows, ctx.observed_outcome_indices[c].n_elem);
        EXPECT_EQ(est_no_fe.G.n_cols, ctx.r);
        EXPECT_FALSE(est_no_fe.has_fixed_effects());
        EXPECT_TRUE(est_fe.has_fixed_effects());
        EXPECT_EQ(est_fe.G.n_rows, ctx.observed_outcome_indices[c].n_elem);
        EXPECT_EQ(est_fe.G.n_cols, ctx.r);

        arma::mat G0_true = ctx.G_true.rows(ctx.observed_outcome_indices[c]);
        arma::mat P_est_no_fe = apm::internal::projection_matrix(est_no_fe.G);
        arma::mat P_est_fe    = apm::internal::projection_matrix(est_fe.G);
        arma::mat P_true      = apm::internal::projection_matrix(G0_true);
        if (!arma::approx_equal(P_est_no_fe, P_true, "absdiff", 1e-9)) {
            largest_diff = std::max(largest_diff, arma::abs(P_est_no_fe - P_true).max());
            // std::cerr << "P_est_no_fe:\n" << P_est_no_fe << "\nP_true:\n" << P_true
            //     << "\nabsdiff: " << arma::abs(P_est_no_fe - P_true).max() << std::endl;
        }
        // Align with R tests: assert span match for no-FE estimator only
        expect_same_subspace(est_no_fe.G, G0_true);
    }
    if (largest_diff > 0.0) {
        std::cerr << "largest_diff: " << largest_diff << std::endl;
    }

    const auto& oms0 = out.cohort_outcome_mean_ests[0].suff_stat_estimates;
    arma::vec expected_means(oms0.observed_outcome_means.n_elem, arma::fill::zeros);
    for (arma::uword k = 0; k < ctx.observed_outcome_indices[0].n_elem; ++k) {
        arma::uword t = ctx.observed_outcome_indices[0][k];
        double sum_m = 0.0;
        for (arma::uword u = 0; u < ctx.units_per; ++u) {
            sum_m += arma::as_scalar(ctx.G_true.row(t) * ctx.l_unit[static_cast<std::size_t>(u)]);
        }
        expected_means(static_cast<arma::uword>(k)) = sum_m / static_cast<double>(ctx.units_per);
    }
    ASSERT_TRUE(arma::approx_equal(oms0.observed_outcome_means, expected_means, "absdiff", 1e-12));
    EXPECT_FALSE(oms0.has_covar_means());

    // Cohort weights: default is equal => 1/C, no bootstrap
    ASSERT_EQ(out.cohort_weights.size(), 2u);
    arma::vec eq = arma::ones(ctx.C) / static_cast<double>(ctx.C);
    {
        const auto& w = out.cohort_weights.at("pca");
        ASSERT_TRUE(w.bootstrap_cohort_weights.empty());
        ASSERT_TRUE(arma::approx_equal(w.cohort_weights, eq, "absdiff", 1e-12));
    }
    {
        const auto& w = out.cohort_weights.at("pca_fe");
        ASSERT_TRUE(w.bootstrap_cohort_weights.empty());
        ASSERT_TRUE(arma::approx_equal(w.cohort_weights, eq, "absdiff", 1e-12));
    }
}

TEST(CohortSpecificRawTest, YXAssembly_WithCovariates_DimensionsAndMeans) {
    auto ctx = make_staircase_panel_context(/*T=*/5, /*r=*/2, /*T_c=*/3, /*units_per=*/0, /*q=*/2, /*with_covariates=*/true);
    auto rp = make_raw_panel(ctx);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, ctx.r});
    specs.emplace("pca_fe", apm::EstimatorSpecification{"principal_components", true, ctx.r});

    std::vector<const double*> covar_cols;
    covar_cols.push_back(rp.cov1.data());
    covar_cols.push_back(rp.cov2.data());

    std::vector<const double*> auxiliary_cols; // d=0
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);
    apm::CohortSpecificEstimates out = apm::estimate_cohort_specific_params_from_internal_panel_rep(
        panel, specs, /*bootstrap=*/nullptr, /*num_threads=*/std::nullopt, /*mask=*/apm::CohortOutcomeMask());

    const auto& oms0 = out.cohort_outcome_mean_ests[0].suff_stat_estimates;
    ASSERT_TRUE(oms0.has_covar_means());
    const arma::mat& cm = *(oms0.covar_means);
    // C++ impl uses global T across cohorts when q>0
    ASSERT_EQ(cm.n_rows, ctx.T);
    ASSERT_EQ(cm.n_cols, ctx.q);

    // Expected covariate means reflect generation in make_raw_panel:
    // cov1 = (u+1)*(t+1) ⇒ E[cov1 | t] = (t+1) * mean_{u}(u+1)
    // cov2 = (u+1)*(t+1) * ( (t+1) + 0.5*(u+1) + 1 ) + 0.7*(c+1)
    //      ⇒ E[cov2 | t, c=0] = (t+1) * [ (t+1+1)*E[U] + 0.5*E[U^2] ] + 0.7*1, with U = u+1 ~ {1..units_per}
    const double n_units = static_cast<double>(ctx.units_per);
    const double mean_u = (n_units + 1.0) / 2.0; // mean of 1..n
    const double mean_u2 = ((n_units + 1.0) * (2.0 * n_units + 1.0)) / 6.0; // mean of squares of 1..n
    for (arma::uword t = 0; t < ctx.T; ++t) {
        const double tt = static_cast<double>(t) + 1.0;
        const double exp_cov1 = tt * mean_u;
        const double exp_cov2 = tt * ((tt + 1.0) * mean_u + 0.5 * mean_u2) + 0.7 * 1.0;
        EXPECT_NEAR(cm(t, 0), exp_cov1, 1e-12);
        EXPECT_NEAR(cm(t, 1), exp_cov2, 1e-12);
    }
}

TEST(CohortSpecificRawTest, Bootstrap_DeterministicReplicates_NoCovariates) {
    auto ctx = make_staircase_panel_context(/*T=*/5, /*r=*/2, /*T_c=*/3);
    auto rp = make_raw_panel(ctx, /*with_covariates=*/false);

    // 3 cohorts x units_per units; build W with two bootstrap draws:
    // draw 1 uses cohort 0 units only; draw 2 uses cohort 2 units only
    arma::mat W(static_cast<arma::uword>(ctx.C * ctx.units_per), 2, arma::fill::zeros);
    for (int i = 0; i < static_cast<int>(ctx.units_per); ++i) {
        W(i, 0) = 1.0 / static_cast<double>(ctx.units_per); // units 0..units_per-1 (cohort 0)
    }
    for (int i = static_cast<int>(2 * ctx.units_per); i < static_cast<int>(ctx.C * ctx.units_per); ++i) {
        W(i, 1) = 1.0 / static_cast<double>(ctx.units_per); // units of cohort 2 only
    }
    auto boot = std::make_shared<TestBootstrap>(W);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, ctx.r});
    specs.emplace("pca_fe", apm::EstimatorSpecification{"principal_components", true, ctx.r});
    // Set cohort_weighting: by_size for pca, equal for pca_fe
    specs.at("pca").cohort_weighting = std::string("by_size");
    specs.at("pca_fe").cohort_weighting = std::string("equal");

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols; // d=0
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);
    apm::CohortSpecificEstimates out = apm::estimate_cohort_specific_params_from_internal_panel_rep(
        panel, specs, boot, /*num_threads=*/std::nullopt, /*mask=*/apm::CohortOutcomeMask());

    // Check suff stat bootstrap replicates exist and have expected lengths for cohort 0
    const auto& oms0 = out.cohort_outcome_mean_ests[0];
    ASSERT_EQ(oms0.bootstrap_replicates.size(), 2u);
    ASSERT_EQ(oms0.suff_stat_estimates.observed_outcome_means.n_elem, ctx.observed_outcome_indices[0].n_elem);
    ASSERT_EQ(oms0.bootstrap_replicates[0].observed_outcome_means.n_elem, ctx.observed_outcome_indices[0].n_elem);

    // Factor estimator replicates exist for both specs
    const auto& pca_vec = out.cohort_specific_factor_ests.at("pca");
    const auto& pca_fe_vec = out.cohort_specific_factor_ests.at("pca_fe");
    ASSERT_EQ(pca_vec[0].bootstrap_replicates.size(), 2u);
    ASSERT_EQ(pca_fe_vec[0].bootstrap_replicates.size(), 2u);

    // Cohort weights by spec
    ASSERT_EQ(out.cohort_weights.size(), 2u);
    const auto& w_by = out.cohort_weights.at("pca");
    const auto& w_eq = out.cohort_weights.at("pca_fe");

    // Expected S columns per draw given W
    // Draw 1: all mass on cohort 0; Draw 2: all mass on cohort 2
    arma::vec S1 = {1.0, 0.0, 0.0};
    arma::vec S2 = {0.0, 0.0, 1.0};
    arma::vec meanS = 0.5 * (S1 + S2);

    // by_size: point = mean across draws; bootstrap replicates equal S columns
    ASSERT_EQ(w_by.bootstrap_cohort_weights.size(), 2u);
    ASSERT_TRUE(arma::approx_equal(w_by.cohort_weights, meanS, "absdiff", 1e-12));
    ASSERT_TRUE(arma::approx_equal(w_by.bootstrap_cohort_weights[0], S1, "absdiff", 1e-12));
    ASSERT_TRUE(arma::approx_equal(w_by.bootstrap_cohort_weights[1], S2, "absdiff", 1e-12));

    // equal: point and each bootstrap replicate are 1/C
    arma::vec eq = arma::ones(ctx.C) / static_cast<double>(ctx.C);
    ASSERT_EQ(w_eq.bootstrap_cohort_weights.size(), 2u);
    ASSERT_TRUE(arma::approx_equal(w_eq.cohort_weights, eq, "absdiff", 1e-12));
    ASSERT_TRUE(arma::approx_equal(w_eq.bootstrap_cohort_weights[0], eq, "absdiff", 1e-12));
    ASSERT_TRUE(arma::approx_equal(w_eq.bootstrap_cohort_weights[1], eq, "absdiff", 1e-12));
}


TEST(CohortSpecificRawTest, AuxiliaryMeans_NoCovariates) {
    auto ctx = make_staircase_panel_context(/*T=*/5, /*r=*/2, /*T_c=*/3);
    auto rp = make_raw_panel(ctx, /*with_auxiliary=*/true);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, ctx.r});

    // Build two auxiliary columns as deterministic functions of unit and cohort
    const std::vector<double>& aux1 = rp.aux1;
    const std::vector<double>& aux2 = rp.aux2;

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols;
    auxiliary_cols.push_back(aux1.data());
    auxiliary_cols.push_back(aux2.data());

    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    apm::CohortSpecificEstimates out = apm::estimate_cohort_specific_params_from_internal_panel_rep(
        panel, specs, /*bootstrap=*/nullptr, /*num_threads=*/std::nullopt, /*mask=*/apm::CohortOutcomeMask());

    ASSERT_EQ(out.cohort_auxiliary_means.size(), ctx.C);

    // Compute expected per-cohort pop shares (by unit share) and means directly from raw panel
    // Determine total number of unique units
    int max_unit = -1;
    for (std::size_t i = 0; i < rp.unit_idx.size(); ++i) if (rp.unit_idx[i] > max_unit) max_unit = rp.unit_idx[i];
    const std::size_t total_units = static_cast<std::size_t>(max_unit + 1);
    for (std::size_t c = 0; c < ctx.C; ++c) {
        // Population share by unit: fraction of unique units in cohort c
        std::vector<char> seen(total_units, 0);
        std::size_t units_c = 0;
        for (std::size_t i = 0; i < rp.unit_idx.size(); ++i) {
            if (static_cast<std::size_t>(rp.cohort_id[i]) == c) {
                const std::size_t u = static_cast<std::size_t>(rp.unit_idx[i]);
                if (!seen[u]) { seen[u] = 1; ++units_c; }
            }
        }
        double share_exp = (total_units > 0u) ? (static_cast<double>(units_c) / static_cast<double>(total_units)) : 0.0;

        const auto& est = out.cohort_auxiliary_means[c].estimates;
        EXPECT_NEAR(est.cohort_pop_share, share_exp, 1e-12);

        // Dimensions
        arma::uword Tc = static_cast<arma::uword>(ctx.T);
        ASSERT_EQ(est.auxiliary_means.n_rows, ctx.T);
        ASSERT_EQ(est.auxiliary_means.n_cols, 2u);

        // Means per outcome row t
        for (arma::uword t = 0; t < ctx.T; ++t) {
            double s1 = 0.0, s2 = 0.0; std::size_t n = 0;
            for (std::size_t i = 0; i < rp.y.size(); ++i) {
                if (static_cast<std::size_t>(rp.cohort_id[i]) == c && static_cast<arma::uword>(rp.outcome_idx[i]) == t) {
                    s1 += aux1[i]; s2 += aux2[i]; ++n;
                }
            }
            if (n == 0) {
                EXPECT_TRUE(std::isnan(est.auxiliary_means(t, 0)));
                EXPECT_TRUE(std::isnan(est.auxiliary_means(t, 1)));
            } else {
                EXPECT_NEAR(est.auxiliary_means(t, 0), s1 / static_cast<double>(n), 1e-12);
                EXPECT_NEAR(est.auxiliary_means(t, 1), s2 / static_cast<double>(n), 1e-12);
            }
        }
    }
}

TEST(CohortSpecificRawTest, AuxiliaryMeans_WithBootstrapReplicatesExist) {
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3);
    auto rp = make_raw_panel(ctx, /*with_auxiliary=*/true);

    // Two bootstrap draws with nontrivial partitions over units
    arma::mat W(static_cast<arma::uword>(ctx.C * ctx.units_per), 2, arma::fill::zeros);
    for (int i = 0; i < static_cast<int>(ctx.units_per); ++i) W(i, 0) = 1.0 / 3.0;
    for (int i = static_cast<int>(ctx.units_per); i < static_cast<int>(ctx.C * ctx.units_per); ++i) W(i, 1) = 1.0 / 3.0;
    auto boot = std::make_shared<TestBootstrap>(W);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, ctx.r});

    std::vector<double> aux1(rp.y.size(), 1.0);
    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols{aux1.data()};

    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);

    apm::CohortSpecificEstimates out = apm::estimate_cohort_specific_params_from_internal_panel_rep(
        panel, specs, boot, /*num_threads=*/std::nullopt, /*mask=*/apm::CohortOutcomeMask());

    ASSERT_EQ(out.cohort_auxiliary_means.size(), ctx.C);
    for (std::size_t c = 0; c < ctx.C; ++c) {
        const auto& aux = out.cohort_auxiliary_means[c];
        ASSERT_EQ(aux.bootstrap_replicates.size(), 2u);
        arma::uword Tc = static_cast<arma::uword>(ctx.T);
        ASSERT_EQ(aux.estimates.auxiliary_means.n_rows, ctx.T);
        ASSERT_EQ(aux.estimates.auxiliary_means.n_cols, 1u);
        ASSERT_EQ(aux.bootstrap_replicates[0].auxiliary_means.n_rows, ctx.T);
        ASSERT_EQ(aux.bootstrap_replicates[0].auxiliary_means.n_cols, 1u);
    }

    std::vector<double> exp_draw1(ctx.C, 0.0);
    exp_draw1[0] = 1.0;
    std::vector<double> exp_draw2(ctx.C, ctx.C > 1 ? 1.0 / (ctx.C - 1.0) : 0.0);
    if (!exp_draw2.empty()) exp_draw2[0] = 0.0;
    for (std::size_t c = 0; c < ctx.C; ++c) {
        const auto& aux = out.cohort_auxiliary_means[c];
        EXPECT_NEAR(aux.bootstrap_replicates[0].cohort_pop_share, exp_draw1[c], 1e-12);
        EXPECT_NEAR(aux.bootstrap_replicates[1].cohort_pop_share, exp_draw2[c], 1e-12);
    }
}


TEST(CohortSpecificRawTest, Masking_LastCohort_Outcome5) {
    auto ctx = make_staircase_panel_context(/*T=*/7, /*r=*/2, /*T_c=*/3);
    // Raw panel with observed outcomes only (q=0)
    auto rp = make_raw_panel(ctx);

    // Estimator spec: principal components without fixed effects
    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, ctx.r});

    // Build mask: for last cohort (index C-1), mask its last observed outcome
    apm::CohortOutcomeMask mask;
    const std::size_t last_cohort = static_cast<std::size_t>(ctx.C - 1);
    arma::uword t_to_mask = ctx.observed_outcome_indices[last_cohort].tail(1)(0);
    mask.emplace(static_cast<int>(last_cohort), arma::uvec{t_to_mask});

    // Estimate with mask provided (bootstrap omitted, num_threads set to 1 explicitly)
    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols; // d=0
    apm::InMemoryUnbalancedPanel panel(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), ctx.observed_outcome_indices, /*one_indexed=*/false);
    apm::CohortSpecificEstimates out = apm::estimate_cohort_specific_params_from_internal_panel_rep(
        panel, specs, /*bootstrap=*/nullptr, /*num_threads=*/static_cast<std::optional<std::size_t>>(1), /*mask=*/mask);

    // 1) Remaining cohort-specific factor estimates are correct (up to rotation):
    const auto& pca_vec = out.cohort_specific_factor_ests.at("pca");
    ASSERT_EQ(pca_vec.size(), ctx.C);

    // Cohort 0 and 1 unchanged
    {
        const auto& est0 = pca_vec[0].parameter_estimates;
        expect_same_subspace(est0.G, ctx.G_true.rows(ctx.observed_outcome_indices[0]));
        const auto& est1 = pca_vec[1].parameter_estimates;
        expect_same_subspace(est1.G, ctx.G_true.rows(ctx.observed_outcome_indices[1]));
    }

    // Last cohort should drop its last observed outcome
    {
        const auto& est_last = pca_vec[last_cohort].parameter_estimates;
        ASSERT_EQ(est_last.G.n_rows, ctx.T_c - 1);
        arma::uvec kept = ctx.observed_outcome_indices[last_cohort].head(ctx.T_c - 1);
        arma::mat G_true_masked = ctx.G_true.rows(kept);
        expect_same_subspace(est_last.G, G_true_masked);
    }

    // 2) Masked outcome mean is estimated correctly for cohort 2, outcome 4
    ASSERT_TRUE(out.masked_observed_outcome_indices.has_value());
    const auto& ooi_eff = *(out.masked_observed_outcome_indices);
    ASSERT_EQ(ooi_eff[last_cohort].n_elem, ctx.T_c - 1);
    for (arma::uword i = 0; i < ctx.T_c - 1; ++i) {
        EXPECT_EQ(ooi_eff[last_cohort][i], ctx.observed_outcome_indices[last_cohort][i]);
    }

    // Compute expected mean of outcome 4 among units in cohort 2 from raw panel
    double sum_y = 0.0; std::size_t count = 0;
    for (std::size_t i = 0; i < rp.y.size(); ++i) {
        if (rp.cohort_id[i] == static_cast<int>(last_cohort) && rp.outcome_idx[i] == static_cast<int>(t_to_mask)) {
            sum_y += rp.y[i];
            ++count;
        }
    }
    ASSERT_GT(count, 0u);
    double expected_mean = sum_y / static_cast<double>(count);

    // Retrieve masked sufficient statistics for cohort 2
    auto it = out.masked_cohort_outcome_means.find(static_cast<int>(last_cohort));
    ASSERT_TRUE(it != out.masked_cohort_outcome_means.end());
    const apm::OutcomeMeanSufficientStatistics& masked_stats = it->second;
    ASSERT_EQ(masked_stats.observed_outcome_means.n_elem, 1u);
    EXPECT_NEAR(masked_stats.observed_outcome_means[0], expected_mean, 1e-12);
}

