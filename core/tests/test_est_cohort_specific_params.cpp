#include <gtest/gtest.h>
#include <armadillo>
#include <memory>
#include <unordered_map>

#include "est_cohort_specific_params.h"
#include "apm_core.h"
#include "linear_algebra_utils.h"
#include "bootstrap.h"

namespace {

class TestBootstrap : public apm::WeightedBootstrap {
public:
    explicit TestBootstrap(const arma::mat& W) : apm::WeightedBootstrap(W) {}
};

arma::mat proj(const arma::mat& X) { return apm::internal::projection_matrix(X); }
void expect_same_subspace(const arma::mat& G1, const arma::mat& G2, double tol = 1e-9) {
    ASSERT_TRUE(arma::approx_equal(proj(G1), proj(G2), "absdiff", tol));
}

struct PanelCtx {
    arma::uword T = 5, r = 2, C = 3, units_per = 2, q = 2;
    arma::mat G_true;                    // T x r
    std::vector<arma::uvec> T_idx;       // size C
    arma::vec g0_true;                   // length T
    arma::vec a_true;                    // length q
    std::vector<arma::vec> l_unit;       // per unit (global) length r
    std::vector<arma::mat> cohort_G_list; // per cohort, T_c x r rotated factors
};

PanelCtx make_ctx() {
    PanelCtx ctx;
    ctx.G_true.set_size(ctx.T, ctx.r);
    for (arma::uword t = 0; t < ctx.T; ++t) {
        double v = 0.1 * (t + 1);
        ctx.G_true(t, 0) = v;
        ctx.G_true(t, 1) = v + 0.5;
    }
    ctx.T_idx = { arma::uvec{0, 1, 2}, arma::uvec{1, 2, 3}, arma::uvec{2, 3, 4} };
    ctx.g0_true = arma::linspace(0.1, 0.5, ctx.T);
    ctx.a_true = arma::vec({0.5, 1.0});
    for (int u = 0; u < 6; ++u) ctx.l_unit.push_back(arma::vec({1.0 + u, 2.0 + u}));

    // Build cohort-specific rotated factors to mirror R helper build_factor_model_context
    ctx.cohort_G_list.resize(ctx.C);
    for (arma::uword c = 0; c < ctx.C; ++c) {
        arma::mat R(ctx.r, ctx.r);
        for (arma::uword i = 0; i < ctx.r; ++i) {
            for (arma::uword j = 0; j < ctx.r; ++j) {
                R(i, j) = 0.1 * static_cast<double>(c + 1) * static_cast<double>(i + 1)
                          + 0.1 * static_cast<double>(j + 1);
            }
        }
        R = 0.5 * (R + R.t());
        R.diag() += static_cast<double>(ctx.r);
        ctx.cohort_G_list[c] = ctx.G_true.rows(ctx.T_idx[c]) * R;
    }
    return ctx;
}

struct RawPanel {
    std::vector<int> unit_idx, cohort_id, outcome_idx;
    std::vector<double> y, cov1, cov2; // q=2
};

RawPanel make_raw(const PanelCtx& ctx, bool with_covars) {
    RawPanel rp;
    int global_unit = 0;
    for (int c = 0; c < static_cast<int>(ctx.C); ++c) {
        for (int u = 0; u < static_cast<int>(ctx.units_per); ++u, ++global_unit) {
            for (arma::uword k = 0; k < ctx.T_idx[c].n_elem; ++k) {
                int t = static_cast<int>(ctx.T_idx[c][k]);
                // Match R tests: y contains only factor component using cohort-rotated factors
                double y_val = arma::as_scalar(ctx.cohort_G_list[static_cast<std::size_t>(c)].row(static_cast<arma::uword>(k))
                                               * ctx.l_unit[global_unit]);
                rp.unit_idx.push_back(global_unit);
                rp.cohort_id.push_back(c);
                rp.outcome_idx.push_back(t);
                rp.y.push_back(y_val);
                if (with_covars) {
                    rp.cov1.push_back(global_unit + 1);
                    rp.cov2.push_back(c + 1);
                }
            }
        }
    }
    return rp;
}

} // namespace

TEST(CohortSpecificRawTest, InvalidEstimatorNameThrows) {
    PanelCtx ctx = make_ctx();
    RawPanel rp = make_raw(ctx, /*with_covars=*/false);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("bad", apm::EstimatorSpecification{"not_supported", false, 1});

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols; // d=0
    EXPECT_THROW(
        (void)apm::estimate_cohort_specific_params_from_raw(
            rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
            covar_cols, auxiliary_cols, rp.y.size(), specs, ctx.T_idx, nullptr),
        std::invalid_argument);
}

TEST(CohortSpecificRawTest, RGreaterThanTcThrows) {
    PanelCtx ctx = make_ctx();
    RawPanel rp = make_raw(ctx, /*with_covars=*/false);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, /*r=*/10});

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols; // d=0
    EXPECT_THROW(
        (void)apm::estimate_cohort_specific_params_from_raw(
            rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
            covar_cols, auxiliary_cols, rp.y.size(), specs, ctx.T_idx, nullptr),
        std::invalid_argument);
}

TEST(CohortSpecificRawTest, IntegratesEstimators_NoCovariates) {
    PanelCtx ctx = make_ctx();
    RawPanel rp = make_raw(ctx, /*with_covars=*/false);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, ctx.r});
    specs.emplace("pca_fe", apm::EstimatorSpecification{"principal_components", true, ctx.r});

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols; // d=0
    apm::CohortSpecificEstimates out = apm::estimate_cohort_specific_params_from_raw(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), specs, ctx.T_idx, nullptr);

    ASSERT_EQ(out.cohort_specific_factor_ests.size(), 2u);
    ASSERT_EQ(out.cohort_outcome_mean_ests.size(), ctx.C);

    const auto& pca_vec = out.cohort_specific_factor_ests.at("pca");
    const auto& pca_fe_vec = out.cohort_specific_factor_ests.at("pca_fe");
    ASSERT_EQ(pca_vec.size(), ctx.C);
    ASSERT_EQ(pca_fe_vec.size(), ctx.C);

    const auto& est_no_fe = pca_vec[0].parameter_estimates;
    const auto& est_fe    = pca_fe_vec[0].parameter_estimates;
    EXPECT_EQ(est_no_fe.G.n_rows, ctx.T_idx[0].n_elem);
    EXPECT_EQ(est_no_fe.G.n_cols, ctx.r);
    EXPECT_FALSE(est_no_fe.has_fixed_effects());
    EXPECT_TRUE(est_fe.has_fixed_effects());
    EXPECT_EQ(est_fe.G.n_rows, ctx.T_idx[0].n_elem);
    EXPECT_EQ(est_fe.G.n_cols, ctx.r);

    arma::mat G0_true = ctx.cohort_G_list[0];
    arma::mat P_est_no_fe = proj(est_no_fe.G);
    arma::mat P_est_fe    = proj(est_fe.G);
    arma::mat P_true      = proj(G0_true);
    if (!arma::approx_equal(P_est_no_fe, P_true, "absdiff", 1e-9)) {
        std::cerr << "P_est_no_fe:\n" << P_est_no_fe << "\nP_true:\n" << P_true << std::endl;
    }
    // Align with R tests: assert span match for no-FE estimator only
    expect_same_subspace(est_no_fe.G, G0_true);

    const auto& oms0 = out.cohort_outcome_mean_ests[0].suff_stat_estimates;
    arma::vec expected_means(oms0.observed_outcome_means.n_elem, arma::fill::zeros);
    int u0 = 0, u1 = 1;
    for (arma::uword k = 0; k < ctx.T_idx[0].n_elem; ++k) {
        double m0 = arma::as_scalar(ctx.cohort_G_list[0].row(k) * ctx.l_unit[u0]);
        double m1 = arma::as_scalar(ctx.cohort_G_list[0].row(k) * ctx.l_unit[u1]);
        expected_means(static_cast<arma::uword>(k)) = 0.5 * (m0 + m1);
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
    PanelCtx ctx = make_ctx();
    RawPanel rp = make_raw(ctx, /*with_covars=*/true);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, ctx.r});
    specs.emplace("pca_fe", apm::EstimatorSpecification{"principal_components", true, ctx.r});

    std::vector<const double*> covar_cols;
    covar_cols.push_back(rp.cov1.data());
    covar_cols.push_back(rp.cov2.data());

    std::vector<const double*> auxiliary_cols; // d=0
    apm::CohortSpecificEstimates out = apm::estimate_cohort_specific_params_from_raw(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), specs, ctx.T_idx, nullptr);

    const auto& oms0 = out.cohort_outcome_mean_ests[0].suff_stat_estimates;
    ASSERT_TRUE(oms0.has_covar_means());
    const arma::mat& cm = *(oms0.covar_means);
    // C++ impl uses T = 1 + max(T_idx) per cohort when q>0
    ASSERT_EQ(cm.n_rows, static_cast<arma::uword>(ctx.T_idx[0].max() + 1));
    ASSERT_EQ(cm.n_cols, ctx.q);

    for (arma::uword t = 0; t <= ctx.T_idx[0].max(); ++t) {
        EXPECT_NEAR(cm(t, 0), 1.5, 1e-12);
        EXPECT_NEAR(cm(t, 1), 1.0, 1e-12);
    }
}

TEST(CohortSpecificRawTest, Bootstrap_DeterministicReplicates_NoCovariates) {
    PanelCtx ctx = make_ctx();
    RawPanel rp = make_raw(ctx, /*with_covars=*/false);

    // 6 units total; build W with two bootstrap draws: first uses units 0..2, second uses 3..5 equally
    arma::mat W(6, 2, arma::fill::zeros);
    for (int i = 0; i < 3; ++i) W(i, 0) = 1.0 / 3.0;
    for (int i = 3; i < 6; ++i) W(i, 1) = 1.0 / 3.0;
    auto boot = std::make_shared<TestBootstrap>(W);

    std::unordered_map<std::string, apm::EstimatorSpecification> specs;
    specs.emplace("pca", apm::EstimatorSpecification{"principal_components", false, ctx.r});
    specs.emplace("pca_fe", apm::EstimatorSpecification{"principal_components", true, ctx.r});
    // Set cohort_weighting: by_size for pca, equal for pca_fe
    specs.at("pca").cohort_weighting = std::string("by_size");
    specs.at("pca_fe").cohort_weighting = std::string("equal");

    std::vector<const double*> covar_cols; // q=0
    std::vector<const double*> auxiliary_cols; // d=0
    apm::CohortSpecificEstimates out = apm::estimate_cohort_specific_params_from_raw(
        rp.unit_idx.data(), rp.cohort_id.data(), rp.outcome_idx.data(), rp.y.data(),
        covar_cols, auxiliary_cols, rp.y.size(), specs, ctx.T_idx, boot);

    // Check suff stat bootstrap replicates exist and have expected lengths for cohort 0
    const auto& oms0 = out.cohort_outcome_mean_ests[0];
    ASSERT_EQ(oms0.bootstrap_replicates.size(), 2u);
    ASSERT_EQ(oms0.suff_stat_estimates.observed_outcome_means.n_elem, ctx.T_idx[0].n_elem);
    ASSERT_EQ(oms0.bootstrap_replicates[0].observed_outcome_means.n_elem, ctx.T_idx[0].n_elem);

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
    arma::vec S1 = {2.0/3.0, 1.0/3.0, 0.0};
    arma::vec S2 = {0.0, 1.0/3.0, 2.0/3.0};
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


