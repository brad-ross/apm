#ifndef APM_TEST_HELPERS_H
#define APM_TEST_HELPERS_H

#ifdef USING_R
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include <vector>
#include <set>

#include "apm_core.h"
#include "est_outcome_means.h"

// -------- Utilities and assertions --------

std::vector<arma::mat> generate_rotation_matrices(size_t C, arma::uword size);

std::vector<std::vector<std::set<arma::uword>>> canonicalize_o3_output(
    std::vector<std::vector<std::set<arma::uword>>> output);

void expect_same_subspace(const arma::mat& G1, const arma::mat& G2, double tol = 1e-6);

// -------- Estimation fixtures --------

struct EstimationTestData {
    arma::uword T = 4, r = 2, C = 2, q = 2;
    arma::mat G;
    arma::vec a;
    arma::vec g_0;
    std::vector<arma::mat> X_c_vec;
    std::vector<arma::uvec> observed_outcome_indices;
    std::vector<arma::vec> l_c;
};

EstimationTestData setup_estimation_test_data();

apm::FactorModelEstimates duplicate_bootstrap(const apm::FactorModelParameters& point, std::size_t B);

std::vector<apm::OutcomeMeanSuffStatEstimates> duplicate_bootstrap_suff(
    const std::vector<apm::OutcomeMeanSufficientStatistics>& suff_stats_point,
    std::size_t B);

std::vector<apm::OutcomeMeanSufficientStatistics> make_suff_stats_vec(
    const arma::mat& true_m,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::mat>& X_c_vec);

std::vector<apm::FactorModelEstimates> build_cohort_estimates_all_with_bootstrap(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::mat>& rotation_matrices,
    const arma::vec& g0_true,
    arma::uword q,
    std::size_t B,
    std::vector<arma::vec>& out_a_c_vec);

void run_alignment_test(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const arma::vec& cohort_weights);

void run_alignment_test(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices);

// -------- Staircase panel helpers --------

struct StaircaseData {
    arma::mat true_factors;                       // T x r
    std::vector<arma::uvec> observed_outcome_indices; // size C, sliding windows
    arma::vec g0_true;                            // length T
};

StaircaseData make_staircase_data(arma::uword T = 5, arma::uword r = 2, arma::uword C = 3);

// Build a staircase pattern with window length T_c; C is implied as T - T_c + 1.
// Precondition: T_c <= T.
std::vector<arma::uvec> make_staircase_observed_indices(arma::uword T, arma::uword T_c);

struct StaircasePanelContext {
    arma::uword T = 5;
    arma::uword r = 2;
    arma::uword C = 3;
    arma::uword T_c = 3; // window length per cohort
    arma::uword units_per = 2; // units per cohort
    arma::uword q = 2; // number of covariate columns when included
    arma::mat G_true; // T x r
    std::vector<arma::uvec> observed_outcome_indices; // size C
    arma::vec g0_true; // length T
    arma::vec a_true; // length q
    std::vector<arma::vec> l_unit; // length C*units_per, each length r
};

// Create a simple increasing factor structure and unit loadings matching earlier tests
StaircasePanelContext make_staircase_panel_context(
    arma::uword T,
    arma::uword r,
    arma::uword T_c,
    arma::uword units_per = 0,
    arma::uword q = 2,
    bool with_covariates = false);

struct RawPanelData {
    std::vector<int> unit_idx, cohort_id, outcome_idx;
    std::vector<double> y, cov1, cov2; // q up to 2
    std::vector<double> aux1, aux2;    // d up to 2
};

// Build raw long-format panel from context.
// If with_covariates or with_auxiliary are true, expand to full T per unit with NaN y for unobserved.
RawPanelData make_raw_panel(
    const StaircasePanelContext& ctx,
    bool with_auxiliary = false);

#endif // APM_TEST_HELPERS_H


