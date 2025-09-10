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

// Helper data structures and function declarations used across tests

std::vector<arma::mat> generate_rotation_matrices(size_t C, arma::uword size);

struct StaircaseData {
    arma::mat true_factors;                       // T x r
    std::vector<arma::uvec> observed_outcome_indices; // size C, sliding windows
    arma::vec g0_true;                            // length T
};

StaircaseData make_staircase_data(arma::uword T = 5, arma::uword r = 2, arma::uword C = 3);

std::vector<apm::OutcomeMeanSufficientStatistics> make_suff_stats_vec(
    const arma::mat& true_m,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::mat>& X_c_vec);

apm::FactorModelEstimates duplicate_bootstrap(const apm::FactorModelParameters& point, std::size_t B);

std::vector<apm::OutcomeMeanSuffStatEstimates> duplicate_bootstrap_suff(
    const std::vector<apm::OutcomeMeanSufficientStatistics>& suff_stats_point,
    std::size_t B);

std::vector<apm::FactorModelEstimates> build_cohort_estimates_all_with_bootstrap(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const std::vector<arma::mat>& rotation_matrices,
    const arma::vec& g0_true,
    arma::uword q,
    std::size_t B,
    std::vector<arma::vec>& out_a_c_vec);

std::vector<std::vector<std::set<arma::uword>>> canonicalize_o3_output(
    std::vector<std::vector<std::set<arma::uword>>> output);

void run_alignment_test(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices,
    const arma::vec& cohort_weights);

void run_alignment_test(
    const arma::mat& true_factors,
    const std::vector<arma::uvec>& observed_outcome_indices);

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

#endif // APM_TEST_HELPERS_H


