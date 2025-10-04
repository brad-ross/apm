#include <gtest/gtest.h>
#include <numeric>
#include "bootstrap.h"

using namespace apm;

TEST(BootstrapTest, MultinomialDimensionsAndSums) {
    std::size_t N = 25;
    std::size_t B = 100;
    MultinomialBootstrap mb(N, B, 123);

    ASSERT_EQ(mb.n_obs(), N);
    ASSERT_EQ(mb.n_bootstraps(), B);

    const arma::mat& W = mb.weights();
    ASSERT_EQ(W.n_rows, N);
    ASSERT_EQ(W.n_cols, B);

    arma::rowvec sums = arma::sum(W, 0);
    for (arma::uword j = 0; j < sums.n_elem; ++j) {
        EXPECT_NEAR(sums[j], 1.0, 1e-12);
    }

    arma::vec d0 = mb.draw(0);
    EXPECT_EQ(d0.n_rows, N);

    arma::vec oi = mb.obs(3);
    EXPECT_EQ(oi.n_rows, B);
}

TEST(BootstrapTest, BayesianDimensionsAndSums) {
    std::size_t N = 30;
    std::size_t B = 80;
    BayesianBootstrap bb(N, B, 321);

    ASSERT_EQ(bb.n_obs(), N);
    ASSERT_EQ(bb.n_bootstraps(), B);

    const arma::mat& W = bb.weights();
    arma::rowvec sums = arma::sum(W, 0);
    for (arma::uword j = 0; j < sums.n_elem; ++j) {
        EXPECT_NEAR(sums[j], 1.0, 1e-12);
    }
}

TEST(BootstrapTest, RowSelectionOverloads) {
    std::size_t N = 10;
    std::size_t B = 5;
    MultinomialBootstrap mb(N, B, 7);

    arma::uvec idxA = {0, 2, 2, 9};
    arma::mat subA = mb.obs(idxA);
    EXPECT_EQ(subA.n_rows, idxA.n_elem);
    EXPECT_EQ(subA.n_cols, B);

    std::vector<std::size_t> idxB = {1, 5, 5};
    arma::mat subB = mb.obs(idxB);
    EXPECT_EQ(subB.n_rows, idxB.size());
    EXPECT_EQ(subB.n_cols, B);
}

TEST(BootstrapTest, InvalidIndicesThrow) {
    std::size_t N = 4;
    std::size_t B = 3;
    MultinomialBootstrap mb(N, B, 9);

    EXPECT_THROW(mb.draw(3), std::out_of_range);
    EXPECT_NO_THROW(mb.draw(2));

    EXPECT_THROW(mb.obs(4), std::out_of_range);
    EXPECT_NO_THROW(mb.obs(3));

    arma::uvec bad = {0, 5};
    EXPECT_THROW(mb.obs(bad), std::out_of_range);
}

TEST(BootstrapTest, NegativeWeightsValidation) {
    // Create a tiny derived class in-test to bypass RNG and inject negatives
    struct FakeBootstrap : public WeightedBootstrap {
        explicit FakeBootstrap(const arma::mat& W) : WeightedBootstrap(W) {}
    };

    arma::mat Wpos = {{0.2, 0.8}, {0.8, 0.2}}; // columns sum to 1 already
    EXPECT_NO_THROW({ FakeBootstrap fb(Wpos); (void)fb; });

    arma::mat Wneg = {{-0.1, 1.1}, {1.1, -0.1}}; // has negatives
    EXPECT_THROW({ FakeBootstrap fb(Wneg); (void)fb; }, std::invalid_argument);
}

TEST(BootstrapInferenceTest, RepeatedBootstrapCoverageAndPValues) {
    const std::size_t p = 5;
    const std::size_t B = 1000;
    const std::size_t N = 100;
    const double sig_level = 0.05;
    const std::size_t n_sims = 10000;
    
    arma::arma_rng::set_seed(12345);

    const double sqrtN = std::sqrt(static_cast<double>(N));
    
    // Track coverage rates
    std::size_t simult_coverage_count = 0;  // All p parameters covered simultaneously
    arma::uvec pointwise_coverage_count = arma::zeros<arma::uvec>(p);  // Each parameter
    
    // Store p-values in p x n_sims matrix (each row is a parameter, each column is a simulation)
    arma::mat p_value_matrix(p, n_sims);
    
    for (std::size_t sim = 0; sim < n_sims; ++sim) {
        // True parameter is zero; point estimates are N(0, 1/sqrt(N))
        arma::vec point_ests = arma::randn(p) / sqrtN;
        
        // Bootstrap replicates: centered at point_ests, each column is one bootstrap
        // Simulates resampling from data with true variance 1/N
        arma::mat bootstrap_replicates = arma::randn(p, B) / sqrtN;
        bootstrap_replicates.each_col() += point_ests;
        
        // Run bootstrap inference
        auto results = get_bootstrap_inference(point_ests, bootstrap_replicates, N, sig_level);
        
        // Check simultaneous coverage: do ALL bands contain zero?
        bool all_covered = true;
        for (arma::uword i = 0; i < p; ++i) {
            bool covered = (results.cb_lb(i) <= 0.0) && (0.0 <= results.cb_ub(i));
            if (!covered) {
                all_covered = false;
            }
            
            // Check pointwise coverage for each parameter
            bool pointwise_covered = (results.ci_lb(i) <= 0.0) && (0.0 <= results.ci_ub(i));
            if (pointwise_covered) {
                pointwise_coverage_count(i)++;
            }
            
            // Store p-value
            p_value_matrix(i, sim) = results.pointwise_p_vals(i);
        }
        
        if (all_covered) {
            simult_coverage_count++;
        }
    }
    
    // Test 1: Simultaneous band coverage should be around 95%
    double simult_coverage_rate = static_cast<double>(simult_coverage_count) / n_sims;
    EXPECT_GT(simult_coverage_rate, 0.94) 
        << "Simultaneous coverage rate too low: " << simult_coverage_rate;
    EXPECT_LT(simult_coverage_rate, 0.96) 
        << "Simultaneous coverage rate too high: " << simult_coverage_rate;
    
    // Test 2: Pointwise coverage should be around 95% for each parameter
    for (arma::uword i = 0; i < p; ++i) {
        double pointwise_rate = static_cast<double>(pointwise_coverage_count(i)) / n_sims;
        EXPECT_GT(pointwise_rate, 0.94) 
            << "Pointwise coverage too low for param " << i << ": " << pointwise_rate;
        EXPECT_LT(pointwise_rate, 0.96) 
            << "Pointwise coverage too high for param " << i << ": " << pointwise_rate;
    }
    
    // Test 3: P-values should be uniformly distributed for each parameter
    // Check empirical CDF shares against theoretical uniform quantiles
    // Check all interior deciles: 0.1, 0.2, 0.3, ..., 0.9
    const double tolerance = 0.02;  // Allow 2% deviation
    
    for (arma::uword i = 0; i < p; ++i) {
        // Extract p-values for this parameter across all simulations
        arma::vec param_pvals = p_value_matrix.row(i).t();  // column vector of length n_sims

        // Precompute deciles and empirical shares below each decile
        arma::vec deciles = arma::regspace(0.1, 0.1, 0.9); // 0.1, 0.2, ..., 0.9
        for (arma::uword d = 0; d < deciles.n_elem; ++d) {
            double q = deciles(d);
            // empirical share = proportion of p-values < q
            arma::uvec below = arma::find(param_pvals < q);
            double share = static_cast<double>(below.n_elem) / static_cast<double>(param_pvals.n_elem);

            EXPECT_NEAR(share, q, tolerance)
                << "Param " << i << ": empirical CDF at " << q
                << " should be near " << q << ", got " << share;
        }
    }
}