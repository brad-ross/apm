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


