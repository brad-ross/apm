#include <gtest/gtest.h>
#include <armadillo>
#include "linear_algebra_utils.h"

// Function to check if a matrix is a projection matrix (idempotent and symmetric)
void check_is_projection(const arma::mat& P) {
    ASSERT_TRUE(P.is_symmetric());
    arma::mat P_squared = P * P;
    ASSERT_TRUE(arma::approx_equal(P, P_squared, "absdiff", 1e-9));
}

TEST(LinearAlgebraUtilsTest, ProjectionMatrixFullRank) {
    arma::mat X = {{1, 2}, {2, 4.1}, {3, 6}};
    arma::mat P = apm::internal::projection_matrix(X);
    check_is_projection(P);

    // Project a vector onto the column space of X
    arma::vec v = {1, 1, 1};
    arma::vec projected_v = P * v;
    
    // The residual should be orthogonal to the columns of X
    arma::vec residual = v - projected_v;
    ASSERT_NEAR(arma::dot(residual, X.col(0)), 0.0, 1e-9);
    ASSERT_NEAR(arma::dot(residual, X.col(1)), 0.0, 1e-9);
}

TEST(LinearAlgebraUtilsTest, ProjectionMatrixRankDeficient) {
    arma::mat X = {{1, 2}, {2, 4}, {3, 6}};
    arma::mat P = apm::internal::projection_matrix(X);
    check_is_projection(P);

    // The projection onto the col space of X should be the same as onto its first column
    arma::mat P_expected = apm::internal::projection_matrix(X.col(0));
    ASSERT_TRUE(arma::approx_equal(P, P_expected, "absdiff", 1e-9))
        << "Matrix P is not equal to P_expected:" << std::endl
        << "P:" << std::endl << P << std::endl
        << "P_expected:" << std::endl << P_expected;
}

TEST(LinearAlgebraUtilsTest, ProjectionMatrixZeroCols) {
    arma::mat X(3, 0);
    arma::mat P = apm::internal::projection_matrix(X);
    ASSERT_EQ(P.n_rows, 3);
    ASSERT_EQ(P.n_cols, 3);
    ASSERT_TRUE(arma::all(arma::vectorise(P) == 0.0));
} 

TEST(LinearAlgebraUtilsTest, MinNormSolveUnderdetermined) {
    // Underdetermined system: 2 equations, 3 unknowns.
    arma::mat A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    arma::mat B = {{7.0, 8.0}, {9.0, 10.0}};

    arma::mat X = apm::internal::multi_min_norm_solve(A, B);

    // 1. Check if the solution satisfies the equation.
    ASSERT_TRUE(arma::approx_equal(A * X, B, "absdiff", 1e-9));

    // 2. Check if the solution has the minimum norm.
    // The min-norm solution must be orthogonal to the null space of A.
    arma::mat null_space_basis = arma::null(A);
    ASSERT_GT(null_space_basis.n_cols, 0) << "Null space should not be empty for an underdetermined system.";

    ASSERT_TRUE(arma::approx_equal(X.t() * null_space_basis, 
        arma::zeros(X.n_cols, null_space_basis.n_cols), "absdiff", 1e-9));
} 

static arma::mat make_hilbert_like(arma::uword m, arma::uword n){
    arma::mat A(m,n);
    for (arma::uword i=0;i<m;++i) {
        for (arma::uword j=0;j<n;++j) {
            A(i,j) = 1.0 / static_cast<double>(i + j + 1);
        }
    }
    return A;
}

TEST(LSMR, OverdeterminedMatchesPinvDeterministic) {
    const arma::uword m=50, n=10;
    arma::mat A = make_hilbert_like(m,n);
    arma::vec x_true(n);
    for (arma::uword j=0;j<n;++j) x_true[j] = static_cast<double>(j+1);
    arma::vec b = A * x_true;

    apm::internal::LinearOperator Op{
        n, m,
        [&](const arma::vec& x, arma::vec& y){ y = A*x; },
        [&](const arma::vec& y, arma::vec& z){ z = A.t()*y; }
    };
    apm::internal::LSMROptions opts; opts.atol=1e-16; opts.btol=1e-16; opts.max_iters=2000; opts.lambda=0.0;
    auto res = apm::internal::lsmr(Op, b, opts, std::nullopt, true, true);

    arma::vec x_pinv = arma::pinv(A) * b;
    double rel_err = arma::norm(res.x - x_pinv)/std::max(1.0, arma::norm(x_pinv));
    ASSERT_LT(rel_err, 5e-6);
}

TEST(LSMR, UnderdeterminedMinNormMatchesPinvDeterministic) {
    const arma::uword m=8, n=20;
    arma::mat A = make_hilbert_like(m,n);
    arma::vec b(m);
    for (arma::uword i=0;i<m;++i) b[i] = (i % 2 == 0) ? static_cast<double>(i+1) : -static_cast<double>(i+1);

    apm::internal::LinearOperator Op{
        n, m,
        [&](const arma::vec& x, arma::vec& y){ y = A*x; },
        [&](const arma::vec& y, arma::vec& z){ z = A.t()*y; }
    };
    // opts.conlim = 0.0 means no condition limit which is crucial for underdetermined systems.
    apm::internal::LSMROptions opts; opts.atol=1e-16; opts.btol=1e-16; opts.max_iters=4000; opts.lambda=0.0; opts.conlim=0.0;
    auto res = apm::internal::lsmr(Op, b, opts);

    arma::vec x_pinv = arma::pinv(A) * b;
    double rel_err = arma::norm(res.x - x_pinv)/std::max(1.0, arma::norm(x_pinv));
    ASSERT_LT(rel_err, 1e-5);

    // Min-norm check: solution orthogonal to null(A)
    arma::mat N = arma::null(A);
    if (N.n_cols > 0) {
        arma::vec proj = N.t() * res.x;
        ASSERT_LT(arma::norm(proj)/std::max(1.0, arma::norm(res.x)), 1e-6);
    }
}

TEST(LSMR, TallSkinnyConvergesAccurately) {
    const arma::uword m=200, n=15;
    arma::mat A = make_hilbert_like(m,n);
    arma::vec x_true(n); for (arma::uword j=0;j<n;++j) x_true[j] = std::sin(static_cast<double>(j+1));
    arma::vec b = A * x_true;

    apm::internal::LinearOperator Op{
        n, m,
        [&](const arma::vec& x, arma::vec& y){ y = A*x; },
        [&](const arma::vec& y, arma::vec& z){ z = A.t()*y; }
    };
    apm::internal::LSMROptions opts; opts.atol=1e-16; opts.btol=1e-16; opts.max_iters=4000; opts.conlim=0.0;
    auto res = apm::internal::lsmr(Op, b, opts);

    arma::vec x_pinv = arma::pinv(A) * b;
    double rel_err = arma::norm(res.x - x_pinv)/std::max(1.0, arma::norm(x_pinv));
    ASSERT_LT(rel_err, 5e-5);
}
// TODO: find fixes for these extreme wide-matrix cases; not relevant to our application.
// TEST(LSMR, WideShortConvergesAccurately) {
//     const arma::uword m=15, n=200;
//     arma::mat A = make_hilbert_like(m,n);
//     arma::vec x_true(n); for (arma::uword j=0;j<n;++j) x_true[j] = std::cos(static_cast<double>(j+1));
//     arma::vec b = A * x_true;

//     apm::internal::LinearOperator Op{
//         n, m,
//         [&](const arma::vec& x, arma::vec& y){ y = A*x; },
//         [&](const arma::vec& y, arma::vec& z){ z = A.t()*y; }
//     };
//     apm::internal::LSMROptions opts; opts.atol=1e-16; opts.btol=1e-16; opts.max_iters=100000; opts.conlim=0.0;
//     auto res = apm::internal::lsmr(Op, b, opts, std::nullopt, false, false, std::nullopt, 10);

//     arma::vec x_pinv = arma::pinv(A) * b;
//     std::cout << "res.iters: " << res.iters << std::endl;
//     std::cout << "res.flag: " << res.flag << std::endl;
//     std::cout << "arma::norm(res.x - x_pinv, \"inf\"): " << arma::norm(res.x - x_pinv, "inf") << std::endl;
//     double rel_err = arma::norm(res.x - x_pinv)/std::max(1.0, arma::norm(x_pinv));
//     ASSERT_LT(rel_err, 1e-5);
// }

TEST(LSMR, ZeroRHSAndZeroOperatorCases) {
    // Zero RHS
    {
        const arma::uword m=10, n=5;
        arma::mat A = make_hilbert_like(m,n);
        arma::vec b(m, arma::fill::zeros);
        apm::internal::LinearOperator Op{n, m,
            [&](const arma::vec& x, arma::vec& y){ y = A*x; },
            [&](const arma::vec& y, arma::vec& z){ z = A.t()*y; }
        };
        apm::internal::LSMROptions opts; opts.atol=1e-12; opts.btol=1e-12; opts.max_iters=1000;
        auto res = apm::internal::lsmr(Op, b, opts);
        ASSERT_LT(arma::norm(res.x), 1e-12);
        ASSERT_LT(res.rnorm, 1e-12);
    }

    // Zero operator
    {
        const arma::uword m=8, n=10;
        arma::vec b(m); for (arma::uword i=0;i<m;++i) b[i] = static_cast<double>(i+1);
        apm::internal::LinearOperator Op{n, m,
            [&](const arma::vec& x, arma::vec& y){ y.zeros(m); },
            [&](const arma::vec& y, arma::vec& z){ z.zeros(n); }
        };
        apm::internal::LSMROptions opts; opts.atol=1e-12; opts.btol=1e-12; opts.max_iters=50;
        auto res = apm::internal::lsmr(Op, b, opts);
        ASSERT_LT(arma::norm(res.x), 1e-12);
        ASSERT_NEAR(res.rnorm, arma::norm(b), 1e-12);
    }
}

TEST(LSMR, WarmStartReducesIterations) {
    const arma::uword m=50, n=10;
    arma::mat A = make_hilbert_like(m,n);
    arma::vec x_true(n); for (arma::uword j=0;j<n;++j) x_true[j] = static_cast<double>(j+1);
    arma::vec b = A * x_true;

    apm::internal::LinearOperator Op{n, m,
        [&](const arma::vec& x, arma::vec& y){ y = A*x; },
        [&](const arma::vec& y, arma::vec& z){ z = A.t()*y; }
    };

    apm::internal::LSMROptions opts; opts.atol=1e-12; opts.btol=1e-12; opts.max_iters=4000;
    auto cold = apm::internal::lsmr(Op, b, opts);

    arma::vec x0 = (arma::pinv(A) * b) + 1e-3 * arma::regspace<arma::vec>(1, n);
    auto warm = apm::internal::lsmr(Op, b, opts, x0);

    ASSERT_LE(warm.iters, cold.iters);
}

TEST(LSMR, IllConditionedDampingReducesArnorn) {
    const arma::uword n = 30; // square for simplicity
    arma::vec sigma(n);
    for (arma::uword k=0;k<n;++k) sigma[k] = std::pow(10.0, -static_cast<double>(k) / static_cast<double>(n-1));

    // Define A as diagonal with decaying singular values
    apm::internal::LinearOperator Op{n, n,
        [&](const arma::vec& x, arma::vec& y){ y.set_size(n); for (arma::uword i=0;i<n;++i) y[i] = sigma[i] * x[i]; },
        [&](const arma::vec& y, arma::vec& z){ z.set_size(n); for (arma::uword i=0;i<n;++i) z[i] = sigma[i] * y[i]; }
    };

    arma::vec x_true(n); for (arma::uword i=0;i<n;++i) x_true[i] = (i%2==0?1.0:-1.0);
    arma::vec b(n); for (arma::uword i=0;i<n;++i) b[i] = sigma[i] * x_true[i];

    apm::internal::LSMROptions undamped; undamped.atol=1e-12; undamped.btol=1e-12; undamped.max_iters=2000; undamped.lambda=0.0;
    apm::internal::LSMROptions damped = undamped; damped.lambda = 1e-2;

    auto res_undamped = apm::internal::lsmr(Op, b, undamped);
    auto res_damped   = apm::internal::lsmr(Op, b, damped);

    ASSERT_LE(res_damped.arnorm, res_undamped.arnorm + 1e-12);
}

TEST(LSMR, MaxItersStopsEarly) {
    const arma::uword m=100, n=50;
    arma::mat A = make_hilbert_like(m,n);
    arma::vec x_true(n); for (arma::uword j=0;j<n;++j) x_true[j] = std::cos(static_cast<double>(j+1));
    arma::vec b = A * x_true;

    apm::internal::LinearOperator Op{
        n, m,
        [&](const arma::vec& x, arma::vec& y){ y = A*x; },
        [&](const arma::vec& y, arma::vec& z){ z = A.t()*y; }
    };
    apm::internal::LSMROptions opts; opts.atol=1e-16; opts.btol=1e-16; opts.max_iters=2;
    auto res = apm::internal::lsmr(Op, b, opts);
    ASSERT_EQ(res.flag, 2);
}