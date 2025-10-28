#include "linear_algebra_utils.h"
#include <cmath>

namespace apm {
namespace internal {

arma::mat projection_matrix(const arma::mat& X) {
    if (X.n_cols == 0) {
        return arma::mat(X.n_rows, X.n_rows, arma::fill::zeros);
    }

    arma::mat U, V;
    arma::vec s;
    arma::svd_econ(U, s, V, X, "left");

    // Determine the rank by finding the number of singular values greater than a tolerance.
    double tol = std::max(X.n_rows, X.n_cols) * s.max() * arma::Datum<double>::eps;
    arma::uword r = arma::sum(s > tol);

    if (r == 0) {
        return arma::mat(X.n_rows, X.n_rows, arma::fill::zeros);
    }

    // Use only the first 'r' columns of U, which form a basis for the column space.
    arma::mat U_r = U.cols(0, r - 1);

    return U_r * U_r.t();
}

arma::mat multi_min_norm_solve(const arma::mat& A, const arma::mat& B) {
    if (A.n_rows != B.n_rows) {
        throw std::invalid_argument("The number of rows in A must match the number of rows in B.");
    }
    return arma::pinv(A) * B;
}

arma::vec min_norm_solve(const arma::mat& A, const arma::vec& b) {
    return arma::pinv(A) * b;
}

arma::vec exact_column_norms(const LinearOperator& A, double eps) {
    arma::vec d(A.domain_dim, arma::fill::zeros);
    arma::vec ej(A.domain_dim, arma::fill::zeros);
    arma::vec y(A.range_dim, arma::fill::zeros);
    for (arma::uword j = 0; j < A.domain_dim; ++j) {
        ej.zeros(); ej[j] = 1.0;
        A.apply(ej, y);
        d[j] = std::max(eps, arma::norm(y));
    }
    return d;
}

arma::vec hutchinson_column_norms(const LinearOperator& A, std::size_t K, double eps) {
    arma::uword n = A.domain_dim, m = A.range_dim;
    arma::vec diag_est(n, arma::fill::zeros);
    arma::vec s(n), As(m), Ats(n);
    for (std::size_t k = 0; k < K; ++k) {
        s.randn();
        s = arma::sign(s);
        s.replace(0.0, 1.0);
        A.apply(s, As);
        A.apply_transpose(As, Ats);
        diag_est += Ats % s;
    }
    diag_est /= static_cast<double>(K);
    arma::vec d = arma::sqrt(arma::clamp(diag_est, eps*eps, std::numeric_limits<double>::infinity()));
    return d;
}

std::pair<LinearOperator, arma::vec> make_column_scaled_operator(
    const LinearOperator& A,
    bool exact_col_scaling,
    std::optional<std::size_t> K,
    double eps)
{
    arma::vec d = exact_col_scaling ? exact_column_norms(A, eps)
                                    : hutchinson_column_norms(A, K.value_or(16), eps);

    LinearOperator S;
    S.domain_dim = A.domain_dim;
    S.range_dim  = A.range_dim;
    S.apply = [A, d](const arma::vec& xs, arma::vec& y) {
        arma::vec x = xs / d;
        A.apply(x, y);
    };
    S.apply_transpose = [A, d](const arma::vec& y, arma::vec& zs) {
        arma::vec z(A.domain_dim, arma::fill::zeros);
        A.apply_transpose(y, z);
        zs = z / d;
    };
    return {S, d};
}

static inline double safe_div(double num, double den) {
    const double eps = std::numeric_limits<double>::epsilon();
    return num / std::max(den, eps);
}

namespace {

// Symmetric Givens rotation helper: given a and b, returns c, s, and r such that
// [c s; s -c]^T * [a; b] = [r; 0] with r >= 0
static inline void sym_ortho(double a, double b, double& c, double& s, double& r) {
    if (b == 0.0) {
        c = (a >= 0.0) ? 1.0 : -1.0;
        s = 0.0;
        r = std::abs(a);
    } else if (a == 0.0) {
        c = 0.0;
        s = (b >= 0.0) ? 1.0 : -1.0;
        r = std::abs(b);
    } else if (std::abs(b) > std::abs(a)) {
        double tau = a / b;
        s = ((b >= 0.0) ? 1.0 : -1.0) / std::sqrt(1.0 + tau * tau);
        c = s * tau;
        r = b / s;
    } else {
        double tau = b / a;
        c = ((a >= 0.0) ? 1.0 : -1.0) / std::sqrt(1.0 + tau * tau);
        s = c * tau;
        r = a / c;
    }
}

LSMRResult lsmr_core(const LinearOperator& A,
                            const arma::vec& b,
                            const LSMROptions& o,
                            std::optional<arma::vec> x0)
{
    const arma::uword n = A.domain_dim;
    const arma::uword m = A.range_dim;
    if (b.n_rows != m) {
        throw std::invalid_argument("lsmr: dimension mismatch: b length must equal A.range_dim");
    }

    arma::vec x = x0.value_or(arma::vec(n, arma::fill::zeros));

    arma::vec u = b;
    double normb = arma::norm(b);
    double beta;
    if (x0) {
        arma::vec Ax(m, arma::fill::zeros);
        A.apply(x, Ax);
        u -= Ax;
        beta = arma::norm(u);
    } else {
        beta = normb;
    }

    arma::vec v(n, arma::fill::zeros);
    double alpha = 0.0;
    if (beta > 0.0) {
        u /= beta;
        arma::vec Atu(n, arma::fill::zeros);
        A.apply_transpose(u, Atu);
        alpha = arma::norm(Atu);
        if (alpha > 0.0) v = Atu / alpha;
    }

    // Initialize variables for 1st iteration (mirror SciPy)
    std::size_t itn = 0;
    double damp = o.lambda;

    double zetabar = alpha * beta;
    double alphabar = alpha;
    double rho = 1.0;
    double rhobar = 1.0;
        double cbar = 1.0;
        double sbar = 0.0;

    arma::vec h = v;
    arma::vec hbar(n, arma::fill::zeros);

    // For ||r|| estimate
    double betadd = beta;
    double betad = 0.0;
    double rhodold = 1.0;
    double tautildeold = 0.0;
    double thetatilde = 0.0;
    double zeta = 0.0;
    double d = 0.0;

    // For ||A|| and cond(A)
    double normA2 = alpha * alpha;
    double maxrbar = 0.0;
    double minrbar = 1e100;
    double normA = std::sqrt(normA2);
    double condA = 1.0;
    double normx = 0.0;

    // Stopping
    int istop = 0;
    double ctol = 0.0;
    if (o.conlim > 0.0) ctol = 1.0 / o.conlim;
    double normr = beta;
    double normar = alpha * beta;

    if (normar == 0.0) {
        return {x, itn, normr, normar, normA, condA, 0};
    }
    if (normb == 0.0) {
        x.zeros();
        return {x, itn, normr, normar, normA, condA, 0};
    }

    // Main iteration loop
    arma::vec temp_m(m), temp_n(n);
    while (itn < o.max_iters) {
        itn += 1;

        // Bidiagonalization step
        u *= -alpha;
        A.apply(v, temp_m); // temp_m = A*v
        u += temp_m;
        beta = arma::norm(u);
        if (beta > 0.0) {
            u /= beta;
            v *= -beta;
            A.apply_transpose(u, temp_n); // temp_n = A^T*u
            v += temp_n;
            alpha = arma::norm(v);
            if (alpha > 0.0) v /= alpha;
        }

        // Qhat rotation for damping
        double chat, shat, alphahat;
        sym_ortho(alphabar, damp, chat, shat, alphahat);

        // Rotation Q_i
        double c, s;
        double rhoold = rho;
        sym_ortho(alphahat, beta, c, s, rho);
        double thetanew = s * alpha;
        alphabar = c * alpha;

        // Rotation Qbar_i
        double rhobarold = rhobar;
        double zetaold = zeta;
        double thetabar = sbar * rho;
        double rhotemp = cbar * rho;
        sym_ortho(cbar * rho, thetanew, cbar, sbar, rhobar);
        zeta = cbar * zetabar;
        zetabar = -sbar * zetabar;

        // Update h, hbar, x
        // hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
        hbar *= -(thetabar * rho / (rhoold * rhobarold));
        hbar += h;
        x += (zeta / (rho * rhobar)) * hbar;
        h *= -(thetanew / rho);
        h += v;

        // ||r|| estimate
        double betaacute = chat * betadd;
        double betacheck = -shat * betadd;
        double betahat = c * betaacute;
        betadd = -s * betaacute;

        double thetatildeold = thetatilde;
        double ctildeold, stildeold, rhotildeold;
        sym_ortho(rhodold, thetabar, ctildeold, stildeold, rhotildeold);
        thetatilde = stildeold * rhobar;
        rhodold = ctildeold * rhobar;
        betad = -stildeold * betad + ctildeold * betahat;

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
        double taud = (zeta - thetatilde * tautildeold) / rhodold;
        d = d + betacheck * betacheck;
        normr = std::sqrt(d + (betad - taud) * (betad - taud) + betadd * betadd);

        // ||A|| estimate
        normA2 = normA2 + beta * beta;
        normA = std::sqrt(normA2);
        normA2 = normA2 + alpha * alpha;

        // Condition estimate
        maxrbar = std::max(maxrbar, rhobarold);
        if (itn > 1) minrbar = std::min(minrbar, rhobarold);
        condA = std::max(maxrbar, rhotemp) / std::min(minrbar, rhotemp);

        // Convergence tests
        normar = std::abs(zetabar);
        normx = arma::norm(x);

        double test1 = normr / std::max(1e-300, normb);
        double test2 = (normA * normr != 0.0) ? (normar / (normA * normr)) : std::numeric_limits<double>::infinity();
        double test3 = 1.0 / std::max(condA, 1e-300);
        double t1 = test1 / (1.0 + (normA * normx) / std::max(1e-300, normb));
        double rtol = o.btol + o.atol * normA * normx / std::max(1e-300, normb);

        if (itn >= o.max_iters) istop = 7;
        if (1.0 + test3 <= 1.0) istop = 6;
        if (1.0 + test2 <= 1.0) istop = 5;
        if (1.0 + t1 <= 1.0) istop = 4;
        if (test3 <= ctol) istop = 3;
        // Require both tests to pass to ensure approach to min-norm solution
        bool ok2 = (test2 <= o.atol);
        bool ok1 = (test1 <= rtol);
        if (ok1 && ok2) istop = 1;

        if (istop > 0) break;
    }

    int flag;
    if (istop == 7) flag = 2; // max iters
    else if (istop == 3 || istop == 6) flag = 1; // condition limit
    else flag = 0; // converged

    return {x, itn, normr, normar, normA, condA, flag};
}

} // anonymous namespace

LSMRResult lsmr(const LinearOperator& A,
                const arma::vec& b,
                const LSMROptions& o,
                std::optional<arma::vec> x0,
                bool diagonal_precond,
                bool exact_col_scaling,
                std::optional<std::size_t> K,
                std::size_t homotopy_iters)
{
    LinearOperator Ao = A;
    arma::vec d; // column scaling for preconditioned coordinates
    if (diagonal_precond) {
        auto pair = make_column_scaled_operator(A, exact_col_scaling, K);
        Ao = pair.first;
        d  = std::move(pair.second);
    }

    // Map x0 into scaled coordinates if needed
    std::optional<arma::vec> x0_s;
    if (x0) {
        x0_s = diagonal_precond ? std::optional<arma::vec>((*x0) % d) : x0;
    }

    if (homotopy_iters == 0) {
        auto r = lsmr_core(Ao, b, o, x0_s);
        if (diagonal_precond) r.x /= d;
        return r;
    }

    // Homotopy: run damped stages then final undamped
    arma::vec x_s = x0_s.value_or(arma::vec(Ao.domain_dim, arma::fill::zeros));

    double lambda0 = (o.lambda > 0.0) ? o.lambda : 1e-2;
    const double decay = 0.1;
    std::size_t per_stage = std::max<std::size_t>(50, o.max_iters / (homotopy_iters + 1));

    LSMROptions stage = o;
    for (std::size_t k = 0; k < homotopy_iters; ++k) {
        stage = o;
        stage.lambda = lambda0 * std::pow(decay, homotopy_iters - 1 - k);
        stage.max_iters = per_stage;
        auto rk = lsmr_core(Ao, b, stage, std::optional<arma::vec>(x_s));
        x_s = rk.x;
    }

    stage = o; stage.lambda = 0.0;
    auto rfinal = lsmr_core(Ao, b, stage, std::optional<arma::vec>(x_s));
    if (diagonal_precond) rfinal.x /= d;
    return rfinal;
}

} // namespace internal

} // namespace apm 