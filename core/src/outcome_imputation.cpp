#include "outcome_imputation.h"
#include "outcome_imputation_helpers.h"
#include "linear_algebra_utils.h"
#include <stdexcept>
#include <limits>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <vector>
#include "panels/CohortLevelUnbalancedPanel.h"
#include "panels/InMemoryUnbalancedPanel.h"
#ifdef APM_HAS_TBB
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#endif
#include "utils.h"
#include "bootstrap.h"

namespace apm {

namespace {

static arma::vec assemble_w_i(
	const UnitRun& ur,
	const arma::uvec& T_idxs,
	const std::unordered_map<int, std::size_t>& pos_map,
    const AbstractUnbalancedPanel& panel,
	const VariableSpec& var,
	arma::vec& Y_buf,
	arma::mat& X_obs_buf)
{
	if (var.kind == VariableSpec::Kind::Outcome) {
		panel.assemble_Y_for_unit(ur, T_idxs, pos_map, Y_buf);
		return Y_buf;
	} else {
		panel.assemble_X_for_unit(ur, panel.T(), T_idxs, /*X_full=*/nullptr, /*X_obs=*/&X_obs_buf, var.covariate_index);
		return X_obs_buf.col(0);
	}
}

} // anonymous namespace

namespace internal {

arma::vec comp_lambda_i(const arma::mat& G_c, const arma::vec& w_i)
{
    // Shortcut: if G_c is a single column of all ones, return average of w_i
    if (G_c.n_cols == 1) {
        bool all_ones = true;
        for (arma::uword i = 0; i < G_c.n_rows; ++i) {
            if (G_c(i, 0) != 1.0) { all_ones = false; break; }
        }
        if (all_ones) {
            arma::vec lambda(1);
            lambda[0] = arma::mean(w_i);
            return lambda;
        }
    }
    return ::apm::internal::min_norm_solve(G_c, w_i);
}

} // namespace internal

std::pair<std::optional<arma::vec>, std::optional<arma::mat>> apm::internal::get_unit_and_outcome_specific_params(
    std::optional<arma::vec> g_0_prev_opt,
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    std::optional<arma::vec> covar_coefs_for_residualization,
    bool store_unit_params)
{
    const ObservedOutcomeIndices& ooi = effective_ooi_opt ? *effective_ooi_opt
                                                          : panel.observed_outcome_indices();
    const arma::mat& G = factor_model_params.G;
    const std::size_t T = panel.T();
    const std::size_t r = static_cast<std::size_t>(G.n_cols);

    if (var.kind == VariableSpec::Kind::Covariate && var.covariate_index >= panel.q()) {
        throw std::invalid_argument("covariate_index out of range for panel.q()");
    }

    const bool update_g0 = static_cast<bool>(g_0_prev_opt);
    if (update_g0) {
        if (g_0_prev_opt->n_elem != static_cast<arma::uword>(T)) {
        throw std::invalid_argument("g_0_prev has incompatible length with panel.T()");
        }
    } else {
        if (!store_unit_params) {
            throw std::invalid_argument("When g_0_prev is not provided, store_unit_params must be true to return lambda.");
        }
    }

    if (unit_weights_opt && unit_weights_opt->n_elem != static_cast<arma::uword>(panel.num_units())) {
        throw std::invalid_argument("unit_weights_opt has incompatible length with panel.num_units()");
    }

    arma::vec g_0_in_progress;
    arma::vec total_weight_per_outcome;
    if (update_g0) {
        g_0_in_progress.set_size(static_cast<arma::uword>(T));
        g_0_in_progress.fill(0.0);
        total_weight_per_outcome.set_size(static_cast<arma::uword>(T));
        total_weight_per_outcome.fill(0.0);
    }

    std::optional<arma::mat> lambda_opt;
    if (store_unit_params) {
        const std::size_t n_units = panel.num_units();
        lambda_opt.emplace(static_cast<arma::uword>(n_units), static_cast<arma::uword>(r), arma::fill::zeros);
    }

    arma::vec Y;
    arma::mat X_obs;

    for (const auto& blk : panel.cohort_blocks()) {
        const auto cohort0 = blk.cohort;
        const arma::uvec& T_idxs = ooi.at(cohort0);
        const std::size_t T_c = static_cast<std::size_t>(T_idxs.n_elem);
        if (T_c == 0) continue;

        arma::mat G_c = G.rows(T_idxs); // T_c x r
        arma::vec g_0c_prev;
        if (update_g0) {
            g_0c_prev = g_0_prev_opt->elem(T_idxs); // T_c
        }
        const auto& pos_map = panel.pos_T_idx_for_cohort(cohort0);

        for (const auto& ur : blk.unit_runs) {
            arma::vec w_i = assemble_w_i(ur, T_idxs, pos_map, panel, var, Y, X_obs);

            // Optional residualization by covariates for outcomes
            if (covar_coefs_for_residualization && var.kind == VariableSpec::Kind::Outcome && panel.q() > 0) {
                arma::mat X_obs_q;
                panel.assemble_X_for_unit(ur, panel.T(), T_idxs, /*X_full=*/nullptr, /*X_obs=*/&X_obs_q, std::nullopt);
                arma::vec covar_term = X_obs_q * (*covar_coefs_for_residualization);
                w_i -= covar_term;
            }

            // Solve for lambda_i using residual against previous g_0
            arma::vec r_for_lambda = update_g0 ? (w_i - g_0c_prev) : w_i;
            arma::vec lambda_i = apm::internal::comp_lambda_i(G_c, r_for_lambda); // r x 1

            if (store_unit_params) {
                // ur.unit is 0-based index of the unit
                (*lambda_opt).row(static_cast<arma::uword>(ur.unit)) = lambda_i.t();
            }

            // Residual used to update g_0 via online weighted averaging
            if (update_g0) {
            arma::vec r_i = w_i - (G_c * lambda_i); // T_c x 1

                const double u_w = unit_weights_opt ? (*unit_weights_opt)[static_cast<arma::uword>(ur.unit)] : 1.0;
                if (std::isfinite(u_w) && u_w > 0.0) {
                    arma::vec w_old = total_weight_per_outcome.elem(T_idxs) / (total_weight_per_outcome.elem(T_idxs) + u_w);
                    arma::vec g_old = g_0_in_progress.elem(T_idxs);
                    g_0_in_progress.elem(T_idxs) = w_old % g_old + (1.0 - w_old) % r_i;
                    total_weight_per_outcome.elem(T_idxs) += u_w;
                }
            }
        }
    }

    if (update_g0) {
    // normalize g_0_in_progress to be orthogonal to the rows of G:
        arma::vec g_0_result = g_0_in_progress - G * ::apm::internal::min_norm_solve(G, g_0_in_progress);
        return std::make_pair(std::optional<arma::vec>(std::move(g_0_result)), std::move(lambda_opt));
    } else {
        return std::make_pair(std::optional<arma::vec>(), std::move(lambda_opt));
    }
}

arma::vec apm::internal::get_outcome_specific_params(
    const arma::vec& g_0_prev,
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    std::optional<arma::vec> covar_coefs_for_residualization)
{
    auto res = apm::internal::get_unit_and_outcome_specific_params(
        std::optional<arma::vec>(g_0_prev), panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, covar_coefs_for_residualization, /*store_unit_params=*/false);
    if (!res.first.has_value()) {
        throw std::runtime_error("Expected g_0 in get_outcome_specific_params result");
    }
    return std::move(*res.first);
}

namespace internal {

std::pair<std::optional<arma::vec>, std::optional<arma::mat>> comp_unit_and_outcome_specific_params_fixed_point(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp,
    std::optional<arma::vec> covar_coefs_for_residualization,
    bool store_unit_params)
{
    const std::size_t T = panel.T();

    if (var.kind == VariableSpec::Kind::Covariate && var.covariate_index >= panel.q()) {
        throw std::invalid_argument("covariate_index out of range for panel.q()");
    }

    arma::vec g_0(static_cast<arma::uword>(T), arma::fill::zeros);

    auto F = [&](const arma::vec& g) -> arma::vec {
        return apm::internal::get_outcome_specific_params(
            g, panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, covar_coefs_for_residualization);
    };

    auto apply_k = [&](const arma::vec& x, std::size_t k) -> arma::vec {
        arma::vec y = x;
        for (std::size_t i = 0; i < k; ++i) y = F(y);
        return y;
    };

    auto aitken = [&](const arma::vec& x0, const arma::vec& x1, const arma::vec& x2) -> arma::vec {
        arma::vec d1 = x1 - x0;
        arma::vec d2 = x2 - x1;
        arma::vec dd = d2 - d1;
        double denom = arma::dot(dd, dd);
        if (denom <= 0.0) return x2;
        double tau = - arma::dot(d2, dd) / denom;
        if (!std::isfinite(tau)) return x2;
        return x2 + tau * d2;
    };

    std::size_t iter = 0;
    for (; iter < fp.max_iters; ++iter) {
        // 1) Always start with one projection
        arma::vec g_proj = F(g_0);

        // Optional extra simple projections before acceleration
        for (std::size_t e = 0; e < fp.extra_proj; ++e) {
            g_proj = F(g_proj);
        }

        const bool use_accel = (fp.method == apm::AccelMethod::IronsTuck);
        const bool do_grand = use_accel && fp.grand_period > 0 && ((iter + 1) % fp.grand_period == 0);

        arma::vec g_next;
        if (!use_accel) {
            g_next = g_proj;
        } else if (do_grand) {
            // Grand acceleration on h(X)=f^k(X)
            arma::vec h1 = apply_k(g_0, fp.grand_k);
            arma::vec h2 = apply_k(h1, fp.grand_k);
            g_next = aitken(g_0, h1, h2);
        } else {
            // Regular acceleration on X, f(X), f(f(X))
            arma::vec g2 = F(g_proj);
            g_next = aitken(g_0, g_proj, g2);
        }

        // Stabilization: one extra projection after acceleration once running long
        if (fp.stabilize_after > 0 && (iter + 1) >= fp.stabilize_after && use_accel) {
            g_next = F(g_next);
        }

        arma::vec r_k = g_next - g_0;
        if (arma::norm(r_k, "inf") <= fp.tol) {
            g_0 = std::move(g_next);
            break;
        }

        g_0 = std::move(g_next);
    }

    if (iter == fp.max_iters) {
        std::cerr << "Warning: comp_unit_and_outcome_specific_params_fixed_point did not converge within max_iters="
                  << fp.max_iters << ", tol=" << fp.tol << std::endl;
    }

    return apm::internal::get_unit_and_outcome_specific_params(
        std::optional<arma::vec>(g_0), panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, covar_coefs_for_residualization, store_unit_params);
}

arma::vec comp_outcome_specific_params_fixed_point(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp)
{
    auto res = comp_unit_and_outcome_specific_params_fixed_point(
        panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, fp, std::nullopt, /*store_unit_params=*/false);
    if (!res.first.has_value()) {
        throw std::runtime_error("Expected g_0 in comp_outcome_specific_params_fixed_point result");
    }
    return std::move(*res.first);
}

static std::pair<std::optional<arma::vec>, std::optional<arma::mat>> comp_unit_and_outcome_specific_params_lsmr(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp,
    std::optional<arma::vec> covar_coefs_for_residualization,
    bool store_unit_params)
{
    const ObservedOutcomeIndices& ooi = effective_ooi_opt ? *effective_ooi_opt
                                                          : panel.observed_outcome_indices();
    const arma::mat& G = factor_model_params.G;
    const std::size_t T = panel.T();

    // Compute null-space basis N of G^T (T x d)
    arma::mat N = arma::null(G.t());
    const arma::uword d = N.n_cols;
    if (d == 0) {
        // Degenerate: no null space; g_0 must be zero
        arma::vec g0(T, arma::fill::zeros);
        std::optional<arma::mat> L_opt;
        if (store_unit_params) {
            const std::size_t r = static_cast<std::size_t>(G.n_cols);
            L_opt.emplace(static_cast<arma::uword>(panel.num_units()), static_cast<arma::uword>(r), arma::fill::zeros);
            // Compute per-unit lambdas against zero g0
            arma::vec Y; arma::mat X_obs;
            for (const auto& blk : panel.cohort_blocks()) {
                const auto cohort0 = blk.cohort;
                const arma::uvec& T_idxs = ooi.at(cohort0);
                arma::mat G_c = G.rows(T_idxs);
                const auto& pos_map = panel.pos_T_idx_for_cohort(cohort0);
                for (const auto& ur : blk.unit_runs) {
                    arma::vec w_i = assemble_w_i(ur, T_idxs, pos_map, panel, var, Y, X_obs);
                    arma::vec lambda_i = apm::internal::comp_lambda_i(G_c, w_i);
                    (*L_opt).row(static_cast<arma::uword>(ur.unit)) = lambda_i.t();
                }
            }
        }
        return std::make_pair(std::optional<arma::vec>(std::move(g0)), std::move(L_opt));
    }

    // Build cohort-level sufficient stats: s_c and ybar_c
    struct CohStats { arma::uvec idx; double s_sqrt; arma::vec ybar; };
    std::vector<CohStats> cohorts;
    cohorts.reserve(ooi.size());

    arma::vec Y; arma::mat X_obs;
    for (const auto& blk : panel.cohort_blocks()) {
        const auto cohort0 = blk.cohort;
        const arma::uvec& T_idxs = ooi.at(cohort0);
        if (T_idxs.n_elem == 0) continue;

        arma::vec accum(T_idxs.n_elem, arma::fill::zeros);
        double s = 0.0;
        const auto& pos_map = panel.pos_T_idx_for_cohort(cohort0);

        for (const auto& ur : blk.unit_runs) {
			arma::vec w_i = assemble_w_i(ur, T_idxs, pos_map, panel, var, Y, X_obs);
			if (covar_coefs_for_residualization && var.kind == VariableSpec::Kind::Outcome && panel.q() > 0) {
                arma::mat X_obs_q;
                panel.assemble_X_for_unit(ur, panel.T(), T_idxs, /*X_full=*/nullptr, /*X_obs=*/&X_obs_q, std::nullopt);
                arma::vec covar_term = X_obs_q * (*covar_coefs_for_residualization);
                w_i -= covar_term;
            }
            double u_w = unit_weights_opt ? (*unit_weights_opt)[static_cast<arma::uword>(ur.unit)] : 1.0;
            if (!(std::isfinite(u_w) && u_w > 0.0)) continue;
            accum += u_w * w_i;
            s += u_w;
        }

        if (s <= 0.0) continue;
        arma::vec ybar = accum / s;
        cohorts.push_back(CohStats{T_idxs, std::sqrt(s), std::move(ybar)});
    }

    // Construct operator A' via matrix-free apply/apply_transpose
    apm::internal::LinearOperator Aop;
    {
        arma::uword mprime = 0;
        for (const auto& cs : cohorts) mprime += cs.idx.n_elem;
        Aop.domain_dim = d;
        Aop.range_dim = mprime;
        Aop.apply = [&, G, N, cohorts](const arma::vec& v, arma::vec& y) {
            y.set_size(Aop.range_dim);
            arma::uword off = 0;
            for (const auto& cs : cohorts) {
                arma::mat G_c = G.rows(cs.idx);
                arma::vec Nc_v = N.rows(cs.idx) * v;
                arma::vec proj = Nc_v - G_c * ::apm::internal::min_norm_solve(G_c, Nc_v);
                y.subvec(off, off + cs.idx.n_elem - 1) = cs.s_sqrt * proj;
                off += cs.idx.n_elem;
            }
        };
        Aop.apply_transpose = [&, G, N, cohorts](const arma::vec& u, arma::vec& z) {
            z.zeros(d);
            arma::uword off = 0;
            for (const auto& cs : cohorts) {
                arma::mat G_c = G.rows(cs.idx);
                arma::vec uc = u.subvec(off, off + cs.idx.n_elem - 1) * cs.s_sqrt;
                arma::vec proj = uc - G_c * ::apm::internal::min_norm_solve(G_c, uc);
                z += N.rows(cs.idx).t() * proj;
                off += cs.idx.n_elem;
            }
        };
    }

    // Build b' stacked over cohorts: sqrt(s_c) * P_c ybar_c
    arma::vec bprime(Aop.range_dim, arma::fill::zeros);
    {
        arma::uword off = 0;
        for (const auto& cs : cohorts) {
            arma::mat G_c = G.rows(cs.idx);
            arma::vec proj = cs.ybar - G_c * ::apm::internal::min_norm_solve(G_c, cs.ybar);
            bprime.subvec(off, off + cs.idx.n_elem - 1) = cs.s_sqrt * proj;
            off += cs.idx.n_elem;
        }
    }

    // Solve for z with LSMR
    apm::internal::LSMROptions lopts;
    lopts.atol = fp.lsmr_atol;
    lopts.btol = fp.lsmr_btol;
    lopts.conlim = fp.lsmr_conlim;
    lopts.max_iters = (fp.lsmr_max_iters > 0)
        ? fp.lsmr_max_iters
        : (fp.max_iters == std::numeric_limits<std::size_t>::max() ? static_cast<std::size_t>(2 * (T + d)) : fp.max_iters);
    lopts.lambda = fp.lsmr_lambda;

    apm::internal::LSMRResult lres = apm::internal::lsmr(
        Aop,
        bprime,
        lopts,
        /*x0=*/std::nullopt,
        /*diagonal_precond=*/fp.lsmr_diagonal_precond,
        /*num_diag_approx_draws=*/fp.lsmr_num_diag_approx_draws,
        /*homotopy_iters=*/fp.lsmr_homotopy_iters);

    arma::vec g0 = N * lres.x;
    // Re-orthogonalize for numerical stability
    g0 -= G * ::apm::internal::min_norm_solve(G, g0);

    std::optional<arma::mat> L_opt;
    if (store_unit_params) {
        const std::size_t r = static_cast<std::size_t>(G.n_cols);
        L_opt.emplace(static_cast<arma::uword>(panel.num_units()), static_cast<arma::uword>(r), arma::fill::zeros);
        arma::vec Y2; arma::mat X_obs2;
        for (const auto& blk : panel.cohort_blocks()) {
            const auto cohort0 = blk.cohort;
            const arma::uvec& T_idxs = ooi.at(cohort0);
            if (T_idxs.n_elem == 0) continue;
            arma::mat G_c = G.rows(T_idxs);
            arma::vec g0c = g0.elem(T_idxs);
            const auto& pos_map = panel.pos_T_idx_for_cohort(cohort0);
            for (const auto& ur : blk.unit_runs) {
				arma::vec w_i = assemble_w_i(ur, T_idxs, pos_map, panel, var, Y2, X_obs2);
				if (covar_coefs_for_residualization && var.kind == VariableSpec::Kind::Outcome && panel.q() > 0) {
                    arma::mat X_obs_q;
                    panel.assemble_X_for_unit(ur, panel.T(), T_idxs, /*X_full=*/nullptr, /*X_obs=*/&X_obs_q, std::nullopt);
                    arma::vec covar_term = X_obs_q * (*covar_coefs_for_residualization);
                    w_i -= covar_term;
                }
                arma::vec r_for_lambda = w_i - g0c;
                arma::vec lambda_i = apm::internal::comp_lambda_i(G_c, r_for_lambda);
                (*L_opt).row(static_cast<arma::uword>(ur.unit)) = lambda_i.t();
            }
        }
    }

    return std::make_pair(std::optional<arma::vec>(std::move(g0)), std::move(L_opt));
}

// New: g0-only wrapper for LSMR
arma::vec comp_outcome_specific_params_lsmr(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp)
{
    auto res = comp_unit_and_outcome_specific_params_lsmr(
        panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, fp, std::nullopt, /*store_unit_params=*/false);
    if (!res.first.has_value()) {
        throw std::runtime_error("Expected g_0 in comp_outcome_specific_params_lsmr result");
    }
    return std::move(*res.first);
}

std::pair<std::optional<arma::vec>, std::optional<arma::mat>> comp_unit_and_outcome_specific_params(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp,
    std::optional<arma::vec> covar_coefs_for_residualization,
    bool store_unit_params)
{
    // If there are no fixed effects, ignore LSMR and only compute unit-specific params when requested
    if (!factor_model_params.has_fixed_effects()) {
        if (store_unit_params) {
            auto res = apm::internal::get_unit_and_outcome_specific_params(
                std::optional<arma::vec>(), panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, covar_coefs_for_residualization, /*store_unit_params=*/true);
            return std::make_pair(std::optional<arma::vec>(), std::move(res.second));
        }
        return std::make_pair(std::optional<arma::vec>(), std::optional<arma::mat>());
    }

    // With fixed effects present: dispatch to requested solver
    if (fp.solver == apm::ImputationSolver::LSMR) {
        return comp_unit_and_outcome_specific_params_lsmr(
            panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, fp, covar_coefs_for_residualization, store_unit_params);
    }

    return apm::internal::comp_unit_and_outcome_specific_params_fixed_point(
        panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, fp, covar_coefs_for_residualization, store_unit_params);
}

arma::vec comp_outcome_specific_params(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp)
{
    auto res = apm::internal::comp_unit_and_outcome_specific_params(
        panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, fp, std::nullopt, /*store_unit_params=*/false);
    if (!res.first.has_value()) {
        throw std::runtime_error("Expected g_0 in comp_outcome_specific_params result");
    }
    return std::move(*res.first);
}

std::optional<arma::mat> get_unit_specific_params(
    const AbstractUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    std::optional<arma::vec> covar_coefs_for_residualization)
{
    auto res = apm::internal::get_unit_and_outcome_specific_params(
        std::optional<arma::vec>(), panel, var, factor_model_params, unit_weights_opt, effective_ooi_opt, covar_coefs_for_residualization, /*store_unit_params=*/true);
    return std::move(res.second);
}

arma::vec comp_covar_coefs(
    const AbstractUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> g_0_init,
    std::vector<arma::vec> g_0_init_covars,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt)
{
    const std::size_t q = factor_model_params.q();
    if (q == 0) return arma::vec();
    if (!g_0_init_covars.empty() && !g_0_init.has_value()) {
        throw std::invalid_argument("g_0_init_covars provided but g_0_init is not; inconsistent inputs");
    }

    const ObservedOutcomeIndices& ooi = effective_ooi_opt ? *effective_ooi_opt
                                                          : panel.observed_outcome_indices();
    const arma::mat& G = factor_model_params.G;

    if (unit_weights_opt && unit_weights_opt->n_elem != static_cast<arma::uword>(panel.num_units())) {
        throw std::invalid_argument("unit_weights_opt has incompatible length with panel.num_units()");
    }

    // Online weighted averages for outer products and cross terms
    arma::mat XTX_avg(static_cast<arma::uword>(q), static_cast<arma::uword>(q), arma::fill::zeros);
    arma::vec XTy_avg(static_cast<arma::uword>(q), arma::fill::zeros);
    double total_weight_accum = 0.0;

    arma::vec Y; arma::mat X_full; arma::mat X_obs;

    for (const auto& blk : panel.cohort_blocks()) {
        const auto cohort0 = blk.cohort;
        const arma::uvec& T_idxs = ooi.at(cohort0);
        if (T_idxs.n_elem == 0) continue;

        arma::mat G_c = G.rows(T_idxs);
        arma::vec g0_c;
        if (g_0_init) g0_c = g_0_init->elem(T_idxs);

        std::vector<arma::vec> g0cov_c;
        if (!g_0_init_covars.empty()) {
            g0cov_c.reserve(q);
            for (std::size_t j = 0; j < q; ++j) {
                g0cov_c.push_back(g_0_init_covars[j].elem(T_idxs));
            }
        }

        const auto& pos_map = panel.pos_T_idx_for_cohort(cohort0);
        for (const auto& ur : blk.unit_runs) {
            panel.assemble_YX_for_unit(ur, panel.T(), T_idxs, pos_map, Y, X_full, X_obs);
            if (X_obs.n_cols != static_cast<arma::uword>(q)) continue;

            const double u_w = unit_weights_opt ? (*unit_weights_opt)[static_cast<arma::uword>(ur.unit)] : 1.0;
            if (!(std::isfinite(u_w) && u_w > 0.0)) continue;

            arma::vec y_tilde = g_0_init ? (Y - g0_c) : Y;
            arma::vec lambda_y = apm::internal::comp_lambda_i(G_c, y_tilde);
            arma::vec y_res = y_tilde - (G_c * lambda_y);

            arma::mat X_res = X_obs; // T_c x q
            for (std::size_t j = 0; j < q; ++j) {
                arma::vec x_j = X_obs.col(static_cast<arma::uword>(j));
                if (!g_0_init_covars.empty()) {
                    x_j -= g0cov_c[j];
                }
                arma::vec lambda_xj = apm::internal::comp_lambda_i(G_c, x_j);
                X_res.col(static_cast<arma::uword>(j)) = x_j - (G_c * lambda_xj);
            }

            for (arma::uword k = 0; k < X_res.n_rows; ++k) {
                double yk = y_res[k];
                if (!std::isfinite(yk)) continue;
                arma::rowvec xk = X_res.row(k);
                bool ok = true;
                for (arma::uword jj = 0; jj < xk.n_cols; ++jj) { if (!std::isfinite(xk[jj])) { ok = false; break; } }
                if (!ok) continue;

                arma::mat outer = xk.t() * xk; // q x q
                arma::vec cross = xk.t() * yk; // q x 1

                double new_total = total_weight_accum + u_w;
                double update_weight = u_w / new_total;
                XTX_avg = XTX_avg + update_weight * (outer - XTX_avg);
                XTy_avg = XTy_avg + update_weight * (cross - XTy_avg);
                total_weight_accum = new_total;
            }
        }
    }

arma::vec alpha = ::apm::internal::min_norm_solve(XTX_avg, XTy_avg);
    return alpha;
}

} // namespace internal

namespace {

static void validate_unit_weights_opt(const std::optional<arma::vec>& unit_weights_opt,
                                      std::size_t expected_len)
{
    if (!unit_weights_opt) return;
    if (unit_weights_opt->n_elem != static_cast<arma::uword>(expected_len)) {
        throw std::invalid_argument("unit_weights_opt has incompatible length with expected number of units");
    }
    double sum_w = 0.0;
    for (arma::uword i = 0; i < unit_weights_opt->n_elem; ++i) {
        double w = (*unit_weights_opt)[i];
        if (w < 0.0) {
            throw std::invalid_argument("unit_weights_opt contains negative weight(s)");
        }
        sum_w += w;
    }
    if (sum_w == 0.0) {
        throw std::invalid_argument("unit_weights_opt sum is zero but weights are non-negative");
    }
}
} // anonymous namespace

FactorModelParameters comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    const std::vector<OutcomeMeanSufficientStatistics>& cohort_outcome_mean_suff_stats,
    std::optional<arma::vec> unit_weights_opt,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp)
{
    std::optional<arma::vec> alpha_opt;

    // Validate unit-level weights once (used for pre-computations on the original panel)
    validate_unit_weights_opt(unit_weights_opt, panel.num_units());

    // Prepare cohort-level panel and weights once, if provided and sized correctly
    const bool have_cohort_stats = !cohort_outcome_mean_suff_stats.empty();
    const ObservedOutcomeIndices& ooi = effective_ooi_opt ? *effective_ooi_opt
                                                          : panel.observed_outcome_indices();
    const std::size_t n_cohorts = ooi.size();

    std::optional<CohortLevelUnbalancedPanel> cohort_panel_opt;
    std::optional<arma::vec> cohort_weights_opt;

    if (have_cohort_stats) {
        if (cohort_outcome_mean_suff_stats.size() != n_cohorts) {
            throw std::invalid_argument("cohort_outcome_mean_suff_stats length must equal number of cohorts");
        }
        cohort_panel_opt.emplace(cohort_outcome_mean_suff_stats, ooi);

        arma::vec cw(static_cast<arma::uword>(cohort_outcome_mean_suff_stats.size()));
        for (std::size_t i = 0; i < cohort_outcome_mean_suff_stats.size(); ++i) {
            cw[static_cast<arma::uword>(i)] = cohort_outcome_mean_suff_stats[i].cohort_pop_share;
        }
        cohort_weights_opt = std::move(cw);
    }

    // Pre-computations always use the original panel and unit-level weights
    if (factor_model_params.has_fixed_effects()) {
        if (factor_model_params.has_covariate_coefs() && factor_model_params.q() > 0) {
			arma::vec g_0_init = internal::comp_outcome_specific_params(
				panel, VariableSpec::outcome(), factor_model_params, unit_weights_opt, effective_ooi_opt, fp);

            const std::size_t q = factor_model_params.q();
            std::vector<arma::vec> g_0_init_covars(q);
            for (std::size_t j = 0; j < q; ++j) {
				g_0_init_covars[j] = internal::comp_outcome_specific_params(
					panel, VariableSpec::covariate(j), factor_model_params, unit_weights_opt, effective_ooi_opt, fp);
            }

            std::optional<arma::vec> g_0_init_opt = std::move(g_0_init);
            arma::vec alpha = apm::internal::comp_covar_coefs(panel, factor_model_params, g_0_init_opt, g_0_init_covars, unit_weights_opt, effective_ooi_opt);
            alpha_opt = std::move(alpha);
        }

        // Final pass: use cohort-level panel and weights if present; else original
        const AbstractUnbalancedPanel& final_panel = cohort_panel_opt ? static_cast<const AbstractUnbalancedPanel&>(*cohort_panel_opt)
                                                                      : panel;
        std::optional<arma::vec> final_weights = cohort_panel_opt ? cohort_weights_opt : unit_weights_opt;

		std::pair<std::optional<arma::vec>, std::optional<arma::mat>> final_pair = internal::comp_unit_and_outcome_specific_params(
			final_panel, VariableSpec::outcome(), factor_model_params, final_weights, effective_ooi_opt, fp, alpha_opt, /*store_unit_params=*/true);

        FactorModelParameters out;
        out.G = factor_model_params.G;
        if (final_pair.first.has_value()) out.g_0 = *final_pair.first; else out.g_0 = std::nullopt;
        out.a = alpha_opt;
        out.L = std::move(final_pair.second);
        return out;
    } else {
        if (factor_model_params.has_covariate_coefs() && factor_model_params.q() > 0) {
            arma::vec alpha = apm::internal::comp_covar_coefs(panel, factor_model_params, std::nullopt, {}, unit_weights_opt, effective_ooi_opt);
            alpha_opt = std::move(alpha);
        }

        const AbstractUnbalancedPanel& final_panel = cohort_panel_opt ? static_cast<const AbstractUnbalancedPanel&>(*cohort_panel_opt)
                                                                      : panel;
        std::optional<arma::vec> final_weights = cohort_panel_opt ? cohort_weights_opt : unit_weights_opt;

		std::optional<arma::mat> L_opt;
		{
			auto pair_l = internal::comp_unit_and_outcome_specific_params(
				final_panel, VariableSpec::outcome(), factor_model_params, final_weights, effective_ooi_opt, fp, alpha_opt, /*store_unit_params=*/true);
			L_opt = std::move(pair_l.second);
		}

        FactorModelParameters out;
        out.G = factor_model_params.G;
        out.a = alpha_opt;
        out.g_0 = std::nullopt;
        out.L = std::move(L_opt);
        return out;
    }
}

FactorModelEstimates comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelEstimates& factor_model_ests,
    const std::vector<OutcomeMeanSuffStatEstimates>& cohort_outcome_mean_suff_stat_ests,
    std::shared_ptr<const WeightedBootstrap> wb,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp,
    std::optional<std::size_t> num_threads)
{
    apm::ParallelismScope par_scope(num_threads);
    std::size_t nt = par_scope.nt;

    // If cohort stats are provided, construct point slice. If empty, pass through empty vector.
    std::vector<OutcomeMeanSufficientStatistics> suff_stats_point;
    if (!cohort_outcome_mean_suff_stat_ests.empty()) {
        const std::size_t C = static_cast<std::size_t>(cohort_outcome_mean_suff_stat_ests.size());
        suff_stats_point.reserve(C);
        for (std::size_t c = 0; c < C; ++c) {
            suff_stats_point.push_back(cohort_outcome_mean_suff_stat_ests[c].suff_stat_estimates);
        }
    }

    // Point estimate via parameter-based overload (no unit weights for point estimate)
    FactorModelParameters point_params = comp_imputation_components(
        panel,
        factor_model_ests.parameter_estimates,
        suff_stats_point,
        std::nullopt,
        effective_ooi_opt,
        fp);

    // Bootstrap replicates
    const bool has_param_boot = factor_model_ests.has_bootstrap_replicates();
    std::size_t B = has_param_boot ? factor_model_ests.n_bootstrap_replicates() : 0;

    if (has_param_boot && !cohort_outcome_mean_suff_stat_ests.empty()) {
        // Validate each cohort has B suff-stat replicates (mirror est_outcome_means.cpp)
        const std::size_t C = static_cast<std::size_t>(cohort_outcome_mean_suff_stat_ests.size());
        for (std::size_t c = 0; c < C; ++c) {
            if (cohort_outcome_mean_suff_stat_ests[c].n_bootstrap_replicates() != B) {
                throw std::invalid_argument(
                    "All cohorts must have the same number of bootstrap replicates as the factor model estimates.");
            }
        }
    }

    std::vector<FactorModelParameters> boot_out;
    if (B > 0) {
        if (!wb) {
            throw std::invalid_argument("WeightedBootstrap must be provided when bootstrap replicates are present.");
        }
        if (wb->n_obs() != panel.num_units()) throw std::invalid_argument("WeightedBootstrap n_obs must equal panel.num_units()");
        
        boot_out.resize(B);

        auto worker = [&](std::size_t b) {
            std::vector<OutcomeMeanSufficientStatistics> suff_stats_b;
            if (!cohort_outcome_mean_suff_stat_ests.empty()) {
                const std::size_t C = static_cast<std::size_t>(cohort_outcome_mean_suff_stat_ests.size());
                suff_stats_b.reserve(C);
                for (std::size_t c = 0; c < C; ++c) {
                    suff_stats_b.push_back(cohort_outcome_mean_suff_stat_ests[c].bootstrap_replicates[b]);
                }
            }

            const FactorModelParameters& params_b = factor_model_ests.bootstrap_replicates[b];
            arma::vec unit_weights_b = wb->draw(b);
            boot_out[b] = comp_imputation_components(
                panel,
                params_b,
                suff_stats_b,
                std::optional<arma::vec>(std::move(unit_weights_b)),
                effective_ooi_opt,
                fp);
        };

#ifdef APM_HAS_TBB
        if (nt <= 1) {
            for (std::size_t b = 0; b < B; ++b) worker(b);
        } else {
            oneapi::tbb::parallel_for(std::size_t(0), B, [&](std::size_t b){ worker(b); });
        }
#else
        for (std::size_t b = 0; b < B; ++b) worker(b);
#endif
    }

    return FactorModelEstimates(std::move(point_params), std::move(boot_out));
}

} // namespace apm

namespace apm {

std::unordered_map<std::string, FactorModelEstimates> comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, FactorModelEstimates>& factor_model_estimates_map,
    const std::vector<OutcomeMeanSuffStatEstimates>& cohort_outcome_mean_suff_stat_ests,
    std::shared_ptr<const WeightedBootstrap> wb,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    const apm::ImputationOptions& fp,
    std::optional<std::size_t> num_threads)
{
    std::unordered_map<std::string, FactorModelEstimates> out;
    out.reserve(factor_model_estimates_map.size());

    for (const auto& kv : factor_model_estimates_map) {
        const std::string& key = kv.first;
        const FactorModelEstimates& ests = kv.second;
        FactorModelEstimates res = comp_imputation_components(
            panel,
            ests,
            cohort_outcome_mean_suff_stat_ests,
            wb,
            effective_ooi_opt,
            fp,
            num_threads);
        out.emplace(key, std::move(res));
    }

    return out;
}

} // namespace apm