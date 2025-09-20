#include "outcome_imputation.h"
#include "outcome_imputation_helpers.h"
#include "linear_algebra_utils.h"
#include <stdexcept>
#include <limits>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <vector>

namespace apm {

namespace {

static arma::vec assemble_w_i(
	const UnitRun& ur,
	const arma::uvec& T_idxs,
	const std::unordered_map<int, std::size_t>& pos_map,
	const InMemoryUnbalancedPanel& panel,
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

std::pair<std::optional<arma::vec>, std::optional<arma::mat>> apm::internal::comp_unit_and_outcome_specific_params(
    std::optional<arma::vec> g_0_prev_opt,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
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

    arma::vec g_0_in_progress;
    arma::uvec num_units_per_outcome;
    if (update_g0) {
        g_0_in_progress.set_size(static_cast<arma::uword>(T));
        g_0_in_progress.fill(0.0);
        num_units_per_outcome.set_size(static_cast<arma::uword>(T));
        num_units_per_outcome.fill(0u);
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

            // Residual used to update g_0
            if (update_g0) {
            arma::vec r_i = w_i - (G_c * lambda_i); // T_c x 1

            arma::uvec counts_u = num_units_per_outcome.elem(T_idxs);
            arma::uvec new_counts_u = counts_u + 1u;
            arma::vec w_old = arma::conv_to<arma::vec>::from(counts_u) / arma::conv_to<arma::vec>::from(new_counts_u);

            arma::vec g_old = g_0_in_progress.elem(T_idxs);
            g_0_in_progress.elem(T_idxs) = w_old % g_old + (1.0 - w_old) % r_i;

            num_units_per_outcome.elem(T_idxs) = new_counts_u;
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

arma::vec apm::internal::comp_outcome_specific_params(
    const arma::vec& g_0_prev,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt)
{
    auto res = apm::internal::comp_unit_and_outcome_specific_params(
        std::optional<arma::vec>(g_0_prev), panel, var, factor_model_params, effective_ooi_opt, std::nullopt, /*store_unit_params=*/false);
    if (!res.first.has_value()) {
        throw std::runtime_error("Expected g_0 in comp_outcome_specific_params result");
    }
    return std::move(*res.first);
}

namespace {

static arma::vec comp_vanilla_fixed_point_update(
    const arma::vec& g_0,
    const arma::vec& g_1,
    const arma::vec& /*r_prev*/, bool /*have_prev*/)
{
    (void)g_0;
    return g_1;
}

static arma::vec comp_irons_tuck_fixed_point_update(
    const arma::vec& g_0,
    const arma::vec& g_1,
    const arma::vec& r_prev,
    bool have_prev)
{
    arma::vec r_k = g_1 - g_0;
    if (!have_prev) {
        return g_1;
    }
    arma::vec dr = r_k - r_prev;
    double denom = arma::dot(dr, dr);
    if (denom <= 0.0) {
        return g_1;
    }
    double omega = - arma::dot(r_k, dr) / denom;
    if (!std::isfinite(omega)) {
        return g_1;
    }
    if (omega < 0.0) omega = 0.0;
    if (omega > 2.0) omega = 2.0;
    return g_0 + omega * r_k;
}

} // anonymous namespace

std::pair<std::optional<arma::vec>, std::optional<arma::mat>> apm::internal::comp_unit_and_outcome_specific_params_fixed_point(
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    double tol,
    std::size_t max_iters,
    const std::string& fixed_point_method,
    std::optional<arma::vec> covar_coefs_for_residualization,
    bool store_unit_params)
{
    const std::size_t T = panel.T();

    if (var.kind == VariableSpec::Kind::Covariate && var.covariate_index >= panel.q()) {
        throw std::invalid_argument("covariate_index out of range for panel.q()");
    }

    arma::vec g_0(static_cast<arma::uword>(T), arma::fill::zeros);
    arma::vec r_prev;
    bool have_prev = false;
    bool converged = false;

    std::size_t iter = 0;
    for (; iter < max_iters; ++iter) {
        arma::vec g_1 = apm::internal::comp_outcome_specific_params(g_0, panel, var, factor_model_params, effective_ooi_opt);

        arma::vec r_k = g_1 - g_0;
        if (arma::norm(r_k, "inf") <= tol) {
            g_0 = std::move(g_1);
            converged = true;
            break;
        }

        arma::vec g_next;
        if (fixed_point_method == "irons-tuck") {
            g_next = comp_irons_tuck_fixed_point_update(g_0, g_1, r_prev, have_prev);
        } else {
            g_next = comp_vanilla_fixed_point_update(g_0, g_1, r_prev, have_prev);
        }

        r_prev = std::move(r_k);
        have_prev = true;
        g_0 = std::move(g_next);
    }
    
    if (iter == max_iters) {
        std::cerr << "Warning: comp_unit_and_outcome_specific_params_vanilla_fixed_point did not converge within max_iters="
                  << max_iters << ", tol=" << tol << std::endl;
    }

    return apm::internal::comp_unit_and_outcome_specific_params(
        std::optional<arma::vec>(g_0), panel, var, factor_model_params, effective_ooi_opt, covar_coefs_for_residualization, store_unit_params);
}

arma::vec apm::internal::comp_outcome_specific_params_fixed_point(
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    double tol,
    std::size_t max_iters,
    const std::string& fixed_point_method)
{
    auto res = apm::internal::comp_unit_and_outcome_specific_params_fixed_point(
        panel, var, factor_model_params, effective_ooi_opt, tol, max_iters, fixed_point_method, std::nullopt, /*store_unit_params=*/false);
    if (!res.first.has_value()) {
        throw std::runtime_error("Expected g_0 in comp_outcome_specific_params_fixed_point result");
    }
    return std::move(*res.first);
}

namespace internal {

std::optional<arma::mat> comp_unit_specific_params(
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    std::optional<arma::vec> covar_coefs_for_residualization)
{
    auto res = apm::internal::comp_unit_and_outcome_specific_params(
        std::optional<arma::vec>(), panel, var, factor_model_params, effective_ooi_opt, covar_coefs_for_residualization, /*store_unit_params=*/true);
    return std::move(res.second);
}

arma::vec comp_covar_coefs(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    std::optional<arma::vec> g_0_init,
    std::vector<arma::vec> g_0_init_covars,
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

    arma::mat XTX(static_cast<arma::uword>(q), static_cast<arma::uword>(q), arma::fill::zeros);
    arma::vec XTy(static_cast<arma::uword>(q), arma::fill::zeros);

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
                XTX += xk.t() * xk;
                XTy += xk.t() * yk;
            }
        }
    }

    arma::vec alpha = ::apm::internal::min_norm_solve(XTX, XTy);
    return alpha;
}

} // namespace internal

// High-level orchestration
FactorModelParameters comp_imputation_components(
    const InMemoryUnbalancedPanel& panel,
    const FactorModelParameters& factor_model_params,
    const std::vector<OutcomeMeanSufficientStatistics>& /*cohort_outcome_mean_suff_stats*/,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    double tol,
    std::size_t max_iters,
    const std::string& fixed_point_method)
{
    std::optional<arma::vec> alpha_opt;
    std::optional<arma::vec> g_0_init_opt;
    std::vector<arma::vec> g_0_init_covars;

    if (factor_model_params.has_fixed_effects()) {
        arma::vec g_0_init = apm::internal::comp_outcome_specific_params_fixed_point(
            panel, VariableSpec::outcome(), factor_model_params, effective_ooi_opt, tol, max_iters, fixed_point_method);
        g_0_init_opt = g_0_init;

        if (factor_model_params.has_covariate_coefs() && factor_model_params.q() > 0) {
            const std::size_t q = factor_model_params.q();
            g_0_init_covars.resize(q);
            for (std::size_t j = 0; j < q; ++j) {
                g_0_init_covars[j] = apm::internal::comp_outcome_specific_params_fixed_point(
                    panel, VariableSpec::covariate(j), factor_model_params, effective_ooi_opt, tol, max_iters, fixed_point_method);
            }
        }

        if (factor_model_params.has_covariate_coefs() && factor_model_params.q() > 0) {
            arma::vec alpha = apm::internal::comp_covar_coefs(panel, factor_model_params, g_0_init_opt, g_0_init_covars, effective_ooi_opt);
            alpha_opt = std::move(alpha);
        }

        auto final_pair = apm::internal::comp_unit_and_outcome_specific_params_fixed_point(
            panel, VariableSpec::outcome(), factor_model_params, effective_ooi_opt, tol, max_iters, fixed_point_method, alpha_opt, /*store_unit_params=*/true);

        FactorModelParameters out;
        out.G = factor_model_params.G;
        if (final_pair.first.has_value()) out.g_0 = *final_pair.first; else out.g_0 = std::nullopt;
        out.a = alpha_opt;
        out.L = std::move(final_pair.second);
        return out;
    } else {
        if (factor_model_params.has_covariate_coefs() && factor_model_params.q() > 0) {
            arma::vec alpha = apm::internal::comp_covar_coefs(panel, factor_model_params, std::nullopt, {}, effective_ooi_opt);
            alpha_opt = std::move(alpha);
        }

        auto L_opt = apm::internal::comp_unit_specific_params(
            panel, VariableSpec::outcome(), factor_model_params, effective_ooi_opt, alpha_opt);

        FactorModelParameters out;
        out.G = factor_model_params.G;
        out.a = alpha_opt;
        out.g_0 = std::nullopt;
        out.L = std::move(L_opt);
        return out;
    }
}

} // namespace apm