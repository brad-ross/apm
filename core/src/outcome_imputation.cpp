#include "outcome_imputation.h"
#include "outcome_imputation_helpers.h"
#include "linear_algebra_utils.h"
#include <stdexcept>
#include <limits>
#include <unordered_map>
#include <utility>
#include <iostream>

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

std::pair<arma::vec, std::optional<arma::mat>> apm::internal::comp_unit_and_outcome_specific_params(
    const arma::vec& g_0_prev,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
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
    if (g_0_prev.n_elem != static_cast<arma::uword>(T)) {
        throw std::invalid_argument("g_0_prev has incompatible length with panel.T()");
    }

    arma::vec g_0_in_progress(static_cast<arma::uword>(T), arma::fill::zeros);
    arma::uvec num_units_per_outcome(static_cast<arma::uword>(T), arma::fill::zeros);

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
        arma::vec g_0c_prev = g_0_prev.elem(T_idxs);         // T_c
        const auto& pos_map = panel.pos_T_idx_for_cohort(cohort0);

        for (const auto& ur : blk.unit_runs) {
            arma::vec w_i = assemble_w_i(ur, T_idxs, pos_map, panel, var, Y, X_obs);

            // Solve for lambda_i using residual against previous g_0
            arma::vec r_for_lambda = w_i - g_0c_prev;
            arma::vec lambda_i = internal::min_norm_solve(G_c, r_for_lambda); // r x 1

            if (store_unit_params) {
                // ur.unit is 0-based index of the unit
                (*lambda_opt).row(static_cast<arma::uword>(ur.unit)) = lambda_i.t();
            }

            // Residual used to update g_0
            arma::vec r_i = w_i - (G_c * lambda_i); // T_c x 1

            arma::uvec counts_u = num_units_per_outcome.elem(T_idxs);
            arma::uvec new_counts_u = counts_u + 1u;
            arma::vec w_old = arma::conv_to<arma::vec>::from(counts_u) / arma::conv_to<arma::vec>::from(new_counts_u);

            arma::vec g_old = g_0_in_progress.elem(T_idxs);
            g_0_in_progress.elem(T_idxs) = w_old % g_old + (1.0 - w_old) % r_i;

            num_units_per_outcome.elem(T_idxs) = new_counts_u;
        }
    }

    // normalize g_0_in_progress to be orthogonal to the rows of G:
    arma::vec g_0_result = g_0_in_progress - G * apm::internal::min_norm_solve(G, g_0_in_progress);
    return std::make_pair(std::move(g_0_result), std::move(lambda_opt));
}

arma::vec apm::internal::comp_outcome_specific_params(
    const arma::vec& g_0_prev,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt)
{
    auto res = apm::internal::comp_unit_and_outcome_specific_params(
        g_0_prev, panel, var, factor_model_params, effective_ooi_opt, /*store_unit_params=*/false);
    return std::move(res.first);
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

arma::vec apm::internal::comp_outcome_specific_params_fixed_point(
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt,
    double tol,
    std::size_t max_iters,
    const std::string& fixed_point_method)
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

    return g_0;
}

} // namespace apm