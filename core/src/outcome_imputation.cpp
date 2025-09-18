#include "outcome_imputation.h"
#include "linear_algebra_utils.h"
#include <stdexcept>
#include <limits>
#include <unordered_map>

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

arma::mat comp_unit_specific_params(
    const arma::vec& g_0,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt)
{
    const ObservedOutcomeIndices& ooi = effective_ooi_opt ? *effective_ooi_opt
                                                         : panel.observed_outcome_indices();

    const std::size_t r = static_cast<std::size_t>(factor_model_params.G.n_cols);
    const std::size_t N = panel.num_units();

    if (var.kind == VariableSpec::Kind::Covariate && var.covariate_index >= panel.q()) {
        throw std::invalid_argument("covariate_index out of range for panel.q()");
    }

    arma::mat lambda(static_cast<arma::uword>(N), static_cast<arma::uword>(r), arma::fill::zeros);

    for (const auto& blk : panel.cohort_blocks()) {
        const auto cohort0 = blk.cohort;
        const arma::uvec& T_idxs = ooi.at(cohort0);
        const std::size_t T_c = static_cast<std::size_t>(T_idxs.n_elem);
        if (T_c == 0) continue;

        arma::mat G_c = factor_model_params.G.rows(T_idxs); // T_c x r
        arma::vec g_0c = g_0.elem(T_idxs);                  // T_c

		arma::vec Y;            // reused for outcomes
		arma::mat X_obs;        // reused for covariate column
        const auto& pos_map = panel.pos_T_idx_for_cohort(cohort0);

        for (const auto& ur : blk.unit_runs) {
			arma::vec w_i = assemble_w_i(ur, T_idxs, pos_map, panel, var, Y, X_obs);

            arma::vec r_i = w_i - g_0c;
            arma::vec lambda_i = internal::min_norm_solve(G_c, r_i);
            lambda.row(static_cast<arma::uword>(ur.unit)) = lambda_i.t();
        }
    }

    return lambda;
}

arma::vec comp_outcome_specific_params(
    const arma::mat& lambda,
    const InMemoryUnbalancedPanel& panel,
    const VariableSpec& var,
    const FactorModelParameters& factor_model_params,
    std::optional<ObservedOutcomeIndices> effective_ooi_opt)
{
    const ObservedOutcomeIndices& ooi = effective_ooi_opt ? *effective_ooi_opt
                                                         : panel.observed_outcome_indices();

    const std::size_t T = panel.T();
    const std::size_t r = static_cast<std::size_t>(factor_model_params.G.n_cols);
    const std::size_t N = panel.num_units();

    if (var.kind == VariableSpec::Kind::Covariate && var.covariate_index >= panel.q()) {
        throw std::invalid_argument("covariate_index out of range for panel.q()");
    }
    if (lambda.n_rows != static_cast<arma::uword>(N) || lambda.n_cols != static_cast<arma::uword>(r)) {
        throw std::invalid_argument("lambda has incompatible dimensions with panel.num_units() or G.n_cols");
    }

    arma::vec g_0_in_progress(static_cast<arma::uword>(T), arma::fill::zeros);
    arma::uvec num_units_per_outcome(static_cast<arma::uword>(T), arma::fill::zeros);

    arma::vec Y;
    arma::mat X_obs;

    for (const auto& blk : panel.cohort_blocks()) {
        const auto cohort0 = blk.cohort;
        const arma::uvec& T_idxs = ooi.at(cohort0);
        const std::size_t T_c = static_cast<std::size_t>(T_idxs.n_elem);
        if (T_c == 0) continue;

        arma::mat G_c = factor_model_params.G.rows(T_idxs); // T_c x r
        const auto& pos_map = panel.pos_T_idx_for_cohort(cohort0);

		for (const auto& ur : blk.unit_runs) {
			arma::vec w_i = assemble_w_i(ur, T_idxs, pos_map, panel, var, Y, X_obs);

            arma::vec lambda_i = lambda.row(static_cast<arma::uword>(ur.unit)).t(); // r x 1
            arma::vec r_i = w_i - (G_c * lambda_i);                                  // T_c x 1

            // Online convex average over outcomes in this cohort
            arma::uvec counts_u = num_units_per_outcome.elem(T_idxs);
            arma::uvec new_counts_u = counts_u + 1u;
            arma::vec w_old = arma::conv_to<arma::vec>::from(counts_u) / arma::conv_to<arma::vec>::from(new_counts_u);

            arma::vec g_old = g_0_in_progress.elem(T_idxs);
            g_0_in_progress.elem(T_idxs) = w_old % g_old + (1.0 - w_old) % r_i;

            // Increment counts
            num_units_per_outcome.elem(T_idxs) = new_counts_u;
        }
    }

    return g_0_in_progress;
}

} // namespace apm
