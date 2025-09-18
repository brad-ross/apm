#include "outcome_imputation.h"
#include "linear_algebra_utils.h"
#include <stdexcept>
#include <limits>

namespace apm {

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
            arma::vec w_i;

            if (var.kind == VariableSpec::Kind::Outcome) {
                panel.assemble_Y_for_unit(ur, T_idxs, pos_map, Y);
                w_i = Y;
            } else {
                panel.assemble_X_for_unit(ur, panel.T(), T_idxs, /*X_full=*/nullptr, /*X_obs=*/&X_obs, var.covariate_index);
                w_i = X_obs.col(0);
            }

            arma::vec r_i = w_i - g_0c;
            arma::vec lambda_i = internal::min_norm_solve(G_c, r_i);
            lambda.row(static_cast<arma::uword>(ur.unit)) = lambda_i.t();
        }
    }

    return lambda;
}

} // namespace apm
