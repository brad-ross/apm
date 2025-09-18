#include "est_cohort_specific_params.h"
#include <algorithm>
#include <limits>
#include <unordered_set>
#ifdef APM_HAS_TBB
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/global_control.h>
#endif
#include "factor_model_estimators/FactorModelEstimator.h"
#include "factor_model_estimators/pc_estimators.h"
#include "OutcomeMeanSuffStatEstimator.h"
#include "nuisance_param_estimators.h"
#include "panels/InMemoryUnbalancedPanel.h"

namespace apm {
namespace {

//==============================
// Helpers: masked pos map
//==============================

static std::unordered_map<int, std::size_t> make_pos_map(const arma::uvec& idx0) {
    std::unordered_map<int, std::size_t> mp;
    mp.reserve(static_cast<std::size_t>(idx0.n_elem) * 2);
    for (arma::uword k = 0; k < idx0.n_elem; ++k)
        mp.emplace(static_cast<int>(idx0[k]), static_cast<std::size_t>(k));
    return mp;
}

//==============================
// Helpers: estimator factories
//==============================
static std::unordered_map<std::string, std::unique_ptr<FactorModelEstimator>>
make_factor_estimators_for_cohort(
    const std::unordered_map<std::string, EstimatorSpecification>& specs,
    std::size_t T_c,
    std::size_t q,
    const std::shared_ptr<const WeightedBootstrap>& bootstrap)
{
    std::unordered_map<std::string, std::unique_ptr<FactorModelEstimator>> out;
    out.reserve(specs.size());
    for (const auto& kv : specs) {
        const auto& sp = kv.second;
        if (sp.factor_model_estimator != "principal_components") {
            throw std::invalid_argument("Only 'principal_components' is supported");
        }
        if (sp.include_outcome_fes) {
            out.emplace(kv.first, std::make_unique<PCEstimatorWithFEs>(sp.r, T_c, bootstrap, q));
        } else {
            out.emplace(kv.first, std::make_unique<PCEstimator>(sp.r, T_c, bootstrap, q));
        }
    }
    return out;
}

//==============================
// Helpers: finalize outputs
//==============================

static std::vector<CohortAuxiliaryDataMeanEstimates> finalize_auxiliary_outputs(
    const std::vector<std::optional<CohortAuxiliaryDataMeanEstimator>>& tmp_aux,
    std::size_t total_units)
{
    std::vector<CohortAuxiliaryDataMeanEstimates> aux_out;
    const std::size_t C = tmp_aux.size();
    if (C == 0) return aux_out;

    aux_out.reserve(C);
    for (std::size_t c = 0; c < C; ++c) {
        aux_out.emplace_back(tmp_aux[c]->estimate(total_units));
    }
    return aux_out;
}

static CohortSpecificEstimates build_cohort_specific_estimates(
    std::unordered_map<std::string, std::vector<std::optional<FactorModelEstimates>>>& tmp_factor,
    std::vector<std::optional<OutcomeMeanSuffStatEstimates>>& tmp_outcome,
    std::unordered_map<std::string, CohortWeightEstimates> cohort_weights,
    std::vector<CohortAuxiliaryDataMeanEstimates>&& aux_out,
    const std::optional<ObservedOutcomeIndices>& masked_indices_opt,
    std::unordered_map<int, OutcomeMeanSufficientStatistics>&& masked_means)
{
    const std::size_t C = tmp_outcome.size();

    CohortSpecificEstimates out;
    out.cohort_outcome_mean_ests.reserve(C);
    for (std::size_t i = 0; i < C; ++i) {
        out.cohort_outcome_mean_ests.push_back(std::move(*tmp_outcome[i]));
    }

    for (auto& kv : tmp_factor) {
        auto& name = kv.first;
        auto& vec_opt = kv.second;
        std::vector<FactorModelEstimates> vec;
        vec.reserve(C);
        for (std::size_t i = 0; i < C; ++i) vec.push_back(std::move(*vec_opt[i]));
        out.cohort_specific_factor_ests.emplace(name, std::move(vec));
    }

    // Attach cohort weights
    out.cohort_weights = std::move(cohort_weights);
    // Attach auxiliary outputs
    out.cohort_auxiliary_means = std::move(aux_out);

    // Attach masked results when provided
    if (masked_indices_opt.has_value()) {
        out.masked_observed_outcome_indices = masked_indices_opt;
        out.masked_cohort_outcome_means = std::move(masked_means);
    }
    return out;
}

static std::unordered_map<int, OutcomeMeanSufficientStatistics> build_masked_means_map(
    const std::vector<CohortBlock>& blocks,
    const std::vector<std::optional<OutcomeMeanSufficientStatistics>>& tmp_masked_means)
{
    std::unordered_map<int, OutcomeMeanSufficientStatistics> out;
    const std::size_t C = tmp_masked_means.size();
    out.reserve(C);
    for (std::size_t cidx = 0; cidx < C; ++cidx) {
        if (tmp_masked_means[cidx].has_value()) {
            out.emplace(blocks[cidx].cohort, *tmp_masked_means[cidx]);
        }
    }
    return out;
}

//==============================
// Helpers: cohort weights
//==============================

static std::unordered_map<std::string, CohortWeightEstimates> est_cohort_weights(
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    const std::vector<CohortBlock>& blocks,
    const std::shared_ptr<const WeightedBootstrap>& bootstrap)
{
    const std::size_t C = blocks.size();

    // Build per-cohort unique unit indices
    std::vector<arma::uvec> cohort_unit_idxs;
    cohort_unit_idxs.reserve(C);
    for (const auto& blk : blocks) {
        arma::uvec idxs(blk.unit_runs.size());
        for (std::size_t i = 0; i < blk.unit_runs.size(); ++i) {
            idxs(i) = static_cast<arma::uword>(blk.unit_runs[i].unit);
        }
        cohort_unit_idxs.push_back(std::move(idxs));
    }

    const bool any_by_size = std::any_of(
        est_specs.begin(), est_specs.end(),
        [](const auto& kv){ return kv.second.cohort_weighting == "by_size"; }
    );

    // Optional precompute of S (C x B) only if bootstrap present and needed
    std::size_t B = bootstrap ? bootstrap->n_bootstraps() : 0;
    arma::mat S;
    if (bootstrap && any_by_size) {
        S.set_size(C, B);
        for (std::size_t c = 0; c < C; ++c) {
            arma::mat Wc = bootstrap->obs(cohort_unit_idxs[c]); // n_c x B
            S.row(static_cast<arma::uword>(c)) = arma::sum(Wc, 0);
        }
        // Each column already sums to 1 across cohorts
    }

    // Build per-spec CohortWeightEstimates
    std::unordered_map<std::string, CohortWeightEstimates> out;
    out.reserve(est_specs.size());
    for (const auto& kv : est_specs) {
        const auto& name = kv.first;
        const auto& sp = kv.second;
        CohortWeightEstimates w;
        if (sp.cohort_weighting == "equal") {
            w.cohort_weights = arma::ones(C) / static_cast<double>(C);
            if (B > 0) {
                w.bootstrap_cohort_weights.reserve(B);
                for (std::size_t b = 0; b < B; ++b) {
                    w.bootstrap_cohort_weights.emplace_back(w.cohort_weights);
                }
            }
        } else if (sp.cohort_weighting == "by_size") {
            if (bootstrap && any_by_size) {
                w.cohort_weights = arma::mean(S, 1);
                w.bootstrap_cohort_weights.reserve(B);
                for (std::size_t b = 0; b < B; ++b) {
                    w.bootstrap_cohort_weights.emplace_back(S.col(static_cast<arma::uword>(b)));
                }
            } else {
                arma::vec counts(C, arma::fill::zeros);
                for (std::size_t c = 0; c < C; ++c) counts(c) = static_cast<double>(cohort_unit_idxs[c].n_elem);
                const double tot = arma::accu(counts);
                if (tot > 0.0) {
                    w.cohort_weights = counts / tot;
                } else {
                    w.cohort_weights = arma::ones(C) / static_cast<double>(C);
                }
            }
        } else {
            throw std::invalid_argument("EstimatorSpecification.cohort_weighting must be 'equal' or 'by_size'");
        }
        out.emplace(name, std::move(w));
    }

    return out;
}

} // anonymous namespace

//==============================
// Main entry (panel-based)
//==============================

CohortSpecificEstimates estimate_cohort_specific_params_from_internal_panel_rep(
    const InMemoryUnbalancedPanel& panel,
    const std::unordered_map<std::string, EstimatorSpecification>& est_specs,
    std::shared_ptr<const WeightedBootstrap> bootstrap,
    std::optional<std::size_t> num_threads,
    const CohortOutcomeMask& cohort_outcomes_to_mask)
{
    const std::size_t q = panel.q();
    const std::size_t d = panel.d();

#ifdef APM_HAS_TBB
    std::size_t nt = num_threads.has_value() ? *num_threads : oneapi::tbb::info::default_concurrency();
    std::unique_ptr<oneapi::tbb::global_control> tbb_gc;
    if (nt > 1) {
        tbb_gc = std::make_unique<oneapi::tbb::global_control>(
            oneapi::tbb::global_control::max_allowed_parallelism,
            static_cast<std::size_t>(nt)
        );
    }
#else
    std::size_t nt = num_threads.has_value() ? *num_threads : 1;
#endif

    // Determine effective observed outcome indices after optional masking
    const bool has_mask = !cohort_outcomes_to_mask.empty();
    ObservedOutcomeIndices ooi_effective = get_masked_observed_outcome_indices(panel.observed_outcome_indices(), cohort_outcomes_to_mask);

    const std::size_t T = panel.T();
    const std::size_t C = panel.cohort_blocks().size();

    // Prepare thread-safe output buffers
    std::unordered_map<std::string, std::vector<std::optional<FactorModelEstimates>>> tmp_factor;
    tmp_factor.reserve(est_specs.size());
    for (const auto& kv : est_specs) {
        tmp_factor.emplace(kv.first, std::vector<std::optional<FactorModelEstimates>>(C));
    }
    std::vector<std::optional<OutcomeMeanSuffStatEstimates>> tmp_outcome(C);
    std::vector<std::optional<CohortAuxiliaryDataMeanEstimator>> tmp_aux(C);
    std::vector<std::optional<OutcomeMeanSufficientStatistics>> tmp_masked_means(C);

    // Parallelize across cohorts
    auto process_cohort = [&](std::size_t cidx) {
        const CohortBlock& blk = panel.cohort_blocks()[cidx];

        // Cohort observed outcome indices (respect masking via ooi_effective)
        const arma::uvec& T_idxs_for_cohort = ooi_effective.at(blk.cohort);
        const std::size_t T_c = static_cast<std::size_t>(T_idxs_for_cohort.n_elem);

        // Estimators
        OutcomeMeanSuffStatEstimator omsse(
            T_c,
            /*T=*/(q > 0 ? T : 0),
            /*q=*/q,
            bootstrap
        );
        auto ests = make_factor_estimators_for_cohort(est_specs, T_c, q, bootstrap);
        std::optional<CohortAuxiliaryDataMeanEstimator> aux_est;
        if (d > 0) aux_est.emplace(T, d, bootstrap);

        // Reusable buffers (pre-sized once per cohort)
        arma::vec Y(static_cast<arma::uword>(T_c));
        arma::mat X_full;
        arma::mat X_obs;
        if (q > 0) {
            X_full.set_size(static_cast<arma::uword>(T), static_cast<arma::uword>(q));
            X_obs.set_size(static_cast<arma::uword>(T_c), static_cast<arma::uword>(q));
        }

        // Optional masked outcomes estimator for this cohort (computed in the same pass)
        bool compute_masked = false;
        arma::uvec masked_T_idxs;
        std::optional<OutcomeMeanSuffStatEstimator> omsse_masked;
        arma::vec Y_mask;
        if (has_mask) {
            auto itM = cohort_outcomes_to_mask.find(blk.cohort);
            if (itM != cohort_outcomes_to_mask.end() && itM->second.n_elem > 0) {
                compute_masked = true;
                masked_T_idxs = itM->second;
                const std::size_t K_mask = static_cast<std::size_t>(itM->second.n_elem);
                omsse_masked.emplace(K_mask, /*T=*/0, /*q=*/0, bootstrap);
                Y_mask.set_size(static_cast<arma::uword>(K_mask));
            }
        }

        // Stream units
        for (const auto& ur : blk.unit_runs) {
            panel.assemble_YX_for_unit(
                ur,
                T,
                T_idxs_for_cohort,
                make_pos_map(T_idxs_for_cohort),
                Y, X_full, X_obs);

            // Add datum to factor estimators
            for (auto& kv : ests) {
                if (q > 0) {
                    kv.second->add_datum(static_cast<std::size_t>(ur.unit), Y, X_obs);
                } else {
                    kv.second->add_datum(static_cast<std::size_t>(ur.unit), Y);
                }
            }

            // Add datum to outcome mean estimator
            if (q > 0) {
                omsse.add_datum(static_cast<std::size_t>(ur.unit), Y, X_full);
            } else {
                omsse.add_datum(static_cast<std::size_t>(ur.unit), Y);
            }

            // Add datum to auxiliary data mean estimator
            if (d > 0) {
                arma::mat A;
                panel.assemble_aux_for_unit(ur, T, A);
                aux_est->add_datum(static_cast<std::size_t>(ur.unit), A);
            }

            // Also accumulate masked outcome means if requested
            if (compute_masked) {
                auto pos_mask = make_pos_map(masked_T_idxs);
                panel.assemble_Y_for_unit(ur, masked_T_idxs, pos_mask, Y_mask);
                omsse_masked->add_datum(static_cast<std::size_t>(ur.unit), Y_mask);
            }

            // Reset buffers for next unit
            Y.fill(std::numeric_limits<double>::quiet_NaN());
            if (q > 0) {
                X_full.fill(std::numeric_limits<double>::quiet_NaN());
                X_obs.fill(std::numeric_limits<double>::quiet_NaN());
            }
        }

        if (compute_masked) {
            tmp_masked_means[cidx].emplace(omsse_masked->estimate().suff_stat_estimates);
        }

        // Estimate and store
        tmp_outcome[cidx].emplace(omsse.estimate());
        for (auto& kv : ests) {
            tmp_factor[kv.first][cidx].emplace(kv.second->estimate());
        }
        if (d > 0) {
            tmp_aux[cidx].emplace(std::move(*aux_est));
        }
    };

    // Parallelize across cohorts if requested; otherwise use serial loop
#ifdef APM_HAS_TBB
    if (nt <= 1) {
        for (std::size_t cidx = 0; cidx < C; ++cidx) {
            process_cohort(cidx);
        }
    } else {
        oneapi::tbb::parallel_for(std::size_t(0), C, [&](std::size_t cidx) {
            process_cohort(cidx);
        });
    }
#else
    for (std::size_t cidx = 0; cidx < C; ++cidx) {
        process_cohort(cidx);
    }
#endif

    // Compute cohort weights first, then build outputs with weights attached
    auto weights_by_spec = est_cohort_weights(est_specs, panel.cohort_blocks(), bootstrap);

    // Finalize auxiliary outputs
    std::vector<CohortAuxiliaryDataMeanEstimates> aux_out;
    if (d > 0) {
        // Compute total number of unique units across all cohorts
        std::size_t total_units = 0;
        for (const auto& blk : panel.cohort_blocks()) {
            total_units += blk.unit_runs.size();
        }
        aux_out = finalize_auxiliary_outputs(tmp_aux, total_units);
    }

    // Build masked means map if any
    std::unordered_map<int, OutcomeMeanSufficientStatistics> masked_means_map;
    std::optional<ObservedOutcomeIndices> masked_indices_opt = std::nullopt;
    if (has_mask) {
        masked_means_map = build_masked_means_map(panel.cohort_blocks(), tmp_masked_means);
        masked_indices_opt = ooi_effective;
    }

    return build_cohort_specific_estimates(
        tmp_factor,
        tmp_outcome,
        std::move(weights_by_spec),
        std::move(aux_out),
        masked_indices_opt,
        std::move(masked_means_map)
    );
}

} // namespace apm